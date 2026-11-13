# The model for hyperelastic materials on truss elements (truss_element_line2.py).
#
# A truss carries one force, along its axis, so its material law is 1D (material_dim=1, the default):
#     cal_1pk_stress(stretch, material, create_graph, return_W)
#         stretch.shape (M,): deformed length / undeformed length of each truss, M is the number of elements
#         material.shape (M,?) or (1,?)
#         return: S, the 1PK stress along the truss (force per undeformed area), shape (M,)
#                 and W, the strain energy per undeformed volume, shape (M,), when return_W is True
#     S is the derivative of W with respect to stretch.
# The force along a truss is S*area with the area of the undeformed cross-section, so the model does not need to
# know how the cross-section changes.
#
# One 1D law is in this file: cal_1pk_stress_linear_truss, FEBio's linear truss material.
#
# A 3D material (material_dim=3, the default is material_dim=1) can be used in place of a 1D law, as FEBio's "elastic-truss"
# domain does: the law of the solid models, cal_1pk_stress(F, material, create_graph, return_W), takes
#     F=diag(stretch, stretch^(-v), stretch^(-v)), shape (M,1,3,3), in the axes of the truss (axis 1 along the truss)
# and returns S, shape (M,1,3,3), and W, shape (M,1). v is chosen by the caller: the cross-section changes by stretch^(-v)
# in each direction. The force is the derivative of the strain energy of W(F), but the truss is in uniaxial tension only when
# v is the lateral contraction of the material: exactly for an incompressible isotropic material (NeoHookean, ...) and
# v=0.5; a compressible material contracts less, and its 1D law is better (see truss_element_line2.py).
# A fibre model of a tissue wall (GOH, UFD) is not for a truss, which is thin and carries one axial force: a fibrous truss
# takes a 1D law of the stretch.
# V is not given to the law (one integration point: the mean dilatation is J).
# The stiffness matrix functions are in model_stiffness.py.
#
# Example:
#     material=torch.tensor([[E]], dtype=dtype, device=device)
#     force, stretch, S, W, H=cal_1pk_stress_force(node_x, element, node_X, area, material, cal_1pk_stress_linear_truss,
#                                                  return_stretch_S_W=True, return_stiffness="sparse")
#     energy=cal_strain_energy(W, node_X, element, area).sum()
# Example with a 3D material:
#     material=torch.tensor([[C10, D1]], dtype=dtype, device=device)
#     force, stretch, S, W=cal_1pk_stress_force(node_x, element, node_X, area, material, material_NeoHookean.cal_1pk_stress,
#                                               return_stretch_S_W=True, material_dim=3, v=0.5)
#%%
import torch
import torch_scatter
import torch_fea.element.truss_element_line2 as truss_element_line2
from torch_fea.model.model_stiffness import cal_dense_stiffness_matrix, cal_sparse_stiffness_matrix, cal_diagonal_stiffness_matrix
#%% 1D material law: FEBio's linear truss material (FELinearTrussMaterial)
def cal_strain_energy_density_linear_truss(stretch, material):
    #stretch.shape (M,)
    #material.shape (M,1) or (1,1): E, Young's modulus
    #W=0.5*E*log(stretch)^2, strain energy per undeformed volume
    E=material[:,0]
    log_stretch=torch.log(stretch)
    W=0.5*E*log_stretch*log_stretch
    return W #(M,)
#%%
def cal_1pk_stress_linear_truss(stretch, material, create_graph, return_W):
    #stretch.shape (M,)
    #material.shape (M,1) or (1,1): E, Young's modulus
    #create_graph: not used, the stress is written out and stays a function of stretch;
    #    it is an input so that every 1D law has the same inputs (a law may take the derivative of W by autograd)
    #Kirchhoff stress tau=E*log(stretch) in FEBio; 1PK stress S=tau/stretch, which is dW/d(stretch)
    E=material[:,0]
    log_stretch=torch.log(stretch)
    S=E*log_stretch/stretch
    #S.shape is (M,)
    if return_W == True:
        W=0.5*E*log_stretch*log_stretch
        return S, W
    else:
        return S
#%%
def cal_stretch(node_x, element, node_X):
    #node_x.shape (N,3), N is the number of nodes, x is deformed
    #element.shape (M,2), M is the number of elements, the two nodes of each truss
    #node_X.shape (N,3), X is undeformed
    if element.shape[1] != 2:
        raise ValueError("only support the truss element with 2 nodes")
    stretch=truss_element_line2.cal_stretch(node_x[element], node_X[element])
    return stretch #(M,)
#%% forward: node_x is unknown, node_X is known
def cal_1pk_stress_force(node_x, element, node_X, area, material, cal_1pk_stress=cal_1pk_stress_linear_truss,
                         return_stretch_S_W=False, return_stiffness=None, return_force_at_element=False,
                         material_dim=1, v=0.5):
    #node_x.shape (N,3), N is the number of nodes, x is deformed
    #element.shape (M,2), M is the number of elements, the two nodes of each truss
    #node_X.shape (N,3), X is undeformed
    #area: cross-section area of the undeformed truss, a number or a tensor of shape (M,) or (M,1)
    #material.shape (M,?) or (1,?)
    #cal_1pk_stress: the material law, see the top of this file
    #material_dim: 1, cal_1pk_stress is a 1D law, S.shape (M,), W.shape (M,);
    #              3, cal_1pk_stress is a 3D law, S.shape (M,1,3,3), W.shape (M,1)
    #v: used only for material_dim=3, F=diag(stretch, stretch^(-v), stretch^(-v))
    if element.shape[1] != 2:
        raise ValueError("only support the truss element with 2 nodes")
    if material_dim not in [1, 3]:
        raise ValueError("material_dim must be 1 or 3")
    x=node_x[element].requires_grad_(True)
    X=node_X[element]
    stretch=truss_element_line2.cal_stretch(x, X)
    #stretch.shape: (M,)
    if material_dim == 1:
        S, W=cal_1pk_stress(stretch, material, create_graph=True, return_W=True)
        #S.shape: (M,), W.shape: (M,)
    else:
        F=truss_element_line2.cal_F_tensor(x, X, v)
        S, W=cal_1pk_stress(F, material, create_graph=True, return_W=True)
        #S.shape: (M,1,3,3), W.shape: (M,1)
    force_element=truss_element_line2.cal_nodal_force_from_1pk_stress(S, x, X, area, material_dim, v)
    #force_element.shape: (M,2,3)
    #-----------------------------------------------------------------------------
    N=node_x.shape[0]
    #int64 index: with an int32 index on the CPU, the backward of scatter gives wrong gradient rows (torch 2.12.1)
    force = torch_scatter.scatter(force_element.reshape(-1,3), element.reshape(-1).to(torch.int64), dim=0, dim_size=N, reduce="sum")
    #force.shape: (N, 3)
    #-----------------------------------------------------------------------------
    if return_stiffness =="none" or return_stiffness is None:
        return_stiffness = None
    elif return_stiffness == "dense":
        H=cal_dense_stiffness_matrix(N, element, force_element, x)
    elif return_stiffness == "sparse":
        H=cal_sparse_stiffness_matrix(N, element, force_element, x)
    elif return_stiffness == "diagonal":
        H=cal_diagonal_stiffness_matrix(N, element, force_element, x)
    else:
        raise ValueError("return_stiffness unkown")
    #-----------------------------------------------------------------------------
    #out=[force, stretch, S, W, H, force_element]
    out=[force]
    if return_stretch_S_W == True:
        out.extend([stretch,S,W])
    if return_stiffness is not None:
        out.append(H)
    if return_force_at_element == True:
        out.append(force_element)
    return out
#%%
def cal_strain_energy(W, node_X, element, area):
    #W: strain energy per undeformed volume, shape (M,) from a 1D law or (M,1) from a 3D law (material_dim=3)
    #node_X.shape (N,3), X is undeformed
    #element.shape (M,2)
    #area: cross-section area of the undeformed truss, a number or a tensor of shape (M,) or (M,1)
    #return: energy, the strain energy of each element, shape (M,)
    #    energy.sum() is the strain energy of the mesh
    if element.shape[1] != 2:
        raise ValueError("only support the truss element with 2 nodes")
    energy=truss_element_line2.cal_strain_energy(W, node_X[element], area)
    return energy #(M,)
