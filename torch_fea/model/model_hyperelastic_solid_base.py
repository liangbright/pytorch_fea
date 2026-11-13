# The functions of the model for hyperelastic materials on solid elements, with the element functions as inputs.
# model_hyperelastic_solid.py picks the element functions and calls the functions of this file.
# The stiffness matrix functions are in model_stiffness.py, the pressure load is in model_surface.py.
# The material functions always get V, the volume of each integration point, for the mean dilatation:
#     cal_1pk_stress(F, material, create_graph, return_W, V) and cal_cauchy_stress(F, material, create_graph, return_W, V)
# F0 (optional) is the pre/residual deformation gradient, F=Fu*F0, a field stored at the integration points:
#     F0.shape must be (M,K,3,3), the same as F (M elements, K integration points); check_F0 raises ValueError otherwise
#%%
import torch
from torch_fea.utility.small_matmul import matmul #torch.matmul on the CPU, elementwise for small matrices on the GPU
import torch_scatter
from torch_fea.model.model_stiffness import cal_dense_stiffness_matrix, cal_sparse_stiffness_matrix, cal_diagonal_stiffness_matrix
#%%
class PotentialEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, force, u):
        ctx.save_for_backward(force, u)
        energy=(force*u).sum()
        return energy
    @staticmethod
    def backward(ctx, grad_output):
        force, u = ctx.saved_tensors
        grad_force = None #grad_output*u
        grad_u = grad_output*force
        return grad_force, grad_u
#---------------------------------
cal_potential_energy=PotentialEnergy.apply
#%%
def check_F0(F0, shape):
    #F0 is the pre/residual deformation gradient at the integration points, F=Fu*F0
    #shape: the shape F0 must have, (M,K,3,3), the shape of F (M elements, K integration points)
    #F0 is generally not the gradient of a displacement field, so it is one 3x3 tensor per integration point;
    #an F0 per element (M,1,3,3) or for the whole mesh (3,3) has to be expanded to (M,K,3,3) by the caller
    shape=tuple(shape)
    if tuple(F0.shape) != shape:
        raise ValueError("F0.shape is "+str(tuple(F0.shape))+", but it must be "+str(shape)
                         +" (M,K,3,3): one 3x3 tensor per integration point, the same shape as F")
#%%
def cal_F_tensor(node_x, element, node_X, n_integration_points, d_sf_dX, F0,
                 get_integration_point, cal_F_tensor_at_element, cal_F_tensor_with_d_sf_dX_at_element):
    #element.shape: (M, A), M is the number of elements, A is the number of nodes of an element
    #node_X.shape: (N,3), N is the number of nodes, X is undeformed
    #node_x.shape: (N,3), N is the number of nodes, x is deformed
    x=node_x[element]
    X=node_X[element]
    if d_sf_dX is None:
        r=get_integration_point(n_integration_points, dtype=node_x.dtype, device=node_x.device)
        F=cal_F_tensor_at_element(r, x, X)
    else:
        F=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual/pre deformation, F0.shape must be F.shape (M,K,3,3)
        check_F0(F0, F.shape)
        F=matmul(F, F0)
    #F.shape (M,K,3,3)
    return F
#%%
def cal_d_sf_dx_and_dx_dr(node_x, element, n_integration_points,
                          get_integration_point, cal_d_sf_dh):
    x=node_x[element]
    r=get_integration_point(n_integration_points, dtype=node_x.dtype, device=node_x.device)
    d_sf_dx, dx_dr, det_dx_dr = cal_d_sf_dh(r, x)
    return d_sf_dx, dx_dr, det_dx_dr
#%%
def cal_d_sf_dX_and_dX_dr(node_X, element, n_integration_points,
                          get_integration_point, cal_d_sf_dh):
    X=node_X[element]
    r=get_integration_point(n_integration_points, dtype=node_X.dtype, device=node_X.device)
    d_sf_dX, dX_dr, det_dX_dr = cal_d_sf_dh(r, X)
    return d_sf_dX, dX_dr, det_dX_dr
#%%
def cal_integration_point_volume(det_dX_dr, F0, get_integration_weight):
    #V_k = w_k*det(dX/dr) at integration point k: the volume that the force and the strain energy integrate over
    #with F0 (F=F_u*F0) it is the stress-free volume V_k/det(F0_k)
    #det_dX_dr.shape (M,K,1,1) or (M,K)
    #F0.shape (M,K,3,3) or F0 is None
    #V.shape (M,K): the V input of the material functions, theta_e = sum_k V_k*J_k / sum_k V_k
    M=det_dX_dr.shape[0]
    V=det_dX_dr.reshape(M,-1)
    K=V.shape[1]
    V=get_integration_weight(K, dtype=V.dtype, device=V.device).view(1,K)*V
    if F0 is not None:
        check_F0(F0, (M,K,3,3))
        V=V/torch.linalg.det(F0).reshape(M,K)
    return V
#%% forward: node_x is unknown, node_X is known
def cal_cauchy_stress_force(node_x, element, d_sf_dX, material, cal_cauchy_stress,
                            F0, return_F_S_W, return_stiffness, return_force_at_element,
                            get_integration_point, cal_F_tensor_with_d_sf_dX_at_element,
                            cal_d_sf_dh, cal_nodal_force_from_cauchy_stress, get_integration_weight):
    x=node_x[element].requires_grad_(True)
    #d_sf_dX.shape (M,K,3,A), K is the number of integration points
    F=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual/pre deformation, F0.shape must be F.shape (M,K,3,3)
        check_F0(F0, F.shape)
        F=matmul(F, F0)
    #-----------------------------------------------------------------------------
    #F.shape: (M,K,3,3)
    #M=element.shape[0]
    #material.shape: (1, ?) or (M, ?)
    n_integration_points=d_sf_dX.shape[1]
    r=get_integration_point(n_integration_points, dtype=node_x.dtype, device=node_x.device)
    d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #V from det(dx/dr)/det(F) = det(dX/dr)/det(F0): its value does not depend on node_x, and it keeps its graph
    #for the derivatives with respect to F0 and node_X (its derivative with respect to node_x is zero)
    V=cal_integration_point_volume(det_dx_dr/torch.linalg.det(F).view(det_dx_dr.shape), None, get_integration_weight)
    #the material differentiates (V*W) with respect to its F input: it gets a copy of F, so that V, which is
    #computed from F, is held fixed there and keeps its graph only for the derivatives outside the material
    S,W=cal_cauchy_stress(F.clone(), material, create_graph=True, return_W=True, V=V)
    #S.shape: (M,K,3,3)
    force_element=cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr)
    #force_element.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    N=node_x.shape[0]
    #M=element.shape[0]
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
    #out=[force, F, S, W, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F,S,W])
    if return_stiffness is not None:
        out.append(H)
    if return_force_at_element == True:
        out.append(force_element)
    return out
#%% node_x is known, node_X or material is unknown
def cal_cauchy_stress_force_inverse(d_sf_dx, det_dx_dr, n_nodes, element, F, material, cal_cauchy_stress, F0, return_S_W,
                                    cal_nodal_force_from_cauchy_stress, get_integration_weight):
    if F0 is not None:
        #F0 is residual/pre deformation, F0.shape must be F.shape (M,K,3,3)
        check_F0(F0, F.shape)
        F=matmul(F, F0)
    #-----------------------------------------------------------------------------
    #V from det(dx/dr)/det(F) = det(dX/dr)/det(F0); node_X may be unknown, so V keeps its graph
    V=cal_integration_point_volume(det_dx_dr/torch.linalg.det(F).view(det_dx_dr.shape), None, get_integration_weight)
    #the material differentiates (V*W) with respect to its F input: it gets a copy of F, so that V, which is
    #computed from F, is held fixed there and keeps its graph only for the derivatives outside the material
    S, W=cal_cauchy_stress(F.clone(), material, create_graph=True, return_W=True, V=V)
    #S.shape: (M,K,3,3)
    force_element=cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr)
    #force_element.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    #M=element.shape[0]
    #int64 index: with an int32 index on the CPU, the backward of scatter gives wrong gradient rows (torch 2.12.1)
    force = torch_scatter.scatter(force_element.reshape(-1,3), element.reshape(-1).to(torch.int64), dim=0, dim_size=n_nodes, reduce="sum")
    #force.shape: (n_nodes, 3)
    #-----------------------------------------------------------------------------
    if return_S_W == False:
        return force
    else:
        return force, S, W
#%%
def cal_1pk_stress_force(node_x, element, d_sf_dX, det_dX_dr, material, cal_1pk_stress,
                         F0, return_F_S_W, return_stiffness, return_force_at_element,
                         cal_F_tensor_with_d_sf_dX_at_element, cal_nodal_force_from_1pk_stress, get_integration_weight):
    x=node_x[element].requires_grad_(True)
    #d_sf_dX.shape (M,K,3,A)
    F=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual/pre deformation, F0.shape must be F.shape (M,K,3,3)
        check_F0(F0, F.shape)
        F=matmul(F, F0)
    #F.shape: (M,K,3,3))
    #material.shape: (M,?) or (1,?)
    V=cal_integration_point_volume(det_dX_dr, F0, get_integration_weight)
    S, W=cal_1pk_stress(F, material, create_graph=True, return_W=True, V=V)
    #S.shape: (M,K,3,3)
    if F0 is not None:
        #S=dW/dF with F=Fu*F0 is the 1PK stress per unit volume of the stress-free state,
        #but the force is integrated over the mesh X (d_sf_dX and det_dX_dr of X), so it needs
        #dW/dFu per unit volume of X: S_X = S*F0^T/det(F0); the returned S is not changed
        S_X=matmul(S, F0.transpose(-1,-2))/torch.linalg.det(F0)[...,None,None]
    else:
        S_X=S
    force_element=cal_nodal_force_from_1pk_stress(S_X, d_sf_dX, det_dX_dr)
    #force_element.shape: (M,A,3)
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
    #out=[force, F, S, W, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F,S,W])
    if return_stiffness is not None:
        out.append(H)
    if return_force_at_element == True:
        out.append(force_element)
    return out
#%%
def cal_diagonal_hessian(loss_fn, node_x):
    #the diagonal of the Hessian of the scalar loss_fn(node) with respect to node, node.shape (N,3); Hdiag.shape (3N,)
    #it takes 3N backward passes through the graph of the loss. For the strain energy of a mesh, return_stiffness="diagonal"
    #of the force functions (cal_diagonal_stiffness_matrix) gives the same values with 3A passes, A nodes per element
    #a derivative that has no graph is a constant, so the next derivative is zero: a constant loss, a loss that does not
    #use node, or an affine loss (whose gradient has no graph) gives zeros
    node=node_x.detach().clone().requires_grad_(True) #clone: node does not share memory with node_x
    N=node.shape[0]
    Hdiag=torch.zeros((3*N,), dtype=node.dtype, device=node.device)
    loss=loss_fn(node)
    if not loss.requires_grad:
        return Hdiag
    g=torch.autograd.grad(loss, node, create_graph=True, allow_unused=True)[0]
    if g is None or not g.requires_grad:
        return Hdiag
    for n in range(0, N):
        for i in range(0, 3):
            gg=torch.autograd.grad(g[n,i], node, retain_graph=True, allow_unused=True)[0]
            if gg is not None:
                Hdiag[3*n+i]=gg[n,i]
    return Hdiag

