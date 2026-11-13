# The model for hyperelastic materials on shell elements, FEBio's shell formulation.
#
# shell elements (6):  quad4, quad8, tri3, tri6     FEBio's shell domain (FEElasticShellDomain)
#                      q4ans                        assumed strain (FEElasticANSShellDomain)
#                      q4eas                        assumed and enhanced strain (FEElasticEASShellDomain)
#
# element_type, the first input of the functions of this file, names the shell element.
#
# Nodes: a shell node is on the front face of the shell and has a back point on the back face,
#     back point = shell node - director                                        (cal_director)
# In this file the back points are nodes of the mesh as well:
#     node_X, node_x: the nodes of the mesh and then the back points            (add_back_node)
#     element.shape (M,2A): the A shell nodes of an element and then their A back points
# So a shell element is a solid element that is linear through the thickness. The unknowns of a shell node are the
# position of the node and the position of its back point (displacement and shell displacement in FEBio), and the
# force and the stiffness matrix have rows for both. A solid element can use shell nodes or back points as its nodes.
#
# Material: a 3D material, as for the solid elements,
#     cal_1pk_stress(F, material, create_graph, return_W) or cal_cauchy_stress(F, material, create_graph, return_W)
#     quad4, quad8, tri3, tri6: F is the deformation gradient of the element.
#     q4ans, q4eas: the material gets an F that has the assumed strain E of the element (F^T*F = 2*E+I) and the nodal
#         force comes from its 2PK stress. The F and the stress that the functions of this file return are the
#         deformation gradient of the element and that 2PK stress pushed forward with it, as FEBio stores them.
#
# q4eas has 7 more unknowns per element, alpha.shape (M,7), without nodes. As in FEBio they are removed from the
#     equations of the element: the force functions return the nodal force and the stiffness with alpha removed,
#     and cal_alpha_increment gives the change of alpha that belongs to a change of the nodes. alpha starts at zero.
#
# Pressure on the front face: cal_pressure_force(pressure, node_x, element[:,0:A], n_integration_points) of
#     model_surface.py, which can be imported from this file as well.
# The stiffness matrix functions are in model_stiffness.py.
#
# Where this file or the element files do not do what FEBio does:
#     tri3 with 9 integration points: the element file has the Gauss points +-sqrt(3/5) through the
#         thickness, FEBio (FEShellTri3G9) has +-2/3 with the Gauss weights, which is not a Gauss rule
#     q4ans, q4eas: the F for the material is the Cholesky factor of 2*E+I, FEBio takes the square root; the strain
#         energy and the 2PK stress are the same
#     q4eas: the matrices that remove alpha are made at every call, FEBio keeps those of its last stiffness matrix
#     only nodal directors (FEBio's default); FEBio's option of one normal per element is not here
#
# Example for quad4, with element.shape (M,4) holding the shell nodes:
#     director=cal_director('quad4', node_X, element, thickness)
#     node_X, element, back_node=add_back_node(node_X, element, director)
#     force, F, S, W, H=cal_1pk_stress_force('quad4', node_x, element, node_X, 8, material, cal_1pk_stress,
#                                            return_F_S_W=True, return_stiffness="sparse")
#     d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dX_and_dX_dr('quad4', node_X, element, 8)
#     energy=cal_strain_energy('quad4', W, det_dX_dr).sum()
#%%
import torch
from torch_fea.utility.small_matmul import matmul #torch.matmul on the CPU, elementwise for small matrices on the GPU
from torch.linalg import det, solve
import torch_scatter
import torch_fea.element.shell_element_quad4 as shell_element_quad4
import torch_fea.element.shell_element_quad8 as shell_element_quad8
import torch_fea.element.shell_element_tri3 as shell_element_tri3
import torch_fea.element.shell_element_tri6 as shell_element_tri6
import torch_fea.element.shell_element_q4ans as shell_element_q4ans
import torch_fea.element.shell_element_q4eas as shell_element_q4eas
from torch_fea.model.model_stiffness import cal_dense_stiffness_matrix, cal_sparse_stiffness_matrix, cal_diagonal_stiffness_matrix
from torch_fea.model.model_surface import cal_pressure_force
#%%
#element_type -> (the number of shell nodes per element, the element file)
shell_element={'quad4': (4, shell_element_quad4),
               'quad8': (8, shell_element_quad8),
               'tri3':  (3, shell_element_tri3),
               'tri6':  (6, shell_element_tri6),
               'q4ans': (4, shell_element_q4ans),
               'q4eas': (4, shell_element_q4eas)}
#%%
def get_shell_element(element_type, n_points_per_element=None):
    #element_type: a key of shell_element ('quad4', 'q4ans', ...)
    #n_points_per_element: element.shape[1] of an element array with back points; if given, it must be
    #    2 times the number of shell nodes of that element
    #return: the element file (a module), e.g. shell_element_quad4
    if element_type not in shell_element:
        raise ValueError("unknown element_type "+str(element_type)+", only support "+", ".join(shell_element))
    n_nodes, element_file=shell_element[element_type]
    if n_points_per_element is not None and n_points_per_element != 2*n_nodes:
        raise ValueError(element_type+" has "+str(n_nodes)+" shell nodes and "+str(n_nodes)+" back points, but element.shape[1] is "
                         +str(n_points_per_element))
    return element_file
#%%
def cal_director(element_type, node_X, element, thickness):
    #FEModel::InitShells in FEBio
    #node_X.shape (N,3), N is the number of nodes, X is undeformed
    #element.shape (M,A): the A shell nodes of each element, without back points
    #thickness: a number, or a tensor of shape (M,1) (one per element) or (M,A) (one per node of an element)
    #the director of a node is the average of thickness*normal over the elements that have the node; normal is the
    #    unit normal that an element has at the node, from the node and its two neighbours in the element.
    #    As in FEBio the average is not made as long as the thickness again: where the elements at a node have
    #    different normals the shell is thinner than thickness.
    #return: director, director.shape (N,3), zero for a node that no element has
    element_file=get_shell_element(element_type)
    n_nodes=shell_element[element_type][0]
    if element.shape[1] != n_nodes:
        raise ValueError(element_type+" has "+str(n_nodes)+" shell nodes, but element.shape[1] is "+str(element.shape[1]))
    d=element_file.cal_director(node_X[element], thickness)
    #d.shape (M,A,3)
    N=node_X.shape[0]
    #int64 index: with an int32 index on the CPU, the backward of scatter gives wrong gradient rows (torch 2.12.1)
    director = torch_scatter.scatter(d.reshape(-1,3), element.reshape(-1).to(torch.int64), dim=0, dim_size=N, reduce="mean")
    return director #(N,3)
#%%
def add_back_node(node_X, element, director):
    #node_X.shape (N,3)
    #element.shape (M,A): the A shell nodes of each element, without back points
    #director.shape (N,3), from cal_director
    #every node that an element has gets a back point, back point = node - director; the back points are put after
    #    the N nodes, in the order of their nodes
    #return: node_X_shell, shape (N+B,3): the nodes and then the B back points
    #        element_shell, shape (M,2A): the shell nodes of each element and then their back points
    #        back_node, shape (N,): the index of the back point of each node in node_X_shell, -1 for a node without one
    N=node_X.shape[0]
    shell_node=torch.unique(element)
    back_node=torch.full((N,), -1, dtype=element.dtype, device=element.device)
    back_node[shell_node]=N+torch.arange(shell_node.shape[0], dtype=element.dtype, device=element.device)
    node_X_shell=torch.cat([node_X, node_X[shell_node]-director[shell_node]], dim=0)
    element_shell=torch.cat([element, back_node[element]], dim=1)
    return node_X_shell, element_shell, back_node
#%%
def cal_F_tensor(element_type, node_x, element, node_X, n_integration_points):
    #node_x.shape (N,3), deformed; node_X.shape (N,3), undeformed: the nodes and the back points
    #element.shape (M,2A): the shell nodes of each element and then their back points
    #the deformation gradient of the element. For q4ans and q4eas it is that of quad4: their assumed and enhanced
    #strain is not in it (see the top of this file)
    e=get_shell_element(element_type, element.shape[1])
    if element_type == 'q4ans' or element_type == 'q4eas':
        e=shell_element_quad4
    r=e.get_integration_point(n_integration_points, dtype=node_x.dtype, device=node_x.device)
    F=e.cal_F_tensor(r, node_x[element], node_X[element])
    return F #(M,K,3,3)
#%%
def cal_d_sf_dx_and_dx_dr(element_type, node_x, element, n_integration_points):
    e=get_shell_element(element_type, element.shape[1])
    r=e.get_integration_point(n_integration_points, dtype=node_x.dtype, device=node_x.device)
    d_sf_dx, dx_dr, det_dx_dr = e.cal_d_sf_dh(r, node_x[element])
    return d_sf_dx, dx_dr, det_dx_dr
#%%
def cal_d_sf_dX_and_dX_dr(element_type, node_X, element, n_integration_points):
    e=get_shell_element(element_type, element.shape[1])
    r=e.get_integration_point(n_integration_points, dtype=node_X.dtype, device=node_X.device)
    d_sf_dX, dX_dr, det_dX_dr = e.cal_d_sf_dh(r, node_X[element])
    return d_sf_dX, dX_dr, det_dX_dr
#%% forward: node_x is unknown, node_X is known
def cal_stress_force(element_type, node_x, element, node_X, n_integration_points, material, cal_stress, stress,
                     alpha, return_F_S_W, return_stiffness, return_force_at_element, return_eas):
    #the work of cal_1pk_stress_force (stress is '1pk') and cal_cauchy_stress_force (stress is 'cauchy')
    e=get_shell_element(element_type, element.shape[1])
    if element_type == 'q4eas':
        if alpha is None:
            raise ValueError("q4eas needs alpha, alpha.shape (M,7)")
    elif alpha is not None or return_eas == True:
        raise ValueError("alpha and return_eas are for q4eas, not for "+element_type)
    x=node_x[element].requires_grad_(True)
    X=node_X[element]
    M=x.shape[0]
    r=e.get_integration_point(n_integration_points, dtype=node_x.dtype, device=node_x.device)
    #-----------------------------------------------------------------------------
    if element_type == 'q4ans' or element_type == 'q4eas':
        #F_material has the assumed strain, the nodal force comes from the 2PK stress
        if element_type == 'q4eas':
            alpha=alpha.detach().requires_grad_(True)
            F_material=e.cal_F_tensor(r, x, X, alpha)
        else:
            F_material=e.cal_F_tensor(r, x, X)
        S_material, W=cal_stress(F_material, material, create_graph=True, return_W=True)
        if stress == '1pk':
            S_2pk=solve(F_material, S_material)
        else:
            S_2pk=e.cal_2pk_stress_from_cauchy_stress(F_material, S_material)
        force_element=e.cal_nodal_force_from_2pk_stress(S_2pk, r, x, X)
        #force_element.shape: (M,8,3)
        if element_type == 'q4eas':
            eas_force=e.cal_eas_force_from_2pk_stress(S_2pk, r, X)
            #eas_force.shape: (M,7), the derivative of the strain energy with respect to alpha
            #K_aa: derivative of eas_force with respect to alpha; K_au: with respect to x
            #K_ua, the derivative of force_element with respect to alpha, is the transpose of K_au, because
            #force_element and eas_force are the derivatives of one strain energy
            K_aa=[]
            K_au=[]
            for n in range(0, 7):
                gx, ga=torch.autograd.grad(eas_force[:,n].sum(), [x, alpha], retain_graph=True)
                K_au.append(gx.reshape(M,1,24))
                K_aa.append(ga.view(M,1,7))
            K_aa=torch.cat(K_aa, dim=1).detach() #(M,7,7)
            K_ua=torch.cat(K_au, dim=1).transpose(1,2).detach() #(M,24,7)
            #alpha removed: force_element - K_ua*K_aa^-1*eas_force; its derivative with respect to x, taken
            #below for the stiffness, is K_uu - K_ua*K_aa^-1*K_au
            force_element=e.cal_condensed_nodal_force(force_element, eas_force, K_ua, K_aa)
        if return_F_S_W == True:
            #as FEBio stores them: the deformation gradient of the element, and the 2PK stress pushed forward with it
            F=shell_element_quad4.cal_F_tensor(r, x, X)
            if stress == '1pk':
                S=matmul(F, S_2pk)
            else:
                S=matmul(matmul(F, S_2pk), F.transpose(2,3))/det(F).view(M,-1,1,1)
    else:
        F=e.cal_F_tensor(r, x, X)
        #F.shape: (M,K,3,3)
        S, W=cal_stress(F, material, create_graph=True, return_W=True)
        #S.shape: (M,K,3,3)
        if stress == '1pk':
            d_sf_dX, dX_dr, det_dX_dr=e.cal_d_sf_dh(r, X)
            force_element=e.cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr)
        else:
            d_sf_dx, dx_dr, det_dx_dr=e.cal_d_sf_dh(r, x)
            force_element=e.cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr)
        #force_element.shape: (M,2A,3)
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
    #out=[force, F, S, W, H, force_element, eas_force, K_ua, K_aa]
    out=[force]
    if return_F_S_W == True:
        out.extend([F,S,W])
    if return_stiffness is not None:
        out.append(H)
    if return_force_at_element == True:
        out.append(force_element)
    if return_eas == True:
        out.extend([eas_force.detach(),K_ua,K_aa])
    return out
#%%
def cal_cauchy_stress_force(element_type, node_x, element, node_X, n_integration_points, material, cal_cauchy_stress,
                            alpha=None, return_F_S_W=False, return_stiffness=None, return_force_at_element=False,
                            return_eas=False):
    #the inputs and outputs of cal_1pk_stress_force, with the Cauchy stress in place of the 1PK stress
    return cal_stress_force(element_type, node_x, element, node_X, n_integration_points, material, cal_cauchy_stress, 'cauchy',
                            alpha, return_F_S_W, return_stiffness, return_force_at_element, return_eas)
#%%
def cal_1pk_stress_force(element_type, node_x, element, node_X, n_integration_points, material, cal_1pk_stress,
                         alpha=None, return_F_S_W=False, return_stiffness=None, return_force_at_element=False,
                         return_eas=False):
    #node_x.shape (N,3), deformed; node_X.shape (N,3), undeformed: the nodes and the back points
    #element.shape (M,2A): the shell nodes of each element and then their back points
    #n_integration_points: K, the number of integration points of an element
    #material.shape (M,?) or (1,?)
    #cal_1pk_stress(F, material, create_graph, return_W): a 3D material
    #alpha.shape (M,7): for q4eas only, where it must be given; None for the other elements
    #return: out=[force, F, S, W, H, force_element, eas_force, K_ua, K_aa], of which force is always there
    #    force.shape (N,3)
    #    F.shape (M,K,3,3), S.shape (M,K,3,3), W.shape (M,K)      when return_F_S_W is True
    #    H.shape (3N,3N)                                          when return_stiffness is "dense", "sparse"; (3N,) for "diagonal"
    #    force_element.shape (M,2A,3)                             when return_force_at_element is True
    #    eas_force.shape (M,7), K_ua.shape (M,24,7), K_aa.shape (M,7,7), the inputs of cal_alpha_increment
    #                                                             when return_eas is True, for q4eas only
    #q4eas: force, H and force_element have alpha removed (FEElasticEASShellDomain in FEBio). eas_force is zero when
    #    alpha has its value for node_x; then force is the derivative of the strain energy with respect to node_x
    return cal_stress_force(element_type, node_x, element, node_X, n_integration_points, material, cal_1pk_stress, '1pk',
                            alpha, return_F_S_W, return_stiffness, return_force_at_element, return_eas)
#%%
def cal_alpha_increment(eas_force, K_ua, K_aa, node_dx, element):
    #q4eas, FEElasticEASShellDomain::UpdateEAS in FEBio
    #eas_force.shape (M,7), K_ua.shape (M,24,7), K_aa.shape (M,7,7): from cal_1pk_stress_force or
    #    cal_cauchy_stress_force with return_eas=True, for the nodes and the alpha before the change
    #node_dx.shape (N,3): the change of the nodes and back points, new node_x - old node_x
    #element.shape (M,8)
    #return: d_alpha, d_alpha.shape (M,7); alpha+d_alpha goes with the new node_x
    if element.shape[1] != 8:
        raise ValueError("q4eas has 4 shell nodes and 4 back points, but element.shape[1] is "+str(element.shape[1]))
    d_alpha=shell_element_q4eas.cal_alpha_increment(eas_force, K_ua, K_aa, node_dx[element])
    return d_alpha #(M,7)
#%%
def cal_strain_energy(element_type, W, det_dX_dr):
    #element_type: a key of shell_element ('quad4', 'q4ans', ...)
    #W.shape (M,K), strain energy density at the integration points
    #det_dX_dr.shape (M,K,1,1), from cal_d_sf_dX_and_dX_dr
    #return: energy, the strain energy of each element, shape (M,)
    #    energy.sum() is the strain energy of the mesh
    e=get_shell_element(element_type)
    energy=e.cal_strain_energy(W, det_dX_dr)
    return energy #(M,)
#%%
def interpolate(element_type, r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,2A,3), M elements, the A shell nodes of an element and then their A back points
    e=get_shell_element(element_type, h.shape[1])
    return e.interpolate(r, h)
