# this is the implementation of direct selective-reduced integration(SRI)
from torch import matmul
import torch_scatter
from torch_fea.model.FEModelC3_base import cal_F_tensor, cal_d_sf_dx_and_dx_dr, cal_d_sf_dX_and_dX_dr
from torch_fea.model.FEModelC3_base import cal_pressure_force, cal_potential_energy
from torch_fea.model.FEModelC3_base import cal_dense_stiffness_matrix, cal_sparse_stiffness_matrix, cal_diagonal_stiffness_matrix
#%%
#dev: deviatoric
#vol: volumetric
def cal_cauchy_stress_force(node_x, element, d_sf_dX_dev, d_sf_dX_vol, material, cal_cauchy_stress,
                            F0_dev, F0_vol, return_F_S_W, return_stiffness, return_force_at_element,
                            get_integration_point, cal_F_tensor_with_d_sf_dX_at_element,
                            cal_d_sf_dh, cal_nodal_force_from_cauchy_stress):
    #d_sf_dX_dev from cal_d_sf_dX_and_dX_dr()
    #d_sf_dX_vol from cal_d_sf_dX_and_dX_dr()
    x=node_x[element]
    F_dev=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX_dev)
    F_vol=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX_vol)
    if F0_dev is not None and F0_vol is not None:
        #F0 is residual/pre deformation
        F_dev=matmul(F_dev, F0_dev)
        F_vol=matmul(F_vol, F0_vol)
    S_dev, S_vol, W_dev, W_vol=cal_cauchy_stress(F_dev, F_vol, material, create_graph=True, return_W=True)
    #S_dev.shape: (M,K,3,3), S_vol.shape: (M,1,3,3)
    #-----------------------------------------------------------------------------
    r_dev=get_integration_point(d_sf_dX_dev.shape[1], dtype=node_x.dtype, device=node_x.device)
    d_sf_dX_dev, dx_dr_dev, det_dx_dr_dev=cal_d_sf_dh(r_dev, x)
    force_element_S_dev=cal_nodal_force_from_cauchy_stress(S_dev, d_sf_dX_dev, det_dx_dr_dev)
    #force_element_dev.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    r_vol=get_integration_point(d_sf_dX_vol.shape[1], dtype=node_x.dtype, device=node_x.device)
    d_sf_dX_vol, dx_dr_vol, det_dx_dr_vol=cal_d_sf_dh(r_vol, x)
    force_element_S_vol=cal_nodal_force_from_cauchy_stress(S_vol, d_sf_dX_vol, det_dx_dr_vol)
    #force_element_vol.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    force_element=force_element_S_dev+force_element_S_vol
    N=node_x.shape[0]
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
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
    #out=[force, F_dev, F_vol, S_dev, S_vol, W_dev, W_vol, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F_dev, F_vol, S_dev, S_vol, W_dev, W_vol])
    if return_stiffness is not None:
        out.append(H)
    if return_force_at_element == True:
        out.append(force_element)
    return out
#%%
def cal_cauchy_stress_force_inverse(d_sf_dX_dev, d_sf_dX_vol, det_dx_dr_dev, det_dx_dr_vol,
                                    n_nodes, element, F_dev, F_vol, material, cal_cauchy_stress, F0_dev, F0_vol, return_S_W,
                                    cal_nodal_force_from_cauchy_stress):
    if F0_dev is not None and F0_vol is not None:
        #F0 is residual/pre deformation
        F_dev=matmul(F_dev, F0_dev)
        F_vol=matmul(F_vol, F0_vol)
    S_dev, S_vol, W_dev, W_vol=cal_cauchy_stress(F_dev, F_vol, material, create_graph=True, return_W=True)
    #S_dev.shape: (M,K,3,3), S_vol.shape: (M,1,3,3)
    #-----------------------------------------------------------------------------
    force_element_S_dev=cal_nodal_force_from_cauchy_stress(S_dev, d_sf_dX_dev, det_dx_dr_dev)
    #force_element_dev.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    force_element_S_vol=cal_nodal_force_from_cauchy_stress(S_vol, d_sf_dX_vol, det_dx_dr_vol)
    #force_element_vol.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    force_element=force_element_S_dev+force_element_S_vol
    N=n_nodes
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
    #force.shape: (N, 3)
    #-----------------------------------------------------------------------------
    if return_S_W == False:
        return force
    else:
        return force, S_dev, S_vol, W_dev, W_vol
#%%
def cal_1pk_stress_force(node_x, element, d_sf_dX_dev, d_sf_dX_vol, det_dX_dr_dev, det_dX_dr_vol,
                         material, cal_1pk_stress, F0_dev, F0_vol,
                         return_F_S_W, return_stiffness, return_force_at_element,
                         cal_F_tensor_with_d_sf_dX_at_element, cal_nodal_force_from_1pk_stress):
    x=node_x[element]
    F_dev=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX_dev)
    F_vol=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX_vol)
    if F0_dev is not None and F0_vol is not None:
        #F0 is residual/pre deformation
        F_dev=matmul(F_dev, F0_dev)
        F_vol=matmul(F_vol, F0_vol)
    #F_dev.shape: (M,K,3,3)
    #F_vol.shape: (M,1,3,3)
    #material.shape: (M,?) or (1,?)
    S_dev, S_vol, W_dev, W_vol=cal_1pk_stress(F_dev, F_vol, material, create_graph=True, return_W=True)
    #S_dev.shape: (M,8,3,3), S_vol.shape: (M,1,3,3)
    force_element_S_dev=cal_nodal_force_from_1pk_stress(S_dev, d_sf_dX_dev, det_dX_dr_dev)
    force_element_S_vol=cal_nodal_force_from_1pk_stress(S_vol, d_sf_dX_vol, det_dX_dr_vol)
    force_element=force_element_S_dev+force_element_S_vol
    #force_element.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    N=node_x.shape[0]
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
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
    #out=[force, F_dev, F_vol, S_dev, S_vol, W_dev, W_vol, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F_dev, F_vol, S_dev, S_vol, W_dev, W_vol])
    if return_stiffness is not None:
        out.append(H)
    if return_force_at_element == True:
        out.append(force_element)
    return out
