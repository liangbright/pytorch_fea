from torch_fea.element.SolidElement_tet4 import interpolate as interpolate_tet4
from torch_fea.element.SolidElement_tet4 import cal_d_sf_dh as cal_d_sf_dh_tet4
from torch_fea.element.SolidElement_tet4 import cal_F_tensor as cal_F_tensor_at_element_tet4
from torch_fea.element.SolidElement_tet4 import cal_F_tensor_with_d_sf_dX as cal_F_tensor_with_d_sf_dX_at_element_tet4
from torch_fea.element.SolidElement_tet4 import get_integration_point as get_integration_point_tet4
from torch_fea.element.SolidElement_tet4 import cal_nodal_force_from_1pk_stress as cal_nodal_force_from_1pk_stress_tet4
from torch_fea.element.SolidElement_tet4 import cal_nodal_force_from_cauchy_stress as cal_nodal_force_from_cauchy_stress_tet4
from torch_fea.element.SolidElement_tet4 import cal_strain_energy as cal_strain_energy_tet4
#------------------------------------------------------------------------------------------------------------
from torch_fea.element.SolidElement_tet10 import interpolate as interpolate_tet10
from torch_fea.element.SolidElement_tet10 import cal_d_sf_dh as cal_d_sf_dh_tet10
from torch_fea.element.SolidElement_tet10 import cal_F_tensor as cal_F_tensor_at_element_tet10
from torch_fea.element.SolidElement_tet10 import cal_F_tensor_with_d_sf_dX as cal_F_tensor_with_d_sf_dX_at_element_tet10
from torch_fea.element.SolidElement_tet10 import get_integration_point as get_integration_point_tet10
from torch_fea.element.SolidElement_tet10 import cal_nodal_force_from_1pk_stress as cal_nodal_force_from_1pk_stress_tet10
from torch_fea.element.SolidElement_tet10 import cal_nodal_force_from_cauchy_stress as cal_nodal_force_from_cauchy_stress_tet10
from torch_fea.element.SolidElement_tet10 import cal_strain_energy as cal_strain_energy_tet10
#------------------------------------------------------------------------------------------------------------
from torch_fea.element.SolidElement_wedge6 import interpolate as interpolate_wedge6
from torch_fea.element.SolidElement_wedge6 import cal_d_sf_dh as cal_d_sf_dh_wedge6
from torch_fea.element.SolidElement_wedge6 import cal_F_tensor as cal_F_tensor_at_element_wedge6
from torch_fea.element.SolidElement_wedge6 import cal_F_tensor_with_d_sf_dX as cal_F_tensor_with_d_sf_dX_at_element_wedge6
from torch_fea.element.SolidElement_wedge6 import get_integration_point as get_integration_point_wedge6
from torch_fea.element.SolidElement_wedge6 import cal_nodal_force_from_1pk_stress as cal_nodal_force_from_1pk_stress_wedge6
from torch_fea.element.SolidElement_wedge6 import cal_nodal_force_from_cauchy_stress as cal_nodal_force_from_cauchy_stress_wedge6
from torch_fea.element.SolidElement_wedge6 import cal_strain_energy as cal_strain_energy_wedge6
#------------------------------------------------------------------------------------------------------------
from torch_fea.element.SolidElement_hex8 import interpolate as interpolate_hex8
from torch_fea.element.SolidElement_hex8 import cal_d_sf_dh as cal_d_sf_dh_hex8
from torch_fea.element.SolidElement_hex8 import cal_F_tensor as cal_F_tensor_at_element_hex8
from torch_fea.element.SolidElement_hex8 import cal_F_tensor_with_d_sf_dX as cal_F_tensor_with_d_sf_dX_at_element_hex8
from torch_fea.element.SolidElement_hex8 import get_integration_point as get_integration_point_hex8
from torch_fea.element.SolidElement_hex8 import cal_nodal_force_from_1pk_stress as cal_nodal_force_from_1pk_stress_hex8
from torch_fea.element.SolidElement_hex8 import cal_nodal_force_from_cauchy_stress as cal_nodal_force_from_cauchy_stress_hex8
from torch_fea.element.SolidElement_hex8 import cal_strain_energy as cal_strain_energy_hex8
#------------------------------------------------------------------------------------------------------------
from torch_fea.element.pressure_load_on_tri3 import cal_nodal_force_from_pressure as cal_nodal_force_from_pressure_tri3
from torch_fea.element.pressure_load_on_tri6 import cal_nodal_force_from_pressure as cal_nodal_force_from_pressure_tri6
from torch_fea.element.pressure_load_on_quad4 import cal_nodal_force_from_pressure as cal_nodal_force_from_pressure_quad4
import torch_fea.model.FEModelC3_SRI_base as FEModel
from torch_fea.model.FEModelC3_SRI_base import cal_potential_energy
#%%
def cal_pressure_force(pressure, node_x, element, n_integration_points, return_force="dense", return_stiffness=None):
    #element must be tri3 or tri6 or quad4, not mixed
    if element.shape[1] == 3:
        cal_nodal_force_from_pressure=cal_nodal_force_from_pressure_tri3
    elif element.shape[1] == 6:
        cal_nodal_force_from_pressure=cal_nodal_force_from_pressure_tri6
    elif element.shape[1] == 4:
        cal_nodal_force_from_pressure=cal_nodal_force_from_pressure_quad4
    else:
        raise ValueError("only support pressure on quad and triangle")
    return FEModel.cal_pressure_force(pressure=pressure,
                                      node_x=node_x,
                                      element=element,
                                      n_integration_points=n_integration_points,
                                      return_force=return_force,
                                      return_stiffness=return_stiffness,
                                      cal_nodal_force_from_pressure=cal_nodal_force_from_pressure)
#%%
def cal_F_tensor(node_x, element, node_X, n_integration_points, d_sf_dX=None, F0=None):
    #element must be tet4, tet10, wedge6, or hex8, not mixed
    if element.shape[1] == 4:
        get_integration_point=get_integration_point_tet4
        cal_F_tensor_at_element=cal_F_tensor_at_element_tet4
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_tet4
    elif element.shape[1] == 10:
        get_integration_point=get_integration_point_tet10
        cal_F_tensor_at_element=cal_F_tensor_at_element_tet10
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_tet10
    elif element.shape[1] == 6:
        get_integration_point=get_integration_point_wedge6
        cal_F_tensor_at_element=cal_F_tensor_at_element_wedge6
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_wedge6
    elif element.shape[1] == 8:
        get_integration_point=get_integration_point_hex8
        cal_F_tensor_at_element=cal_F_tensor_at_element_hex8
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_hex8
    else:
        raise ValueError("only support tet4, tet10, wedge6, and hex8")
    return FEModel.cal_F_tensor(node_x=node_x,
                                element=element,
                                node_X=node_X,
                                n_integration_points=n_integration_points,
                                d_sf_dX=d_sf_dX,
                                F0=F0,
                                get_integration_point=get_integration_point,
                                cal_F_tensor_at_element=cal_F_tensor_at_element,
                                cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element)
#%%
def cal_d_sf_dx_and_dx_dr(node_x, element, n_integration_points):
    if element.shape[1] == 4:
        get_integration_point=get_integration_point_tet4
        cal_d_sf_dh=cal_d_sf_dh_tet4
    elif element.shape[1] == 10:
        get_integration_point=get_integration_point_tet10
        cal_d_sf_dh=cal_d_sf_dh_tet10
    elif element.shape[1] == 6:
        get_integration_point=get_integration_point_wedge6
        cal_d_sf_dh=cal_d_sf_dh_wedge6
    elif element.shape[1] == 8:
        get_integration_point=get_integration_point_hex8
        cal_d_sf_dh=cal_d_sf_dh_hex8
    else:
        raise ValueError("only support tet4, tet10, wedge6, and hex8")
    return FEModel.cal_d_sf_dx_and_dx_dr(node_x=node_x,
                                         element=element,
                                         n_integration_points=n_integration_points,
                                         get_integration_point=get_integration_point,
                                         cal_d_sf_dh=cal_d_sf_dh)
#%%
def cal_d_sf_dX_and_dX_dr(node_X, element, n_integration_points):
    if element.shape[1] == 4:
        get_integration_point=get_integration_point_tet4
        cal_d_sf_dh=cal_d_sf_dh_tet4
    elif element.shape[1] == 10:
        get_integration_point=get_integration_point_tet10
        cal_d_sf_dh=cal_d_sf_dh_tet10
    elif element.shape[1] == 6:
        get_integration_point=get_integration_point_wedge6
        cal_d_sf_dh=cal_d_sf_dh_wedge6
    elif element.shape[1] == 8:
        get_integration_point=get_integration_point_hex8
        cal_d_sf_dh=cal_d_sf_dh_hex8
    else:
        raise ValueError("only support tet4, tet10, wedge6, and hex8")
    return FEModel.cal_d_sf_dX_and_dX_dr(node_X=node_X,
                                         element=element,
                                         n_integration_points=n_integration_points,
                                         get_integration_point=get_integration_point,
                                         cal_d_sf_dh=cal_d_sf_dh)
#%%
def cal_cauchy_stress_force(node_x, element, d_sf_dX_dev, d_sf_dX_vol, material, cal_cauchy_stress,
                            F0_dev=None, F0_vol=None, return_F_S_W=False, return_stiffness=None, return_force_at_element=False):
    if element.shape[1] == 4:
        get_integration_point=get_integration_point_tet4
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_tet4
        cal_d_sf_dh=cal_d_sf_dh_tet4
        cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress_tet4
    elif element.shape[1] == 10:
        get_integration_point=get_integration_point_tet10
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_tet10
        cal_d_sf_dh=cal_d_sf_dh_tet10
        cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress_tet10
    elif element.shape[1] == 6:
        get_integration_point=get_integration_point_wedge6
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_wedge6
        cal_d_sf_dh=cal_d_sf_dh_wedge6
        cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress_wedge6
    elif element.shape[1] == 8:
        get_integration_point=get_integration_point_hex8
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_hex8
        cal_d_sf_dh=cal_d_sf_dh_hex8
        cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress_hex8
    else:
        raise ValueError("only support tet4, tet10, wedge6, and hex8")
    return FEModel.cal_cauchy_stress_force(node_x=node_x,
                                           element=element,
                                           d_sf_dX_dev=d_sf_dX_dev,
                                           d_sf_dX_vol=d_sf_dX_vol,
                                           material=material,
                                           cal_cauchy_stress=cal_cauchy_stress,
                                           F0_dev=F0_dev,
                                           F0_vol=F0_vol,
                                           return_F_S_W=return_F_S_W,
                                           return_stiffness=return_stiffness,
                                           return_force_at_element=return_force_at_element,
                                           get_integration_point=get_integration_point,
                                           cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element,
                                           cal_d_sf_dh=cal_d_sf_dh,
                                           cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress)
#%%
def cal_cauchy_stress_force_inverse(d_sf_dX_dev, d_sf_dX_vol, det_dx_dr_dev, det_dx_dr_vol,
                                    n_nodes, element, F_dev, F_vol, material, cal_cauchy_stress,
                                    F0_dev=None, F0_vol=None, return_S_W=False):
    if element.shape[1] == 4:
        cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress_tet4
    elif element.shape[1] == 10:
        cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress_tet10
    elif element.shape[1] == 6:
        cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress_wedge6
    elif element.shape[1] == 8:
        cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress_hex8
    else:
        raise ValueError("only support tet4, tet10, wedge6, and hex8")
    return FEModel.cal_cauchy_stress_force_inverse(d_sf_dX_dev=d_sf_dX_dev,
                                                   d_sf_dX_vol=d_sf_dX_vol,
                                                   det_dx_dr_dev=det_dx_dr_dev,
                                                   det_dx_dr_vol=det_dx_dr_vol,
                                                   n_nodes=n_nodes,
                                                   element=element,
                                                   F_dev=F_dev,
                                                   F_vol=F_vol,
                                                   material=material,
                                                   cal_cauchy_stress=cal_cauchy_stress,
                                                   F0_dev=F0_dev,
                                                   F0_vol=F0_vol,
                                                   return_S_W=return_S_W,
                                                   cal_nodal_force_from_cauchy_stress=cal_nodal_force_from_cauchy_stress)
#%%
def cal_1pk_stress_force(node_x, element,  d_sf_dX_dev, d_sf_dX_vol, det_dX_dr_dev, det_dX_dr_vol,
                         material, cal_1pk_stress, F0_dev=None, F0_vol=None,
                         return_F_S_W=False, return_stiffness=None, return_force_at_element=False):
    if element.shape[1] == 4:
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_tet4
        cal_nodal_force_from_1pk_stress=cal_nodal_force_from_1pk_stress_tet4
    elif element.shape[1] == 10:
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_tet10
        cal_nodal_force_from_1pk_stress=cal_nodal_force_from_1pk_stress_tet10
    elif element.shape[1] == 6:
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_wedge6
        cal_nodal_force_from_1pk_stress=cal_nodal_force_from_1pk_stress_wedge6
    elif element.shape[1] == 8:
        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element_hex8
        cal_nodal_force_from_1pk_stress=cal_nodal_force_from_1pk_stress_hex8
    else:
        raise ValueError("only support tet4, tet10, wedge6, and hex8")
    return FEModel.cal_1pk_stress_force(node_x=node_x,
                                        element=element,
                                        d_sf_dX_dev=d_sf_dX_dev,
                                        d_sf_dX_vol=d_sf_dX_vol,
                                        det_dX_dr_dev=det_dX_dr_dev,
                                        det_dX_dr_vol=det_dX_dr_vol,
                                        material=material,
                                        cal_1pk_stress=cal_1pk_stress,
                                        F0_dev=F0_dev,
                                        F0_vol=F0_vol,
                                        return_F_S_W=return_F_S_W,
                                        return_stiffness=return_stiffness,
                                        return_force_at_element=return_force_at_element,
                                        cal_F_tensor_with_d_sf_dX_at_element=cal_F_tensor_with_d_sf_dX_at_element,
                                        cal_nodal_force_from_1pk_stress=cal_nodal_force_from_1pk_stress)
#%%
def cal_strain_energy(element_type, W_dev, W_vol, det_dX_dr_dev, det_dX_dr_vol, reduction):
    if element_type == 'tet4':
        energy_dev=cal_strain_energy_tet4(W_dev, det_dX_dr_dev, reduction)
        energy_vol=cal_strain_energy_tet4(W_vol, det_dX_dr_vol, reduction)
    elif element_type == 'tet10':
        energy_dev=cal_strain_energy_tet10(W_dev, det_dX_dr_dev, reduction)
        energy_vol=cal_strain_energy_tet10(W_vol, det_dX_dr_vol, reduction)
    elif element_type == 'wedge6':
        energy_dev=cal_strain_energy_wedge6(W_dev, det_dX_dr_dev, reduction)
        energy_vol=cal_strain_energy_wedge6(W_vol, det_dX_dr_vol, reduction)
    elif element_type == 'hex8':
        energy_dev=cal_strain_energy_hex8(W_dev, det_dX_dr_dev, reduction)
        energy_vol=cal_strain_energy_hex8(W_vol, det_dX_dr_vol, reduction)
    else:
        raise ValueError("only support tet4, tet10, wedge6, and hex8")
    if reduction == 'sum':
        energy=energy_dev+energy_vol
    elif reduction == 'mean':
        energy=(energy_dev+energy_vol)/2
    else:
        raise ValueError("not supported reduction:"+reduction)
    return energy
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,A,3), M elements, 1 elemnet has A nodes, 1 node has a 3D position
    if h.shape[1] == 4:
        return interpolate_tet4(r, h)
    elif h.shape[1] == 10:
        return interpolate_tet10(r, h)
    elif h.shape[1] == 6:
        return interpolate_wedge6(r, h)
    elif h.shape[1] == 8:
        return interpolate_hex8(r, h)
    else:
        raise ValueError("only support tet4, tet10, wedge6, and hex8")