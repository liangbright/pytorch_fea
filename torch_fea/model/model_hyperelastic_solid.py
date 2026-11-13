# The model for hyperelastic materials on solid elements:
# the functions of model_hyperelastic_solid_base.py with the element picked for the caller.
#
# solid elements (13):   tet4, tet5, tet10, tet15, tet20, wedge6, wedge15, hex8, hex20, hex27, pyramid5, pyramid13, pyramid13gb
#
# element_type, the first input of every function of this file, names the solid element: 'hex8', 'tet10', ...
#     It is not found from the number of nodes per element: 5 nodes (tet5, pyramid5), 13 nodes (pyramid13, pyramid13gb),
#     15 nodes (tet15, wedge15) and 20 nodes (tet20, hex20) belong to two elements each.
#
# The pressure load, cal_pressure_force, is in model_surface.py and can be imported from this file as well.
# The files of torch_fea/element are used as modules: solid_element_<element_type>.py.
#
# The material functions always get V, the volume of each integration point (see cal_integration_point_volume),
# and evaluate the volumetric part at the element mean dilatation (material_GOH.py, material_NeoHookean.py):
#     cal_1pk_stress(F, material, create_graph, return_W, V) and cal_cauchy_stress(F, material, create_graph, return_W, V)
#
# F0 (optional) is the pre/residual deformation gradient, F=Fu*F0, a field stored at the integration points:
#     F0.shape must be (M,K,3,3), the same as F, with K = n_integration_points of the element; other shapes raise ValueError.
#     An F0 from a run with another element type or another K cannot be reused without projecting it to the new points.
#%%
import torch
import torch_fea.element.solid_element_tet4 as solid_element_tet4
import torch_fea.element.solid_element_tet5 as solid_element_tet5
import torch_fea.element.solid_element_tet10 as solid_element_tet10
import torch_fea.element.solid_element_tet15 as solid_element_tet15
import torch_fea.element.solid_element_tet20 as solid_element_tet20
import torch_fea.element.solid_element_wedge6 as solid_element_wedge6
import torch_fea.element.solid_element_wedge15 as solid_element_wedge15
import torch_fea.element.solid_element_hex8 as solid_element_hex8
import torch_fea.element.solid_element_hex20 as solid_element_hex20
import torch_fea.element.solid_element_hex27 as solid_element_hex27
import torch_fea.element.solid_element_pyramid5 as solid_element_pyramid5
import torch_fea.element.solid_element_pyramid13 as solid_element_pyramid13
import torch_fea.element.solid_element_pyramid13gb as solid_element_pyramid13gb
#------------------------------------------------------------------------------------------------------------
import torch_fea.model.model_hyperelastic_solid_base as base
from torch_fea.model.model_hyperelastic_solid_base import cal_potential_energy
from torch_fea.model.model_surface import cal_pressure_force
#%%
#element_type -> (the number of nodes per element, the element file)
solid_element={'tet4':      (4,  solid_element_tet4),
               'tet5':      (5,  solid_element_tet5),
               'tet10':     (10, solid_element_tet10),
               'tet15':     (15, solid_element_tet15),
               'tet20':     (20, solid_element_tet20),
               'wedge6':    (6,  solid_element_wedge6),
               'wedge15':   (15, solid_element_wedge15),
               'hex8':      (8,  solid_element_hex8),
               'hex20':     (20, solid_element_hex20),
               'hex27':     (27, solid_element_hex27),
               'pyramid5':  (5,  solid_element_pyramid5),
               'pyramid13': (13, solid_element_pyramid13),
               'pyramid13gb': (13, solid_element_pyramid13gb)}
#%%
def get_solid_element(element_type, n_nodes_per_element=None):
    #element_type: a key of solid_element ('tet4', 'hex8', ...)
    #n_nodes_per_element: element.shape[1]; if given, it must be the number of nodes of that element
    #return: the element file (a module), e.g. solid_element_hex8
    if element_type not in solid_element:
        raise ValueError("unknown element_type "+str(element_type)+", only support "+", ".join(solid_element))
    n_nodes, element_file=solid_element[element_type]
    if n_nodes_per_element is not None and n_nodes_per_element != n_nodes:
        raise ValueError(element_type+" has "+str(n_nodes)+" nodes, but element.shape[1] is "+str(n_nodes_per_element))
    return element_file
#%%
def cal_F_tensor(element_type, node_x, element, node_X, n_integration_points, d_sf_dX=None, F0=None):
    #element must be one solid element type, not mixed
    e=get_solid_element(element_type, element.shape[1])
    return base.cal_F_tensor(node_x=node_x,
                             element=element,
                             node_X=node_X,
                             n_integration_points=n_integration_points,
                             d_sf_dX=d_sf_dX,
                             F0=F0,
                             get_integration_point=e.get_integration_point,
                             cal_F_tensor_at_element=e.cal_F_tensor,
                             cal_F_tensor_with_d_sf_dX_at_element=e.cal_F_tensor_with_d_sf_dX)
#%%
def cal_integration_point_volume(element_type, det_dX_dr, F0=None):
    #V, shape (M,K): the volume of each integration point, w_k*det(dX/dr), divided by det(F0) if F0 is given
    #it is the V input of the material functions
    e=get_solid_element(element_type)
    return base.cal_integration_point_volume(det_dX_dr, F0, e.get_integration_weight)
#%%
def cal_d_sf_dx_and_dx_dr(element_type, node_x, element, n_integration_points):
    e=get_solid_element(element_type, element.shape[1])
    return base.cal_d_sf_dx_and_dx_dr(node_x=node_x,
                                      element=element,
                                      n_integration_points=n_integration_points,
                                      get_integration_point=e.get_integration_point,
                                      cal_d_sf_dh=e.cal_d_sf_dh)
#%%
def cal_d_sf_dX_and_dX_dr(element_type, node_X, element, n_integration_points):
    e=get_solid_element(element_type, element.shape[1])
    return base.cal_d_sf_dX_and_dX_dr(node_X=node_X,
                                      element=element,
                                      n_integration_points=n_integration_points,
                                      get_integration_point=e.get_integration_point,
                                      cal_d_sf_dh=e.cal_d_sf_dh)
#%%
def cal_cauchy_stress_force(element_type, node_x, element, d_sf_dX, material, cal_cauchy_stress,
                            F0=None, return_F_S_W=False, return_stiffness=None, return_force_at_element=False):
    e=get_solid_element(element_type, element.shape[1])
    return base.cal_cauchy_stress_force(node_x=node_x,
                                        element=element,
                                        d_sf_dX=d_sf_dX,
                                        material=material,
                                        cal_cauchy_stress=cal_cauchy_stress,
                                        F0=F0,
                                        return_F_S_W=return_F_S_W,
                                        return_stiffness=return_stiffness,
                                        return_force_at_element=return_force_at_element,
                                        get_integration_point=e.get_integration_point,
                                        cal_F_tensor_with_d_sf_dX_at_element=e.cal_F_tensor_with_d_sf_dX,
                                        cal_d_sf_dh=e.cal_d_sf_dh,
                                        cal_nodal_force_from_cauchy_stress=e.cal_nodal_force_from_cauchy_stress,
                                        get_integration_weight=e.get_integration_weight)
#%%
def cal_cauchy_stress_force_inverse(element_type, d_sf_dx, det_dx_dr, n_nodes, element, F, material, cal_cauchy_stress,
                                    F0=None, return_S_W=False):
    e=get_solid_element(element_type, element.shape[1])
    return base.cal_cauchy_stress_force_inverse(d_sf_dx=d_sf_dx,
                                                det_dx_dr=det_dx_dr,
                                                n_nodes=n_nodes,
                                                element=element,
                                                F=F,
                                                material=material,
                                                cal_cauchy_stress=cal_cauchy_stress,
                                                F0=F0,
                                                return_S_W=return_S_W,
                                                cal_nodal_force_from_cauchy_stress=e.cal_nodal_force_from_cauchy_stress,
                                                get_integration_weight=e.get_integration_weight)
#%%
def cal_1pk_stress_force(element_type, node_x, element, d_sf_dX, det_dX_dr, material, cal_1pk_stress,
                         F0=None, return_F_S_W=False, return_stiffness=None, return_force_at_element=False):
    e=get_solid_element(element_type, element.shape[1])
    return base.cal_1pk_stress_force(node_x=node_x,
                                     element=element,
                                     d_sf_dX=d_sf_dX,
                                     det_dX_dr=det_dX_dr,
                                     material=material,
                                     cal_1pk_stress=cal_1pk_stress,
                                     F0=F0,
                                     return_F_S_W=return_F_S_W,
                                     return_stiffness=return_stiffness,
                                     return_force_at_element=return_force_at_element,
                                     cal_F_tensor_with_d_sf_dX_at_element=e.cal_F_tensor_with_d_sf_dX,
                                     cal_nodal_force_from_1pk_stress=e.cal_nodal_force_from_1pk_stress,
                                     get_integration_weight=e.get_integration_weight)
#%%
def cal_strain_energy(element_type, W, det_dX_dr, F0=None):
    #element_type: a key of solid_element ('tet4', 'hex8', ...)
    #W.shape (M,K), strain energy density at the integration points, per unit volume of the stress-free state
    #det_dX_dr.shape (M,K,1,1)
    #F0: None or the pre/residual deformation gradient (M,K,3,3) of F=Fu*F0, the same F0 as for the force;
    #    with F0 the stress-free volume of point k is w_k*det(dX/dr)/det(F0) (see cal_integration_point_volume)
    #return: energy, the strain energy of each element, shape (M,)
    #    energy.sum() is the strain energy of the mesh
    e=get_solid_element(element_type)
    if F0 is not None:
        base.check_F0(F0, (W.shape[0], W.shape[1], 3, 3))
        det_dX_dr=det_dX_dr/torch.linalg.det(F0).reshape(det_dX_dr.shape)
    energy=e.cal_strain_energy(W, det_dX_dr)
    return energy #(M,)
#%%
def interpolate(element_type, r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,A,3), M elements, 1 elemnet has A nodes, 1 node has a 3D position
    e=get_solid_element(element_type, h.shape[1])
    return e.interpolate(r, h)
