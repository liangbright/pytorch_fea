from torch_fea.model.FEModelC3 import cal_cauchy_stress_force as _cal_cauchy_stress_force_
from torch_fea.model.FEModelC3 import cal_cauchy_stress_force_inverse as _cal_cauchy_stress_force_inverse_
from torch_fea.model.FEModelC3 import cal_1pk_stress_force as _cal_1pk_stress_force_
from torch_fea.model.FEModelC3 import cal_F_tensor, cal_d_sf_dX_and_dX_dr, cal_d_sf_dx_and_dx_dr
from torch_fea.model.FEModelC3 import cal_pressure_force
from torch_fea.model.FEModelC3 import cal_strain_energy, cal_potential_energy
from torch_fea.model.FEModelC3 import interpolate
#%%
def cal_cauchy_stress_force(node_x, element, d_sf_dX, material, element_orientation, cal_cauchy_stress,
                            F0=None, return_F_S_W=False, return_stiffness=None, return_force_at_element=False):
    def _cal_cauchy_stress_(F, material, create_graph, return_W):
        return cal_cauchy_stress(F, material, element_orientation, create_graph, return_W)
    return _cal_cauchy_stress_force_(node_x, element, d_sf_dX, material, _cal_cauchy_stress_,
                                     F0, return_F_S_W, return_stiffness, return_force_at_element)
#%%
def cal_cauchy_stress_force_inverse(d_sf_dx, det_dx_dr, n_nodes, element,
                                    F, material, element_orientation, cal_cauchy_stress,
                                    F0=None, return_S_W=False):
    def _cal_cauchy_stress_(F, material, create_graph, return_W):
        return cal_cauchy_stress(F, material, element_orientation, create_graph, return_W)
    return _cal_cauchy_stress_force_inverse_(d_sf_dx, det_dx_dr, n_nodes, element,
                                             F, material, _cal_cauchy_stress_,
                                             F0, return_S_W)
#%%
def cal_1pk_stress_force(node_x, element, d_sf_dX, det_dX_dr, material, element_orientation, cal_1pk_stress,
                         F0=None, return_F_S_W=False, return_stiffness=None, return_force_at_element=False):
    def _cal_1pk_stress_(F, material, create_graph, return_W):
        return cal_1pk_stress(F, material, element_orientation, create_graph, return_W)
    return _cal_1pk_stress_force_(node_x, element, d_sf_dX, det_dX_dr, material, _cal_1pk_stress_,
                                  F0, return_F_S_W, return_stiffness, return_force_at_element)
