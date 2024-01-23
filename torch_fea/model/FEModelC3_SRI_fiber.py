from torch_fea.model.FEModelC3_SRI import cal_cauchy_stress_force as _cal_cauchy_stress_force_
from torch_fea.model.FEModelC3_SRI import cal_cauchy_stress_force_inverse as _cal_cauchy_stress_force_inverse_
from torch_fea.model.FEModelC3_SRI import cal_1pk_stress_force as _cal_1pk_stress_force_
from torch_fea.model.FEModelC3_SRI import cal_F_tensor, cal_d_sf_dx_and_dx_dr, cal_d_sf_dX_and_dX_dr
from torch_fea.model.FEModelC3_SRI import cal_pressure_force
from torch_fea.model.FEModelC3_SRI import cal_strain_energy, cal_potential_energy
from torch_fea.model.FEModelC3_SRI import interpolate
#%%
def cal_cauchy_stress_force(node_x, element, d_sf_dX_dev, d_sf_dX_vol, material, element_orientation, cal_cauchy_stress,
                            F0_dev=None, F0_vol=None, return_F_S_W=False, return_stiffness=None, return_force_at_element=False):
    def _cal_cauchy_stress_(F_dev, F_vol, material, create_graph, return_W):
        return cal_cauchy_stress(F_dev, F_vol, material, element_orientation, create_graph, return_W)
    return _cal_cauchy_stress_force_(node_x, element, d_sf_dX_dev, d_sf_dX_vol, material, _cal_cauchy_stress_,
                                     F0_dev, F0_vol, return_F_S_W, return_stiffness, return_force_at_element)
#%%
def cal_cauchy_stress_force_inverse(d_sf_dx_dev, d_sf_dx_vol, det_dx_dr_dev, det_dx_dr_vol,
                                    n_nodes, element, F_dev, F_vol, material, element_orientation, cal_cauchy_stress,
                                    F0_dev=None, F0_vol=None, return_S_W=False):
    def _cal_cauchy_stress_(F_dev, F_vol, material, create_graph, return_W):
        return cal_cauchy_stress(F_dev, F_vol, material, element_orientation, create_graph, return_W)
    return _cal_cauchy_stress_force_inverse_(d_sf_dx_dev, d_sf_dx_vol, det_dx_dr_dev, det_dx_dr_vol,
                                             n_nodes, element, F_dev, F_vol, material, _cal_cauchy_stress_,
                                             F0_dev, F0_vol, return_S_W)
#%%
def cal_1pk_stress_force(node_x, element, d_sf_dX_dev, d_sf_dX_vol, det_dX_dr_dev, det_dX_dr_vol,
                         material, element_orientation, cal_1pk_stress, F0_dev=None, F0_vol=None,
                         return_F_S_W=False, return_stiffness=None, return_force_at_element=False):
    def _cal_1pk_stress_(F_dev, F_vol, material, create_graph, return_W):
        return cal_1pk_stress(F_dev, F_vol, material, element_orientation, create_graph, return_W)
    return _cal_1pk_stress_force_(node_x, element, d_sf_dX_dev, d_sf_dX_vol, det_dX_dr_dev, det_dX_dr_vol,
                                  material, _cal_1pk_stress_, F0_dev, F0_vol,
                                  return_F_S_W, return_stiffness, return_force_at_element)
