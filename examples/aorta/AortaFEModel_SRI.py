import numpy as np
import torch
from torch import matmul
from torch_fea.model.FEModelC3_SRI_fiber import cal_F_tensor, cal_d_sf_dX_and_dX_dr
from torch_fea.model.FEModelC3_SRI_fiber import cal_strain_energy, cal_potential_energy
from torch_fea.model.FEModelC3_SRI_fiber import cal_pressure_force, cal_1pk_stress_force
from torch_fea.utils.functions import cal_cauchy_stress_from_1pk_stress
from aorta_element_orientation import cal_element_orientation
#%%
from scipy.sparse import csr_matrix
def process_H(H, free_node):
    idlist=np.concatenate([3*free_node.reshape(-1,1),
                           3*free_node.reshape(-1,1)+1,
                           3*free_node.reshape(-1,1)+2], axis=1)
    idlist=idlist.reshape(-1)
    A=H.detach().cpu()
    row=A.indices()[0].numpy()
    col=A.indices()[1].numpy()
    value=A.values().numpy()
    A=csr_matrix((value, (row, col)), shape=H.shape)
    A=A[idlist,:]
    A=A[:,idlist]    
    return A
#%%
class AortaFEModel:
    def __init__(self, node_x, element, node_X, boundary0, boundary1, pressure_surface,
                 material, element_orientation, cal_1pk_stress, dtype, device, mode):
        #mode: 'inflation'   to get node_x,   given node_X and material
        #      'inverse_p0'  to get node_X,   given node_x and material
        #      'inverse_mat' to get material, given node_x and node_X
        #This model works for SRI mat_model and hex8 elements
        if mode != 'inflation' and mode != 'inverse_p0' and mode != 'inverse_mat':
            raise ValueError('mode is unknown')
        self.mode=mode
        self.dtype=dtype
        self.device=device
        self.state={}
        if element.shape[1] == 8:
            self.state['element_type']='hex8'
            self.state['n_integration_points_solid']=8
            self.state['n_integration_points_surface']=1
        else:
            raise ValueError('only support hex8')
        self.state['node_x']=node_x
        self.state['element']=element
        self.state['node_X']=node_X
        self.state['boundary0']=boundary0
        self.state['boundary1']=boundary1
        if node_X is not None:
            free_node=np.arange(0, node_X.shape[0], 1)
        elif node_x is not None:
            free_node=np.arange(0, node_x.shape[0], 1)
        else:
            raise ValueError('node_X is None and node_x is None')
        free_node=np.setdiff1d(free_node, boundary0.view(-1).numpy())
        free_node=np.setdiff1d(free_node, boundary1.view(-1).numpy())
        self.state['free_node']=free_node #torch.tensor(free_node, dtype=torch.int64, device=device)
        self.state['pressure_surface']=pressure_surface
        self.state['material']=material
        self.state['element_orientation']=element_orientation
        self.state['F0_dev']=None
        self.state['F0_vol']=None
        self.cal_1pk_stress=cal_1pk_stress
        if mode == 'inflation':
            self.initialize_for_inflation()
        elif mode == 'inverse_p0':
            self.initialize_for_inverse_p0()
        elif mode == 'inverse_mat':
            self.initialize_for_inverse_mat()
        else:
            raise ValueError('mode is unknown')

        if (node_x is not None) and (node_X is not None):
            #init u_field
            if self.state['u_field'] is not None:
                u_field_full=node_x-node_X
                self.state['u_field'].data.copy_(u_field_full.data[self.state['free_node']])

    def initialize_for_inflation(self):
        state=self.state
        state['material']=state['material'].to(self.dtype).to(self.device)
        state['node_X']=state['node_X'].to(self.dtype).to(self.device)
        state['element']=state['element'].to(self.device)
        state['pressure_surface']=state['pressure_surface'].to(self.device)
        state['u_field']=torch.zeros((state['free_node'].shape[0],3),
                                     dtype=self.dtype, device=self.device, requires_grad=True)
        state['disp0']=0 #zero displacement of boundary0
        state['disp1']=0 #zero displacement of boundary1
        d_sf_dX_dev, dX_dr_dev, det_dX_dr_dev=cal_d_sf_dX_and_dX_dr(state['node_X'], state['element'], 8)
        d_sf_dX_vol, dX_dr_vol, det_dX_dr_vol=cal_d_sf_dX_and_dX_dr(state['node_X'], state['element'], 1)
        state['d_sf_dX_dev']=d_sf_dX_dev
        state['d_sf_dX_vol']=d_sf_dX_vol
        state['det_dX_dr_dev']=det_dX_dr_dev
        state['det_dX_dr_vol']=det_dX_dr_vol
        mask=torch.ones_like(state['node_X'])
        mask[state['boundary0']]=0
        mask[state['boundary1']]=0
        state['mask']=mask
        if state['element_orientation'] is None:
            state['element_orientation']=cal_element_orientation(state['node_X'], state['element'])
        state['element_orientation']=state['element_orientation'].to(self.dtype).to(self.device)

    def initialize_for_inverse_p0(self):
        state=self.state
        state['material']=state['material'].to(self.dtype).to(self.device)
        state['node_x']=state['node_x'].to(self.dtype).to(self.device)
        state['element']=state['element'].to(self.device)
        state['pressure_surface']=state['pressure_surface'].to(self.device)
        state['u_field']=torch.zeros((state['free_node'].shape[0],3),
                                     dtype=self.dtype, device=self.device, requires_grad=True)
        state['disp0']=0 #zero displacement of boundary0
        state['disp1']=0 #zero displacement of boundary1
        mask=torch.ones_like(state['node_x'])
        mask[state['boundary0']]=0
        mask[state['boundary1']]=0
        state['mask']=mask

    def initialize_for_inverse_mat(self):
        state=self.state
        state['material']=state['material'].to(self.dtype).to(self.device)
        state['node_x']=state['node_x'].to(self.dtype).to(self.device)
        state['node_X']=state['node_X'].to(self.dtype).to(self.device)
        state['element']=state['element'].to(self.device)
        state['pressure_surface']=state['pressure_surface'].to(self.device)
        state['u_field']=None
        state['disp0']=0 #zero displacement of boundary0
        state['disp1']=0 #zero displacement of boundary1
        d_sf_dX_dev, dX_dr_dev, det_dX_dr_dev=cal_d_sf_dX_and_dX_dr(state['node_X'], state['element'], 8)
        d_sf_dX_vol, dX_dr_vol, det_dX_dr_vol=cal_d_sf_dX_and_dX_dr(state['node_X'], state['element'], 1)
        state['d_sf_dX_dev']=d_sf_dX_dev
        state['d_sf_dX_vol']=d_sf_dX_vol
        state['det_dX_dr_dev']=det_dX_dr_dev
        state['det_dX_dr_vol']=det_dX_dr_vol
        state['element_orientation']=cal_element_orientation(state['node_X'], state['element'])
        mask=torch.ones_like(state['node_X'])
        mask[state['boundary0']]=0
        mask[state['boundary1']]=0
        state['mask']=mask

    def set_material(self, material):
        self.state['material']=material.to(self.dtype).to(self.device)

    def set_node_x(self, node_x):
        if self.mode == 'inverse_p0' or self.mode == 'inverse_mat':
            self.state['node_x']=node_x.to(self.dtype).to(self.device)
        elif self.mode == 'inflation':
            raise ValueError('cannot set node_x when mode is inflation')
        else:
            raise ValueError('mode is unknown')

    def set_node_X(self, node_X):
        if self.mode == 'inflation':
            self.state['node_X']=node_X.to(self.dtype).to(self.device)
            self.initialize_for_inflation()
        elif self.mode == 'inverse_mat':
            self.state['node_X']=node_X.to(self.dtype).to(self.device)
            self.initialize_for_inverse_mat()
        elif self.mode == 'inverse_p0':
            raise ValueError('cannot set node_x when mode is inverse_p0')
        else:
            raise ValueError('mode is unknown')

    def get_node_x(self, clone=True, detach=False):
        if self.mode == 'inverse_p0' or self.mode == 'inverse_mat':
            node_x=self.state['node_x']
        elif self.mode == 'inflation':
            node_X=self.state['node_X']
            u_field=self.get_u_field()
            node_x=node_X+u_field
        else:
            raise ValueError('mode is unknown')
        if clone==True:
            node_x=node_x.clone()
        if detach==True:
            node_x=node_x.detach()
        return node_x

    def get_node_X(self, clone=True, detach=False):
        if self.mode == 'inflation' or self.mode == 'inverse_mat':
            node_X=self.state['node_X']
        elif self.mode == 'inverse_p0':
            node_x=self.state['node_x']
            u_field=self.get_u_field()
            node_X=node_x-u_field
        else:
            raise ValueError('mode is unknown')
        if clone==True:
            node_X=node_X.clone()
        if detach==True:
            node_X=node_X.detach()
        return node_X

    def set_boundary_displacement(self, disp0=None, disp1=None):
        #prescribed displacement of boundary0 and boundary1
        state=self.state
        if disp0 is not None:
            state['disp0']=disp0.to(self.dtype).to(self.device)
        if disp1 is not None:
            state['disp1']=disp1.to(self.dtype).to(self.device)

    def set_F0(self, F0_dev, F0_vol):
        #pre/residual deformation
        if (F0_dev is None and F0_vol is not None) or (F0_dev is not None and F0_vol is None):
            raise ValueError('F0_dev and F0_vol must be None or not None together')
        if F0_dev is not None and F0_vol is not None:
            self.state['F0_dev']=F0_dev.to(self.dtype).to(self.device)
            self.state['F0_vol']=F0_vol.to(self.dtype).to(self.device)
        else:
            self.state['F0_dev']=None
            self.state['F0_vol']=None

    def get_F0(self, clone=True, detach=False):
        #pre/residual deformation
        F0_dev=self.state['F0_dev']
        F0_vol=self.state['F0_vol']
        if clone == True and F0_dev is not None and F0_vol is not None:
            F0_dev=F0_dev.clone()
            F0_vol=F0_vol.clone()
        if detach==True and F0_dev is not None and F0_vol is not None:
            F0_dev=F0_dev.detach()
            F0_vol=F0_vol.detach()
        return F0_dev, F0_vol

    def set_u_field(self, u_field_full=None, u_field_free=None, data_copy_=False, requires_grad=True):
        if (u_field_full is not None) and (u_field_free is not None) :
            raise ValueError('u_field_full is not None and u_field_free is not None')
        if u_field_full is not None:
            if data_copy_ == True:
                self.state['u_field'].data.copy_(u_field_full.data[self.state['free_node']])
            else:
                self.state['u_field']=u_field_full[self.state['free_node']]
        elif u_field_free is not None:
            if data_copy_ == True:
                self.state['u_field'].data.copy_(u_field_free.data)
            else:
                self.state['u_field']=u_field_free
        else:
            raise ValueError('u_field_full and u_field_free are None')
        if requires_grad == True and self.state['u_field'].requires_grad == False:
            self.state['u_field'].requires_grad=True
        elif requires_grad == False and self.state['u_field'].requires_grad == True:
            raise ValueError('requires_grad=False cannot be done because self.state["u_field"].requires_grad is True')

    def get_u_field(self):
        state=self.state
        if state['node_X'] is not None:
            u_field=torch.zeros_like(state['node_X'], dtype=self.dtype, device=self.device)
        elif state['node_x'] is not None:
            u_field=torch.zeros_like(state['node_x'], dtype=self.dtype, device=self.device)
        u_field[state['boundary0']]=state['disp0']
        u_field[state['boundary1']]=state['disp1']
        u_field[state['free_node']]=state['u_field']
        return u_field

    def cal_energy_and_force_for_inflation(self, pressure, return_stiffness=None):
        state=self.state
        u_field=self.get_u_field()
        node_x=state['node_X']+u_field
        Output_ext=cal_pressure_force(pressure, node_x, state['pressure_surface'], 1, return_stiffness=return_stiffness)
        if return_stiffness is None:
            force_ext=Output_ext
        else:
            force_ext, H_ext=Output_ext
            if isinstance(H_ext, torch.Tensor) == True:
                H_ext=process_H(H_ext, state['free_node'])
        Output_int=cal_1pk_stress_force(node_x,
                                        state['element'],
                                        state['d_sf_dX_dev'],
                                        state['d_sf_dX_vol'],
                                        state['det_dX_dr_dev'],
                                        state['det_dX_dr_vol'],
                                        state['material'],
                                        state['element_orientation'],
                                        self.cal_1pk_stress,
                                        F0_dev=state['F0_dev'],
                                        F0_vol=state['F0_vol'],
                                        return_F_S_W=True,
                                        return_stiffness=return_stiffness,
                                        return_force_at_element=True)
        if return_stiffness is None:
            force_int, Fd, Fv, Sd, Sv, Wd, Wv, force_int_at_element=Output_int
        else:
            force_int, Fd, Fv, Sd, Sv, Wd, Wv, H_int, force_int_at_element=Output_int
            H_int=process_H(H_int, state['free_node'])
            H=H_int-H_ext
        SE=cal_strain_energy(state['element_type'], Wd, Wv, state['det_dX_dr_dev'], state['det_dX_dr_vol'], reduction='sum')
        force_ext1=force_ext*state['mask']
        #force_ext2 is exernal force on boundary0 and boundary1
        #it is passive: it will adjust itself to match force_int on boundary0 and boundary1
        force_ext2=force_int*(1-state['mask'])
        force_ext=force_ext1+force_ext2
        TPE1=SE-cal_potential_energy(force_ext1, u_field)
        PE_passive=cal_potential_energy(force_ext2, u_field)
        TPE2=TPE1-PE_passive
        out={'TPE1':TPE1, 'TPE2':TPE2, 'SE':SE,
             'force_int':force_int, 'force_ext':force_ext,
             'force_int_at_element':force_int_at_element,
             'F':Fv, 'u_field':u_field}
        if return_stiffness is not None:
            out['H']=H
        return out
        #------------------------------------------------------------------------------------------
        # TPE1 should be used as the objective for linesearch optimization
        # TPE2 is the full total potential energy, force_int-force_ext is grad(TPE2, u_field)
        # TPE2 contains the passive energy on boundary(0&1) and should not be used for linesearch
        # two special cases:
        # if boundary displacement is 0, then PE_passive is 0 and TPE1 is TPE2
        # if boundary displacement is not 0 and pressure is 0, then TPE1 is SE and TPE1 != TPE2
        #------------------------------------------------------------------------------------------

    def cal_energy_and_force_for_inverse_p0(self, pressure, return_stiffness=None, detach_X=True):
        state=self.state
        node_x=state['node_x']
        u_field=self.get_u_field()
        node_X=node_x-u_field
        if detach_X == True:
            node_X=node_X.detach()
            node_x=node_X+u_field
        Output_ext=cal_pressure_force(pressure, node_x, state['pressure_surface'], 1, return_stiffness=return_stiffness)
        if return_stiffness is None:
            force_ext=Output_ext
        else:
            force_ext, H_ext=Output_ext
            if isinstance(H_ext, torch.Tensor) == True:
                H_ext=process_H(H_ext, state['free_node'])
        d_sf_dX_dev, dX_dr_dev, det_dX_dr_dev=cal_d_sf_dX_and_dX_dr(node_X, state['element'], 8)
        d_sf_dX_vol, dX_dr_vol, det_dX_dr_vol=cal_d_sf_dX_and_dX_dr(node_X, state['element'], 1)
        orientation=cal_element_orientation(node_X, state['element'])
        Output_int=cal_1pk_stress_force(node_x,
                                        state['element'],
                                        d_sf_dX_dev,
                                        d_sf_dX_vol,
                                        det_dX_dr_dev,
                                        det_dX_dr_vol,
                                        state['material'],
                                        orientation,
                                        self.cal_1pk_stress,
                                        F0_dev=state['F0_dev'],
                                        F0_vol=state['F0_vol'],
                                        return_F_S_W=True,
                                        return_stiffness=return_stiffness,
                                        return_force_at_element=True)
        if return_stiffness is None:
            force_int, Fd, Fv, Sd, Sv, Wd, Wv, force_int_at_element=Output_int
        else:
            force_int, Fd, Fv, Sd, Sv, Wd, Wv, H_int, force_int_at_element=Output_int
            H_int=process_H(H_int, state['free_node'])
            H=H_int-H_ext
        SE=cal_strain_energy(state['element_type'], Wd, Wv, det_dX_dr_dev, det_dX_dr_vol, reduction='sum')
        force_ext1=force_ext*state['mask']
        force_ext2=force_int*(1-state['mask'])
        force_ext=force_ext1+force_ext2
        TPE1=SE-cal_potential_energy(force_ext1, u_field)
        PE_passive=cal_potential_energy(force_ext2, u_field)
        TPE2=TPE1-PE_passive
        out={'TPE1':TPE1, 'TPE2':TPE2, 'SE':SE,
             'force_int':force_int, 'force_ext':force_ext,
             'force_int_at_element':force_int_at_element,
             'F':Fv, 'u_field':u_field}
        if return_stiffness is not None:
            out['H']=H
        return out

    def cal_energy_and_force_for_inverse_mat(self, pressure):
        #ex vivo: node_X and node_x are known
        state=self.state
        force_ext=cal_pressure_force(pressure, state['node_x'], state['pressure_surface'], 1)
        output_int=cal_1pk_stress_force(state['node_x'],
                                        state['element'],
                                        state['d_sf_dX_dev'],
                                        state['d_sf_dX_vol'],
                                        state['det_dX_dr_dev'],
                                        state['det_dX_dr_vol'],
                                        state['material'],
                                        state['element_orientation'],
                                        self.cal_1pk_stress,
                                        F0_dev=state['F0_dev'],
                                        F0_vol=state['F0_vol'],
                                        return_F_S_W=True,
                                        return_force_at_element=True)
        force_int, Fd, Fv, Sd, Sv, Wd, Wv, force_int_at_element=output_int
        SE=cal_strain_energy(state['element_type'], Wd, Wv, state['det_dX_dr_dev'], state['det_dX_dr_vol'], reduction='sum')
        u_field=state['node_x']-state['node_X']
        force_ext1=force_ext*state['mask']
        force_ext2=force_int*(1-state['mask'])
        force_ext=force_ext1+force_ext2
        TPE1=SE-cal_potential_energy(force_ext1, u_field)
        PE_passive=cal_potential_energy(force_ext2, u_field)
        TPE2=TPE1-PE_passive
        out={'TPE1':TPE1, 'TPE2':TPE2, 'SE':SE,
             'force_int':force_int, 'force_ext':force_ext,
             'force_int_at_element':force_int_at_element,
             'F':Fv}
        return out

    def cal_energy_and_force(self, pressure, return_stiffness=None):
        if self.mode == 'inflation':
            return self.cal_energy_and_force_for_inflation(pressure, return_stiffness)
        elif self.mode == 'inverse_p0':
            return self.cal_energy_and_force_for_inverse_p0(pressure, return_stiffness)
        elif self.mode == 'inverse_mat':
            return self.cal_energy_and_force_for_inverse_mat(pressure)
        else:
            raise ValueError('mode is unknown')

    def cal_F_tensor(self):
        state=self.state
        if self.mode == 'inflation':
            node_X=state['node_X']
            u_field=self.get_u_field()
            node_x=node_X+u_field
            Fd=cal_F_tensor(node_x, state['element'], node_X, 8, state['d_sf_dX_dev'], state['F0_dev'])
            Fv=cal_F_tensor(node_x, state['element'], node_X, 1, state['d_sf_dX_vol'], state['F0_vol'])
        elif self.mode == 'inverse_p0':
            node_x=state['node_x']
            u_field=self.get_u_field()
            node_X=node_x-u_field
            Fd=cal_F_tensor(node_x, state['element'], node_X, 8, None, state['F0_dev'])
            Fv=cal_F_tensor(node_x, state['element'], node_X, 1, None, state['F0_vol'])
        elif self.mode == 'inverse_mat':
            node_x=state['node_x']
            node_X=state['node_X']
            Fd=cal_F_tensor(node_x, state['element'], node_X, 8, state['d_sf_dX_dev'], state['F0_dev'])
            Fv=cal_F_tensor(node_x, state['element'], node_X, 1, state['d_sf_dX_vol'], state['F0_vol'])
        else:
            raise ValueError('mode is unknown')
        return Fd, Fv

    def cal_stress(self, stress, create_graph, return_W, local_sys=False):
        state=self.state
        if self.mode == 'inflation':
            node_X=state['node_X']
            u_field=self.get_u_field()
            node_x=node_X+u_field
            orientation=self.state['element_orientation']
        elif self.mode == 'inverse_p0':
            node_x=state['node_x']
            u_field=self.get_u_field()
            node_X=node_x-u_field
            orientation=cal_element_orientation(node_X, state['element'])
        elif self.mode == 'inverse_mat':
            node_x=state['node_x']
            node_X=state['node_X']
            orientation=self.state['element_orientation']
        else:
            raise ValueError('mode is unknown')
        Fd, Fv=self.cal_F_tensor()        
        Sd, Sv, Wd, Wv=self.cal_1pk_stress(Fd, Fv, state['material'], orientation,
                                           create_graph=create_graph, return_W=True)
        if stress == '1pk':
            pass
        elif stress == 'cauchy':
            Sd=cal_cauchy_stress_from_1pk_stress(Sd, Fd)
            Sv=cal_cauchy_stress_from_1pk_stress(Sv, Fv)            
        else:
            raise ValueError('unknown stress:'+str(stress))

        if local_sys == True:
            #orientation.shape (M,3,3)
            orientation=orientation.view(-1,1,3,3)
            orientation_t=orientation.permute(0,1,3,2)
            Sd=matmul(matmul(orientation_t, Sd), orientation)
            Sv=matmul(matmul(orientation_t, Sv), orientation)

        S=Sd+Sv
        W=Wd+Wv
        if return_W == False:
            return S
        else:
            return S, W
