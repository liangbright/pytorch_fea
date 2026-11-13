# The finite element model of the aorta examples, for three problems (mode): inflation (forward), inverse_p0 and
# inverse_mat.
#
# The stiffness matrix (return_stiffness) can only be the sparse matrix: return_stiffness='sparse' gives it, and None
# (or 'none') gives no matrix. The solvers of the examples use it as H0, the matrix of a sparse linear solve, on the
# device of the model: on the CPU (pypardiso) if the model is on the CPU, on the GPU (cuDSS) if it is on the GPU
# (process_H). Any other value stops with an error before anything is computed. Mode inverse_mat makes no stiffness and
# ignores return_stiffness: its optimizer needs no H or H0.
#
# A dense matrix is not supported because it does not fit in memory for these meshes. The hex8 and tet4 meshes have
# 10000 nodes, so the matrix has 30000 x 30000 entries, 7.2 GB in float64; the tet10 mesh has 59700 nodes, 257 GB.
# It would hold nothing that the sparse matrix does not; on a small mesh, H.toarray() of the sparse matrix gives it.
#
# A diagonal matrix is not supported because the solver does not converge with it. This was tried on 2026-11-12 with
# FE_lbfgs_residual, which can scale its steps by a diagonal (set_invH0_diag(1/diag(H)) in place of set_H0(H)), on
# the forward example (hex8, shape 171, 16 kPa). Not one load step converged, and the run gave up after 101 tries.
# In the first load step (1% of the pressure) the largest residual, relative to the mean element force, was still
# about 10 after 200 iterations, where 0.005 is needed; with the sparse matrix it was 3e-9 after 100 iterations.
# The diagonal only evens out the scale of the unknowns. What makes this problem hard is the coupling between them:
# the wall is thin and nearly incompressible (k is about 2000 times c0), so the displacements of neighbouring nodes
# depend strongly on each other. The sparse matrix has that coupling, and the few correction pairs of L-BFGS cannot
# build it from a diagonal start.
#%%
import warnings
import numpy as np
import torch
from torch import matmul
from torch_fea.model.model_hyperelastic_solid_with_fiber import cal_F_tensor, cal_d_sf_dX_and_dX_dr
from torch_fea.model.model_hyperelastic_solid_with_fiber import cal_strain_energy, cal_potential_energy
from torch_fea.model.model_hyperelastic_solid_with_fiber import cal_pressure_force, cal_1pk_stress_force
from torch_fea.model.model_hyperelastic_solid_with_fiber import cal_integration_point_volume
from torch_fea.model.model_stiffness import cal_sparse_stiffness_matrix
from torch_fea.utility.function import cal_cauchy_stress_from_1pk_stress
from aorta_element_orientation import cal_element_orientation
#%%
from scipy.sparse import csr_matrix
#from time import time
def process_H(H, free_node):
    #H: the sparse stiffness matrix of all nodes, a torch sparse COO tensor (3N x 3N)
    #returns the rows and columns of the free nodes, in the order of free_node, on the device of H:
    #H on the GPU: a torch sparse CSR tensor with int32 indices on that GPU, for the GPU solver
    #              (cuDSS, torch_fea/optimizer/cuDSS_linear_sparse_solver.py)
    #H on the CPU: a scipy.sparse.csr_matrix, for the CPU solver (pypardiso)
    #the solvers (FE_lbfgs_residual.py, FE_lbfgs_loss.py) pick cuDSS or pypardiso from the type of the matrix
    if not (isinstance(H, torch.Tensor) and H.layout == torch.sparse_coo):
        raise ValueError("process_H: H must be a torch sparse COO tensor (return_stiffness='sparse'), see the top of AortaFEModel.py")
    if H.is_cuda:
        return process_H_gpu(H, free_node)
    else:
        return process_H_cpu(H, free_node)

def process_H_cpu(H, free_node):
    #t0=time()
    idlist=np.concatenate([3*free_node.reshape(-1,1),
                           3*free_node.reshape(-1,1)+1,
                           3*free_node.reshape(-1,1)+2], axis=1)
    idlist=idlist.reshape(-1)
    #t1=time()
    A=H.detach().cpu()
    row=A.indices()[0].numpy()
    col=A.indices()[1].numpy()
    value=A.values().numpy()
    #t2=time()
    A=csr_matrix((value, (row, col)), shape=H.shape)
    #t3=time()
    A=A[idlist,:]
    A=A[:,idlist]
    #t4=time()
    #A=A.tocsr()
    #print('t1-t0', t1-t0, 't2-t1', t2-t1, 't3-t2', t3-t2, 't4-t3', t4-t3)
    return A

def process_H_gpu(H, free_node):
    #the matrix of process_H_cpu, built on the GPU as a torch sparse CSR tensor with int32 indices
    #entries that are 0 are kept (scipy drops the entries of H_int-H_ext that cancel), so that the sparsity pattern
    #is the same at every call and cuDSS keeps its reordering and symbolic factorization
    if not H.is_cuda:
        raise ValueError("process_H_gpu: H must be on the GPU")
    H=H.detach().coalesce()
    device=H.device
    free_node=torch.as_tensor(free_node, dtype=torch.int64, device=device).view(-1)
    idlist=(3*free_node.view(-1,1)+torch.arange(0, 3, device=device).view(1,3)).view(-1)
    #the new index of every dof, -1 for the dofs of the boundary nodes
    new_index=torch.full((H.shape[0],), -1, dtype=torch.int64, device=device)
    new_index[idlist]=torch.arange(0, idlist.shape[0], device=device)
    row=new_index[H.indices()[0]]
    col=new_index[H.indices()[1]]
    keep=(row >= 0)&(col >= 0)
    row=row[keep]
    col=col[keep]
    value=H.values()[keep]
    n=idlist.shape[0]
    #CSR: the entries sorted by row, then by column (they already are if free_node is sorted, as np.setdiff1d makes it)
    key=row*n+col
    if not bool((key[1:] > key[:-1]).all()):
        order=torch.argsort(key)
        row=row[order]
        col=col[order]
        value=value[order]
    crow=torch.zeros(n+1, dtype=torch.int64, device=device)
    crow[1:]=torch.cumsum(torch.bincount(row, minlength=n), dim=0)
    return torch.sparse_csr_tensor(crow.to(torch.int32), col.to(torch.int32), value, size=(n, n))
#%%
class AortaFEModel:
    def __init__(self, element_type, node_x, element, node_X, boundary0, boundary1, pressure_surface,
                 material, element_orientation, cal_1pk_stress, dtype, device, mode):
        #element_type: 'tet4', 'tet10', 'wedge6', or 'hex8'
        #device: also where the stiffness matrix H (return_stiffness='sparse') is built and solved, see process_H
        #mode: 'inflation'   to get node_x,   given node_X and material
        #      'inverse_p0'  to get node_X,   given node_x and material
        #      'inverse_mat' to get material, given node_x and node_X
        #cal_1pk_stress(F, material, orientation, create_graph, return_W, V): the material functions get V,
        #the volume of each integration point, for the mean dilatation (torch_fea/material/material_GOH.py)
        if mode != 'inflation' and mode != 'inverse_p0' and mode != 'inverse_mat':
            raise ValueError('mode is unknown')
        self.mode=mode
        self.dtype=dtype
        self.device=device
        self.state={}
        self.state['element_type']=element_type
        if element_type == 'tet4':
            self.state['n_integration_points_solid']=1
            self.state['n_integration_points_surface']=3    # tri3 faces: FEBio's pressure-load default (FE_TRI3G3)
        elif element_type == 'tet10':
            self.state['n_integration_points_solid']=8    # 8(slow) or 4(fast)
            self.state['n_integration_points_surface']=7    # tri6 faces: FEBio's pressure-load default (FE_TRI6G7)
        elif element_type == 'wedge6':
            self.state['n_integration_points_solid']=6    # as FEBio (FEPenta6G6); with 2, one element alone has a spurious zero-energy mode
            self.state['n_integration_points_surface']=4 if pressure_surface.shape[1] == 4 else 3    # quad4 or tri3 faces: FEBio's pressure-load defaults (FE_QUAD4G4, FE_TRI3G3)
        elif element_type == 'hex8':
            self.state['n_integration_points_solid']=8
            self.state['n_integration_points_surface']=4    # quad4 faces: FEBio's pressure-load default (FE_QUAD4G4)
        else:
            raise ValueError('only support tet4, tet10, wedge6, and hex8')
        self.state['node_x']=node_x
        self.state['element']=element
        self.state['node_X']=node_X
        if not torch.is_tensor(boundary0):
            boundary0=torch.tensor(boundary0, dtype=torch.int64)
        if not torch.is_tensor(boundary1):
            boundary1=torch.tensor(boundary1, dtype=torch.int64) 
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
        #True if the frame is given; else the model computes it from node_X (set_node_X uses this)
        self.state['element_orientation_given']=(element_orientation is not None)
        self.state['F0']=None
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
        d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dX_and_dX_dr(state['element_type'], state['node_X'], state['element'],
                                                        state['n_integration_points_solid'])
        state['d_sf_dX']=d_sf_dX
        state['det_dX_dr']=det_dX_dr
        #elements of the given mesh node_X with det(dX/dr) <= 0 at an integration point are inverted or folded,
        #and their force and energy are wrong: warn, do not stop (the caller decides)
        element_bad=torch.where((det_dX_dr.detach().reshape(det_dX_dr.shape[0],-1) <= 0).any(dim=1))[0].cpu()
        state['element_non_positive_det']=element_bad
        if len(element_bad) > 0:
            warnings.warn('node_X: '+str(len(element_bad))+' elements have det(dX/dr) <= 0 at an integration point'
                          +' (smallest '+'{:.4g}'.format(float(det_dX_dr.min()))+'), elements: '+str(element_bad[:20].tolist()))
        mask=torch.ones_like(state['node_X'])
        mask[state['boundary0']]=0
        mask[state['boundary1']]=0
        state['mask']=mask
        if state['element_orientation'] is None:
            state['element_orientation']=cal_element_orientation(state['element_type'], state['node_X'], state['element'])
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
        #elements of the given mesh node_x (the start of the unknown node_X, u_field=0) with det(dx/dr) <= 0
        #at an integration point are inverted or folded: warn, do not stop (the caller decides)
        _, _, det_dx_dr=cal_d_sf_dX_and_dX_dr(state['element_type'], state['node_x'], state['element'],
                                              state['n_integration_points_solid'])
        element_bad=torch.where((det_dx_dr.detach().reshape(det_dx_dr.shape[0],-1) <= 0).any(dim=1))[0].cpu()
        state['element_non_positive_det']=element_bad
        if len(element_bad) > 0:
            warnings.warn('node_x: '+str(len(element_bad))+' elements have det(dx/dr) <= 0 at an integration point'
                          +' (smallest '+'{:.4g}'.format(float(det_dx_dr.min()))+'), elements: '+str(element_bad[:20].tolist()))

    def initialize_for_inverse_mat(self):
        state=self.state
        state['material']=state['material'].to(self.dtype).to(self.device)
        state['node_x']=state['node_x'].to(self.dtype).to(self.device)
        state['node_X']=state['node_X'].to(self.dtype).to(self.device)
        state['element']=state['element'].to(self.device)
        state['pressure_surface']=state['pressure_surface'].to(self.device)
        state['u_field']=None #not used
        state['disp0']=0 #zero displacement of boundary0
        state['disp1']=0 #zero displacement of boundary1
        d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dX_and_dX_dr(state['element_type'], state['node_X'], state['element'],
                                                        state['n_integration_points_solid'])
        state['d_sf_dX']=d_sf_dX
        state['det_dX_dr']=det_dX_dr
        #elements of the given mesh node_X with det(dX/dr) <= 0 at an integration point are inverted or folded,
        #and their force and energy are wrong: warn, do not stop (the caller decides)
        element_bad=torch.where((det_dX_dr.detach().reshape(det_dX_dr.shape[0],-1) <= 0).any(dim=1))[0].cpu()
        state['element_non_positive_det']=element_bad
        if len(element_bad) > 0:
            warnings.warn('node_X: '+str(len(element_bad))+' elements have det(dX/dr) <= 0 at an integration point'
                          +' (smallest '+'{:.4g}'.format(float(det_dX_dr.min()))+'), elements: '+str(element_bad[:20].tolist()))
        mask=torch.ones_like(state['node_X'])
        mask[state['boundary0']]=0
        mask[state['boundary1']]=0
        state['mask']=mask
        if state['element_orientation'] is None:
            state['element_orientation']=cal_element_orientation(state['element_type'], state['node_X'], state['element'])
        state['element_orientation']=state['element_orientation'].to(self.dtype).to(self.device)

    def set_material(self, material):
        self.state['material']=material.to(self.dtype).to(self.device)

    def set_node_x(self, node_x):
        if self.mode == 'inverse_p0' or self.mode == 'inverse_mat':
            self.state['node_x']=node_x.to(self.dtype).to(self.device)
        elif self.mode == 'inflation':
            raise ValueError('cannot set node_x when mode is inflation')
        else:
            raise ValueError('mode is unknown')

    def set_node_X(self, node_X, element_orientation=None):
        #a new reference geometry node_X for the same mesh (element, boundary, pressure surface)
        #only what depends on node_X changes: d_sf_dX, det_dX_dr (and its check) and the element frame;
        #u_field (the same tensor with the same values, so an optimizer keeps working on it), disp0 and disp1 are kept
        #element_orientation: the frame of the new geometry; if None, the frame is computed from the new node_X
        #  when the model computed it before, and it is kept when it was given (it cannot be computed for tet meshes)
        state=self.state
        if self.mode == 'inflation' or self.mode == 'inverse_mat':
            u_field=state['u_field']; disp0=state['disp0']; disp1=state['disp1']
            state['node_X']=node_X.to(self.dtype).to(self.device)
            if element_orientation is not None:
                state['element_orientation']=element_orientation
                state['element_orientation_given']=True
            elif state['element_orientation_given'] == False:
                state['element_orientation']=None #computed from the new node_X in initialize_for_*
            if self.mode == 'inflation':
                self.initialize_for_inflation()
            else:
                self.initialize_for_inverse_mat()
            state['u_field']=u_field
            state['disp0']=disp0; state['disp1']=disp1
        elif self.mode == 'inverse_p0':
            raise ValueError('cannot set node_X when mode is inverse_p0: node_X is the unknown')
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

    def set_F0(self, F0):
        #pre/residual deformation
        if F0 is not None:
            self.state['F0']=F0.to(self.dtype).to(self.device)
        else:
            self.state['F0']=None

    def get_F0(self, clone=True, detach=False):
        #pre/residual deformation
        F0=self.state['F0']
        if F0 is not None:
            if clone == True:
                F0=F0.clone()
            if detach == True:
                F0=F0.detach()
        return F0

    def set_u_field(self, u_field_full=None, u_field_free=None, data_copy_=True, requires_grad=True): #data_copy_=True: the values go into the same u_field tensor, the one an optimizer holds; False: u_field becomes a new tensor, so an optimizer built on the old one must be rebuilt
        if self.mode == 'inverse_mat':
            raise ValueError('set_u_field: in mode inverse_mat the displacement is node_x-node_X, both given; there is no u_field to set')
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
        if self.mode == 'inverse_mat':
            #both meshes are given, so the displacement of all nodes is known
            return state['node_x']-state['node_X']
        if state['node_X'] is not None:
            u_field=torch.zeros_like(state['node_X'], dtype=self.dtype, device=self.device)
        elif state['node_x'] is not None:
            u_field=torch.zeros_like(state['node_x'], dtype=self.dtype, device=self.device)
        u_field[state['boundary0']]=state['disp0']
        u_field[state['boundary1']]=state['disp1']
        u_field[state['free_node']]=state['u_field']
        return u_field

    def check_return_stiffness(self, return_stiffness):
        #None or 'none': no matrix (returns None); 'sparse': the sparse matrix (returns 'sparse')
        #'dense' and 'diagonal' are not supported, see the top of this file; the error comes before anything is computed
        if return_stiffness is None or return_stiffness == 'none':
            return None
        if return_stiffness == 'sparse':
            return 'sparse'
        raise ValueError("return_stiffness must be None, 'none' or 'sparse', not "+repr(return_stiffness)
                         +": a dense matrix does not fit in memory and the solver does not converge with a diagonal one"
                         +" (see the top of AortaFEModel.py)")

    def process_H_int_ext(self, H_int, H_ext):
        #H=H_int-H_ext, the rows and columns of the free nodes (process_H)
        free_node=self.state['free_node']
        if H_int.is_cuda:
            #torch has no subtraction of sparse CSR tensors: H_int-H_ext is taken first (sparse COO)
            return process_H(H_int-H_ext, free_node)
        else:
            if isinstance(H_ext, torch.Tensor) == True:
                H_ext=process_H(H_ext, free_node)
            return process_H(H_int, free_node)-H_ext

    def cal_energy_and_force_for_inflation(self, pressure, return_stiffness=None):
        return_stiffness=self.check_return_stiffness(return_stiffness)
        state=self.state
        u_field=self.get_u_field()
        node_x=state['node_X']+u_field
        Output_ext=cal_pressure_force(pressure, node_x,
                                      state['pressure_surface'],
                                      state['n_integration_points_surface'],
                                      return_stiffness=return_stiffness)
        if return_stiffness is None:
            force_ext=Output_ext
        else:
            force_ext, H_ext=Output_ext
        Output_int=cal_1pk_stress_force(state['element_type'],
                                        node_x,
                                        state['element'],
                                        state['d_sf_dX'],
                                        state['det_dX_dr'],
                                        state['material'],
                                        state['element_orientation'],
                                        self.cal_1pk_stress,
                                        F0=state['F0'],
                                        return_F_S_W=True,
                                        return_stiffness=return_stiffness,
                                        return_force_at_element=True)
        if return_stiffness is None:
            force_int, F, S, W, force_int_at_element=Output_int
        else:
            force_int, F, S, W, H_int, force_int_at_element=Output_int
            H=self.process_H_int_ext(H_int, H_ext)
        SE=cal_strain_energy(state['element_type'], W, state['det_dX_dr'], state['F0']).sum()
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
             'F':F, 'u_field':u_field}
        if return_stiffness is not None:
            out['H']=H
        return out
        #------------------------------------------------------------------------------------------
        # TPE1 could be used as the objective for linesearch optimization if no pressure
        # TPE2 is the full total potential energy, force_int-force_ext is grad(TPE2, u_field)
        # TPE2 contains the passive energy on boundary(0&1) and should not be used for linesearch
        # two special cases:
        # if boundary displacement is 0, then PE_passive is 0 and TPE1 is TPE2
        # if boundary displacement is not 0 and pressure is 0, then TPE1 is SE and TPE1 != TPE2
        #------------------------------------------------------------------------------------------

    def cal_energy_and_force_for_inverse_p0(self, pressure, return_stiffness=None, detach_X=True, tangent='exact'):
        #tangent selects the matrix H that is returned when return_stiffness is not None
        #  'exact':   H is the derivative of the residual (force_int-force_ext) with respect to u_field
        #             node_x is given and node_X=node_x-u_field, so H is -d(force_int)/d(node_X) at fixed node_x
        #             force_ext is computed on the given node_x and does not depend on u_field
        #             H is not symmetric; return_stiffness must be 'sparse'
        #  'forward': H is the tangent of the forward problem, d(force_int-force_ext)/d(node_x) at fixed node_X
        #             this is the matrix that was returned before the option tangent was added
        if tangent != 'exact' and tangent != 'forward':
            raise ValueError('tangent must be exact or forward, not '+str(tangent))
        return_stiffness=self.check_return_stiffness(return_stiffness)
        state=self.state
        node_x=state['node_x']
        u_field=self.get_u_field()
        node_X=node_x-u_field
        if detach_X == True:
            node_X=node_X.detach()
            node_x=node_X+u_field
        return_exact_tangent=(return_stiffness is not None) and (tangent == 'exact')
        if return_exact_tangent == True:
            if return_stiffness != 'sparse':
                raise ValueError('return_stiffness must be sparse when tangent is exact')
            #the stiffness of the forward problem is not needed
            return_forward_stiffness=None
        else:
            return_forward_stiffness=return_stiffness
        Output_ext=cal_pressure_force(pressure, node_x,
                                      state['pressure_surface'],
                                      state['n_integration_points_surface'],
                                      return_stiffness=return_forward_stiffness)
        if return_forward_stiffness is None:
            force_ext=Output_ext
        else:
            force_ext, H_ext=Output_ext
        if return_exact_tangent == True:
            #X holds the nodes of each element, X.shape (M,A,3)
            #every element gets its own copy of its nodes (node_X_of_element, element_of_X),
            #so that the derivative of the element force with respect to X can be taken element by element
            X=node_X[state['element']]
            if X.requires_grad == False:
                X.requires_grad_(True)
            node_X_of_element=X.reshape(-1,3)
            element_of_X=torch.arange(X.shape[0]*X.shape[1], device=X.device).view(X.shape[0], X.shape[1])
            d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dX_and_dX_dr(state['element_type'], node_X_of_element, element_of_X,
                                                            state['n_integration_points_solid'])
            orientation=cal_element_orientation(state['element_type'], node_X_of_element, element_of_X)
        else:
            d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dX_and_dX_dr(state['element_type'], node_X, state['element'],
                                                            state['n_integration_points_solid'])
            orientation=cal_element_orientation(state['element_type'], node_X, state['element'])
        Output_int=cal_1pk_stress_force(state['element_type'],
                                        node_x,
                                        state['element'],
                                        d_sf_dX,
                                        det_dX_dr,
                                        state['material'],
                                        orientation,
                                        self.cal_1pk_stress,
                                        F0=state['F0'],
                                        return_F_S_W=True,
                                        return_stiffness=return_forward_stiffness,
                                        return_force_at_element=True)
        if return_forward_stiffness is None:
            force_int, F, S, W, force_int_at_element=Output_int
        else:
            force_int, F, S, W, H_int, force_int_at_element=Output_int
            H=self.process_H_int_ext(H_int, H_ext)
        if return_exact_tangent == True:
            #force_int_at_element depends on X through d_sf_dX, det_dX_dr (and the volume V of the mean dilatation),
            #F and orientation; F0 is a constant
            H=cal_sparse_stiffness_matrix(node_x.shape[0], state['element'], force_int_at_element, X)
            H=-process_H(H, state['free_node'])
        SE=cal_strain_energy(state['element_type'], W, det_dX_dr, state['F0']).sum()
        force_ext1=force_ext*state['mask']
        force_ext2=force_int*(1-state['mask'])
        force_ext=force_ext1+force_ext2
        TPE1=SE-cal_potential_energy(force_ext1, u_field)
        PE_passive=cal_potential_energy(force_ext2, u_field)
        TPE2=TPE1-PE_passive
        out={'TPE1':TPE1, 'TPE2':TPE2, 'SE':SE,
             'force_int':force_int, 'force_ext':force_ext,
             'force_int_at_element':force_int_at_element,
             'F':F, 'u_field':u_field}
        if return_stiffness is not None:
            out['H']=H
        return out

    def cal_energy_and_force_for_inverse_mat(self, pressure):
        #ex vivo: node_X and node_x are known
        state=self.state
        force_ext=cal_pressure_force(pressure, state['node_x'],
                                     state['pressure_surface'],
                                     state['n_integration_points_surface'])
        force_int, F, S, W, force_int_at_element=cal_1pk_stress_force(state['element_type'],
                                                                      state['node_x'],
                                                                      state['element'],
                                                                      state['d_sf_dX'],
                                                                      state['det_dX_dr'],
                                                                      state['material'],
                                                                      state['element_orientation'],
                                                                      self.cal_1pk_stress,
                                                                      F0=state['F0'],
                                                                      return_F_S_W=True,
                                                                      return_force_at_element=True)
        SE=cal_strain_energy(state['element_type'], W, state['det_dX_dr'], state['F0']).sum()
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
             'F':F}
        return out

    def cal_energy_and_force(self, pressure, return_stiffness=None, inverse_p0_tangent='exact'):
        #inverse_p0_tangent is used only when mode is inverse_p0, see cal_energy_and_force_for_inverse_p0
        if self.mode == 'inflation':
            return self.cal_energy_and_force_for_inflation(pressure, return_stiffness)
        elif self.mode == 'inverse_p0':
            return self.cal_energy_and_force_for_inverse_p0(pressure, return_stiffness, tangent=inverse_p0_tangent)
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
            F=cal_F_tensor(state['element_type'], node_x, state['element'], node_X,
                           state['n_integration_points_solid'], state['d_sf_dX'], state['F0'])
        elif self.mode == 'inverse_p0':
            node_x=state['node_x']
            u_field=self.get_u_field()
            node_X=node_x-u_field
            F=cal_F_tensor(state['element_type'], node_x, state['element'], node_X,
                           state['n_integration_points_solid'], None, state['F0'])
        elif self.mode == 'inverse_mat':
            node_x=state['node_x']
            node_X=state['node_X']
            F=cal_F_tensor(state['element_type'], node_x, state['element'], node_X,
                           state['n_integration_points_solid'], state['d_sf_dX'], state['F0'])
        else:
            raise ValueError('mode is unknown')
        return F

    def cal_stress(self, stress, create_graph=False, return_W=False, local_sys=False):
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
            orientation=cal_element_orientation(state['element_type'], node_X, state['element'])
        elif self.mode == 'inverse_mat':
            node_X=state['node_X']
            node_x=state['node_x']
            orientation=self.state['element_orientation']
        else:
            raise ValueError('mode is unknown')
        F=self.cal_F_tensor()        
        #V from the stored state: det_dX_dr (stored for inflation and inverse_mat; from node_X for inverse_p0) and F0
        if self.mode == 'inverse_p0':
            _, _, det_dX_dr=cal_d_sf_dX_and_dX_dr(state['element_type'], node_X, state['element'],
                                                  state['n_integration_points_solid'])
        else:
            det_dX_dr=state['det_dX_dr']
        V=cal_integration_point_volume(state['element_type'], det_dX_dr, state['F0'])
        S, W=self.cal_1pk_stress(F, state['material'], orientation, create_graph=create_graph, return_W=True, V=V)
        if stress == '1pk':
            pass
        elif stress == 'cauchy':
            S=cal_cauchy_stress_from_1pk_stress(S, F)
        else:
            raise ValueError('unknown stress:'+str(stress))

        if local_sys == True:
            #orientation.shape (M,3,3)
            orientation=orientation.view(-1,1,3,3)
            orientation_t=orientation.permute(0,1,3,2)
            S=matmul(matmul(orientation_t, S), orientation)

        if return_W == False:
            return S
        else:
            return S, W

