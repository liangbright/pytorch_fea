# The pressure load on surface elements.
# It does not depend on the material or on the solid elements, so every model file can import it from here.
#
# surface elements (7): tri3, tri6, tri7, tri10, quad4, quad8, quad9
# Each of them has its own number of nodes, so the element is found from element.shape[1].
#
# The files of torch_fea/element are used as modules: surface_element_<element>.py.
#%%
import torch
import torch_scatter
import torch_sparse
import torch_fea.element.surface_element_tri3 as surface_element_tri3
import torch_fea.element.surface_element_tri6 as surface_element_tri6
import torch_fea.element.surface_element_tri7 as surface_element_tri7
import torch_fea.element.surface_element_tri10 as surface_element_tri10
import torch_fea.element.surface_element_quad4 as surface_element_quad4
import torch_fea.element.surface_element_quad8 as surface_element_quad8
import torch_fea.element.surface_element_quad9 as surface_element_quad9
from torch_fea.model.model_stiffness import cal_dense_stiffness_matrix, cal_sparse_stiffness_matrix, cal_diagonal_stiffness_matrix
#%%
#the number of nodes per element -> the element file
surface_element={3:  surface_element_tri3,
                 6:  surface_element_tri6,
                 7:  surface_element_tri7,
                 10: surface_element_tri10,
                 4:  surface_element_quad4,
                 8:  surface_element_quad8,
                 9:  surface_element_quad9}
#%%
def cal_pressure_force(pressure, node_x, element, n_integration_points, return_force="dense", return_stiffness=None):
    #pressure: a number, or a tensor of shape (M,1)
    #node_x: all of the nodes of the mesh, N is the total number of nodes of the mesh
    #element.shape: (M, A), M is the number of elements, A is the number of nodes of an element
    #element defines a surface of the mesh
    #element must be tri3, tri6, tri7, tri10, quad4, quad8 or quad9, not mixed
    #--------------------------
    if element.shape[1] not in surface_element:
        raise ValueError("only support pressure on tri3, tri6, tri7, tri10, quad4, quad8 and quad9")
    cal_nodal_force_from_pressure=surface_element[element.shape[1]].cal_nodal_force_from_pressure
    #--------------------------
    #pressure 0 (an int, a float or a tensor) takes the same path as any other pressure: the force and H are zero,
    #with the same types and shapes as for a nonzero pressure
    #--------------------------
    x=node_x[element].requires_grad_(True)
    #--------------------------
    force_element=cal_nodal_force_from_pressure(pressure, x, n_integration_points)
    #force_element.shape is (M,A,3)
    #--------------------------
    N=node_x.shape[0]
    if return_force == "dense":
        #reshape, not view: element may be a non-contiguous slice, e.g. element[:,0:A] of the shell model
        #int64 index: with an int32 index on the CPU, the backward of scatter gives wrong gradient rows (torch 2.12.1)
        force = torch_scatter.scatter(force_element.reshape(-1,3), element.reshape(-1).to(torch.int64), dim=0, dim_size=N, reduce="sum")
        #force.shape is (N,3)
    elif return_force == "sparse":
        row_index=element.reshape(-1).to(torch.int64) #torch_sparse.coalesce needs int64 indices (element may be int32)
        col_index=torch.zeros_like(row_index)
        #the matrix has N rows (node indices go up to N-1) and 1 column
        index, value = torch_sparse.coalesce([row_index, col_index], force_element.reshape(-1,3), N, 1, "add")
        row_index, col_index=index
        force=(row_index, value)
        #row_index: the nodes of the surface, sorted, each once
        #value.shape is (len(row_index),3): the force at each of them
    else:
        raise ValueError("return_force unkown")
    #--------------------------
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
    #--------------------------
    if return_stiffness is None:
        return force
    else:
        return force, H
