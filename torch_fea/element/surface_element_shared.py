# Functions that are the same for every surface element (tri3, tri6, tri7, tri10, quad4, quad8, quad9).
#
# The number of nodes is taken from the tensor shapes. What belongs to an element is passed in:
#     d_sf_dr = cal_d_sf_dr(r)                                   derivatives of the shape functions, shape (1,K,2,A)
#     weight  = get_shape_integration_weight(K, dtype, device)   shape function times integration weight, shape (A,K)
#
# Each surface_element_*.py file keeps what belongs to its element: get_integration_point, get_integration_weight,
# get_shape_integration_weight, the shape functions sf0, sf1, ..., cal_d_sf_dr and interpolate. Its functions
# cal_dh_dr, cal_dh_du_and_dh_dv and cal_normal have the same inputs as before and call the functions of this file,
# and so does cal_nodal_force_from_pressure(pressure, x, n_integration_points), which was in
# pressure_load_on_tri3.py, pressure_load_on_tri6.py and pressure_load_on_quad4.py before.
# surface_element_tri3.py keeps its own cal_dh_dr and cal_dh_du_and_dh_dv: its derivatives are the same at every
# point and are the differences of the nodes, which needs no matrix product. It has cal_d_sf_dr like the other
# elements, and the functions of this file give the same values with it. Its results are bit-identical to the
# functions it had before.
#
# cal_dh_dr is one matrix product for all elements and all integration points. The element files had a product
# for each element and each point (matmul of d_sf_dr with h), which is slow on a GPU, most of all when the
# derivative with respect to x is taken. The sums are the same but are added in another order, so for tri6 and
# quad4 the results agree with the functions the files had before in all but the last digits (about 1e-15 of
# the largest value in float64), for all 5 rules (tri6: 1, 3, 7; quad4: 1, 4).
#
# Example for quad4 with 4 points:
#     import torch_fea.element.surface_element_quad4 as element
#     r=element.get_integration_point(4, dtype, device)
#     d_sf_dr=element.cal_d_sf_dr(r)
#     weight=element.get_shape_integration_weight(4, dtype, device)
#     dx_du, dx_dv=cal_dh_du_and_dh_dv(x, d_sf_dr)               #x=node_x[element_index], shape (M,4,3)
#     force=cal_nodal_force_from_pressure(pressure, dx_du, dx_dv, weight)
#
# Shapes: M elements, K integration points, A nodes per element.
# The functions do not check their inputs; the shapes they expect are written in each function.
#%%
import torch
from torch import cross, matmul
#%%
def cal_dh_dr(h, d_sf_dr, numerator_layout=True):
    #h.shape (M,A,3), h is x or X or u
    #d_sf_dr.shape (1,K,2,A), from cal_d_sf_dr(r) of the element
    #dh_dr when numerator_layout is True
    #  h=[a,b,c]
    #  da/du, da/dv
    #  db/du, db/dv
    #  dc/du, dc/dv
    M=h.shape[0]
    A=h.shape[1]
    K=d_sf_dr.shape[1]
    #one matrix product for all elements and points: (2K,A) times (A,3M)
    dh_dr=matmul(d_sf_dr.reshape(K*2,A), h.permute(1,0,2).reshape(A,M*3)) #shape (2K,3M)
    dh_dr=dh_dr.view(K,2,M,3).permute(2,0,1,3) #shape (M,K,2,3)
    if numerator_layout == True:
        dh_dr=dh_dr.transpose(2,3) #(M,K,3,2)
    return dh_dr
#%%
def cal_dh_du_and_dh_dv(h, d_sf_dr):
    #h.shape is (M,A,3)
    #d_sf_dr.shape (1,K,2,A), from cal_d_sf_dr(r) of the element
    dh_dr=cal_dh_dr(h, d_sf_dr, False)
    dh_du=dh_dr[:,:,0] #(M,K,3)
    dh_dv=dh_dr[:,:,1] #(M,K,3)
    return dh_du, dh_dv
#%%
def cal_normal(dx_du, dx_dv):
    #dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x) of the element, shape (M,K,3)
    normal=cross(dx_du, dx_dv, dim=-1)
    normal=normal/torch.norm(normal, p=2, dim=-1, keepdim=True)
    return normal #(M,K,3)
#%%
def cal_nodal_force_from_pressure(pressure, dx_du, dx_dv, weight):
    #pressure is scalar or tensor with shape (M, 1)
    #dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x) of the element, shape (M,K,3)
    #weight.shape (A,K), from get_shape_integration_weight(K, dtype, device) of the element
    #force_i=Integration(pressure*cross(dx_du, dx_dv)*shape_function_i at r), i=0 to A-1
    M=dx_du.shape[0]
    K=dx_du.shape[1]
    A=weight.shape[0]
    if torch.is_tensor(pressure):
        pressure=pressure.view(-1,1,1) #(M,1,1)
    pc=pressure*cross(dx_du, dx_dv, dim=-1) #(M,K,3)
    pc=pc.view(M,1,K,3)
    weight=weight.view(1,A,K,1)
    force=(weight*pc).sum(dim=2) #(M,A,3)
    return force
