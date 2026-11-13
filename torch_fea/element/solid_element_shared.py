# Functions that are the same for every solid element (tet4, tet10, wedge6, hex8, and every other solid_element file).
#
# The number of nodes is taken from the tensor shapes. The two things that belong to an element are passed in:
#     d_sf_dr = cal_d_sf_dr(r)                          derivatives of the shape functions, shape (1,K,3,A)
#     weight  = get_integration_weight(K, dtype, device) integration weights, shape (K,)
# The same number of points K means different weights for different elements (K=1: 8 for hex8, 1/6 for tet4
# and tet10, 1 for wedge6), so the weight cannot be found from K and has to come from the element file.
#
# Each solid_element_*.py file keeps what belongs to its element: get_integration_point, get_integration_weight,
# the shape functions sf0, sf1, ..., cal_d_sf_dr, interpolate and get_shape_integration_weight. Its functions
# cal_dh_dr, cal_d_sf_dh, cal_F_tensor, cal_F_tensor_with_d_sf_dX, cal_nodal_force_from_cauchy_stress,
# cal_nodal_force_from_1pk_stress, cal_nodal_force_from_2pk_stress and cal_strain_energy have the same inputs
# as before and call the functions of this file. cal_nodal_force_from_body_force(body_force, density, det_dX_dr)
# is new in each file and calls the one here.
#
# For tet4, tet10, wedge6 and hex8 the results are bit-identical to the functions these files had before,
# for all 10 rules (hex8: 1, 8; tet4: 1, 4; tet10: 1, 4, 8; wedge6: 1, 2, 6). See integrate() for how.
#
# Example for hex8 with 8 points:
#     import torch_fea.element.solid_element_hex8 as element
#     r=element.get_integration_point(8, dtype, device)
#     weight=element.get_integration_weight(8, dtype, device)
#     d_sf_dr=element.cal_d_sf_dr(r)
#     d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(X, d_sf_dr)          #X=node_X[element_index], shape (M,8,3)
#     F=cal_F_tensor_with_d_sf_dX(x, d_sf_dX)                    #x=node_x[element_index], shape (M,8,3)
#     force=cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight)
#     energy=cal_strain_energy(W, det_dX_dr, weight)             #shape (M,): the strain energy of each element
#
# Shapes: M elements, K integration points, A nodes per element.
# The functions do not check their inputs; the shapes they expect are written in each function.
#%%
import torch
from torch_fea.utility.small_matmul import matmul #torch.matmul on the CPU, elementwise for small matrices on the GPU
from torch.linalg import det, solve
#%%
def cal_dh_dr(h, d_sf_dr, numerator_layout=True):
    #h.shape (M,A,3), h is x or X or u
    #d_sf_dr.shape (1,K,3,A), from cal_d_sf_dr(r) of the element
    M=h.shape[0]
    A=h.shape[1]
    h=h.view(M,1,A,3)
    dh_dr=matmul(d_sf_dr, h) #shape (M,K,3,3)
    if numerator_layout == True:
        dh_dr=dh_dr.transpose(2,3)
    return dh_dr
#%%
def cal_d_sf_dh(h, d_sf_dr):
    #h is x or X, h.shape (M,A,3)
    #d_sf_dr.shape (1,K,3,A), from cal_d_sf_dr(r) of the element
    dh_dr=cal_dh_dr(h, d_sf_dr, False)
    #dh_dr.shape is (M,K,3,3)
    det_dh_dr=det(dh_dr).view(dh_dr.shape[0],-1,1,1)#shape is (M,K,1,1)
    d_sf_dh=solve(dh_dr, d_sf_dr) # numerator_layout=False for cal_dh_dr
    #d_sf_dh.shape is (M,K,3,A)
    #d_sf_dh[:,:,:,0] is d_sf0_dh
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor(x, X, d_sf_dr):
    #x.shape is (M,A,3)
    #X.shape is (M,A,3)
    #d_sf_dr.shape (1,K,3,A), from cal_d_sf_dr(r) of the element
    dx_dr=cal_dh_dr(x, d_sf_dr)
    dX_dr=cal_dh_dr(X, d_sf_dr)
    F=solve(dX_dr.transpose(2,3), dx_dr.transpose(2,3)).transpose(2,3)
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,A,3)
    #d_sf_dX.shape is (M,K,3,A), from cal_d_sf_dh
    M=x.shape[0]
    A=x.shape[1]
    K=d_sf_dX.shape[1]
    x=x.view(M,1,A,3,1).expand(M,K,A,3,1)
    d_sf_dX=d_sf_dX.transpose(2,3)#shape is (M,K,A,3)
    d_sf_dX=d_sf_dX.view(M,K,A,1,3)
    F=matmul(x, d_sf_dX).sum(dim=2)
    return F #(M,K,3,3)
#%%
def integrate(value, det_dh_dr, weight):
    #value.shape (M,K,...), the integrand at the K integration points
    #det_dh_dr.shape (M,K,1,1), from cal_d_sf_dh
    #weight.shape (K,), from get_integration_weight(K, dtype, device) of the element, on any device
    #return sum_k{weight[k]*value[:,k]*det_dh_dr[:,k]}, shape (M,...)
    #------------------------------------------------------------------
    #The order of the multiplications follows the element files, so that the result is bit-identical to them:
    #  all weights equal and K > 1: the weight is applied after the sum
    #  otherwise:                   the weight is applied to the integrand before det_dh_dr
    K=value.shape[1]
    weight=weight.to(dtype=value.dtype)
    shape=[1,K]+[1]*(value.dim()-2)
    det_dh_dr=det_dh_dr.view([det_dh_dr.shape[0],K]+[1]*(value.dim()-2))
    #the comparison below runs on the device of weight: with weight on the CPU it does not wait for the GPU
    if K > 1 and bool((weight == weight[0]).all()):
        out=weight[0].item()*(value*det_dh_dr).sum(dim=1)
    else:
        weight=weight.to(device=value.device)
        out=(weight.view(shape)*value*det_dh_dr).sum(dim=1)
    return out
#%%
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr, weight):
    #S.shape is (M,K,3,3), Cauchy stress at the integration points
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(x, d_sf_dr)
    #d_sf_dx.shape is (M,K,3,A), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #weight.shape is (K,), from get_integration_weight(K, dtype, device) of the element
    #force_i=integration(S*dNi_dx at r), i=0 to A-1, #(9.15b) in the book
    force=integrate(matmul(S, d_sf_dx), det_dx_dr, weight) #(M,3,A)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force #(M,A,3)
#%%
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight):
    #S.shape is (M,K,3,3), 1PK stress at the integration points
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(X, d_sf_dr)
    #d_sf_dX.shape is (M,K,3,A), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #weight.shape is (K,), from get_integration_weight(K, dtype, device) of the element
    #force_i=integration(S*dNi_dX at r), i=0 to A-1, S_1pk=F*S_2pk
    force=integrate(matmul(S, d_sf_dX), det_dX_dr, weight) #(M,3,A)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force #(M,A,3)
#%%
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr, weight):
    #F.shape is (M,K,3,3), deformation gradient at the integration points
    #S.shape is (M,K,3,3), 2PK stress at the integration points
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(X, d_sf_dr)
    #d_sf_dX.shape is (M,K,3,A), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #weight.shape is (K,), from get_integration_weight(K, dtype, device) of the element
    #force_i=integration(F*S*dNi_dX at r), i=0 to A-1, #(page264) in the book
    force=integrate(matmul(matmul(F, S), d_sf_dX), det_dX_dr, weight) #(M,3,A)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force #(M,A,3)
#%%
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #det_dX_dr.shape (M,K,1,1), from cal_d_sf_dh(X, d_sf_dr) on the undeformed element
    #weight.shape (A,K), from get_shape_integration_weight(K, dtype, device) of the element:
    #    shape function times integration weight, not the integration weight that the other functions here take
    #force_i=integration(density*body_force*Ni at r), i=0 to A-1, #(9.15c) in the book
    M=det_dX_dr.shape[0]
    K=det_dX_dr.shape[1]
    A=weight.shape[0]
    weight=weight.to(dtype=det_dX_dr.dtype, device=det_dX_dr.device)
    if torch.is_tensor(density):
        density=density.view(-1,1,1) #(M,1,1)
    q=density*det_dX_dr.view(M,K,1)*body_force #(M,K,3)
    q=q.view(M,1,K,3)
    force=(weight.view(1,A,K,1)*q).sum(dim=2)
    return force #(M,A,3)
#%%
def cal_strain_energy(W, det_dX_dr, weight):
    #W.shape (M,K), strain energy density at the integration points
    #det_dX_dr.shape (M,K,1,1)
    #weight.shape is (K,), from get_integration_weight(K, dtype, device) of the element
    #energy=integration(W at r), the strain energy of each element
    energy=integrate(W, det_dX_dr, weight)
    return energy #(M,)
