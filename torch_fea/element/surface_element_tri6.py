#%% 6-node triangle surface element, a face of solid_element_tet10
#    v
#    |
#    x2
#    |  \
#   x5  x4
#    |     \
#   x0--x3--x1--->u
# r= [u, v] is the locaiton of a point in the unit element
# r(x0)=[0,0], r(x1)=[1,0], r(x2)=[0,1]
# [r[0], r[1]] = [u, v] ~ [r, s] in FEBio
# integration point:
#    https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FEElementTraits.cpp
# shape function:
#    https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FESurfaceElementShape.cpp
#
# Integration rules: 1, 3 and 7 points. FEBio's default for tri6 faces is 3 points (FE_TRI6G3); FEBio has no 1-point
# rule for tri6. The 1-point rule (the centroid, weight 0.5) is exact only up to degree 1: it gives the exact area and
# the exact total pressure force of a flat face, and nearly the exact total force of a curved one. It cannot share the
# force among the six nodes, because the shape functions are quadratic: with 1 point each corner node gets -1/9 of the
# face force (exact: 0) and each mid-side node 4/9 (exact: 1/3), even on a flat face. So do not use 1 point for nodal
# forces (cal_nodal_force_from_pressure); use 3 or 7.
#%%
import torch
import torch_fea.element.surface_element_shared as shared
from torch import cross, matmul
#%%
def get_integration_point_1i(dtype, device):
    r=[[1/3, 1/3]] #a list of 1 point, as the other rules
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (1, 2)
    return r
#%%
def get_integration_weight_1i(dtype, device):
    w=[0.5] #a list of 1 weight, as the other rules
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (1,)
    return w
#%%
def get_integration_point_3i(dtype, device):
    a=1/6
    b=2/3
    r0=[a,a]
    r1=[b,a]
    r2=[a,b]
    r=[r0, r1, r2]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (3, 2)
    return r
#%%
def get_integration_weight_3i(dtype, device):
    w=[1/6, 1/6, 1/6]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (3,)
    return w
#%%
def get_integration_point_7i(dtype, device):
    r0=[0.333333333333333, 0.333333333333333]
    r1=[0.797426985353087, 0.101286507323456]
    r2=[0.101286507323456, 0.797426985353087]
    r3=[0.101286507323456, 0.101286507323456]
    r4=[0.470142064105115, 0.470142064105115]
    r5=[0.470142064105115, 0.059715871789770]
    r6=[0.059715871789770, 0.470142064105115]
    r=[r0, r1, r2, r3, r4, r5, r6]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (7, 2)
    return r
#%%
def get_integration_weight_7i(dtype, device):
    w=[0.5*0.225000000000000,
       0.5*0.125939180544827,
       0.5*0.125939180544827,
       0.5*0.125939180544827,
       0.5*0.132394152788506,
       0.5*0.132394152788506,
       0.5*0.132394152788506]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (7,)
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,2); a list of the points when dtype or device is None
    if n_points==1:
        return get_integration_point_1i(dtype, device)
    elif n_points==3:
        return get_integration_point_3i(dtype, device)
    elif n_points==7:
        return get_integration_point_7i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 3 or 7")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 1:
        return get_integration_weight_1i(dtype, device)
    elif n_points == 3:
        return get_integration_weight_3i(dtype, device)
    elif n_points == 7:
        return get_integration_weight_7i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 3 or 7")
#%%
def get_shape_integration_weight_1i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 0.5 for j=0
    #return: weight, weight.shape (6,1); a list of its numbers when dtype or device is None
    r0 = get_integration_point_1i(None, None)[0]
    a=0.5
    w00=sf0(r0)*a
    w10=sf1(r0)*a
    w20=sf2(r0)*a
    w30=sf3(r0)*a
    w40=sf4(r0)*a
    w50=sf5(r0)*a
    weight=[w00, w10, w20, w30, w40, w50]
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(6,1)
    return weight
#%%
def get_shape_integration_weight_3i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1/6 for j=0,1,2
    #return: weight, weight.shape (6,3); a list of its numbers when dtype or device is None
    r0, r1, r2 = get_integration_point_3i(None, None)
    a=1/6
    w00=sf0(r0)*a; w01=sf0(r1)*a; w02=sf0(r2)*a
    w10=sf1(r0)*a; w11=sf1(r1)*a; w12=sf1(r2)*a
    w20=sf2(r0)*a; w21=sf2(r1)*a; w22=sf2(r2)*a
    w30=sf3(r0)*a; w31=sf3(r1)*a; w32=sf3(r2)*a
    w40=sf4(r0)*a; w41=sf4(r1)*a; w42=sf4(r2)*a
    w50=sf5(r0)*a; w51=sf5(r1)*a; w52=sf5(r2)*a
    weight=[w00, w01, w02,
            w10, w11, w12,
            w20, w21, w22,
            w30, w31, w32,
            w40, w41, w42,
            w50, w51, w52]
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(6,3)
    return weight
#%%
def get_shape_integration_weight_7i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is a[j] for j = 0 to 6
    #return: weight, weight.shape (6,7); a list of its numbers when dtype or device is None
    r0, r1, r2, r3, r4, r5, r6=get_integration_point_7i(None, None)
    a=get_integration_weight_7i(None, None)
    weight=[]
    for i in range(0, 6):
        for j in range(0, 7):
             wij=eval("sf"+str(i)+"(r"+str(j)+")")*a[j]
             weight.append(wij)
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(6,7)
    return weight
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (6,n_points); a list of its numbers when dtype or device is None
    if n_points == 1:
        return get_shape_integration_weight_1i(dtype, device)
    elif n_points == 3:
        return get_shape_integration_weight_3i(dtype, device)
    elif n_points == 7:
        return get_shape_integration_weight_7i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 3 or 7")
#%% shape function
def sf0(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    a=1-r[0]-r[1]
    return a*(2*a-1)
#%%
def cal_d_sf0_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    a=1-r[0]-r[1]
    da=4*a-1
    dr0=-da
    dr1=-da
    dr=[dr0, dr1]
    return dr
#%%
def sf1(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*(2*r[0]-1)
#%%
def cal_d_sf1_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=4*r[0]-1
    dr1=0*r[1]
    dr=[dr0, dr1]
    return dr
#%%
def sf2(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[1]*(2*r[1]-1)
#%%
def cal_d_sf2_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=0*r[0]
    dr1=4*r[1]-1
    dr=[dr0, dr1]
    return dr
#%%
def sf3(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*(1-r[0]-r[1])*r[0]
#%%
def cal_d_sf3_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-8*r[0]+4*(1-r[1])
    dr1=-4*r[0]
    dr=[dr0, dr1]
    return dr
#%%
def sf4(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*r[0]*r[1]
#%%
def cal_d_sf4_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=4*r[1]
    dr1=4*r[0]
    dr=[dr0, dr1]
    return dr
#%%
def sf5(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*(1-r[0]-r[1])*r[1]
#%%
def cal_d_sf5_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-4*r[1]
    dr1=-8*r[1]+4*(1-r[0])
    dr=[dr0, dr1]
    return dr
#%%
def cal_d_sf_dr(r):
    #r.shape is (K,2), K is the number of integration points
    #return: d_sf_dr, d_sf_dr.shape (1,K,2,6)
    K=r.shape[0]
    r=r.permute(1,0)#shape is (2,K)
    r=r.view(2,K,1,1)
    d_sf0_dr=torch.cat(cal_d_sf0_dr(r), dim=1)#shape is (K,2,1)
    d_sf1_dr=torch.cat(cal_d_sf1_dr(r), dim=1)#shape is (K,2,1)
    d_sf2_dr=torch.cat(cal_d_sf2_dr(r), dim=1)#shape is (K,2,1)
    d_sf3_dr=torch.cat(cal_d_sf3_dr(r), dim=1)#shape is (K,2,1)
    d_sf4_dr=torch.cat(cal_d_sf4_dr(r), dim=1)#shape is (K,2,1)
    d_sf5_dr=torch.cat(cal_d_sf5_dr(r), dim=1)#shape is (K,2,1)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,2,6)#6: sf0 to sf5
    return d_sf_dr
#%%
def interpolate(r, h):
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,6,3), M elements, 1 elemnet has 4 nodes, 1 node has a 3D position
    #hr=sum_k{sf_i(r)*h_i, i=0 to 5}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (2,K)
    r=r.view(2,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r)], dim=1)
    w=w.view(1,K,6,1)
    M=h.shape[0]
    h=h.view(M,1,6,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,6,3), M elements, 1 elemnet has 6 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,2,6)
    #dh_dr when numerator_layout is True
    #  h=[a,b,c]
    #  da/du, da/dv
    #  db/du, db/dv
    #  dc/du, dc/dv
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #(M,K,3,2), or (M,K,2,3) when numerator_layout is False
#%%
def cal_dh_du_and_dh_dv(r, h):
    #r.shape (K,2), K is the number of integration points
    #h.shape is (M,6,3)
    dh_du, dh_dv=shared.cal_dh_du_and_dh_dv(h, cal_d_sf_dr(r))
    return dh_du, dh_dv #(M,K,3), (M,K,3)
#%%
def cal_normal(r, x):
    #r.shape (K,2), K is the number of integration points
    #x.shape is (M,6,3)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x)
    normal=shared.cal_normal(dx_du, dx_dv)
    return normal #(M,K,3)
#%% K integration points, K = 3 or 7 (K = 1 gives wrong nodal forces, see the top of this file)
def cal_nodal_force_from_pressure(pressure, x, n_points):
    #x.shape (M,6,3)
    #pressure is scalar or tensor with shape (M, 1)
    #force_i=Integration(pressure*cross(dx_du, dx_dv)*shape_function_i at r), i=0 to 5
    #n_points: the number of integration points
    r=get_integration_point(n_points, x.dtype, x.device)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x) #(M,K,3), (M,K,3)
    weight=get_shape_integration_weight(n_points, x.dtype, x.device) #(6,K)
    force=shared.cal_nodal_force_from_pressure(pressure, dx_du, dx_dv, weight)
    return force #(M,6,3)
