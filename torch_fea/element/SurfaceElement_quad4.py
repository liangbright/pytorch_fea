#%% 4-node quad surface element, a face of SolidElement_hex8 or SolidElement_wedge6
#     v
#    /|\
#     |
# x3  |  x2
#     0--------->u
# x0     x1
# r=[u,v] is the locaiton of a point in the unit element
# r(x0)=[-1,-1], r(x1)=[1,-1], r(x2)=[1,1], r(x3)=[-1,1]
# [r[0], r[1]] = [u, v] ~ [r, s] in FEBio
# integration point:
#    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEElementTraits.cpp
# shape function:
#    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FESurfaceElementShape.cpp
#%%
import torch
from torch import cross, matmul
from math import sqrt
#%%
def get_integration_point_1i(dtype, device):
    r=[0, 0]
    if dtype is not None and device is not None:
        r=torch.tensor([r], dtype=dtype, device=device) #r.shape (1, 2)
    return r
#%%
def get_integration_weight_1i(dtype, device):
    w=4
    if dtype is not None and device is not None:
        w=torch.tensor([w], dtype=dtype, device=device) #r.shape (1, )
    return w
#%%
def get_integration_point_4i(dtype, device):
    a=1/sqrt(3) #0.577350269189626
    r0=[-a,-a]
    r1=[+a,-a]
    r2=[+a,+a]
    r3=[-a,+a]
    r=[r0, r1, r2, r3]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (4, 2)
    return r
#%%
def get_integration_weight_4i(dtype, device):
    w=[1,1,1,1]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #r.shape (4, )
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    if n_points==1:
        return get_integration_point_1i(dtype, device)
    elif n_points==4:
        return get_integration_point_4i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 4")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    if n_points == 1:
        return get_integration_weight_1i(dtype, device)
    elif n_points == 4:
        return get_integration_weight_4i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 4")
#%%
def sf0(r):
    return 0.25*(1-r[0])*(1-r[1])
#%%
def cal_d_sf0_dr(r):
    dr0=-0.25*(1-r[1])
    dr1=-0.25*(1-r[0])
    dr=[dr0, dr1]
    return dr
#%%
def sf1(r):
    return 0.25*(1+r[0])*(1-r[1])
#%%
def cal_d_sf1_dr(r):
    dr0=+0.25*(1-r[1])
    dr1=-0.25*(1+r[0])
    dr=[dr0, dr1]
    return dr
#%%
def sf2(r):
    return 0.25*(1+r[0])*(1+r[1])
#%%
def cal_d_sf2_dr(r):
    dr0=0.25*(1+r[1])
    dr1=0.25*(1+r[0])
    dr=[dr0, dr1]
    return dr
#%%
def sf3(r):
    return 0.25*(1-r[0])*(1+r[1])
#%%
def cal_d_sf3_dr(r):
    dr0=-0.25*(1+r[1])
    dr1=+0.25*(1-r[0])
    dr=[dr0, dr1]
    return dr
#%%
def get_shape_integration_weight_1i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 4 for j=0
    #r0=get_integration_point_1i(None, None)
    #w00=sf0(r0)*4
    #w10=sf1(r0)*4
    #w20=sf2(r0)*4
    #w30=sf3(r0)*4
    #return (w00, w10, w20, w30)
    weight=[1, 1, 1, 1]
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(4,1)
    return weight
#%%
def get_shape_integration_weight_4i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1 for j=0,1,2,3
    r0, r1, r2, r3= get_integration_point_4i(None, None)
    w00=sf0(r0); w01=sf0(r1); w02=sf0(r2); w03=sf0(r3)
    w10=sf1(r0); w11=sf1(r1); w12=sf1(r2); w13=sf1(r3)
    w20=sf2(r0); w21=sf2(r1); w22=sf2(r2); w23=sf2(r3)
    w30=sf3(r0); w31=sf3(r1); w32=sf3(r2); w33=sf3(r3)
    weight=[w00, w01, w02, w03,
            w10, w11, w12, w13,
            w20, w21, w22, w23,
            w30, w31, w32, w33]
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(4,4)
    return weight
#%%
def get_shape_integration_weight(n_points, dtype, device):
    if n_points == 1:
        return get_shape_integration_weight_1i(dtype, device)
    elif n_points == 4:
        return get_shape_integration_weight_4i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 4")
#%%
def cal_d_sf_dr(r):
    #r.shape is (K,2), K is the number of integration points
    K=r.shape[0]
    r=r.permute(1,0)#shape is (2,K)
    r=r.view(2,K,1,1)
    d_sf0_dr=torch.cat(cal_d_sf0_dr(r), dim=1)#shape is (K,2,1)
    d_sf1_dr=torch.cat(cal_d_sf1_dr(r), dim=1)#shape is (K,2,1)
    d_sf2_dr=torch.cat(cal_d_sf2_dr(r), dim=1)#shape is (K,2,1)
    d_sf3_dr=torch.cat(cal_d_sf3_dr(r), dim=1)#shape is (K,2,1)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,2,4)#4: sf0 to sf3
    return d_sf_dr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,4,3), M elements, 1 elemnet has 4 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,2,4)
    #dh_dr when numerator_layout is True
    #  h=[a,b,c]
    #  da/du, da/dv
    #  db/du, db/dv
    #  dc/du, dc/dv
    M=h.shape[0]
    h=h.view(M,1,4,3)
    dh_dr=matmul(d_sf_dr, h) #shape (M,K,2,3)
    if numerator_layout == True:
        dh_dr=dh_dr.transpose(2,3) #(M,K,3,2)
    return dh_dr
#%%
def cal_dh_du_and_dh_dv(r, h):
    #r.shape (K,2), K is the number of integration points
    #h.shape is (M,4,3)
    dh_dr=cal_dh_dr(r, h, None, False)
    dh_du=dh_dr[:,:,0] #(M,K,3)
    dh_dv=dh_dr[:,:,1] #(M,K,3)
    return dh_du, dh_dv
#%%
def cal_normal(r, x):
    #r.shape (K,2), K is the number of integration points
    #x.shape is (M,4,3)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x)
    normal=cross(dx_du, dx_dv, dim=-1)
    normal=normal/torch.norm(normal, p=2, dim=-1, keepdim=True)
    return normal #(M,K,3)
#%%


