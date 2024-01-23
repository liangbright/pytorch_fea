#%% 3-node triangle surface element, a face of SolidElement_tet4 or SolidElement_wedge6
#    v
#    |
#    x2
#    | \
#   x0--x1--->u
# r= [u, v] is the locaiton of a point in the unit element
# r(x0)=[0,0], r(x1)=[1,0], r(x2)=[0,1]
# [r[0], r[1]] = [u, v] ~ [r, s] in FEBio
# integration point:
#    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEElementTraits.cpp
# shape function:
#    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FESurfaceElementShape.cpp
#%%
import torch
from torch import cross
#%%
def get_integration_point_1i(dtype, device):
    r=[1/3, 1/3]
    if dtype is not None and device is not None:
        r=torch.tensor([r], dtype=dtype, device=device) #r.shape (1, 2)
    return r
#%%
def get_integration_weight_1i(dtype, device):
    w=0.5
    if dtype is not None and device is not None:
        w=torch.tensor([w], dtype=dtype, device=device) #w.shape (1,)
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
#%% integration point location
def get_integration_point(n_points, dtype, device):
    if n_points==1:
        return get_integration_point_1i(dtype, device)
    elif n_points==3:
        return get_integration_point_3i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 3")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    if n_points == 1:
        return get_integration_weight_1i(dtype, device)
    elif n_points == 3:
        return get_integration_weight_3i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 3")
#%% shape function
def sf0(r):
    return 1-r[0]-r[1]
#%%
def sf1(r):
    return r[0]
#%%
def sf2(r):
    return r[1]
#%%
def get_shape_integration_weight_1i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 0.5 for j=0
    #r0 = get_integration_point_1i()
    #w00=sf0(r0)*0.5
    #w10=sf1(r0)*0.5
    #w20=sf2(r0)*0.5
    weight=[1/6, 1/6, 1/6]
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(3,1)
    return weight
#%%
def get_shape_integration_weight_3i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1/6 for j=0,1,2
    r0, r1, r2 = get_integration_point_3i(None, None)
    a=1/6
    w00=sf0(r0)*a; w01=sf0(r1)*a; w02=sf0(r2)*a
    w10=sf1(r0)*a; w11=sf1(r1)*a; w12=sf1(r2)*a
    w20=sf2(r0)*a; w21=sf2(r1)*a; w22=sf2(r2)*a
    weight=[w00, w01, w02,
            w10, w11, w12,
            w20, w21, w22]
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(3,3)
    return weight
#%%
def get_shape_integration_weight(n_points, dtype, device):
    if n_points == 1:
        return get_shape_integration_weight_1i(dtype, device)
    elif n_points == 3:
        return get_shape_integration_weight_3i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 3")
#%%
def interpolate(r, h):
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,3,3), M elements, 1 elemnet has 3 nodes, 1 node has a 3D position
    #hr=sum_k{sf_i(r)*h_i, i=0 to 2}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (2,K)
    r=r.view(2,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r)], dim=1)
    w=w.view(1,K,3,1)
    M=h.shape[0]
    h=h.view(M,1,3,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def interpolate_at(r, h0, h1, h2):
    #r=[u, v], an integration point
    #h.shape (M,3,3), M elements, 1 elemnet has 3 nodes, 1 node has a 3D position, h is x or X or u
    #h0=h[:,0]#(M,3)
    #h1=h[:,1]#(M,3)
    #h2=h[:,2]#(M,3)
    h=(1-r[0]-r[1])*h0+r[0]*h1+r[1]*h2
    return h
#%%
def cal_dh_du(h0, h1, h2):
    #dh_du is the same at every point r=[u, v]
    #h0.shape is (M,3)
    dh_du=h1-h0
    return dh_du #(M,3)
#%%
def cal_dh_dv(h0, h1, h2):
    #dh_dv is the same at every point r=[u, v]
    #h0.shape is (M,3)
    dh_dv=h2-h0
    return dh_dv #(M,3)
#%%
def cal_dh_dr(r, h, numerator_layout=True):
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,3,3), M elements, 1 elemnet has 3 nodes, 1 node has a 3D position, h is x or X or u
    #dh_dr is the same at every point r
    #dh_dr when numerator_layout is True
    #  h=[a,b,c]
    #  da/du, da/dv
    #  db/du, db/dv
    #  dc/du, dc/dv
    #----------------------------------
    h0=h[:,0]#(M,3)
    h1=h[:,1]#(M,3)
    h2=h[:,2]#(M,3)
    M=h0.shape[0]
    dh_du=cal_dh_du(h0, h1, h2).view(M,1,3)
    dh_dv=cal_dh_dv(h0, h1, h2).view(M,1,3)
    dh_dr=torch.cat([dh_du, dh_dv], dim=1)#(M,2,3), NOT numerator_layout
    K=r.shape[0]
    dh_dr=dh_dr.view(M,1,2,3).expand(M,K,2,3)
    if numerator_layout == True:
        dh_dr=dh_dr.transpose(2,3) #(M,K,3,2)
    return dh_dr
#%%
def cal_dh_du_and_dh_dv(r, h):
    #r.shape (K,2), K is the number of integration points
    #h.shape is (M,3,3)
    h0=h[:,0]#(M,3)
    h1=h[:,1]#(M,3)
    h2=h[:,2]#(M,3)
    dh_du=cal_dh_du(h0, h1, h2) #(M,3)
    dh_dv=cal_dh_dv(h0, h1, h2) #(M,3)
    M=h0.shape[0]
    K=r.shape[0]
    dh_du=dh_du.view(M,1,3).expand(M,K,3)
    dh_dv=dh_dv.view(M,1,3).expand(M,K,3)
    return dh_du, dh_dv
#%%
def cal_normal(r, x):
    #r.shape (K,2), K is the number of integration points
    #x.shape is (M,3,3)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x)
    normal=cross(dx_du, dx_dv, dim=-1)
    normal=normal/torch.norm(normal, p=2, dim=-1, keepdim=True)
    return normal #(M,K,3)
