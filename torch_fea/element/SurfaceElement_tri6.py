#%% 6-node triangle surface element, a face of SolidElement_tet10
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
#    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEElementTraits.cpp
# shape function:
#    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FESurfaceElementShape.cpp
#%%
import torch
from torch import cross, matmul
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
    if n_points == 1:
        return get_integration_weight_1i(dtype, device)
    elif n_points == 3:
        return get_integration_weight_3i(dtype, device)
    elif n_points == 7:
        return get_integration_weight_7i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 3 or 7")
#%% shape function
def sf0(r):
    a=1-r[0]-r[1]
    return a*(2*a-1)
#%%
def cal_d_sf0_dr(r):
    a=1-r[0]-r[1]
    da=4*a-1
    dr0=-da
    dr1=-da
    dr=[dr0, dr1]
    return dr
#%%
def sf1(r):
    return r[0]*(2*r[0]-1)
#%%
def cal_d_sf1_dr(r):
    dr0=4*r[0]-1
    dr1=0*r[1]
    dr=[dr0, dr1]
    return dr
#%%
def sf2(r):
    return r[1]*(2*r[1]-1)
#%%
def cal_d_sf2_dr(r):
    dr0=0*r[0]
    dr1=4*r[1]-1
    dr=[dr0, dr1]
    return dr
#%%
def sf3(r):
    return 4*(1-r[0]-r[1])*r[0]
#%%
def cal_d_sf3_dr(r):
    dr0=-8*r[0]+4*(1-r[1])
    dr1=-4*r[0]
    dr=[dr0, dr1]
    return dr
#%%
def sf4(r):
    return 4*r[0]*r[1]
#%%
def cal_d_sf4_dr(r):
    dr0=4*r[1]
    dr1=4*r[0]
    dr=[dr0, dr1]
    return dr
#%%
def sf5(r):
    return 4*(1-r[0]-r[1])*r[1]
#%%
def cal_d_sf5_dr(r):
    dr0=-4*r[1]
    dr1=-8*r[1]+4*(1-r[0])
    dr=[dr0, dr1]
    return dr
#%%
def get_shape_integration_weight_1i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 0.5 for j=0
    r0 = get_integration_point_1i(None, None)
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
    if n_points == 1:
        return get_shape_integration_weight_1i(dtype, device)
    elif n_points == 3:
        return get_shape_integration_weight_3i(dtype, device)
    elif n_points == 7:
        return get_shape_integration_weight_7i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 3 or 7")
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
    d_sf4_dr=torch.cat(cal_d_sf4_dr(r), dim=1)#shape is (K,2,1)
    d_sf5_dr=torch.cat(cal_d_sf5_dr(r), dim=1)#shape is (K,2,1)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,2,6)#4: sf0 to sf5
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
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,6,3), M elements, 1 elemnet has 6 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,2,6)
    #dh_dr when numerator_layout is True
    #  h=[a,b,c]
    #  da/du, da/dv
    #  db/du, db/dv
    #  dc/du, dc/dv
    M=h.shape[0]
    h=h.view(M,1,6,3)
    dh_dr=matmul(d_sf_dr, h) #shape (M,K,2,3)
    if numerator_layout == True:
        dh_dr=dh_dr.transpose(2,3) #(M,K,3,2)
    return dh_dr
#%%
def cal_dh_du_and_dh_dv(r, h):
    #r.shape (K,2), K is the number of integration points
    #h.shape is (M,6,3)
    dh_dr=cal_dh_dr(r, h, None, False)
    dh_du=dh_dr[:,:,0] #(M,K,3)
    dh_dv=dh_dr[:,:,1] #(M,K,3)
    return dh_du, dh_dv
#%%
def cal_normal(r, x):
    #r.shape (K,2), K is the number of integration points
    #x.shape is (M,6,3)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x)
    normal=cross(dx_du, dx_dv, dim=-1)
    normal=normal/torch.norm(normal, p=2, dim=-1, keepdim=True)
    return normal #(M,K,3)
#%%
