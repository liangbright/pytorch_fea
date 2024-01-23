# C3D4 in Abaqus, TET4 in FEBio: F is the same at every point inside an element
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio
# integration point:
#     https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEElementTraits.cpp
# shape function:
#    https://github.com/febiosoftware/FEBio/blob/develop/FECore/FESolidElementShape.cpp
#%%
import torch
from torch import matmul
from torch.linalg import det, solve
#%% integration point location
def get_integration_point_1i(dtype, device):
    r=[0.25,0.25,0.25]
    if dtype is not None and device is not None:
        r=torch.tensor([r], dtype=dtype, device=device) #r.shape (1, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_1i(dtype, device):
    w=1/6
    if dtype is not None and device is not None:
        w=torch.tensor([w], dtype=dtype, device=device) #w.shape (1,)
    return w
#%% integration point location
def get_integration_point_4i(dtype, device):
    a=0.58541020
    b=0.13819660
    r0=[b,b,b]
    r1=[a,b,b]
    r2=[b,a,b]
    r3=[b,b,a]
    r=[r0, r1, r2, r3]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (4, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_4i(dtype, device):
    w=[1/24, 1/24, 1/24, 1/24]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (4,)
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
#%% shape function 0
def sf0(r):
    return 1-r[0]-r[1]-r[2]
#%%
def cal_d_sf0_dr():
    dr0=-1
    dr1=-1
    dr2=-1
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf1(r):
    return r[0]
#%%
def cal_d_sf1_dr():
    dr0=1
    dr1=0
    dr2=0
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf2(r):
    return r[1]
#%%
def cal_d_sf2_dr():
    dr0=0
    dr1=1
    dr2=0
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf3(r):
    return r[2]
#%%
def cal_d_sf3_dr():
    dr0=0
    dr1=0
    dr2=1
    dr=[dr0, dr1, dr2]
    return dr
#%% d_sf_dr is the same at every point r inside the element
def cal_d_sf_dr(r):
    #r.shape is (K,3), K is the number of integration points
    d_sf0_dr=cal_d_sf0_dr()
    d_sf1_dr=cal_d_sf1_dr()
    d_sf2_dr=cal_d_sf2_dr()
    d_sf3_dr=cal_d_sf3_dr()
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr]
    d_sf_dr=torch.tensor(d_sf_dr, dtype=r.dtype, device=r.device) #(4,3)
    d_sf_dr=d_sf_dr.permute(1,0) #(3,4)
    K=r.shape[0]
    d_sf_dr=d_sf_dr.view(1,1,3,4).expand(1,K,3,4)#4: sf0 to sf3
    return d_sf_dr
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,4,3), M elements, 1 elemnet has 4 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_k{sf_i(r)*h_i, k=0 to 3}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r)], dim=1)
    w=w.view(1,K,4,1)
    M=h.shape[0]
    h=h.view(M,1,4,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%% dh_dr is the same at every point r inside the element
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,4,3), M elements, 1 elemnet has 4 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,4)
    M=h.shape[0]
    h=h.view(M,1,4,3)
    dh_dr=matmul(d_sf_dr, h) #shape (M,K,3,3)
    if numerator_layout == True:
        dh_dr=dh_dr.transpose(2,3)
    return dh_dr
#%% d_sf_dh (d_sf_dx, d_sf_dX) is the same at every point r inside the element
def cal_d_sf_dh(r, h):
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,4), K is the number of integration points
    dh_dr=cal_dh_dr(r, h, d_sf_dr, False)
    #dh_dr.shape is (M,K,3,3)
    det_dh_dr=det(dh_dr).view(dh_dr.shape[0],-1,1,1)#shape is (M,K,1,1)
    #-----------------------------------
    #inv_dh_dr=inv(dh_dr)
    #t_inv_dh_dr=inv_dh_dr.permute(0,1,3,2)# (9.6ab) transposed inverse
    #d_sf_dh=matmul(t_inv_dh_dr, d_sf_dr)
    #-----------------------------------
    #d_sf_dh=solve(dh_dr.transpose(2,3), d_sf_dr)
    d_sf_dh=solve(dh_dr, d_sf_dr) # numerator_layout=False for cal_dh_dr
    #d_sf_dh.shape is (M,K,3,4)
    #d_sf_dh[:,:,:,0] is d_sf0_dx
    #-----------------------------------
    return d_sf_dh, dh_dr, det_dh_dr
#%% F is the same at every point r inside the element
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,4,3)
    #X.shape is (M,4,3)
    dx_dr=cal_dh_dr(r, x)
    dX_dr=cal_dh_dr(r, X)
    #F=matmul(dx_dr, inv(dX_dr))
    F=solve(dX_dr.transpose(2,3), dx_dr.transpose(2,3)).transpose(2,3)
    #F.shape is (M,K,3,3)
    return F
#%% F is the same at every point r inside the element
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,4,3)
    #d_sf_dX.shape is (M,K,3,4), from cal_d_sf_dh
    M=x.shape[0]
    K=d_sf_dX.shape[1]
    x=x.view(M,1,4,3,1).expand(M,K,4,3,1)
    d_sf_dX=d_sf_dX.transpose(2,3)#shape is (M,K,4,3)
    d_sf_dX=d_sf_dX.view(M,K,4,1,3)
    F=matmul(x, d_sf_dX).sum(dim=2)
    return F
#%% use 1 integration point
def cal_nodal_force_from_cauchy_stress_1i(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point_1i(dtype, device), r.shape is (K,3), K=1
    #F=cal_F_tensor(...) on the integration point, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration point, S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,4), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 3, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    #integration_weight is 1/6 at the integration point
    #print("shape", S.shape, d_sf_dx.shape, det_dx_dr.shape)
    force=(1/6)*matmul(S, d_sf_dx)*det_dx_dr
    force=force.view(-1,3,4)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force #(M,4,3)
#%% use 4 integration points
def cal_nodal_force_from_cauchy_stress_4i(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point_4i(dtype, device), r.shape is (K,3), K=4
    #F=cal_F_tensor(...) on the four integration points, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) the four integration points， S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,4), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 3, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    #integration_weight is 1/24 at every integration point
    force=(1/24)*(matmul(S, d_sf_dx)*det_dx_dr).sum(dim=1)#force.shape is (M,3,4)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force #(M,4,3)
#%%
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    if det_dx_dr.shape[1] == 1:
        return cal_nodal_force_from_cauchy_stress_1i(S, d_sf_dx, det_dx_dr)
    elif det_dx_dr.shape[1] == 4:
        return cal_nodal_force_from_cauchy_stress_4i(S, d_sf_dx, det_dx_dr)
    else:
        raise ValueError("only support 1i and 4i")
#%% use 1 integration point
def cal_nodal_force_from_2pk_stress_1i(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point_1i(dtype, device), r.shape is (K,3), K=1
    #F=cal_F_tensor(...) on the integration point, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration point, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,4), det_dX_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dx at r), i=0 to 3, #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    #integration_weight is 1/6 at the integration point
    force=(1/6)*matmul(matmul(F, S), d_sf_dX)*det_dX_dr
    force=force.view(-1,3,4)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force #(M,4,3)
#%% use 4 integration points
def cal_nodal_force_from_2pk_stress_4i(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point_4i(dtype, device), r.shape is (K,3), K=4
    #F=cal_F_tensor(...) on the four integration points, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the four integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,4), det_dX_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #---------------------------------------------------------------
    #force_i=integration(F*S*dNi_dx at r), i=0 to 3, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    #integration_weight is 1/24 at every integration point
    force=(1/24)*(matmul(matmul(F, S), d_sf_dX)*det_dX_dr).sum(dim=1)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force #(M,4,3)
#%%
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    if det_dX_dr.shape[1] == 1:
        return cal_nodal_force_from_2pk_stress_1i(F, S, d_sf_dX, det_dX_dr)
    elif det_dX_dr.shape[1] == 4:
        return cal_nodal_force_from_2pk_stress_4i(F, S, d_sf_dX, det_dX_dr)
    else:
        raise ValueError("only support 1i and 4i")
#%% use 1 integration point
def cal_nodal_force_from_1pk_stress_1i(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point_1i(dtype, device), r.shape is (K,3), K=1
    #F=cal_F_tensor(...) on the integration point, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration point, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,4), det_dX_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dx at r), i=0 to 3, #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    #integration_weight is 1/6 at the integration point
    force=(1/6)*matmul(S, d_sf_dX)*det_dX_dr
    force=force.view(-1,3,4)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force #(M,4,3)
#%% use 4 integration points
def cal_nodal_force_from_1pk_stress_4i(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point_8i(dtype, device), r.shape is (K,3), K=4
    #F=cal_F_tensor(...) on the eight integration points, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the eight integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,4), det_dX_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #---------------------------------------------------------------
    #force_i=integration(F*S*dNi_dx at r), i=0 to 3, #(9.15b) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    #integration_weight is 1/24 at every integration point
    force=(1/24)*(matmul(S, d_sf_dX)*det_dX_dr).sum(dim=1)
    force=force.permute(0,2,1)
    force=force.contiguous()
    return force #(M,4,3)
#%%
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    if det_dX_dr.shape[1] == 1:
        return cal_nodal_force_from_1pk_stress_1i(S, d_sf_dX, det_dX_dr)
    elif det_dX_dr.shape[1] == 4:
        return cal_nodal_force_from_1pk_stress_4i(S, d_sf_dX, det_dX_dr)
    else:
        raise ValueError("only support 1i and 4i")
#%% use 1 integration point to calculate the strain energy of each element
def cal_strain_energy_1i(W, det_dX_dr, reduction):
    #W.shape (M,1)
    #det_dX_dr.shape is (M,1,1,1)
    #W=cal_strain_energy_density(F, Mat) on the integration point
    #energy=integration(W at r)
    #integration_weight is 1/6 at the integration point
    det_dX_dr=det_dX_dr.view(W.shape)
    energy=(1/6)*W*det_dX_dr
    #shape is (M,1)
    if reduction is None or reduction == "none":
        pass
    elif reduction == "sum":
        energy=energy.sum()
    elif reduction == "mean":
        energy=energy.mean()
    return energy
#%% use 4 integration points to calculate the strain energy of each element
def cal_strain_energy_4i(W, det_dX_dr, reduction):
    #W.shape (M,4)
    #det_dX_dr.shape (M,4,1,1)
    #W=cal_strain_energy_density(F, Mat) on the 4 integration points
    #energy=integration(W at r)
    #integration_weight is 1/24 at every integration point
    det_dX_dr=det_dX_dr.view(W.shape)
    energy=(1/24)*(W*det_dX_dr).sum(dim=1, keepdim=True)
    #shape is (M,1)
    if reduction is None or reduction == "none":
        pass
    elif reduction == "sum":
        energy=energy.sum()
    elif reduction == "mean":
        energy=energy.mean()
    return energy
#%%
def cal_strain_energy(W, det_dX_dr, reduction):
    if det_dX_dr.shape[1]==1:
        return cal_strain_energy_1i(W, det_dX_dr, reduction)
    elif det_dX_dr.shape[1]==4:
        return cal_strain_energy_4i(W, det_dX_dr, reduction)
    else:
        raise ValueError("only support 1i and 4i, det_dX_dr.shape[1]="+str(det_dX_dr.shape[1]))
#%%  useful for 9.15c in the book
def get_shape_integration_weight_1i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1/6 for j=0
    r0=get_integration_point_1i(None, None)
    a=1/6
    weight=[]
    for i in range(0, 4):
        wi0=eval("sf"+str(i)+"(r0)")*a
        weight.append(wi0)
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(4,1)
    return weight
#%%
def get_shape_integration_weight_4i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1/24 for j=0,1,2,3,4,5,6,7
    r0, r1, r2, r3=get_integration_point_4i(None, None)
    a=1/24
    weight=[]
    for i in range(0, 4):
        for j in range(0, 4):
             wij=eval("sf"+str(i)+"(r"+str(j)+")")*a
             weight.append(wij)
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(4,4)
    return weight
#%%
def get_shape_integration_weight(n_points, dtype, device):
    if n_points==1:
        return get_shape_integration_weight_1i(dtype, device)
    elif n_points==4:
        return get_shape_integration_weight_4i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 4")
#%%
if __name__ == "__main__":
    #%%
    device_gpu=torch.device("cuda:0")
    device_cpu=torch.device("cpu")
    #%%
    dh_dr0=torch.tensor([[1, 2, 3]])
    print("dh_dr0.shape", dh_dr0.shape)
    dh_dr1=torch.tensor([[4, 5, 6]])
    dh_dr2=torch.tensor([[7, 8, 9]])
    dh_dr=torch.cat([dh_dr0, dh_dr1, dh_dr2], dim=1)
    dh_dr=dh_dr.view(1,3,3)
    dh_dr=dh_dr.permute(0,2,1)
    print("dh_dr", dh_dr)
    print(dh_dr.reshape(-1))
    #%%
    dh_dr0=torch.tensor([[1, 2, 3]]).view(1,3,1)
    print("dh_dr0.shape", dh_dr0.shape)
    dh_dr1=torch.tensor([[4, 5, 6]]).view(1,3,1)
    dh_dr2=torch.tensor([[7, 8, 9]]).view(1,3,1)
    dh_dr=torch.cat([dh_dr0, dh_dr1, dh_dr2], dim=2)
    dh_dr=dh_dr.view(1,3,3)
    #dh_dr=dh_dr.permute(0,2,1)
    print("dh_dr", dh_dr)
    print(dh_dr.reshape(-1))
    #%%
    import time
    X=torch.rand(1, 4, 3).to(torch.float64)
    F=torch.rand(1, 3, 3).to(torch.float64)
    x=torch.matmul(F, X.view(1,4,3,1)).view(1,4,3)
    r=get_integration_point_1i(x.dtype, x.device)
    t0=time.time()
    F2=cal_F_tensor(r,  x, X)
    t2=time.time()
    d_sf_dX, dX_d, det_dX_dr=cal_d_sf_dh(r, X)
    t3=time.time()
    F3=cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    t4=time.time()
    print('time cost3', t3-t2)
    print('time cost4', t4-t3)
    print((F2-F).abs().mean().item())
    print((F2-F3).abs().mean().item())
    print((F2-F3).abs().max().item())
    #%%
    x=torch.rand(10000, 4, 3, device=device_gpu)
    S=torch.rand(10000, 1, 3, 3, device=device_gpu)
    t0=time.time()
    r=get_integration_point_1i(x.dtype, x.device)
    d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    force_1i=cal_nodal_force_from_cauchy_stress_1i(S, d_sf_dx, det_dx_dr)
    t1=time.time()
    print('time cost4', t1-t0)
    #%%
    x=torch.rand(10000, 4, 3, device=device_gpu)
    S=torch.rand(10000, 4, 3, 3, device=device_gpu)
    t0=time.time()
    r=get_integration_point_4i(dtype=x.dtype, device=x.device)
    d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    force_4i=cal_nodal_force_from_cauchy_stress_4i(S, d_sf_dx, det_dx_dr)
    t1=time.time()
    print('time cost5', t1-t0)
