# C3D4 in Abaqus, TET4 in FEBio: F is the same at every point inside an element
#
# Integration rules: 1 point (the centroid, exact up to degree 1) and 4 points (exact up to degree 2).
# The 4-point rule has the points (a,b,b), (b,a,b), (b,b,a) and (b,b,b), each with the weight 1/24, where
# a=(5+3*sqrt(5))/20 and b=(5-sqrt(5))/20. They are given here to full double precision, the same numbers as in
# solid_element_tet5.py and solid_element_tet15.py.
# Until 2026-11-12 this file had FEBio's numbers (FETet4G4 in FECore/FEElementTraits.cpp): a=0.58541020 and
# b=0.13819660, with 8 digits. With them the rule is exact only to about 1e-8: the integral of a quadratic function
# had a relative error of up to 7.5e-9. Results with this rule now differ from FEBio's by about that much.
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio
# integration point:
#     https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FEElementTraits.cpp
# shape function:
#    https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FESolidElementShape.cpp
#%%
import torch
import torch_fea.element.solid_element_shared as shared
from torch import matmul
from torch.linalg import det, solve
#%% integration point location
def get_integration_point_1i(dtype, device):
    r=[[0.25,0.25,0.25]] #a list of 1 point, as the other rules
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (1, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_1i(dtype, device):
    w=[1/6] #a list of 1 weight, as the other rules
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (1,)
    return w
#%% integration point location
def get_integration_point_4i(dtype, device):
    a=0.5854101966249685 #(5+3*sqrt(5))/20
    b=0.1381966011250105 #(5-sqrt(5))/20
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
    #return: r, r.shape (n_points,3); a list of the points when dtype or device is None
    if n_points==1:
        return get_integration_point_1i(dtype, device)
    elif n_points==4:
        return get_integration_point_4i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 4")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 1:
        return get_integration_weight_1i(dtype, device)
    elif n_points == 4:
        return get_integration_weight_4i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 4")
#%%  useful for 9.15c in the book
def get_shape_integration_weight_1i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1/6 for j=0
    #return: weight, weight.shape (4,1); a list of its numbers when dtype or device is None
    r0=get_integration_point_1i(None, None)[0]
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
    #return: weight, weight.shape (4,4); a list of its numbers when dtype or device is None
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
    #return: weight, weight.shape (4,n_points); a list of its numbers when dtype or device is None
    if n_points==1:
        return get_shape_integration_weight_1i(dtype, device)
    elif n_points==4:
        return get_shape_integration_weight_4i(dtype, device)
    else:
        raise ValueError("n_points must be 1 or 4")
#%% shape function 0
def sf0(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 1-r[0]-r[1]-r[2]
#%%
def cal_d_sf0_dr(r):
    #r.shape (3,...): not used, the derivatives are the same at every point
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number
    dr0=-1
    dr1=-1
    dr2=-1
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf1(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]
#%%
def cal_d_sf1_dr(r):
    #r.shape (3,...): not used, the derivatives are the same at every point
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number
    dr0=1
    dr1=0
    dr2=0
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf2(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[1]
#%%
def cal_d_sf2_dr(r):
    #r.shape (3,...): not used, the derivatives are the same at every point
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number
    dr0=0
    dr1=1
    dr2=0
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf3(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[2]
#%%
def cal_d_sf3_dr(r):
    #r.shape (3,...): not used, the derivatives are the same at every point
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number
    dr0=0
    dr1=0
    dr2=1
    dr=[dr0, dr1, dr2]
    return dr
#%% d_sf_dr is the same at every point r inside the element
def cal_d_sf_dr(r):
    #r.shape is (K,3), K is the number of integration points
    #return: d_sf_dr, d_sf_dr.shape (1,K,3,4)
    d_sf0_dr=cal_d_sf0_dr(r)
    d_sf1_dr=cal_d_sf1_dr(r)
    d_sf2_dr=cal_d_sf2_dr(r)
    d_sf3_dr=cal_d_sf3_dr(r)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr]
    d_sf_dr=torch.tensor(d_sf_dr, dtype=r.dtype, device=r.device) #(4,3)
    #contiguous: a tensor that is not contiguous makes the matrix products about 10 times slower on a GPU
    d_sf_dr=d_sf_dr.permute(1,0).contiguous() #(3,4)
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
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,4,3), M elements, 1 elemnet has 4 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,4)
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #shape (M,K,3,3)
#%%
def cal_d_sf_dh(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,4,3), M elements, 1 elemnet has 4 nodes, 1 node has a 3D position
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,4), K is the number of integration points
    d_sf_dh, dh_dr, det_dh_dr=shared.cal_d_sf_dh(h, d_sf_dr)
    #d_sf_dh.shape is (M,K,3,4), d_sf_dh[:,:,:,0] is d_sf0_dh
    #dh_dr.shape is (M,K,3,3), numerator_layout=False
    #det_dh_dr.shape is (M,K,1,1)
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,4,3)
    #X.shape is (M,4,3)
    F=shared.cal_F_tensor(x, X, cal_d_sf_dr(r))
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,4,3)
    #d_sf_dX.shape is (M,K,3,4), from cal_d_sf_dh
    F=shared.cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    return F #(M,K,3,3)
#%% K integration points, K = 1 or 4
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,4), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 3, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dx_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr, weight)
    return force #(M,4,3)
#%% K integration points, K = 1 or 4
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,4), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dX at r), i=0 to 3, #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr, weight)
    return force #(M,4,3)
#%% K integration points, K = 1 or 4
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,4), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(S*dNi_dX at r), i=0 to 3, #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight)
    return force #(M,4,3)
#%% K integration points, K = 1 or 4
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #det_dX_dr.shape is (M,K,1,1), det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(density*body_force*Ni at r), i=0 to 3, #(9.15c) in the book
    weight=get_shape_integration_weight(det_dX_dr.shape[1], det_dX_dr.dtype, det_dX_dr.device) #(4,K)
    force=shared.cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight)
    return force #(M,4,3)
#%% K integration points, K = 1 or 4, the strain energy of each element
def cal_strain_energy(W, det_dX_dr):
    #W.shape (M,K)
    #det_dX_dr.shape (M,K,1,1)
    #W=cal_strain_energy_density(F, Mat) on the integration points
    #energy=integration(W at r)
    weight=get_integration_weight(det_dX_dr.shape[1], W.dtype, torch.device("cpu"))
    energy=shared.cal_strain_energy(W, det_dX_dr, weight)
    return energy #(M,)
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
    force_1i=cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr)
    t1=time.time()
    print('time cost4', t1-t0)
    #%%
    x=torch.rand(10000, 4, 3, device=device_gpu)
    S=torch.rand(10000, 4, 3, 3, device=device_gpu)
    t0=time.time()
    r=get_integration_point_4i(dtype=x.dtype, device=x.device)
    d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    force_4i=cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr)
    t1=time.time()
    print('time cost5', t1-t0)
