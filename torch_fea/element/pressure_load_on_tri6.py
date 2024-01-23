import torch
from torch import cross
from torch_fea.element.SurfaceElement_tri6 import cal_dh_du_and_dh_dv, get_integration_point, get_shape_integration_weight
#%%
def cal_nodal_force_from_pressure(pressure, x, n_integration_points):
    #x.shape (M,A,3), A is the number of nodes per element
    #pressure is scalar or tensor with shape (M, 1)
    K=n_integration_points
    M=x.shape[0]
    A=x.shape[1]
    r=get_integration_point(K, x.dtype, x.device)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x) #(M,K,3), (M,K,3)
    #force_i=Integration(pressure*cross(dx_du, dx_dv)*shape_function_i at r)
    if torch.is_tensor(pressure):
        pressure=pressure.view(-1,1,1) #(M,1,1)
    pc=pressure*cross(dx_du, dx_dv, dim=-1) #(M,K,3)
    pc=pc.view(M,1,K,3)
    weight=get_shape_integration_weight(K, x.dtype, x.device) #(A,K)
    weight=weight.view(1,A,K,1)
    force=(weight*pc).sum(dim=2) #(M,A,3)
    return force
#%%
if __name__ == "__main__":
    #%%
    device_gpu=torch.device("cuda:0")
    device_cpu=torch.device("cpu")
    torch.manual_seed(0)
    x=torch.rand(1, 6, 3, device=device_gpu)
    #%%
    import time
    pressure=torch.tensor(16.0, device=device_gpu)
    t0=time.time()
    force_1i = cal_nodal_force_from_pressure(pressure, x, n_integration_points=1)
    force_3i = cal_nodal_force_from_pressure(pressure, x, n_integration_points=3)
    force_7i = cal_nodal_force_from_pressure(pressure, x, n_integration_points=7)
    t1=time.time()
    print('time cost', t1-t0)
    t0=time.time()

