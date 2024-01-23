import torch
from torch import matmul
import numpy as np
import torch_scatter
import torch_sparse
#%%
def cal_dense_stiffness_matrix(n_nodes, element, force_element, x):
    #force_element.shape is (M,A,3)
    #x.shape (M,A,3)
    H=torch.zeros((3*n_nodes,3*n_nodes), dtype=x[0].dtype, device=x[0].device)
    for n in range(0, x.shape[1]):
        a=3*element[:,n]
        gn0=torch.autograd.grad(force_element[:,n,0].sum(), x, retain_graph=True)[0]
        gn1=torch.autograd.grad(force_element[:,n,1].sum(), x, retain_graph=True)[0]
        gn2=torch.autograd.grad(force_element[:,n,2].sum(), x, retain_graph=True)[0]
        for m in range(0, x.shape[1]):
            g0=gn0[:,m].detach()
            g1=gn1[:,m].detach()
            g2=gn2[:,m].detach()
            b=3*element[:,m]
            H[a, b]+=g0[:,0]
            H[a, b+1]+=g0[:,1]
            H[a, b+2]+=g0[:,2]
            H[a+1, b]+=g1[:,0]
            H[a+1, b+1]+=g1[:,1]
            H[a+1, b+2]+=g1[:,2]
            H[a+2, b]+=g2[:,0]
            H[a+2, b+1]+=g2[:,1]
            H[a+2, b+2]+=g2[:,2]
    H=H.detach()
    return H
#%%
import time
def cal_sparse_stiffness_matrix(n_nodes, element, force_element, x):
    #force_element.shape is (M,A,3)
    #x.shape (M,A,3)
    RowIndex=[]
    ColIndex=[]
    Value=[]
    #t0=time.time()
    #tab=0
    for n in range(0, x.shape[1]):
        a=3*element[:,n]
        gn0=torch.autograd.grad(force_element[:,n,0].sum(), x, retain_graph=True)[0]
        gn1=torch.autograd.grad(force_element[:,n,1].sum(), x, retain_graph=True)[0]
        gn2=torch.autograd.grad(force_element[:,n,2].sum(), x, retain_graph=True)[0]
        #ta=time.time()
        for m in range(0, x.shape[1]):
            g0=gn0[:,m].detach()
            g1=gn1[:,m].detach()
            g2=gn2[:,m].detach()
            b=3*element[:,m]
            RowIndex.append(a);   ColIndex.append(b);   Value.append(g0[:,0])
            RowIndex.append(a);   ColIndex.append(b+1); Value.append(g0[:,1])
            RowIndex.append(a);   ColIndex.append(b+2); Value.append(g0[:,2])
            RowIndex.append(a+1); ColIndex.append(b);   Value.append(g1[:,0])
            RowIndex.append(a+1); ColIndex.append(b+1); Value.append(g1[:,1])
            RowIndex.append(a+1); ColIndex.append(b+2); Value.append(g1[:,2])
            RowIndex.append(a+2); ColIndex.append(b);   Value.append(g2[:,0])
            RowIndex.append(a+2); ColIndex.append(b+1); Value.append(g2[:,1])
            RowIndex.append(a+2); ColIndex.append(b+2); Value.append(g2[:,2])
        #tb=time.time()
        #tab=tab+tb-ta
    #t1=time.time()
    with torch.no_grad():
        RowIndex=torch.cat(RowIndex, dim=0).view(1,-1)
        ColIndex=torch.cat(ColIndex, dim=0).view(1,-1)
        Index=torch.cat([RowIndex, ColIndex], dim=0)
        Value=torch.cat(Value, dim=0)
        H=torch.sparse_coo_tensor(Index, Value, (3*n_nodes,3*n_nodes))
        H=H.coalesce()
    #t2=time.time()
    #print("cal_sparse_stiffness_matrix", t1-t0, t2-t1, tab)
    return H
#%%
def cal_diagonal_stiffness_matrix(n_nodes, element, force_element, x):
    #force_element.shape is  (M,A,3)
    #x.shape (M,A,3)
    t0=time.time()
    H=torch.zeros((3*n_nodes,), dtype=x[0].dtype, device=x[0].device)
    for n in range(0, x.shape[1]):
        g0=torch.autograd.grad(force_element[:,n,0].sum(), x, retain_graph=True)[0].detach()
        g1=torch.autograd.grad(force_element[:,n,1].sum(), x, retain_graph=True)[0].detach()
        g2=torch.autograd.grad(force_element[:,n,2].sum(), x, retain_graph=True)[0].detach()
        a=3*element[:,n]
        H[a]=g0[:,n,0]
        H[a+1]=g1[:,n,1]
        H[a+2]=g2[:,n,2]
    H=H.detach()
    t1=time.time()
    print("cal_diagonal_stiffness_matrix", t1-t0)
    return H
#%%
def cal_pressure_force(pressure, node_x, element, n_integration_points, return_force, return_stiffness,
                       cal_nodal_force_from_pressure):
    #node_x: all of the nodes of the mesh, N is the total number of nodes of the mesh
    #element.shape: (M, A), M is the number of elements, A is the number of nodes of an element, A=4 or 3
    #element defines a surface of the mesh
    #--------------------------
    if element.shape[1] != 3 and element.shape[1] != 4 and element.shape[1] != 6:
        raise ValueError("only support pressure on tri3, tri6, and qud4")
    #--------------------------
    if isinstance(pressure, int) == True or isinstance(pressure, float) == True:
        if pressure == 0:
            if return_force == "dense":
                force=torch.zeros_like(node_x)
            else:
                raise ValueError("return_force unkown")
            if return_stiffness is None:
                return force
            else:
                return force, 0
    #--------------------------
    x=node_x[element].requires_grad_(True)
    #--------------------------
    force_element=cal_nodal_force_from_pressure(pressure, x, n_integration_points)
    #force_element.shape is (M,A,3)
    #--------------------------
    N=node_x.shape[0]
    if return_force == "dense":
        force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
        #force.shape is (N,3)
    elif return_force == "sparse":
        row_index=element.view(-1)
        col_index=torch.zeros_like(row_index)
        index, value = torch_sparse.coalesce([row_index, col_index], force_element.view(-1,3), len(row_index), 1, "add")
        row_index, col_index=index
        force=(row_index, value)
        #row_index contains node index in element
        #value.shape is (len(row_index),3)
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
#%%
class PotentialEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, force, u):
        ctx.save_for_backward(force, u)
        energy=(force*u).sum()
        return energy
    @staticmethod
    def backward(ctx, grad_output):
        force, u = ctx.saved_tensors
        grad_force = None #grad_output*u
        grad_u = grad_output*force
        return grad_force, grad_u
#---------------------------------
cal_potential_energy=PotentialEnergy.apply
#%%
def cal_F_tensor(node_x, element, node_X, n_integration_points, d_sf_dX, F0,
                 get_integration_point, cal_F_tensor_at_element, cal_F_tensor_with_d_sf_dX_at_element):
    #element.shape: (M, A), M is the number of elements, A is the number of nodes of an element
    #node_X.shape: (N,3), N is the number of nodes, X is undeformed
    #node_x.shape: (N,3), N is the number of nodes, x is deformed
    x=node_x[element]
    X=node_X[element]
    if d_sf_dX is None:
        r=get_integration_point(n_integration_points, dtype=node_x.dtype, device=node_x.device)
        F=cal_F_tensor_at_element(r, x, X)
    else:
        F=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #F.shape (M,K,3,3)
    return F
#%%
def cal_d_sf_dx_and_dx_dr(node_x, element, n_integration_points,
                          get_integration_point, cal_d_sf_dh):
    x=node_x[element]
    r=get_integration_point(n_integration_points, dtype=node_x.dtype, device=node_x.device)
    d_sf_dx, dx_dr, det_dx_dr = cal_d_sf_dh(r, x)
    return d_sf_dx, dx_dr, det_dx_dr
#%%
def cal_d_sf_dX_and_dX_dr(node_X, element, n_integration_points,
                          get_integration_point, cal_d_sf_dh):
    X=node_X[element]
    r=get_integration_point(n_integration_points, dtype=node_X.dtype, device=node_X.device)
    d_sf_dX, dX_dr, det_dX_dr = cal_d_sf_dh(r, X)
    return d_sf_dX, dX_dr, det_dX_dr
#%% forward: node_x is unknown, node_X is known
def cal_cauchy_stress_force(node_x, element, d_sf_dX, material, cal_cauchy_stress,
                            F0, return_F_S_W, return_stiffness, return_force_at_element,
                            get_integration_point, cal_F_tensor_with_d_sf_dX_at_element,
                            cal_d_sf_dh, cal_nodal_force_from_cauchy_stress):
    x=node_x[element].requires_grad_(True)
    #d_sf_dX.shape (M,K,3,A), K is the number of integration points
    F=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #-----------------------------------------------------------------------------
    #F.shape: (M,K,3,3)
    #M=element.shape[0]
    #material.shape: (1, ?) or (M, ?)
    S,W=cal_cauchy_stress(F, material, create_graph=True, return_W=True)
    #S.shape: (M,K,3,3)
    n_integration_points=d_sf_dX.shape[1]
    r=get_integration_point(n_integration_points, dtype=node_x.dtype, device=node_x.device)
    d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    force_element=cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr)
    #force_element.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    N=node_x.shape[0]
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
    #force.shape: (N, 3)
    #-----------------------------------------------------------------------------
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
    #-----------------------------------------------------------------------------
    #out=[force, F, S, W, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F,S,W])
    if return_stiffness is not None:
        out.append(H)
    if return_force_at_element == True:
        out.append(force_element)
    return out
#%% node_x is known, node_X or material is unknown
def cal_cauchy_stress_force_inverse(d_sf_dx, det_dx_dr, n_nodes, element, F, material, cal_cauchy_stress, F0, return_S_W,
                                    cal_nodal_force_from_cauchy_stress):
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #-----------------------------------------------------------------------------
    S, W=cal_cauchy_stress(F, material, create_graph=True, return_W=True)
    #S.shape: (M,K,3,3)
    force_element=cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr)
    #force_element.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    #M=element.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=n_nodes, reduce="sum")
    #force.shape: (n_nodes, 3)
    #-----------------------------------------------------------------------------
    if return_S_W == False:
        return force
    else:
        return force, S, W
#%%
def cal_1pk_stress_force(node_x, element, d_sf_dX, det_dX_dr, material, cal_1pk_stress,
                         F0, return_F_S_W, return_stiffness, return_force_at_element,
                         cal_F_tensor_with_d_sf_dX_at_element, cal_nodal_force_from_1pk_stress):
    x=node_x[element].requires_grad_(True)
    #d_sf_dX.shape (M,K,3,A)
    F=cal_F_tensor_with_d_sf_dX_at_element(x, d_sf_dX)
    if F0 is not None:
        #F0 is residual deformation
        F=matmul(F, F0)
    #F.shape: (M,K,3,3))
    #material.shape: (M,?) or (1,?)
    S, W=cal_1pk_stress(F, material, create_graph=True, return_W=True)
    #S.shape: (M,K,3,3)
    force_element=cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr)
    #force_element.shape: (M,A,3)
    #-----------------------------------------------------------------------------
    N=node_x.shape[0]
    force = torch_scatter.scatter(force_element.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
    #force.shape: (N, 3)
    #-----------------------------------------------------------------------------
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
    #-----------------------------------------------------------------------------
    #out=[force, F, S, W, H, force_element]
    out=[force]
    if return_F_S_W == True:
        out.extend([F,S,W])
    if return_stiffness is not None:
        out.append(H)
    if return_force_at_element == True:
        out.append(force_element)
    return out
#%%
def cal_diagonal_hessian(loss_fn, node_x):
    node_x=node_x.detach()
    N=node_x.shape[0]
    x=[]
    for n in range(0, N):
        x.append(node_x[n].view(1,3).requires_grad_(True))
    node=torch.cat(x, dim=0)
    loss=loss_fn(node)
    Hdiag=torch.zeros((3*node_x.shape[0],), dtype=node_x.dtype, device=node_x.device)
    g_all=torch.autograd.grad(loss, x, create_graph=True)
    for n in range(0, N):
        g=g_all[n]
        gg0=torch.autograd.grad(g[0,0], x[n], retain_graph=True)[0]
        gg1=torch.autograd.grad(g[0,1], x[n], retain_graph=True)[0]
        gg2=torch.autograd.grad(g[0,2], x[n], retain_graph=True)[0]
        Hdiag[3*n]=gg0[0,0]
        Hdiag[3*n+1]=gg1[0,1]
        Hdiag[3*n+2]=gg2[0,2]
    Hdiag=Hdiag.detach()
    return Hdiag

