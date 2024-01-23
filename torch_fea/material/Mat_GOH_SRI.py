# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 00:19:57 2021

@author: liang
"""
import torch
from torch import matmul, log, cos, sin, exp, pow
from torch.linalg import det
from torch.nn.functional import relu
import numpy as np
#%%
# 4.1.2.6 Holzapfel-Gasser-Ogden in FEBio user manual v3.4
#%% fiber direction in global sys
def cal_fiber_direction(gamma, local_sys):
    #local_sys.shape (M,3,3) or (M,1,3,3) or (M,8,3,3)
    e1=local_sys[...,0]
    e2=local_sys[...,1]
    temp1=cos(gamma)*e1
    temp2=sin(gamma)*e2
    a1=temp1+temp2
    a2=temp1-temp2
    #print(a1.shape, a2.shape)
    #a1.shape (M,3) or (M,1,3) or (M,8,3)
    #a2.shape (M,3) or (M,1,3) or (M,8,3)
    return a1, a2
#%%
def cal_strain_energy_density_deviatoric(F, Mat, Orientation):
    #F.shape (M,K,3,3), M is the number of elements, K is the number of integration points
    #Mat.shape Mx6 or 1x6, 6 is the number of material parameters
    #Orientation.shape (M,3,3), e1=Orientation[:,:,0], e2=Orientation[:,:,1], e3=Orientation[:,:,2]
    if F.requires_grad==False:
        F.requires_grad=True
    c0=Mat[:,0:1]
    k1=Mat[:,1:2]
    k2=Mat[:,2:3]
    kappa=Mat[:,3:4]
    gamma=Mat[:,4:5]
    #k=Mat[:,5:6]
    Ft=F.permute(0,1,3,2)
    C=matmul(Ft, F)
    J2=det(C)+1e-18
    C_=pow(J2.view(F.shape[0],F.shape[1],1,1),-1/3)*C
    I1=C_[:,:,0,0]+C_[:,:,1,1]+C_[:,:,2,2]
    a1, a2=cal_fiber_direction(gamma, Orientation)
    a1=a1.view(a1.shape[0], -1,3,1)
    a1t=a1.view(a1.shape[0],-1,1,3)
    I41=matmul(matmul(a1t, C_), a1)
    I41=I41.view(F.shape[0],F.shape[1])
    a2=a2.view(a2.shape[0], -1,3,1)
    a2t=a2.view(a2.shape[0],-1,1,3)
    I42=matmul(matmul(a2t, C_), a2)
    I42=I42.view(F.shape[0],F.shape[1])
    temp1=I1-3
    temp2=kappa*temp1
    E1=temp2+(1-3*kappa)*(I41-1)
    E2=temp2+(1-3*kappa)*(I42-1)
    #print("max(E1), max(E2)", float(E1.max()), float(E2.max()))
    E1=relu(E1)
    E2=relu(E2)
    #print(c.shape, k1.shape, k2.shape, gamma.shape, kappa.shape, k.shape)
    #print(temp1.shape, E1.shape, E2.shape, J2.shape)
    W=(0.5*c0)*temp1+(0.5*k1/k2)*(exp(k2*E1*E1)+exp(k2*E2*E2)-2)
    #print("J.shape", J.shape)
    #print("W.shape", W.shape)
    #W.shape is (M,K)
    return W
#%%
def cal_strain_energy_density_volumetric(F, Mat):
    #F.shape (M,K,3,3), M is the number of elements, K is the number of integration points
    #Mat.shape Mx6 or 1x6, 6 is the number of material parameters
    if F.requires_grad==False:
        F.requires_grad=True
    k=Mat[:,5:6]
    J=det(F)
    J2=J*J+1e-18
    logJ2=log(J2)
    #W1=(J>=0.001)*(J2-1-logJ2)
    #W2=(J<0.001)*(J-0.001)**2 #handle negative J
    #W=(0.25*k)*(W1+W2)
    W=(0.25*k)*(J2-1-logJ2)
    #print("J.shape", J.shape)
    #print("W.shape", W.shape)
    #W.shape is (M,K)
    return W
#%%
def cal_strain_energy_density(Fd, Fv, Mat, Orientation):
    Wd=cal_strain_energy_density_deviatoric(Fd, Mat, Orientation)
    Wv=cal_strain_energy_density_volumetric(Fv, Mat)
    return Wd, Wv
#%%
def cal_1pk_stress(Fd, Fv, Mat, Orientation, create_graph, return_W):
    #set create_graph=True for optimization task
    with torch.enable_grad():
        Wd, Wv=cal_strain_energy_density(Fd, Fv, Mat, Orientation)
        Sd=torch.autograd.grad(Wd.sum(), Fd, retain_graph=True, create_graph=create_graph)[0]
        Sv=torch.autograd.grad(Wv.sum(), Fv, retain_graph=True, create_graph=create_graph)[0]
    #Sd.shape is (M,K,3,3), M is the number of elements, K is the number of integration points
    if return_W == True:
        return Sd, Sv, Wd, Wv
    else:
        return Sd, Sv
#%%
def cal_cauchy_stress(Fd, Fv, Mat, Orientation, create_graph, return_W):
    Sd, Sv, Wd, Wv=cal_1pk_stress(Fd, Fv, Mat, Orientation, create_graph, True)
    Jd=det(Fd)
    Jd=Jd.view(Jd.shape[0],Jd.shape[1],1,1)
    Fdt=Fd.permute(0,1,3,2)
    Jv=det(Fv)
    Jv=Jv.view(Jv.shape[0],Jv.shape[1],1,1)
    Fvt=Fv.permute(0,1,3,2)
    Sigma_d=(1/Jd)*matmul(Sd, Fdt)
    Sigma_v=(1/Jv)*matmul(Sv, Fvt)
    if return_W == True:
        return Sigma_d, Sigma_v, Wd, Wv
    else:
        return Sigma_d, Sigma_v
#%%
def cal_cauchy_stress_(Fd, Fv, Mat, Orientation, create_graph, return_W):
    with torch.enable_grad():
        Wd=cal_strain_energy_density_deviatoric(Fd, Mat, Orientation)
        Wv=cal_strain_energy_density_volumetric(Fv, Mat)
        #Sd is 1pk stress
        Sd=torch.autograd.grad(Wd.sum(), Fd, retain_graph=True, create_graph=create_graph)[0]
        Jd=det(Fd)
        Jd=Jd.view(Fd.shape[0],Fd.shape[1],1,1)
        Fdt=Fd.permute(0,1,3,2)
        Sigma_d=(1/Jd)*matmul(Sd, Fdt)
        #compute cauchy stress Sigma_v
        k=Mat[:,5:6]
        Jv=det(Fv)
        Jv=Jv.view(Fv.shape[0],Fv.shape[1],1,1)
        Identity=torch.eye(3,device=F.device,dtype=F.dtype).view(1,1,3,3)
        Sigma_v=(0.5*k)*(Jv-1/Jv)*Identity
    if return_W == True:
        return Sigma_d, Sigma_v, Wd, Wv
    else:
        return Sigma_d, Sigma_v
#%%
if __name__ == "__main__":
    #%%
    c=100
    k1=1
    k2=2
    kappa=0.2
    gamma=np.pi*(45/180)
    k=2e5
    Mat=torch.tensor([[c, k1, k2, kappa, gamma, k]])
    #%%
    F=torch.eye(3,3, dtype=torch.float32).view(1,1,3,3)
    Fd=F.expand(2,8,3,3)
    Fv=F.expand(2,1,3,3)
    #Orientation=torch.rand(10,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).expand(2,8,3,3)
    Sd, Sv=cal_cauchy_stress(Fd, Fv, Mat, Orientation, create_graph=False, return_W=False)
    S=Sd+Sv
    print(Sd, Sv)
    #%%
    F=torch.tensor([[[1.23456789, 0., 0.123],
                     [0., 1., 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    Fd=F.expand(1,8,3,3)
    Fv=F.expand(1,1,3,3)
    #Orientation=torch.rand(1,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).view(1,3,3)
    Sd, Sv, Wd, Wv=cal_1pk_stress(Fd, Fv, Mat, Orientation, create_graph=False, return_W=True)
    Sigma_d, Sigma_v, Wd, Wv=cal_cauchy_stress(Fd, Fv, Mat, Orientation, create_graph=False, return_W=True)
    print(Sigma_d, Sigma_v)
    Sigma_d_, Sigma_v_, Wd_, Wv_=cal_cauchy_stress_(Fd, Fv, Mat, Orientation, create_graph=False, return_W=True)
    #%%
    F=torch.tensor([[[2., 0., 0.],
                     [0., 0.5, 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    Fd=F.expand(1,8,3,3)
    Fv=F.expand(1,1,3,3)
    #Orientation=torch.rand(1,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).view(1,3,3)
    Sd, Sv, Wd, Wv=cal_1pk_stress(Fd, Fv, Mat, Orientation, create_graph=False, return_W=True)
    Sigma_d, Sigma_v, Wd, Wv=cal_cauchy_stress(Fd, Fv, Mat, Orientation, create_graph=False, return_W=True)
