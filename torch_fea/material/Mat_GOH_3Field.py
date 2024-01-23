# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 00:19:57 2021

@author: liang
"""
import torch
from torch import matmul, log, cos, sin, exp, pow
from torch.linalg import det, solve, inv
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
    J2=det(C)+1e-20
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
    J=J.mean(dim=1, keepdim=True)
    J2=J*J+1e-20
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
def cal_strain_energy_density(F, Mat, Orientation):
    Wd=cal_strain_energy_density_deviatoric(F, Mat, Orientation)
    Wv=cal_strain_energy_density_volumetric(F, Mat)
    W=Wd+Wv
    return W
#%%
def cal_1pk_stress(F, Mat, Orientation, create_graph, return_W):
    with torch.enable_grad():
        Wd=cal_strain_energy_density_deviatoric(F, Mat, Orientation)
        Wv=cal_strain_energy_density_volumetric(F, Mat)
        W=Wd+Wv
        #Sd is 1pk stress
        Sd=torch.autograd.grad(Wd.sum(), F, retain_graph=True, create_graph=create_graph)[0]
        J=det(F)
        J=J.view(F.shape[0],F.shape[1],1,1)
        k=Mat[:,5:6]
        Jv=J.mean(dim=1, keepdim=True)
        #Sigma_v=(0.5*k*)(Jv-1/Jv)
        #Jv_Sigma_v is Jv*Sigma_v
        Jv_Sigma_v=(0.5*k)*(Jv*Jv-1)
        Ft=F.permute(0,1,3,2)
        Sv=Jv_Sigma_v*inv(Ft)
        S=Sd+Sv
    if return_W == True:
        return S, W
        #return Sd, Sv, Wd, Wv
    else:
        return S
        #return Sd, Sv
#%%
def cal_cauchy_stress(F, Mat, Orientation, create_graph, return_W):
    with torch.enable_grad():
        Wd=cal_strain_energy_density_deviatoric(F, Mat, Orientation)
        Wv=cal_strain_energy_density_volumetric(F, Mat)
        W=Wd+Wv
        #Sd is 1pk stress
        Sd=torch.autograd.grad(Wd.sum(), F, retain_graph=True, create_graph=create_graph)[0]
        J=det(F)
        J=J.view(F.shape[0],F.shape[1],1,1)
        Ft=F.permute(0,1,3,2)
        Sigma_d=(1/J)*matmul(Sd, Ft)
        #compute cauchy stress Sigma_v
        k=Mat[:,5:6]
        Jv=J.mean(dim=1, keepdim=True)
        Identity=torch.eye(3,device=F.device,dtype=F.dtype).view(1,1,3,3)
        Sigma_v=(0.5*k)*(Jv-1/Jv)*Identity
        Sigma=Sigma_d+Sigma_v
    if return_W == True:
        return Sigma, W
        #return Sigma_d, Sigma_v, Wd, Wv
    else:
        return Sigma
        #return Sigma_d, Sigma_v
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
    F=torch.tensor([[[1.23456789, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3).expand(1,8,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).view(1,3,3)
    S=cal_1pk_stress(F, Mat, Orientation, create_graph=False, return_W=False)
    print(S)
    Sd, Sv, Wd, Wv=cal_1pk_stress(F, Mat, Orientation, create_graph=False, return_W=True)
    Sigma_d, Sigma_v, Wd, Wv=cal_cauchy_stress(F, Mat, Orientation, create_graph=False, return_W=True)
    print(Sigma_d, Sigma_v)
    #%%
