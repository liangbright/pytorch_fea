# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 00:38:14 2023

@author: liang
"""
#This is equal to 3Field in theory, see test_GOH.py
import torch
from torch import bmm, matmul, log, cos, sin, exp, pow
from torch.linalg import det
from torch.nn.functional import relu
import numpy as np
#%%
# 4.1.2.6 Holzapfel-Gasser-Ogden in FEBio user manual v3.4
#%% fiber orientation in global sys
def cal_fiber_direction(gamma, local_sys):
    #a1.shape (M,3) or (M,1,3) or (M,8,3)
    #a2.shape (M,3) or (M,1,3) or (M,8,3)
    e1=local_sys[:,:,0]
    e2=local_sys[:,:,1]
    temp1=cos(gamma)*e1
    temp2=sin(gamma)*e2
    a1=temp1+temp2
    a2=temp1-temp2
    #print(a1.shape, a2.shape)
    #a1.shape (M,3) or (M,1,3) or (M,8,3)
    #a2.shape (M,3) or (M,1,3) or (M,8,3)
    return a1, a2
#%%
def cal_strain_energy_density(F, Mat, Orientation):
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
    k=Mat[:,5:6]
    #--------------
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
    E1=relu(E1)
    E2=relu(E2)
    W1=(0.5*c0)*temp1+(0.5*k1/k2)*(exp(k2*E1*E1)+exp(k2*E2*E2)-2)
    #use mean of J values at the K integration points
    J=det(F)
    Jv=J.mean(dim=1, keepdim=True)
    Jv2=Jv*Jv+1e-20
    W2=(0.25*k)*(Jv2-1-log(Jv2))
    W=W1+W2
    return W
#%%
def cal_1pk_stress(F, Mat, Orientation, create_graph, return_W):
    #set create_graph=True for optimization task
    with torch.enable_grad():
        W=cal_strain_energy_density(F, Mat, Orientation)
        S=torch.autograd.grad(W.sum(), F, retain_graph=True, create_graph=create_graph)[0]
    #S.shape is (M,K,3,3), M is the number of elements, K is the number of integration points
    if return_W == True:
        return S, W
    else:
        return S
#%%
def cal_cauchy_stress(F, Mat, Orientation, create_graph, return_W):
    if F.requires_grad==False:
        F.requires_grad=True
    with torch.enable_grad():
        W=cal_strain_energy_density(F, Mat, Orientation)
        S=torch.autograd.grad(W.sum(), F, retain_graph=True, create_graph=create_graph)[0]
    J=det(F).view(F.shape[0],F.shape[1],1,1)
    Ft=F.permute(0,1,3,2)
    Sigma=(1/J)*matmul(S, Ft)
    if return_W == True:
        return Sigma, W
    else:
        return Sigma
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
    F=torch.tensor([[[1.2, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).expand(1,3,3)
    W=cal_strain_energy_density(F, Mat, Orientation)
    #%%
    F=torch.eye(3,3, dtype=torch.float32)
    F=F.view(1,1,3,3).expand(10,8,3,3)
    #Orientation=torch.rand(10,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).expand(10,3,3)
    Stress=cal_cauchy_stress(F, Mat, Orientation, create_graph=False, return_W=False)
    #print(Stress)
    #%%
    F=torch.tensor([[[2., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    #Orientation=torch.rand(1,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).view(1,3,3)
    Stress=cal_cauchy_stress(F, Mat, Orientation, create_graph=False, return_W=False)
    #print(Stress)
    #%%
    F=torch.tensor([[[2., 0., 0.],
                     [0., 0.5, 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    #Orientation=torch.rand(1,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).view(1,3,3)
    Stress=cal_cauchy_stress(F, Mat, Orientation, create_graph=False, return_W=False)
    print(Stress)
