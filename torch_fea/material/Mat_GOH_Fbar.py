# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 00:19:57 2021

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
def cal_F_(F):
    J=det(F)#J.shape (M,K)
    Jv=J.mean(dim=1, keepdim=True)#Jv.shape (M,1)
    alpha=pow(Jv/J, 1/3)
    alpha=alpha.view(F.shape[0],F.shape[1],1,1)
    #alpha=alpha.detach() # FEA does not converge
    F_=alpha*F
    return F_
#%%
def cal_strain_energy_density(F_raw, Mat, Orientation, return_F_=False):
    #F.shape (M,K,3,3), M is the number of elements, K is the number of integration points
    #Mat.shape Mx6 or 1x6, 6 is the number of material parameters
    #Orientation.shape (M,3,3), e1=Orientation[:,:,0], e2=Orientation[:,:,1], e3=Orientation[:,:,2]
    if F_raw.requires_grad==False:
        F_raw.requires_grad=True
    c0=Mat[:,0]
    k1=Mat[:,1]
    k2=Mat[:,2]
    kappa=Mat[:,3]
    gamma=Mat[:,4]
    k=Mat[:,5]
    #--------------
    F=cal_F_(F_raw)
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
    W2=(0.25*k)*(J2-1-log(J2))
    W=W1+W2
    if return_F_ == False:
        return W
    else:
        return W, F
#%%
def cal_1pk_stress(F, Mat, Orientation, create_graph, return_W, use_F_=False):
    #set create_graph=True for optimization task
    with torch.enable_grad():
        W, F_=cal_strain_energy_density(F, Mat, Orientation, True)
        if use_F_ == True:
            F=F_
        S=torch.autograd.grad(W.sum(), F, retain_graph=True, create_graph=create_graph)[0]
    #S.shape is (M,K,3,3), M is the number of elements, K is the number of integration points
    if return_W == True:
        return S, W
    else:
        return S
#%%
def cal_1pk_stress_with_F_(F, Mat, Orientation, create_graph, return_W):
    return cal_1pk_stress(F, Mat, Orientation, create_graph, return_W, True)
#%%
def cal_1pk_stress_with_F(F, Mat, Orientation, create_graph, return_W):
    return cal_1pk_stress(F, Mat, Orientation, create_graph, return_W, False)
#%%
def cal_cauchy_stress(F, Mat, Orientation, create_graph, return_W, use_F_=False):
    if F.requires_grad==False:
        F.requires_grad=True
    with torch.enable_grad():
        W, F_=cal_strain_energy_density(F, Mat, Orientation, True)
        if use_F_ == True:
            F=F_
        S=torch.autograd.grad(W.sum(), F, retain_graph=True, create_graph=create_graph)[0]
    J=det(F).view(F.shape[0],F.shape[1],1,1)
    Ft=F.permute(0,1,3,2)
    Sigma=(1/J)*matmul(S, Ft)
    if return_W == True:
        return Sigma, W
    else:
        return Sigma
#%%
def cal_cauchy_stress_with_F_(F, Mat, Orientation, create_graph, return_W):
    return cal_cauchy_stress(F, Mat, Orientation, create_graph, return_W, True)
#%%
def cal_cauchy_stress_with_F(F, Mat, Orientation, create_graph, return_W):
    return cal_cauchy_stress(F, Mat, Orientation, create_graph, return_W, False)
#%%
if __name__ == "__main__":
    #%%
    c=200
    k1=400
    k2=2
    kappa=0
    gamma=np.pi*(0/180)
    k=2e5
    Mat=torch.tensor([[c, k1, k2, kappa, gamma, k]])
    #%%
    F=torch.tensor([[[1.2, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).expand(1,3,3)
    W=cal_strain_energy_density(F, Mat, Orientation, return_F_=False)
    #%%
    F=torch.eye(3,3, dtype=torch.float32)
    F=F.view(1,1,3,3).expand(10,8,3,3)
    #Orientation=torch.rand(10,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).expand(10,3,3)
    Stress=cal_cauchy_stress(F, Mat, Orientation, create_graph=False, return_W=False, use_F_=True)
    #print(Stress)
    #%%
    F=torch.tensor([[[2., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    #Orientation=torch.rand(1,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).view(1,3,3)
    Stress=cal_cauchy_stress(F, Mat, Orientation, create_graph=False, return_W=False, use_F_=True)
    #print(Stress)
    #%%
    F=torch.tensor([[[2., 0., 0.],
                     [0., 0.5, 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    #Orientation=torch.rand(1,3,3)
    Orientation=torch.eye(3,3, dtype=torch.float32).view(1,3,3)
    Stress=cal_cauchy_stress(F, Mat, Orientation, create_graph=False, return_W=False, use_F_=True)
    #print(Stress)
