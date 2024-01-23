# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 00:19:57 2021

@author: liang
"""
import torch
from torch import matmul, log
from torch.linalg import det
#%%
# https://help.febio.org/FEBio/FEBio_um_2_9/FEBio_um_2-9-4.1.3.16.html
#%%
def cal_strain_energy_density(F, Mat):
    #F.shape (M,K,3,3), M is the number of elements, K is the number of integration points
    #Mat.shape MxA or 1xA, A is the number of material parameters
    #need to handle the case when J is very close to zero or negative
    if F.requires_grad==False:
        F.requires_grad=True
    mu=Mat[:,0:1]
    lamda=Mat[:,1:2]
    Ft=F.permute(0,1,3,2)
    C=matmul(Ft, F)
    J2=det(C)
    logJ=0.5*log(J2+1e-18)
    I1=C[:,:,0,0]+C[:,:,1,1]+C[:,:,2,2]
    W=(mu/2)*(I1-3)-mu*logJ+(lamda/2)*logJ*logJ
    #handle negative J
    J=det(F)
    WJ=(J<0.001)*(J-0.001)**2
    W=(J>=0.001)*W+lamda*WJ
    #print("W.shape", W.shape)
    #W.shape is (M,K)
    return W
#%%
def cal_1pk_stress(F, Mat, create_graph, return_W):
    #set create_graph=True for optimization task
    with torch.enable_grad():
        W=cal_strain_energy_density(F, Mat)
        S=torch.autograd.grad(W.sum(), F, retain_graph=True, create_graph=create_graph)[0]
    #S.shape is (M,K,3,3), M is the number of elements, K is the number of integration points
    if return_W == True:
        return S, W
    else:
        return S
#%%
def cal_cauchy_stress(F, Mat, create_graph, return_W):
    S, W= cal_1pk_stress(F, Mat, create_graph, True)
    J=det(F).view(F.shape[0],F.shape[1],1,1)
    Ft=F.permute(0,1,3,2)
    Sigma=(1/J)*matmul(S, Ft)
    if return_W == True:
        return Sigma, W
    else:
        return Sigma
#%%
def get_mu_lamda_from_E_v(E, v):
    mu=E/(2*(1+v))
    lamda=E*v/((1+v)*(1-2*v))
    return mu, lamda
#%%
def ge_E_v_from_mu_lamda(mu, lamda):
    E=mu*(3*lamda+2*mu)/(mu+lamda)
    v=lamda/(2*(mu+lamda))
    return E, v
#%%
if __name__ == "__main__":
    #%%
    E=500
    v=0.45
    mu, lamda=get_mu_lamda_from_E_v(E, v)
    print(mu, lamda)
    Mat=torch.tensor([[mu, lamda]])
    #%%
    F=torch.eye(3,3, dtype=torch.float32)
    F=F.view(1,1,3,3).expand(10,1,3,3)
    Stress=cal_cauchy_stress(F, Mat, create_graph=False, return_W=False)
    print(Stress)
    #%%
    F=torch.tensor([[[2., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    Stress=cal_cauchy_stress(F, Mat, create_graph=False, return_W=False)
    print(Stress)
    #%%
    F=torch.tensor([[[2., 0., 0.],
                     [0., 0.5, 0.],
                     [0., 0., 1.]]])
    F=F.view(1,1,3,3)
    Stress=cal_cauchy_stress(F, Mat, create_graph=False, return_W=False)
    print(Stress)
