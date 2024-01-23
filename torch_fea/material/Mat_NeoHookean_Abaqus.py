# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 00:19:57 2021

@author: liang
"""
import torch
from torch import matmul, pow, sum
from torch.linalg import inv, det
#%%
def cal_strain_energy_density(F, Mat):
    #F.shape (M,K,3,3), M is the number of elements, K is the number of integration points
    #Mat.shape MxA or 1xA, A is the number of material parameters
    mu=Mat[:,0:1]
    D=Mat[:,1:2]
    Ft=F.permute(0,1,3,2)
    C=matmul(Ft, F)
    J=det(F)
    J2=det(C)+1e-18
    #print("J.shape", J.shape, "C.shape", C.shape)
    C_=pow(J2.view(F.shape[0],F.shape[1],1,1),-1/3)*C
    I1_=C_[:,:,0,0]+C_[:,:,1,1]+C_[:,:,2,2]
    #print("I1_.shape", I1_.shape)
    W=mu*(I1_-3)+(1/D)*(J-1)**2
    return W
#%%
def cal_1pk_stress(F, Mat, create_graph, return_W):
    #set create_graph=True for optimization task
    if F.requires_grad==False:
        F.requires_grad=True
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
    S, W = cal_1pk_stress(F, Mat, create_graph, True)
    J=torch.det(F).view(F.shape[0],F.shape[1],1,1)
    Ft=F.permute(0,1,3,2)
    Sigma=(1/J)*matmul(S, Ft)
    if return_W == True:
        return Sigma, W
    else:
        return Sigma
#%%
def cal_2pk_stress1(F, Mat, create_graph):
    mu=Mat[:,0:1]
    D=Mat[:,1:2]
    Ft=F.permute(0,1,3,2)
    J=det(F).view(F.shape[0],F.shape[1],1,1)
    C=matmul(Ft, F)
    invC=inv(C)
    #print("C.shape", C.shape)
    W1_=mu
    Identity=torch.eye(3,3, device=F.device, dtype=F.dtype).view(1,1,3,3)
    S_=2*(W1_*Identity)
    Svol=(1/D)*(2*J**2-2*J)*invC
    Siso=pow(J,-2/3)*(S_-(1/3)*sum(S_*C, dim=(2,3), keepdim=True)*invC)
    S=Svol+Siso
    return S
#%%
def cal_cauchy_stress1(F, Mat, create_graph):
    S=cal_2pk_stress1(F, Mat, create_graph)
    Ft=F.permute(0,1,3,2)
    J=det(F).view(F.shape[0],F.shape[1],1,1)
    Sigma=(1/J)*matmul(matmul(F, S), Ft)
    return Sigma
#%%
if __name__ == "__main__":
    #%%
    F=torch.eye(3,3, dtype=torch.float32)
    F=F.expand(10,1,3,3)
    Mat=torch.tensor([[1, 1e-5]])
    Sigma=cal_cauchy_stress1(F, Mat, create_graph=True)
    print("Sigma.shape", Sigma.shape)
    #%%
    F=torch.eye(3,3, dtype=torch.float32)
    F=F.expand(10,1,3,3)
    Mat=torch.tensor([[100, 1e-4]])
    Stress=cal_cauchy_stress1(F, Mat, create_graph=False)
    print(Stress)
    #%%
    dtype=torch.float64
    F=torch.tensor([[[[ 1.0557e+00,  2.6524e-02,  2.6524e-02],
                      [-5.5827e-03,  9.7396e-01, -8.4368e-03],
                      [-5.6253e-03, -9.2437e-04,  9.7392e-01]]]], dtype=dtype)
    Mat=torch.tensor([[75, 5e-4]], dtype=dtype)
    Stress=cal_cauchy_stress(F, Mat, create_graph=False, return_W=False)
    print(Stress)
    Stress1=cal_cauchy_stress1(F, Mat, create_graph=False)
    print(Stress1)
    #%%
    F=torch.tensor([[[[2., 0., 0.],
                      [0., 0.5, 0.],
                      [0., 0., 1.]]]])
    Mat=torch.tensor([[1, 1e-5]])
    Stress=cal_cauchy_stress1(F, Mat, create_graph=False)
    print(Stress)
