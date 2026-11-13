# -*- coding: utf-8 -*-
"""
Mat_NeoHookean_Abaqus.py with an optional V input for mean dilatation (sandbox), built like Mat_GOH.py
"""
import torch
from torch import pow, sum
from torch_fea.utility.small_matmul import matmul #torch.matmul on the CPU, elementwise for small matrices on the GPU
from torch.linalg import inv, det
#%%
# Abaqus neo-Hookean: W = C10*(I1_-3) + 1/D1*(J-1)^2, I1_ computed from the isochoric C_=J^(-2/3)*C
# Mat = [C10, D1], the same parameters as Mat_NeoHookean_Abaqus.py, where column 0 is named mu;
#   initial shear modulus mu0 = 2*C10, initial bulk modulus K0 = 2/D1
# With V (the volume of each integration point, shape (M,K)), the volumetric term is evaluated at the element
#   mean dilatation theta_e = sum_k V_k*J_k / sum_k V_k instead of the pointwise J; only ratios of V within an element matter
# With V=None, all functions give the same values as Mat_NeoHookean_Abaqus.py
#%%
def cal_mean_dilatation(F, V):
    #theta_e = sum_k V_k*J_k / sum_k V_k over the K integration points of each element (dim 1 of F)
    #V.shape (M,K): the volume of each integration point; only ratios within an element matter
    J=det(F)
    V=V.reshape(J.shape)
    return (V*J).sum(dim=1, keepdim=True)/V.sum(dim=1, keepdim=True)
#%%
def cal_strain_energy_density(F, Mat, V=None):
    #F.shape (M,K,3,3), M is the number of elements, K is the number of integration points
    #Mat.shape MxA or 1xA, A is the number of material parameters
    #V: None (pointwise) or the volume of each integration point, shape (M,K), for the mean dilatation
    mu=Mat[:,0:1] #C10
    D=Mat[:,1:2]  #D1
    Ft=F.permute(0,1,3,2)
    C=matmul(Ft, F)
    J=det(F)
    J2=det(C)+1e-18
    C_=pow(J2.view(F.shape[0],F.shape[1],1,1),-1/3)*C
    I1_=C_[:,:,0,0]+C_[:,:,1,1]+C_[:,:,2,2]
    if V is None:
        W=mu*(I1_-3)+(1/D)*(J-1)**2
    else:
        theta=cal_mean_dilatation(F, V)
        W=mu*(I1_-3)+(1/D)*(theta-1)**2
    #W.shape is (M,K)
    return W
#%%
def cal_1pk_stress(F, Mat, create_graph, return_W, V=None):
    #set create_graph=True for optimization task
    if F.requires_grad==False:
        F.requires_grad=True
    with torch.enable_grad():
        W=cal_strain_energy_density(F, Mat, V)
        if V is None:
            S=torch.autograd.grad(W.sum(), F, retain_graph=True, create_graph=create_graph)[0]
        else:
            #P_k = (1/V_k) dE/dF_k, E = sum_k V_k*W_k is the strain energy of the element
            V=V.reshape(F.shape[0],F.shape[1])
            S=torch.autograd.grad((V*W).sum(), F, retain_graph=True, create_graph=create_graph)[0]
            S=S/V.view(F.shape[0],F.shape[1],1,1)
    #S.shape is (M,K,3,3), M is the number of elements, K is the number of integration points
    if return_W == True:
        return S, W
    else:
        return S
#%%
def cal_cauchy_stress(F, Mat, create_graph, return_W, V=None):
    S, W = cal_1pk_stress(F, Mat, create_graph, True, V)
    J=torch.det(F).view(F.shape[0],F.shape[1],1,1)
    Ft=F.permute(0,1,3,2)
    Sigma=(1/J)*matmul(S, Ft)
    if return_W == True:
        return Sigma, W
    else:
        return Sigma
#%%
def cal_2pk_stress1(F, Mat, create_graph, V=None):
    #explicit 2nd PK stress (no autograd), an independent check of cal_1pk_stress: P = F*S
    #S_vol = U'(J)*J*inv(C) with U'(J) = 2/D1*(J-1); with V, U'(theta_e) replaces U'(J)
    mu=Mat[:,0:1].view(-1,1,1,1)
    D=Mat[:,1:2].view(-1,1,1,1)
    Ft=F.permute(0,1,3,2)
    J=det(F).view(F.shape[0],F.shape[1],1,1)
    C=matmul(Ft, F)
    invC=inv(C)
    Identity=torch.eye(3,3, device=F.device, dtype=F.dtype).view(1,1,3,3)
    S_=2*(mu*Identity)
    if V is None:
        Svol=(1/D)*(2*J**2-2*J)*invC
    else:
        theta=cal_mean_dilatation(F, V).view(-1,1,1,1)
        Svol=(2/D)*(theta-1)*J*invC
    Siso=pow(J,-2/3)*(S_-(1/3)*sum(S_*C, dim=(2,3), keepdim=True)*invC)
    S=Svol+Siso
    return S
#%%
def cal_cauchy_stress1(F, Mat, create_graph, V=None):
    S=cal_2pk_stress1(F, Mat, create_graph, V)
    Ft=F.permute(0,1,3,2)
    J=det(F).view(F.shape[0],F.shape[1],1,1)
    Sigma=(1/J)*matmul(matmul(F, S), Ft)
    return Sigma
#%%
if __name__ == "__main__":
    #%% autograd vs explicit stress, with and without V, on random F
    torch.manual_seed(0)
    dtype=torch.float64
    M, K = 5, 8
    F=torch.eye(3,3, dtype=dtype)+0.1*torch.randn(M,K,3,3, dtype=dtype)
    V=0.5+torch.rand(M,K, dtype=dtype)
    Mat=torch.tensor([[27.4, 2e-5]], dtype=dtype)
    for v in [None, V]:
        Sigma=cal_cauchy_stress(F.clone(), Mat, create_graph=False, return_W=False, V=v)
        Sigma1=cal_cauchy_stress1(F, Mat, create_graph=False, V=v)
        print("V is None:", v is None, "max|Sigma-Sigma1|/max|Sigma1| =", float((Sigma-Sigma1).detach().abs().max()/Sigma1.abs().max()))
