# -*- coding: utf-8 -*-
"""
UFD (unified-fiber-distribution) material, built like material_GOH.py, with the optional V input for mean dilatation
"""
import torch
from torch import log, exp, pow, sum
from torch_fea.utility.small_matmul import matmul #torch.matmul on the CPU, elementwise for small matrices on the GPU
from torch.linalg import inv, det
from torch.nn.functional import relu
#%%
# UFD model: Dong et al. (2023), Ann Biomed Eng 51:2441-2452, doi:10.1007/s10439-023-03275-1, Eqs. (1)-(4)
#   W = c0/2*(I1_-3) + k1/(2*k2)*[exp(k2*A)-1] + U
#   A = zeta^2*<I41_-1>^2 + (1-zeta)^2*<I42_-1>^2, ONE exponential for the unified fiber distribution
#   I41_ = a1.C_.a1 and I42_ = a2.C_.a2: a1=e1 is circumferential, a2=e2 is axial (no fiber angle, no dispersion)
#   I1_, I41_, I42_ are computed from the isochoric C_=J^(-2/3)*C; <.> is the Macaulay bracket,
#   i.e. the Heaviside switch of Eq. (4): a direction is off if I4_<1 (fibers under compression)
#   zeta in [0,1]: zeta=1 means all fibers along e1, zeta=0 means all fibers along e2
# The paper assumes J=1. For nearly incompressible FEA, the volumetric part of material_GOH.py is added:
#   U(J) = k/4*(J^2-1-ln(J^2)) = k/2*((J^2-1)/2-ln(J))
# Mat = [c0, k1, k2, zeta, k]; valid inputs: det(F)>0, c0>=0, k1>=0, k2>=0, 0<=zeta<=1, k>0 (k2=0 is the limit W2=k1/2*A)
# With V (the volume of each integration point, shape (M,K)), U is evaluated at the element mean dilatation theta_e
#   instead of the pointwise J (mean dilatation, the same as material_GOH.py); only ratios of V within an element matter
# Relation to material_GOH.py (Mat = [c0, k1, k2, kappa, gamma, k]):
#   UFD [c0, k1, k2, 1, k] = GOH [c0, k1/2, k2, 0, 0,    k]  (all fibers along e1)
#   UFD [c0, k1, k2, 0, k] = GOH [c0, k1/2, k2, 0, pi/2, k]  (all fibers along e2)
#%% fiber orientation in global sys
def cal_fiber_direction(local_sys):
    #local_sys.shape (M,3,3) or (M,K,3,3): the columns are e1 (circumferential), e2 (axial), e3 (thickness)
    a1=local_sys[...,0]
    a2=local_sys[...,1]
    #a1.shape (M,3) or (M,K,3)
    #a2.shape (M,3) or (M,K,3)
    return a1, a2
#%%
def cal_mean_dilatation(F, V):
    #theta_e = sum_k V_k*J_k / sum_k V_k over the K integration points of each element (dim 1 of F)
    #V.shape (M,K): reference volume of each integration point; only ratios within an element matter
    J=det(F)
    V=V.reshape(J.shape)
    return (V*J).sum(dim=1, keepdim=True)/V.sum(dim=1, keepdim=True)
#%%
def cal_invariant(F, Orientation):
    #I1_, I41_, I42_ of C_=J^(-2/3)*C, and C, J2=det(C)
    Ft=F.permute(0,1,3,2)
    C=matmul(Ft, F)
    J2=det(C)
    C_=pow(J2.view(J2.shape[0],J2.shape[1],1,1),-1/3)*C
    I1=C_[:,:,0,0]+C_[:,:,1,1]+C_[:,:,2,2]
    a1, a2=cal_fiber_direction(Orientation)
    a1=a1.view(a1.shape[0],-1,3,1)
    a1t=a1.view(a1.shape[0],-1,1,3)
    I41=matmul(matmul(a1t, C_), a1)
    I41=I41.view(F.shape[0],F.shape[1])
    a2=a2.view(a2.shape[0],-1,3,1)
    a2t=a2.view(a2.shape[0],-1,1,3)
    I42=matmul(matmul(a2t, C_), a2)
    I42=I42.view(F.shape[0],F.shape[1])
    return I1, I41, I42, C, J2
#%%
def cal_strain_energy_density(F, Mat, Orientation, V=None):
    #V: None (pointwise J) or the reference volume of each integration point, shape (M,K)
    #F.shape (M,K,3,3), M is the number of elements, K is the number of integration points
    #Mat.shape Mx5 or 1x5: c0, k1, k2, zeta, k
    #Orientation.shape (M,3,3), e1=Orientation[:,:,0], e2=Orientation[:,:,1], e3=Orientation[:,:,2]
    if F.requires_grad==False:
        F.requires_grad=True
    c0=Mat[:,0:1]
    k1=Mat[:,1:2]
    k2=Mat[:,2:3]
    zeta=Mat[:,3:4]
    k=Mat[:,4:5]
    I1, I41, I42, C, J2=cal_invariant(F, Orientation)
    logJ2=log(J2+1e-18)
    temp1=I1-3
    E1=relu(I41-1)
    E2=relu(I42-1)
    A=zeta*zeta*E1*E1+(1-zeta)*(1-zeta)*E2*E2
    W1=(0.5*c0)*temp1
    #W2=k1/(2*k2)*[exp(k2*A)-1]; for abs(k2) <= eps_k2 its series in k2 to k2^2, whose k2 = 0 limit is k1/2*A
    #(0/0 = NaN before); the branch depends on k2 only, not on F, so W and the force are as smooth in the deformation as before
    eps_k2=1e-6 #below it, k2*A is too small for expm1(k2*A)/k2 to give an accurate dW/dk2; the series error is ~1e-13
    k2_large=(k2.abs() > eps_k2)
    #the branch not taken must be finite with a finite gradient, or torch.where gives 0*inf or 0*NaN = NaN in the gradient:
    #it divides by 1 (no 0/0) and takes expm1(0) (expm1(1*A) overflows to inf for A > ~709.8)
    k2_safe=torch.where(k2_large, k2, torch.ones_like(k2))
    k2_arg=torch.where(k2_large, k2, torch.zeros_like(k2))
    W2=(0.5*k1)*torch.where(k2_large, torch.expm1(k2_arg*A)/k2_safe, A+(0.5*k2)*A*A+(k2*k2/6)*A*A*A)
    if V is None:
        W3=(0.25*k)*(J2-1-logJ2)
    else:
        theta=cal_mean_dilatation(F, V)
        theta2=theta*theta
        W3=(0.25*k)*(theta2-1-log(theta2+1e-18))
    W=W1+W2+W3
    #W.shape is (M,K)
    return W
#%%
def cal_1pk_stress(F, Mat, Orientation, create_graph, return_W, V=None):
    #set create_graph=True for optimization task
    with torch.enable_grad():
        W=cal_strain_energy_density(F, Mat, Orientation, V)
        if V is None:
            S=torch.autograd.grad(W.sum(), F, retain_graph=True, create_graph=create_graph)[0]
        else:
            #P_k = (1/V_k) dE/dF_k with E = sum_k V_k*W_k, the energy of the element
            V=V.reshape(F.shape[0],F.shape[1])
            S=torch.autograd.grad((V*W).sum(), F, retain_graph=True, create_graph=create_graph)[0]
            S=S/V.view(F.shape[0],F.shape[1],1,1)
    #S.shape is (M,K,3,3), M is the number of elements, K is the number of integration points
    if return_W == True:
        return S, W
    else:
        return S
#%%
def cal_cauchy_stress(F, Mat, Orientation, create_graph, return_W, V=None):
    S, W=cal_1pk_stress(F, Mat, Orientation, create_graph, True, V)
    J=det(F).view(F.shape[0],F.shape[1],1,1)
    Ft=F.permute(0,1,3,2)
    Sigma=(1/J)*matmul(S, Ft)
    if return_W == True:
        return Sigma, W
    else:
        return Sigma
#%%
def cal_2pk_stress1(F, Mat, Orientation, V=None):
    #explicit 2nd PK stress (no autograd), an independent check of cal_1pk_stress: P = F*S
    #S_ = 2*dW/dC_ = c0*I + 2*k1*exp(k2*A)*(zeta^2*E1*a1a1 + (1-zeta)^2*E2*a2a2)
    #S_iso = J^(-2/3)*(S_-(1/3)*(S_:C)*inv(C)),  S_vol = U'(J)*J*inv(C) with U'(J) = k/2*(J-1/J)
    #with V, U'(theta_e) replaces U'(J)
    c0=Mat[:,0:1].view(-1,1,1,1)
    k1=Mat[:,1:2].view(-1,1,1,1)
    k2=Mat[:,2:3].view(-1,1,1,1)
    zeta=Mat[:,3:4].view(-1,1,1,1)
    k=Mat[:,4:5].view(-1,1,1,1)
    I1, I41, I42, C, J2=cal_invariant(F, Orientation)
    J=det(F).view(F.shape[0],F.shape[1],1,1)
    invC=inv(C)
    E1=relu(I41-1).view(F.shape[0],F.shape[1],1,1)
    E2=relu(I42-1).view(F.shape[0],F.shape[1],1,1)
    A=zeta*zeta*E1*E1+(1-zeta)*(1-zeta)*E2*E2
    a1, a2=cal_fiber_direction(Orientation)
    a1=a1.view(a1.shape[0],-1,3,1)
    a2=a2.view(a2.shape[0],-1,3,1)
    a1a1=matmul(a1, a1.transpose(2,3))
    a2a2=matmul(a2, a2.transpose(2,3))
    Identity=torch.eye(3,3, device=F.device, dtype=F.dtype).view(1,1,3,3)
    S_=c0*Identity+(2*k1*exp(k2*A))*(zeta*zeta*E1*a1a1+(1-zeta)*(1-zeta)*E2*a2a2)
    Siso=pow(J,-2/3)*(S_-(1/3)*sum(S_*C, dim=(2,3), keepdim=True)*invC)
    if V is None:
        Svol=(0.5*k)*(J*J-1)*invC
    else:
        theta=cal_mean_dilatation(F, V).view(-1,1,1,1)
        Svol=(0.5*k)*(theta-1/theta)*J*invC
    S=Siso+Svol
    return S
#%%
def cal_cauchy_stress1(F, Mat, Orientation, V=None):
    S=cal_2pk_stress1(F, Mat, Orientation, V)
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
    Orientation=torch.linalg.qr(torch.randn(M,3,3, dtype=dtype))[0]
    Mat=torch.tensor([[20, 100, 10, 0.6, 1e5]], dtype=dtype)
    for v in [None, V]:
        Sigma=cal_cauchy_stress(F.clone(), Mat, Orientation, create_graph=False, return_W=False, V=v)
        Sigma1=cal_cauchy_stress1(F, Mat, Orientation, V=v)
        print("V is None:", v is None, "max|Sigma-Sigma1|/max|Sigma1| =", float((Sigma-Sigma1).detach().abs().max()/Sigma1.abs().max()))
