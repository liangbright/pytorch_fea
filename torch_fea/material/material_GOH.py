# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 00:19:57 2021

@author: liang
"""
import torch
from torch import bmm, log, cos, sin, exp, pow
from torch_fea.utility.small_matmul import matmul #torch.matmul on the CPU, elementwise for small matrices on the GPU
from torch.linalg import det
from torch.nn.functional import relu
import numpy as np
#%%
# GOH model: Gasser, Ogden, Holzapfel (2006), J R Soc Interface 3:15-35, with fibre dispersion kappa.
#   W = c0/2*(I1_-3) + k1/(2*k2)*sum_a[exp(k2*<E_a>^2)-1] + U
#   E_a = kappa*(I1_-3) + (1-3*kappa)*(I4a_-1), a = 1,2: two fibre families at +-gamma from e1 in the e1-e2 plane
#   I1_ and I4a_ are computed from the isochoric C_=J^(-2/3)*C; <.> is the Macaulay bracket (a family is off if E_a<0)
# This is not the HGO model of Holzapfel, Gasser, Ogden (2000), which has no dispersion (kappa=0 here).
# FEBio and Abaqus implement this GOH form under the name "Holzapfel-Gasser-Ogden":
#   FEBio user manual v3.4 section 4.1.2.6 (material "Holzapfel-Gasser-Ogden");
#   Abaqus anisotropic hyperelastic "Holzapfel-Gasser-Ogden form".
# Volumetric part: U(J) = k/4*(J^2-1-ln(J^2)) = k/2*((J^2-1)/2-ln(J))
#   FEBio: pressure_model 3 "Abaqus (GOH)", the default of its Holzapfel-Gasser-Ogden material; Abaqus: 1/D*((J^2-1)/2-ln(J))
# Mat = [c0, k1, k2, kappa, gamma, k], gamma in radians; k2>=0 (k2=0 is the limit W2=k1/2*sum_a <E_a>^2)
#   FEBio: c=c0, k1, k2, kappa, gamma (degrees), k;  Abaqus: C10=c0/2, k1, k2, kappa, D=2/k
# With V (the volume of each integration point), U is evaluated at the element mean dilatation theta_e
#   instead of the pointwise J (mean dilatation; cf. FEBio three-field solid domain)
#%% fiber orientation in global sys
def cal_fiber_direction(gamma, local_sys):
    #local_sys.shape (M,3,3) or (M,1,3,3) or (M,8,3,3)
    e1=local_sys[...,0]
    e2=local_sys[...,1]
    #gamma.shape (M,1) or (1,1) -> (M,1) for e1 (M,3), (M,1,1) for e1 (M,1,3) or (M,8,3): gamma[m] is used for element m
    gamma=gamma.view(gamma.shape[0], *([1]*(e1.dim()-1)))
    temp1=cos(gamma)*e1
    temp2=sin(gamma)*e2
    a1=temp1+temp2
    a2=temp1-temp2
    #print(a1.shape, a2.shape)
    #a1.shape (M,3) or (M,1,3) or (M,8,3)
    #a2.shape (M,3) or (M,1,3) or (M,8,3)
    return a1, a2
#%%
def cal_mean_dilatation(F, V):
    #theta_e = sum_k V_k*J_k / sum_k V_k over the K integration points of each element (dim 1 of F)
    #V.shape (M,K): reference volume of each integration point; only ratios within an element matter
    J=det(F)
    V=V.reshape(J.shape)
    return (V*J).sum(dim=1, keepdim=True)/V.sum(dim=1, keepdim=True)
#%%
def cal_strain_energy_density(F, Mat, Orientation, V=None):
    #V: None (pointwise GOH) or the reference volume of each integration point, shape (M,K)
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
    Ft=F.permute(0,1,3,2)
    C=matmul(Ft, F)
    J2=det(C)
    #print(J2[0])
    logJ2=log(J2+1e-18)
    C_=pow(J2.view(J2.shape[0],J2.shape[1],1,1),-1/3)*C
    #print(det(C_))
    I1=C_[:,:,0,0]+C_[:,:,1,1]+C_[:,:,2,2]
    a1, a2=cal_fiber_direction(gamma, Orientation)
    a1=a1.view(a1.shape[0], -1,3,1)
    a1t=a1.view(a1.shape[0],-1,1,3)
    #print("C_.shape", C_.shape)
    I41=matmul(matmul(a1t, C_), a1)
    #print("I41.shape", I41.shape)
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
    #print("c0.shape, k1.shape, k2.shape, gamma.shape, kappa.shape, k.shape")
    #print(c0.shape, k1.shape, k2.shape, gamma.shape, kappa.shape, k.shape)
    #print("temp1.shape, E1.shape, E2.shape, J2.shape")
    #print(temp1.shape, E1.shape, E2.shape, J2.shape)
    W1=(0.5*c0)*temp1
    #W2=k1/(2*k2)*sum_a[exp(k2*E_a^2)-1]; for abs(k2) <= eps_k2 its series in k2 to k2^2, whose k2 = 0 limit is k1/2*sum_a E_a^2
    #(0/0 = NaN before); the branch depends on k2 only, not on F, so W and the force are as smooth in the deformation as before
    eps_k2=1e-6 #below it, k2*E^2 is too small for expm1(k2*E^2)/k2 to give an accurate dW/dk2; the series error is ~1e-13
    k2_large=(k2.abs() > eps_k2)
    #the branch not taken must be finite with a finite gradient, or torch.where gives 0*inf or 0*NaN = NaN in the gradient:
    #it divides by 1 (no 0/0) and takes expm1(0) (expm1(1*E^2) overflows to inf for E^2 > ~709.8)
    k2_safe=torch.where(k2_large, k2, torch.ones_like(k2))
    k2_arg=torch.where(k2_large, k2, torch.zeros_like(k2))
    E1sq=E1*E1
    E2sq=E2*E2
    W2_exact=(torch.expm1(k2_arg*E1sq)+torch.expm1(k2_arg*E2sq))/k2_safe
    W2_series=(E1sq+E2sq)+(0.5*k2)*(E1sq*E1sq+E2sq*E2sq)+(k2*k2/6)*(E1sq*E1sq*E1sq+E2sq*E2sq*E2sq)
    W2=(0.5*k1)*torch.where(k2_large, W2_exact, W2_series)
    if V is None:
        W3=(0.25*k)*(J2-1-logJ2)
    else:
        theta=cal_mean_dilatation(F, V)
        theta2=theta*theta
        W3=(0.25*k)*(theta2-1-log(theta2+1e-18))
    W=W1+W2+W3
    #handle negative J
    #J=det(F)
    #WJ=(J<0.001)*(J-0.001)**2
    #W=(J>=0.001)*W+k*WJ
    #W=(0.25*k)*(J2-1-logJ2)
    #print("J.shape", J.shape)
    #print("W.shape", W.shape)
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
