import numpy as np
import torch_scatter
import torch
from torch_fea.utils.polar_decomposition import *
from torch_fea.utils.principal_stress import *
from torch_fea.utils.principal_strain import *
#%%
def cal_attribute_on_node(n_nodes, element, element_attribute):
    #n_nodes: the number of nodes
    #element.shape (M,A)
    #element_attribute can be stress, F tensor, detF, shape is (M,?,?,...)
    M=element.shape[0]
    A=element.shape[1]
    a_shape=list(element_attribute.shape)
    a=1
    if len(a_shape) > 1:
        a=np.prod(a_shape[1:])
    attribute=element_attribute.view(M,1,a).expand(M,A,a).contiguous()
    attribute=torch_scatter.scatter(attribute.view(-1,a), element.view(-1), dim=0, dim_size=n_nodes, reduce="mean")
    a_shape[0]=n_nodes
    attribute=attribute.view(a_shape)
    return attribute
#%%
def cal_cauchy_stress_from_1pk_stress(S, F):
    #S.shape (M,K,3,3)
    #F.shape (M,K,3,3)
    J=torch.det(F).view(F.shape[0],F.shape[1],1,1)
    Ft=F.permute(0,1,3,2)
    Sigma=(1/J)*torch.matmul(S, Ft)
    return Sigma
#%%
def cal_von_mises_stress(S, apply_sqrt=True):
    #S is cauchy stress
    Sxx=S[...,0,0]
    Syy=S[...,1,1]
    Szz=S[...,2,2]
    Sxy=S[...,0,1]
    Syz=S[...,1,2]
    Szx=S[...,2,0]
    VM=Sxx**2+Syy**2+Szz**2-Sxx*Syy-Syy*Szz-Szz*Sxx+3*(Sxy**2+Syz**2+Szx**2)
    VM[VM<0]=0
    if apply_sqrt == True:
        if isinstance(S, torch.Tensor):
            VM=torch.sqrt(VM)
        elif isinstance(S, np.ndarray):
            VM=np.sqrt(VM)
        else:
            raise ValueError("unkown type(S):"+str(type(S)))
    return VM
