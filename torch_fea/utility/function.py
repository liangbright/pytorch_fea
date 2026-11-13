import numpy as np
import torch_scatter
import torch
from torch_fea.utility.polar_decomposition import polar_decomposition, polar_decomposition_RU, polar_decomposition_VR
from torch_fea.utility.principal_stress import cal_principal_stress, cal_max_abs_principal_stress, cal_min_abs_principal_stress
from torch_fea.utility.principal_strain import cal_principal_strain, cal_max_abs_principal_strain, cal_min_abs_principal_strain
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
    attribute=element_attribute.reshape(M,1,a).expand(M,A,a).contiguous() #reshape: the inputs may be non-contiguous
    #int64 index: with an int32 index on the CPU, the backward of scatter gives wrong gradient rows (torch 2.12.1)
    attribute=torch_scatter.scatter(attribute.view(-1,a), element.reshape(-1).to(torch.int64), dim=0, dim_size=n_nodes, reduce="mean")
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
    #S is cauchy stress (symmetric), shape (...,3,3), a torch tensor or a numpy array
    #VM^2 = 3/2*s:s, s = S-tr(S)/3*I the deviatoric stress; for a symmetric S that is
    #VM^2 = 0.5*((Sxx-Syy)^2+(Syy-Szz)^2+(Szz-Sxx)^2) + 3*(Sxy^2+Syz^2+Szx^2)
    #the differences cancel the hydrostatic stress exactly; the expanded form Sxx^2+Syy^2+Szz^2-Sxx*Syy-Syy*Szz-Szz*Sxx+...
    #is the same in exact arithmetic, but loses VM when the hydrostatic stress is much larger (0 for diag(1,0,-1)+1e8*I)
    #VM^2 is a sum of squares, so it is never negative
    Sxx=S[...,0,0]
    Syy=S[...,1,1]
    Szz=S[...,2,2]
    Sxy=S[...,0,1]
    Syz=S[...,1,2]
    Szx=S[...,2,0]
    VM=0.5*((Sxx-Syy)**2+(Syy-Szz)**2+(Szz-Sxx)**2)+3*(Sxy**2+Syz**2+Szx**2)
    if apply_sqrt == True:
        if isinstance(S, torch.Tensor):
            VM=torch.sqrt(VM)
        elif isinstance(S, np.ndarray):
            VM=np.sqrt(VM)
        else:
            raise ValueError("unkown type(S):"+str(type(S)))
    return VM
