import torch
from torch import matmul
from torch.linalg import svd, eigh, eigvalsh
#%%
def cal_principal_stress(S):
    #S is stress, shape (?,...,3,3)
    if S.shape[-1] != 3 and S.shape[-2] !=3:
        raise ValueError("wrong S shape")
    Sp = eigvalsh(S)
    return Sp
#%%
def cal_max_abs_principal_stress(S):
    #S is stress, shape (?,...,3,3)
    if S.shape[-1] != 3 and S.shape[-2] !=3:
        raise ValueError("wrong S shape")
    Sp = eigvalsh(S)
    Sp=Sp.abs().max(dim=-1, keepdim=True)[0]
    return Sp
#%%
def cal_min_abs_principal_stress(S):
    #S is stress, shape (?,...,3,3)
    if S.shape[-1] != 3 and S.shape[-2] !=3:
        raise ValueError("wrong S shape")
    Sp = eigvalsh(S)
    Sp=Sp.abs().min(dim=-1, keepdim=True)[0]
    return Sp
#%%
if __name__ == "__main__":
    S=torch.rand(1,1,3,3, dtype=torch.float64)
    S=S+S.transpose(2,3)
    U, k, Vt = svd(S)
    k=torch.diag_embed(k)
    SS=matmul(matmul(U,k), Vt)
    a, b =  eigh(S)
    a=torch.diag_embed(a)
    c=matmul(matmul(b,a), b.transpose(2,3))

    d=eigvalsh(S)
    Sp=cal_max_abs_principal_stress(S)
