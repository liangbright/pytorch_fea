import torch
from torch import matmul
from torch.linalg import svd, eigh, eigvalsh
#S must be a symmetric strain, e.g. the Green strain: eigvalsh reads one triangle only, so the symmetric part (S+S^T)/2 is used;
#a displacement gradient is not symmetric, and the result for it is not its principal values (use a strain tensor)
#%%
def cal_principal_strain(S):
    #S is strain, shape (?,...,3,3)
    if S.shape[-1] != 3 or S.shape[-2] != 3:
        raise ValueError("wrong S shape")
    Sp = eigvalsh(0.5*(S+S.transpose(-1,-2)))
    return Sp
#%%
def cal_max_abs_principal_strain(S):
    #S is strain, shape (?,...,3,3)
    if S.shape[-1] != 3 or S.shape[-2] != 3:
        raise ValueError("wrong S shape")
    Sp = eigvalsh(0.5*(S+S.transpose(-1,-2)))
    Sp=Sp.abs().max(dim=-1, keepdim=True)[0]
    return Sp
#%%
def cal_min_abs_principal_strain(S):
    #S is strain, shape (?,...,3,3)
    if S.shape[-1] != 3 or S.shape[-2] != 3:
        raise ValueError("wrong S shape")
    Sp = eigvalsh(0.5*(S+S.transpose(-1,-2)))
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
    Sp=cal_max_abs_principal_strain(S)
