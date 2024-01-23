import torch
from torch import matmul, diag_embed
from torch.linalg import svd
#%%
def polar_decomposition(F, return_lambda=False):
    #F.shape (M, K, 3, 3)
    #F=RU=VR
    A, L, B = svd(F) # F = ALB = ALA'AB
    Le=diag_embed(L)
    #F'F=UU=B'LLB, then U=B'LB
    #FF'=VV=ALLA', then V=ALA'
    #R=AB
    At=A.permute(0,1,3,2)
    Bt=B.permute(0,1,3,2)
    U=matmul(Bt, matmul(Le, B))
    V=matmul(A,  matmul(Le, At))
    R=matmul(A, B)
    if return_lambda == False:
        return V, R, U
    else:
        return V, R, U, L
#%%
def polar_decomposition_RU(F, return_lambda=False):
    A, L, B = svd(F)
    Le=diag_embed(L)
    Bt=B.permute(0,1,3,2)
    U=matmul(Bt, matmul(Le, B))
    R=matmul(A, B)
    if return_lambda == False:
        return R, U
    else:
        return R, U, L
#%%
def polar_decomposition_VR(F, return_lambda=False):
    A, L, B = svd(F)
    Le=diag_embed(L)
    At=A.permute(0,1,3,2)
    V=matmul(A, matmul(Le, At))
    R=matmul(A, B)
    if return_lambda == False:
        return V, R
    else:
        return V, R, L
#%%
if __name__ == "__main__":
    F=torch.rand(10,8,3,3, dtype=torch.float64)
    V, R, U, L=polar_decomposition(F, return_lambda=True)
    F1=matmul(R,U)
    F2=matmul(V,R)
    print((F2-F).abs().max(), (F1-F).abs().max())
