# Polar decomposition of the deformation gradient F: F = R*U = V*R
#     R: the rotation;  U and V: the right and the left stretch, symmetric;  L: the principal stretches
#     F.shape (3,3), (M,3,3), (M,K,3,3), or any (...,3,3). Another matrix size raises ValueError.
#
# The values come from the svd F = A*diag(L)*B:
#     R = A*B,   U = B'*diag(L)*B = R'*F,   V = A*diag(L)*A' = F*R',   L from large to small
#     If det(F) < 0, R is a reflection (det(R) = -1).
#
# The derivatives of R, U and V do not go through the svd (changed on 2026-11-12).
#   Why: R, U and V are smooth functions of F, but the singular vectors A and B are not where two singular values
#     are equal (F = I, a rotation, a uniaxial stretch, ...): their derivative has the terms 1/(L_i^2 - L_j^2).
#     With autograd through the svd, the gradient through R, U or V was
#       NaN where two singular values are exactly equal (F = I, diag(1.2, 1, 1));
#       finite but wrong where they are equal up to rounding (a rotation, a rotated uniaxial stretch: 10 to 35%
#         off in the tests);
#       inaccurate where they are close: a relative error of about 1e-16/(L_i - L_j) in the gradient and about
#         1e-16/(L_i - L_j)^2 in the second derivative (5e-8 and 99 for a difference of 1e-9).
#   Now: R has its own derivative (_PolarRotation below), and U = R'*F and V = F*R' are computed from R.
#     dF = dR*U + R*dU, and dR = R*hat(w), where hat(w) is the skew matrix of the vector w.
#     The skew part of R'*dF gives hat(w)*U + U*hat(w) = R'*dF - dF'*R, which is the 3x3 system
#         (tr(U)*I - U) w = the axial vector of (R'*dF - dF'*R)
#     tr(U)*I - U has the eigenvalues L_i + L_j, so equal singular values do no harm.
#     The formula has only R and F in it, not A and B, so it can be differentiated again: the second derivative
#     (e.g. the stiffness matrix of an energy written with U, such as the Biot strain U - I) is right as well.
#   Tested (float64): the gradient and the second derivative are right to rounding (1e-15) at F = I, at rotations,
#     with two or three equal singular values and with det(F) < 0; with torch.autograd.grad, create_graph=True and
#     is_grads_batched=True (what model_stiffness.py uses), on the CPU and the GPU.
#
# Limits:
#   L (return_lambda=True) keeps the derivative of the svd. Its first derivative is right. Its second derivative is
#     NaN or wrong where singular values are equal, because the singular values themselves are not smooth there.
#     For an energy, use U (e.g. tr(U), tr(U*U), det(U)) instead of L.
#   Forward-mode derivatives (torch.func.jacfwd, torch.func.hessian) stop with NotImplementedError.
#     Use reverse mode: torch.autograd.grad, torch.func.grad, torch.func.jacrev.
#   F needs at least two nonzero singular values; with fewer, R is not unique.
#%%
import torch
from torch.linalg import svd, det
from torch_fea.utility.small_matmul import matmul #torch.matmul on the CPU, elementwise for small matrices on the GPU
#%%
def _sym(M):
    #the symmetric part of M
    return 0.5*(M+M.transpose(-1,-2))
#%%
def _cal_rotation_derivative(F, R, X):
    #F=RU, X.shape is F.shape
    #return R*hat(w), with (tr(U)*I - U) w = the axial vector of (R'X - X'R); see the top of this file
    #    X=dF: the result is dR, the change of R
    #    X=dLoss/dR: the result is dLoss/dF. It is the same formula, because the map dF -> dR is self-adjoint
    Rt=R.transpose(-1,-2)
    U=_sym(matmul(Rt, F))
    K=matmul(Rt, X)
    k=torch.stack([K[...,2,1]-K[...,1,2], K[...,0,2]-K[...,2,0], K[...,1,0]-K[...,0,1]], dim=-1)
    #w = inv(tr(U)*I - U) k, and by Cayley-Hamilton inv(tr(U)*I - U) = (U*U + I2*I)/(I1*I2 - I3)
    #I1, I2, I3 are the invariants of U; I1*I2 - I3 = (L1+L2)*(L2+L3)*(L3+L1) > 0
    U2=matmul(U, U)
    I1=U[...,0,0]+U[...,1,1]+U[...,2,2]
    I2=0.5*(I1*I1-(U2[...,0,0]+U2[...,1,1]+U2[...,2,2]))
    I3=det(U)
    w=(matmul(U2, k.unsqueeze(-1)).squeeze(-1)+I2.unsqueeze(-1)*k)/(I1*I2-I3).unsqueeze(-1)
    #R*hat(w): its row i is (row i of R) x w
    return torch.linalg.cross(R, w.unsqueeze(-2), dim=-1)
#%%
class _PolarRotation(torch.autograd.Function):
    #R=AB, the rotation of F=ALB (svd). A and B are inputs without a derivative (detached by the caller).
    #backward is written with the input F and the output R, so autograd can differentiate it again
    #no jvp (forward mode): with one, torch.func.jacfwd(jacfwd(...)) gave wrong values without an error
    generate_vmap_rule=True #for torch.func.vmap
    @staticmethod
    def forward(F, A, B):
        return matmul(A, B)
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0], output)
    @staticmethod
    def backward(ctx, grad_R):
        F, R = ctx.saved_tensors
        return _cal_rotation_derivative(F, R, grad_R), None, None
#%%
def _cal_rotation(F):
    #return R and L of F=RU; L: the singular values of F (the principal stretches), from large to small
    if tuple(F.shape[-2:]) != (3,3):
        raise ValueError("F.shape is "+str(tuple(F.shape))+", but the last two dimensions must be (3,3): "
                         "the derivative of R is written for 3x3 matrices")
    A, L, B = svd(F) # F = ALB = ALA'AB
    R=_PolarRotation.apply(F, A.detach(), B.detach()) #R=AB
    return R, L
#%%
def polar_decomposition(F, return_lambda=False):
    #F.shape (3,3), (M,3,3), (M,K,3,3), or any (...,3,3): the last two dimensions are the matrix
    #F=RU=VR
    R, L = _cal_rotation(F)
    Rt=R.transpose(-1,-2)
    #F'F=UU, then U=R'F (=B'LB)
    #FF'=VV, then V=FR' (=ALA')
    U=_sym(matmul(Rt, F))
    V=_sym(matmul(F, Rt))
    if return_lambda == False:
        return V, R, U
    else:
        return V, R, U, L
#%%
def polar_decomposition_RU(F, return_lambda=False):
    R, L = _cal_rotation(F) #F.shape (...,3,3), as polar_decomposition
    U=_sym(matmul(R.transpose(-1,-2), F))
    if return_lambda == False:
        return R, U
    else:
        return R, U, L
#%%
def polar_decomposition_VR(F, return_lambda=False):
    R, L = _cal_rotation(F) #F.shape (...,3,3), as polar_decomposition
    V=_sym(matmul(F, R.transpose(-1,-2)))
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
    #the gradient where singular values are equal (F = I in half of the batch):
    #sum(U*U) = sum(F*F), so its gradient is 2F; sum(R*F) = tr(U), and its gradient is R
    F=torch.eye(3, dtype=torch.float64).repeat(10,8,1,1)
    F[5:]=torch.rand(5,8,3,3, dtype=torch.float64)
    F.requires_grad_(True)
    V, R, U=polar_decomposition(F)
    g1=torch.autograd.grad((U*U).sum(), F, retain_graph=True)[0]
    g2=torch.autograd.grad((R*F).sum(), F)[0]
    print((g1-2*F.detach()).abs().max(), (g2-R.detach()).abs().max())
