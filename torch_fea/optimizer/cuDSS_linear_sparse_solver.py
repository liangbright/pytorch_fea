# -*- coding: utf-8 -*-
# GPU sparse direct solver: NVIDIA cuDSS through nvmath-python, with an API like pypardiso.
#
# API
#   spsolve(A, b)     solve A x = b; spsolve(A, b, ir_num_steps=2) with 2 steps of iterative refinement (see Accuracy)
#   factorized(A)     factorize A and return a function that solves A x = b for a given b
#   cuDSSSolver       the class behind them: factorize(A), solve(A, b, ir_num_steps=0), free_memory()
#   A: square torch sparse CSR tensor on the GPU (torch.sparse_csr_tensor), float64 or float32, int32 or int64 indices
#   b: torch tensor on the same GPU, shape (n,) or (n, k)
#
# Install (the env pytorch2.12.1cuda13.2py3.14, CUDA 13)
#   pip install nvmath-python "nvidia-cudss-cu13==0.8.*"   (nvmath-python 1.0 works only with cuDSS 0.8)
#   pip install "cuda-core==1.2.2"   (cuda-core 1.3 needs cuda-bindings >= 13.5, newer than the one torch uses)
#
# Reuse of the factorization: like pypardiso, the solver keeps the factorization of the last matrix
#   the same matrix again       -> solve only
#   same sparsity, new values   -> numeric factorization only; the reordering and symbolic factorization (the "plan")
#                                  are kept
#   new sparsity                -> plan and factorization
#   The plan is the slow part, so keep the sparsity pattern the same from one stiffness matrix to the next: keep the
#   entries that happen to be 0. process_H in example/aorta/AortaFEModel.py does that for a matrix on the GPU.
#   If the plan or the factorization fails, the solver is cleared and the next call starts with a new plan.
#
# Time and memory (aorta meshes, A800)
#   plan: hex8 ~110 ms, tet10 ~770 ms
#   one new matrix and one solve: hex8 (29400 unknowns) 16-19 ms, tet10 (177300 unknowns) ~200 ms
#   (pypardiso, with the conversion to scipy: 110-130 ms and 1.3 s)
#   The factorization stays in GPU memory (hex8 ~0.4 GB, tet10 ~2.2 GB) until free_memory() is called.
#
# GPU device
#   nvmath works on torch's current CUDA device, so its calls are made inside torch.cuda.device(A.device): the solver
#   uses only the GPU of A. The scripts select the GPU with torch.device('cuda:N') and do not call
#   torch.cuda.set_device; without this, every solve also opened a CUDA context on device 0.
#
# Accuracy
#   By default the solver uses the default options of cuDSS: no iterative refinement and no check of the residual
#   (factorization_info is not read; it is 0 even for a singular matrix).
#   - Matrices with a strong diagonal, such as the stiffness matrices of the aorta examples, are solved as accurately
#     as by pypardiso (hex8 9e-14, tet10 1.6e-10).
#   - A matrix with a weak diagonal, whose pivots must come from off the diagonal, can be solved inaccurately without
#     any warning. Tested 2026-11-12 with n = 3000: one zero diagonal entry gave the residual 1.5e-4 and a
#     saddle-point matrix [K B^T; B 0] 3.4e-4, where pypardiso gave 1e-16.
#   - A simple rule: after a solve, compute the relative residual norm(A x - b)/norm(b). In float64 it should be near
#     rounding (the aorta matrices: hex8 9e-14, tet10 1.6e-10, the same with pypardiso). If it is well above about
#     1e-10, turn on iterative refinement.
#   - For such matrices (mixed u-p or other saddle-point systems), check the residual, or turn on iterative refinement:
#     ir_num_steps, the number of steps (an integer >= 0, 0 is off), in spsolve(A, b, ir_num_steps=2), solve(A, b,
#     ir_num_steps=2) and the function from factorized(A). It applies to that call only and needs no new plan or
#     factorization; any other value (True, a float, a tensor, None) raises ValueError. On the two cases above,
#     2 steps gave 7.5e-13 and 2.4e-10, 5 steps 1.1e-15 and 2.2e-16. Cost, aorta hex8 matrix (29400 unknowns), A800,
#     factorization kept: 1.3 ms per solve without, 2.6 ms with 2 steps, 5.0 ms with 5 steps.
#   - As H0 of the L-BFGS optimizers, an inexact solve costs only iterations: the line search and the convergence test
#     use the true residual.
#%%
import functools
import logging
import numbers
import torch

#nvmath logs a warning for every new solver because no multithreading library is given for the CPU part of the plan
_logger=logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())
_logger.propagate=False

def _nvmath_sparse():
    #nvmath is imported on first use, so that importing this file works without it
    try:
        from nvmath.sparse import advanced
    except ImportError as err:
        raise ImportError('cuDSS_linear_sparse_solver needs nvmath-python and cuDSS: '
                          'pip install nvmath-python "nvidia-cudss-cu13==0.8.*"') from err
    return advanced

class cuDSSSolver:
    def __init__(self, mtype='general', logger=None):
        #mtype: 'general' (LU, as the default mtype=11 of pypardiso), 'symmetric' (LDL^T), 'spd' (Cholesky)
        #logger: a logging.Logger for the messages of nvmath; None: no messages
        if mtype not in ('general', 'symmetric', 'spd'):
            raise ValueError("mtype must be 'general', 'symmetric' or 'spd'")
        self.mtype=mtype
        self.logger=_logger if logger is None else logger
        self.solver=None  #nvmath DirectSolver
        self.A=None       #the solver's own copy of the matrix (int32 CSR); the values of a new matrix are copied into it
        self.b=None       #the solver's right-hand side, shape (n,); b is copied into it
        self.factorized=False

    def factorize(self, A):
        #factorize A and keep the factorization; the plan is made only if the sparsity pattern of A is new
        #if anything here fails (e.g. out of memory in the plan or the factorization), the solver is cleared before the
        #error is raised again: the next call then makes a new plan, and never uses a half-made solver or the
        #factorization of an earlier matrix
        self._check_A(A)
        self.factorized=False
        try:
            with torch.cuda.device(A.device): #nvmath works on the current device: the GPU of A
                if self._is_same_pattern(A):
                    #in place: nvmath keeps the plan only while the matrix buffers stay the same (reset_operands(a=...) drops it)
                    self.A.values().copy_(A.values())
                else:
                    self.free_memory()
                    if A.values().numel() >= 2**31:
                        raise ValueError('cuDSSSolver: more than 2^31-1 nonzeros')
                    self.A=torch.sparse_csr_tensor(A.crow_indices().to(torch.int32, copy=True),
                                                   A.col_indices().to(torch.int32, copy=True),
                                                   A.values().clone(), size=A.shape)
                    self.b=torch.zeros(A.shape[0], dtype=A.dtype, device=A.device)
                    advanced=_nvmath_sparse()
                    mtype={'general':advanced.DirectSolverMatrixType.GENERAL,
                           'symmetric':advanced.DirectSolverMatrixType.SYMMETRIC,
                           'spd':advanced.DirectSolverMatrixType.SPD}[self.mtype]
                    options=advanced.DirectSolverOptions(sparse_system_type=mtype, logger=self.logger)
                    self.solver=advanced.DirectSolver(self.A, self.b, options=options)
                    self.solver.plan()
                self.solver.factorize()
        except BaseException:
            try:
                self.free_memory()
            except Exception:
                pass #free_memory clears the solver in any case; the first error is the one to raise
            raise
        self.factorized=True

    def solve(self, A, b, ir_num_steps=0):
        #solve A x = b; A is factorized first unless it is the matrix of the last factorization
        #ir_num_steps: the number of steps of iterative refinement for this solve, 0 (off) or more; it is set at every
        #call, so it does not carry over to the next one, and it needs no new plan or factorization
        #returns x with the shape of b, in the dtype of A
        self._check_A(A)
        #only an int or a numpy integer >= 0: a bool (also numpy's), a float, a tensor or None raises ValueError
        if not (isinstance(ir_num_steps, numbers.Integral) and not isinstance(ir_num_steps, bool) and ir_num_steps >= 0):
            raise ValueError('ir_num_steps must be an integer >= 0 (int or numpy integer), not '+repr(ir_num_steps))
        if not self._is_already_factorized(A):
            self.factorize(A)
        self._check_b(A, b)
        if b.dim() == 1:
            return self._solve_vector(b, int(ir_num_steps))
        #the right-hand side of the solver has the shape (n,): the columns of b are solved one by one
        return torch.stack([self._solve_vector(b[:,j], int(ir_num_steps)) for j in range(0, b.shape[1])], dim=1)

    def _solve_vector(self, b, ir_num_steps):
        self.b.copy_(b)
        with torch.cuda.device(self.b.device): #nvmath works on the current device: the GPU of A
            self.solver.solution_config.ir_num_steps=ir_num_steps
            return self.solver.solve()  #a new tensor at every call

    def remove_stored_factorization(self):
        #the next solve factorizes again (the plan is kept)
        self.factorized=False

    def free_memory(self):
        #free the plan, the factorization and the solver's copy of the matrix
        #the solver is cleared even if free() raises, so that the next factorize starts from a new plan
        try:
            if self.solver is not None:
                if self.A is not None:
                    with torch.cuda.device(self.A.device):
                        self.solver.free()
                else:
                    self.solver.free()
        finally:
            self.solver=None
            self.A=None
            self.b=None
            self.factorized=False

    def _is_same_pattern(self, A):
        if self.A is None:
            return False
        if A.shape != self.A.shape or A.dtype != self.A.dtype or A.device != self.A.device:
            return False
        if A.col_indices().numel() != self.A.col_indices().numel():
            return False
        return (torch.equal(A.crow_indices().to(torch.int32), self.A.crow_indices())
                and torch.equal(A.col_indices().to(torch.int32), self.A.col_indices()))

    def _is_already_factorized(self, A):
        return self.factorized and self._is_same_pattern(A) and torch.equal(A.values(), self.A.values())

    def _check_A(self, A):
        if not (torch.is_tensor(A) and A.layout == torch.sparse_csr):
            raise TypeError('A must be a torch sparse CSR tensor (torch.sparse_csr_tensor)')
        if not A.is_cuda:
            raise ValueError('A must be on the GPU')
        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('A must be a square matrix')
        if A.dtype not in (torch.float64, torch.float32):
            raise TypeError('A must be float64 or float32')

    def _check_b(self, A, b):
        if not torch.is_tensor(b):
            raise TypeError('b must be a torch tensor')
        if b.device != A.device:
            raise ValueError('b must be on the device of A')
        if b.dim() not in (1, 2) or b.shape[0] != A.shape[0]:
            raise ValueError('b must have the shape (n,) or (n, k), n = A.shape[0]')

#the solver used by spsolve and factorized, as pypardiso_solver in pypardiso
cudss_solver=cuDSSSolver()

def spsolve(A, b, factorize=True, squeeze=True, solver=None, ir_num_steps=0):
    #solve A x = b, as pypardiso.spsolve
    #A: torch sparse CSR tensor on the GPU; b: torch tensor (n,) or (n, k) on the same GPU
    #factorize: kept for the same call as pypardiso; the factorization is always kept for the next call
    #squeeze: x.squeeze() is returned, as scipy and pypardiso do: shape (n,) if b has the shape (n, 1)
    #solver: a cuDSSSolver; None: cudss_solver
    #ir_num_steps: the number of steps of iterative refinement for this call only, an integer >= 0; 0 (default) is off
    #    (see "Accuracy" at the top of this file)
    if solver is None:
        solver=cudss_solver
    x=solver.solve(A, b, ir_num_steps=ir_num_steps)
    if squeeze:
        return x.squeeze()
    return x

def factorized(A, solver=None):
    #factorize A and return a function that solves A x = b for b, as pypardiso.factorized
    if solver is None:
        solver=cudss_solver
    solver.factorize(A)
    return functools.partial(solver.solve, A)
