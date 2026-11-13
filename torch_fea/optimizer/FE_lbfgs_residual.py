# -*- coding: utf-8 -*-
# optimization target: residual
#   the line search looks for a zero of d.R along the search direction d (R from closure(output='R'));
#   it is not for energy: use FE_lbfgs_loss
"""
Created on Fri Jun 11 17:06:15 2021

@author: liang
"""

#alpha α
#rho ρ
#beta β
import torch
#from torch import matmul

from functools import reduce
from torch.optim import Optimizer
#https://github.com/haasad/PyPardisoProject
#download pypardiso to the example folder
from pypardiso import spsolve
#the GPU solver (cuDSS), with the API of pypardiso; nvmath-python is imported on its first use
from torch_fea.optimizer.cuDSS_linear_sparse_solver import spsolve as cudss_spsolve
import numpy as np
import time
import scipy


def linear_spsolve_cpu(H, r, verbose=False):
    #H is the hessian/stiffness, a sparse matrix
    #q=inv(H)*r
    #q=solve(H, r)
    if not isinstance(H, scipy.sparse.csr_matrix):
        raise ValueError("linear_spsolve: H must be scipy.sparse.csr_matrix")
    rr=r.detach().cpu().numpy().reshape(-1).astype("float64")
    t0=time.time()
    q=spsolve(H, rr)
    t1=time.time()
    if verbose == True:
        print("done: linear_spsolve", t1-t0)
        print("r", r)
        print("q", q)
    q=torch.tensor(q, dtype=r.dtype, device=r.device).reshape(-1) #shape (n,): spsolve squeezes 1 unknown to shape ()
    return q

def linear_spsolve_cuda_torch(H, r, verbose=False):
    #the original linear_spsolve_cuda, with torch.sparse.spsolve
    #assume H is scipy.sparse.csr_matrix
    #to use torch.sparse.spsolve: compile PyTorch with cuDSS
    H=H.tocsr()
    #CSR: the row pointers (indptr), not the row index of each entry
    H=torch.sparse_csr_tensor(torch.from_numpy(H.indptr).long(), torch.from_numpy(H.indices).long(),
                              torch.from_numpy(H.data), size=H.shape)
    H=H.to(r.device)
    q=torch.sparse.spsolve(H, r)
    return q

def linear_spsolve_cuda(H, r, verbose=False):
    #H: torch sparse CSR tensor on the GPU (AortaFEModel on the GPU, process_H),
    #   or scipy.sparse.csr_matrix, which is copied to the GPU of r at every call
    #q=solve(H, r) with cuDSS (torch_fea/optimizer/cuDSS_linear_sparse_solver.py); like pypardiso, the solver keeps
    #the factorization: the same H again is only solved, a new H with the same sparsity pattern is only factorized
    if not torch.is_tensor(H):
        H=H.tocsr()
        H=torch.sparse_csr_tensor(torch.from_numpy(H.indptr).to(torch.int32), torch.from_numpy(H.indices).to(torch.int32),
                                  torch.from_numpy(H.data), size=H.shape)
        H=H.to(r.device)
    rr=r.detach().reshape(-1).to(H.dtype)
    t0=time.time()
    q=cudss_spsolve(H, rr)
    if verbose == True:
        torch.cuda.synchronize(H.device)
        t1=time.time()
        print("done: linear_spsolve_cuda", t1-t0)
        print("r", r)
        print("q", q)
    q=q.to(r.dtype).reshape(-1) #shape (n,): spsolve squeezes 1 unknown to shape ()
    return q

def linear_spsolve(H, r, use_cuda=False, verbose=False):
    #H on the GPU (torch sparse CSR tensor) is solved on the GPU (cuDSS);
    #a scipy.sparse.csr_matrix on the CPU (pypardiso), or on the GPU if use_cuda is True
    if use_cuda == True or (torch.is_tensor(H) and H.is_cuda):
        return linear_spsolve_cuda(H, r, verbose)
    else:
        return linear_spsolve_cpu(H, r, verbose)

def update_invHd_slow(s, y, ρ, d, invH0_diag, H0):
    #the same as update_invHd below, written plainly, kept for reference
    #t0=time.time()
    r=d
    n=len(s)
    α=[None]*n
    for i in range(n-1, -1, -1):
        α[i]=ρ[i]*s[i].dot(r)
        r-=α[i]*y[i]
    if H0 is not None:
        r=linear_spsolve(H0, r)
    else:
        r*=invH0_diag
    for i in range(0, n):
        β=ρ[i]*y[i].dot(r)
        r+=(α[i]-β)*s[i]
    #t1=time.time()
    #print("done: update_invHd_slow", t1-t0)
    return r

def update_invHd(s, y, ρ, d, invH0_diag, H0):
    #the same as update_invHd_slow, with fused in-place updates: r.addcmul_(y[i], α[i], value=-1) in place of r-=α[i]*y[i],
    #so no temporary vector for each pair; α[i] stays a 0-d tensor, so there is no GPU synchronization
    #(r.add_(y[i], alpha=-α[i]) would need α[i] as a number)
    r=d
    n=len(s)
    α=[None]*n
    for i in range(n-1, -1, -1):
        α[i]=ρ[i]*s[i].dot(r)
        r.addcmul_(y[i], α[i], value=-1)
    if H0 is not None:
        r=linear_spsolve(H0, r)
    else:
        r*=invH0_diag
    for i in range(0, n):
        β=ρ[i]*y[i].dot(r)
        r.addcmul_(s[i], α[i]-β)
    return r

class LBFGS(Optimizer):

    def __init__(self,
                 params,
                 lr=1,
                 tolerance_grad=1e-7,
                 history_size=100,
                 reset_state_per_step=False,
                 line_search_fn=None,
                 backtracking={"c":0.5, "t_list": [1, 0.5, 0.1, 0.05, 0.01],
                               "t_default":0.5, "t_default_init":'auto', "max_iter":10, "verbose":False}
                 ):
        defaults = dict(
            lr=lr,
            tolerance_grad=tolerance_grad,
            history_size=history_size,
            reset_state_per_step=reset_state_per_step,
            line_search_fn=line_search_fn,
            backtracking=backtracking.copy())
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        self.reset_state()

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().reshape(-1)
            else:
                view = p.grad.reshape(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    def backtracking(self, closure, gd, d, t_list, t_default, c, verbose, max_iter):
        #x=x+td such that loss=>0
        closure = torch.no_grad()(closure)
        p_init = self._clone_param()
        gd=float(gd)
        #-------------------------------
        evaluated=[] #(t, R) of every evaluation, for the choices below when R is NaN or inf
        def call_R(t):
            self._add_grad(t, d)
            R = closure(output='R')
            self._set_param(p_init)
            R=R.reshape(-1)
            #print("shape", d.shape, R.shape)
            R=float((d*R).sum())
            evaluated.append((t, R))
            return R
        #--------------------------------
        #R0=call_R(0) # it is gd
        R0=gd
        if verbose == True:
            print("R0", R0)
            #print("R0", R0, "gd", gd)
        #--------------------------------
        flag0=-1
        R_list=[]
        for n, t in enumerate(t_list):
            Rt = call_R(t)
            R_list.append(Rt)
            if abs(Rt) < c*abs(R0):
                flag0=1
                t1=t2=t
                R1=R2=Rt
                break
            if len(R_list) > 1:
                if R_list[n-1]*R_list[n]<=0:
                    t1=t_list[n-1]; R1=R_list[n-1]
                    t2=t_list[n];   R2=R_list[n]
                    flag0=0
                    break
        if flag0 == -1:
            R_abs=np.abs(np.array(R_list))
            if np.all(np.isnan(R_abs)):
                n=0 #every listed t gives NaN; the fallback at the end decides
            else:
                n=int(np.nanargmin(R_abs)) #the best listed t; np.argmin would return the first NaN
            if (n ==0 or n == len(R_list)-1
                or not np.isfinite(R_abs[n-1]) or not np.isfinite(R_abs[n+1])):
                #the best listed t is at an end of t_list, or next to a NaN or inf: no refinement
                t1=t2=t=t_list[n]
                R1=R2=Rt=R_list[n]
                if verbose == True:
                    print("no zero-crossing, no V shape, R_list", R_list)
            else:
                t1=t_list[n-1]; R1=R_list[n-1]
                t=t_list[n];    Rt=R_list[n] #start from the best listed t
                t2=t_list[n+1]; R2=R_list[n+1]
                if t1 > t2:
                    t1, t2 = t2, t1
                    R1, R2 = R2, R1
                if verbose == True:
                    print("start binary search: t1", t1, "t2", t2, "R1", R1, "R2", R2)
                #t1<t<t2
                for k in range(0, max_iter):
                    if abs(Rt) < c*abs(R0):
                        break
                    #a new point in the larger of the intervals [t1, t] and [t, t2], never t itself
                    if t-t1 > t2-t:
                        t3=(t1+t)/2
                    else:
                        t3=(t+t2)/2
                    R3=abs(call_R(t3))
                    if t3 < t:
                        #t1<t3<t<t2
                        if abs(R3) < abs(Rt):
                            t2=t; R2=Rt
                            t=t3; Rt=R3
                        else:
                            t1=t3; R1=R3
                    else:
                        #t1<t<t3<t2
                        if abs(R3) < abs(Rt):
                            t1=t; R1=Rt
                            t=t3; Rt=R3
                        else:
                            t2=t3; R2=R3
        elif flag0 == 0:
            if verbose == True:
                print("start zero-crossing search: t1", t1, "t2", t2, "R1", R1, "R2", R2)
            t=(t1+t2)/2
            Rt = call_R(t)
            for k in range(0, max_iter):
                if abs(Rt) < c*abs(R0):
                    break
                if Rt*R1 > 0:
                    t1=t
                    R1=Rt
                elif Rt*R2 > 0:
                    t2=t
                    R2=Rt
                else:
                    break
                t=(t1+t2)/2
                Rt = call_R(t)
        if abs(Rt) < abs(R0):
            flag=1
            if verbose == True:
                print("backtracking(good): t", t, "t1", t1, "t2", t2)
                print("backtracking(good): t", t, "Rt", Rt, "R1", R1, "R2", R2)
        elif Rt == R0:
            flag=0
            print("backtracking(ok): t", t, "t1", t1, "t2", t2)
            print("backtracking(ok): t", t, "Rt", Rt, "R1", R1, "R2", R2)
        else: #abs(Rt) > abs(R0) or Rt is nan
            flag=-1
            #no optimal t is found, make the best guess
            #small t is bad for Rmax
            t_found=t
            t=t_default
            #the choices below change nothing when no evaluation gave NaN or inf
            if not np.isfinite(Rt):
                #the search ended on NaN or inf: take the evaluated t with the smallest finite abs(R) < abs(R0)
                good=[(tt, RR) for (tt, RR) in evaluated if np.isfinite(RR) and abs(RR) < abs(R0)]
                if len(good) > 0:
                    t, Rt=min(good, key=lambda e: abs(e[1]))
                    flag=1
            if flag == -1 and any(tt == t_default and not np.isfinite(RR) for (tt, RR) in evaluated):
                #t_default itself gave NaN or inf: take the largest evaluated t with a finite R
                finite=[tt for (tt, RR) in evaluated if np.isfinite(RR)]
                if len(finite) > 0:
                    t=max(finite)
            if verbose == True:
                print("backtracking(bad): t", t_found, "Rt", Rt, "set t to", t, "(t_default =", t_default, ")")
        return t, flag


    def set_backtracking(self, t_list=None, t_default=None, t_default_init=None, c=None, max_iter=None, verbose=None):
        backtracking = self.param_groups[0]['backtracking']
        if t_list is not None:
            backtracking["t_list"]=t_list.copy()
        if t_default is not None:
            backtracking["t_default"]=t_default
        if t_default_init is not None:
            backtracking["t_default_init"]=t_default_init
        if c is not None:
            backtracking["c"]=c
        if max_iter is not None:
            backtracking["max_iter"]=max_iter
        if verbose is not None:
            backtracking['verbose']=verbose

    def set_invH0_diag(self, invH0_diag):
        state = self.state[self._params[0]]
        state["invH0_diag"]=invH0_diag

    def set_H0(self, H0):
        state = self.state[self._params[0]]
        state["H0"]=H0

    def get_H0(self):
        state = self.state[self._params[0]]
        return state["H0"]

    def get_n_iter(self):
        state = self.state[self._params[0]]
        return state['n_iter']

    def reset_state_n_iter(self):
        #the next iteration starts a new memory (s, y, ρ) from H0, as after reset_state, but keeps H0
        state = self.state[self._params[0]]
        state['n_iter']=0

    def reset_state(self):
        state = self.state[self._params[0]]
        state['func_evals']=0
        state['n_iter']=0
        state['ys_counter']=0
        state['t'] = None
        state['d'] = None
        state['s'] = []
        state['y'] = []
        state['ρ'] = []
        state['g_prev'] = None
        state['H_diag'] = None
        state['H0'] = None
        state['invH0_diag'] = None
        state['linesearch_flag']=0

    def set_linesearch_fn(self, line_search_fn):
        group = self.param_groups[0]
        group['line_search_fn']=line_search_fn

    def get_linesearch_fn(self):
        group = self.param_groups[0]
        line_search_fn = group['line_search_fn']
        return line_search_fn

    def get_linesearch_flag(self):
        state = self.state[self._params[0]]
        return state['linesearch_flag']

    def get_linesearch_t(self):
        state = self.state[self._params[0]]
        return state['t']

    def set_lr(self, lr):
        group = self.param_groups[0]
        group['lr']=lr

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        #closure_grad: closure is called with grad enabled
        #do loss.backward() inside closure; with output='R', return R before any loss.backward(): backtracking calls closure(output='R') with grad disabled
        closure_grad = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        tolerance_grad = group['tolerance_grad']
        history_size = group['history_size']
        reset_state_per_step=group["reset_state_per_step"]
        line_search_fn = group['line_search_fn']
        backtracking= group['backtracking']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        if reset_state_per_step == True:
            self.reset_state_n_iter()

        # evaluate initial f(x) and df/dx
        loss = float(closure_grad())
        if np.isnan(loss):
            print("abort: loss is nan, optimizer will return 'nan'")
            return "nan"
        if np.isinf(loss):
            print("abort: loss is inf, optimizer will return 'inf'")
            return "inf"

        current_evals = 1
        state['func_evals'] += 1

        g = self._gather_flat_grad()
        opt_cond = bool(g.abs().max() <= tolerance_grad)

        # optimal condition
        if opt_cond:
            print("optimal condition1, break the current iteration")
            return opt_cond

        t = state.get('t')
        d = state.get('d')
        s = state.get('s')
        y = state.get('y')
        ρ = state.get('ρ')
        g_prev = state.get('g_prev')
        H_diag = state.get('H_diag')
        invH0_diag = state.get('invH0_diag')
        H0 = state.get('H0')

        ys_counter=state.get('ys_counter')

        # keep track of nb of iterations
        state['n_iter'] += 1

        ############################################################
        # compute gradient descent direction
        ############################################################
        if state['n_iter'] == 1:
            s = []
            y = []
            ρ = []
            H_diag = 1
            d = -g
            if H0 is not None:
                d=linear_spsolve(H0, d)
                t=1
            else:
                if invH0_diag is not None:
                    d*=invH0_diag
                    t=1
        else:
            # do lbfgs update (update memory)
            y_new = g-g_prev
            s_new = t*d # t and d from the previous iteration
            ys = y_new.dot(s_new)
            yy = y_new.dot(y_new)
            ss = s_new.dot(s_new)
            #curvature test relative to |s|*|y|, so that it does not depend on the units of force and length
            if ys > 1e-10*torch.sqrt(ss*yy) and yy > 0:
                # updating memory
                if len(s) == history_size:
                    # shift history by one (limited-memory)
                    s.pop(0)
                    y.pop(0)
                    ρ.pop(0)
                # store new direction/step
                s.append(s_new)
                y.append(y_new)
                ρ.append(1./ys)
                # update scale of initial Hessian approximation
                H_diag = ys / yy # y.dot(y)  # (y*y)
                ys_counter=0
            else:
                ys_counter+=1
            # compute the approximate (L-BFGS) inverse Hessian multiplied by the gradient
            # also with an empty memory, so that H0 (or the diagonal scaling) is applied to -g
            # the scaling is the current H_diag unless invH0_diag is set by set_invH0_diag
            d=-g
            d=update_invHd(s, y, ρ, d, invH0_diag if invH0_diag is not None else H_diag, H0)
        #if ys_counter > 10:
        #    print("ys_counter", ys_counter)
        #------- record grad and loss ----------------------------
        if g_prev is None:
            g_prev = g.clone(memory_format=torch.contiguous_format)
        else:
            g_prev.copy_(g)
        prev_loss = loss
        #----------------------------------------------------------
        # directional derivative
        gd = g.dot(d)
        #gd > 0 is not turned into a descent direction here: the line search looks for a zero of d.R along d,
        #which a Newton step with a non-symmetric tangent (e.g. inverse p0, no energy) has also when gd > 0;
        #turning d around there makes the step go the wrong way

        ############################################################
        # Line Search
        ############################################################

        ls_func_evals = 0
        if line_search_fn is not None and line_search_fn != "none":
            # perform line search, using user function
            if line_search_fn == "backtracking":
                t_default=backtracking["t_default"]
                if state['n_iter'] == 1:
                    t_default_init=backtracking["t_default_init"]
                    if isinstance(t_default_init, str):
                        if t_default_init == "auto":
                            t_default_init=lr*float(min(1., 1. / g.abs().sum()))
                            t_default_init=min(t_default_init, min(backtracking["t_list"]))
                            t_default=min(t_default_init, t_default)
                        else:
                            raise ValueError("unknown t_default_init:"+t_default_init)
                    else: # t_default_init should be float or int
                        t_default=float(t_default_init)
                t, flag = self.backtracking(closure, gd, d,
                                            t_list=backtracking["t_list"],
                                            t_default=t_default,
                                            c=backtracking["c"],
                                            max_iter=backtracking["max_iter"],
                                            verbose=backtracking['verbose'])
                #---------------------
                state['linesearch_flag']=flag
                state['t']=t
                self._add_grad(t, d)
                loss = float(closure_grad())
                flat_grad = self._gather_flat_grad()
                opt_cond = bool(flat_grad.abs().max() <= tolerance_grad)
                g = flat_grad #the gradient at the new point, for the next iteration
            else:
                raise RuntimeError(line_search_fn+" not supported")
        else:
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t=lr*float(min(1., 1. / g.abs().sum()))
            else:
                t = lr
            # no line search, simply move with fixed-step
            self._add_grad(t, d)

        # update func eval
        current_evals += ls_func_evals
        state['func_evals'] += ls_func_evals

        ############################################################
        # check conditions
        ############################################################
        # optimal condition
        if opt_cond:
            print("optimal condition2, break the current iteration")

        state['t'] = t
        state['d'] = d
        state['s'] = s
        state['y'] = y
        state['ρ'] = ρ
        state['g_prev'] = g_prev
        state['ys_counter']=ys_counter
        state['H_diag']=H_diag
        return opt_cond

    def load_state_dict(self, state_dict):
        #disabled: this optimizer is not saved and loaded (FEBio does not do that either), and torch's
        #load_state_dict cannot load a scipy H0 (set_H0) back: it rebuilds an iterable state value item by item
        raise NotImplementedError("load_state_dict is disabled for this optimizer")
