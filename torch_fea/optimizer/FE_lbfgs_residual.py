# -*- coding: utf-8 -*-
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
    q=torch.tensor(q, dtype=r.dtype, device=r.device)
    return q

def linear_spsolve_cuda(H, r, verbose=False):
    #assume H is scipy.sparse.csr_matrix
    #to use torch.sparse.spsolve: compile PyTorch with cuDSS
    H=H.tocoo()
    H=torch.sparse_csr_tensor(H.row, H.col, H.data, size=H.shape)
    H=H.to(r.device)
    q=torch.sparse.spsolve(H, r)
    return q

def linear_spsolve(H, r, use_cuda=False, verbose=False):
    if use_cuda == True:
        return linear_spsolve_cuda(H, r, verbose)
    else:
        return linear_spsolve_cpu(H, r, verbose)

def update_invHd(s, y, ρ, d, invH0_diag, H0):
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
    #print("done: update_invHd", t1-t0)
    return r

class LBFGS(Optimizer):

    def __init__(self,
                 params,
                 lr=1,
                 max_iter=20,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 reset_state_per_step=False,
                 line_search_fn=None,
                 backtracking={"c":0.5, "t_list": [1, 0.5, 0.1, 0.05, 0.01],
                               "t_default":0.5, "t_default_init":'auto', "max_iter":10, "verbose":False}
                 ):
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
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
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
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
        def call_R(t):
            self._add_grad(t, d)
            R = closure(output='R')
            self._set_param(p_init)
            R=R.view(-1)
            #print("shape", d.shape, R.shape)
            R=float((d*R).sum())
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
            n=np.argmin(np.abs(np.array(R_list)))
            if n ==0 or n == len(R_list)-1:
                t1=t2=t=t_list[n]
                R1=R2=Rt=R_list[n]
                if verbose == True:
                    print("no zero-crossing, no V shape, R_list", R_list)
            else:
                t1=t_list[n-1]; R1=R_list[n-1]
                t=t_list[n-1];  Rt=R_list[n-1]
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
                    t3=(t1+t2)/2
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
            if verbose == True:
                print("backtracking(bad): t", t, "Rt", Rt, "set t to t_default =", t_default)
            flag=-1
            #no optimal t is found, make the best guess
            #small t is bad for Rmax
            t=t_default
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

    def set_tolerance_change(self, tolerance_change):
        group = self.param_groups[0]
        group['tolerance_change']=tolerance_change

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        #closure_grad: closure is called with grad enabled
        #do loss.backward() inside closure
        closure_grad = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
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

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
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
                if ys > 1e-10 and yy > 1e-10:
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
                d=-g
                if len(s) > 0:
                    if invH0_diag is None:
                        invH0_diag=H_diag
                    d=update_invHd(s, y, ρ, d, invH0_diag, H0)
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
            # directional derivative is below tolerance
            #if gtd > -tolerance_change:
            #    print("gtd > -tolerance_change("+str(-tolerance_change)+"), break")
            #    break

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
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    loss = float(closure_grad())
                    flat_grad = self._gather_flat_grad()
                    opt_cond = bool(flat_grad.abs().max() <= tolerance_grad)
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            #if n_iter == max_iter:
            #    print("n_iter == max_iter")
            #    break

            #if current_evals >= max_eval:
            #    print("current_evals >= max_eval")
            #    break

            # optimal condition
            if opt_cond:
                print("optimal condition2, break the current iteration")
                break

            # lack of progress
            #if d.mul(t).abs().max() <= tolerance_change:
            #    break

            #if abs(loss - prev_loss) < tolerance_change:
            #    print("abs(loss - prev_loss) < tolerance_change")
            #    break

        state['t'] = t
        state['d'] = d
        state['s'] = s
        state['y'] = y
        state['ρ'] = ρ
        state['g_prev'] = g_prev
        state['ys_counter']=ys_counter
        state['H_diag']=H_diag
        return opt_cond
