# optimization target: a loss, e.g. the energy (forward, GPA) or a misfit (inverse_mat)
#   the line search compares the loss returned by closure at trial steps;
#   it is not for the residual: use FE_lbfgs_residual
import torch
from torch import matmul

from functools import reduce
from torch.optim import Optimizer
from scipy.sparse import coo_matrix
from pypardiso import spsolve
#the GPU solver (cuDSS), with the API of pypardiso; nvmath-python is imported on its first use
from torch_fea.optimizer.cuDSS_linear_sparse_solver import spsolve as cudss_spsolve
import numpy as np
import time
import scipy
import copy

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    #a NaN or inf value (a trial step where the loss is NaN or inf, see _strong_wolfe) gives no interpolation: bisect
    if not all(np.isfinite(float(v)) for v in (f1, g1, f2, g2)):
        return (xmin_bound + xmax_bound) / 2.

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    #a loss that is NaN or inf counts as +inf: the step is too long, so the search steps back.
    #In torch.optim.LBFGS a NaN passes none of the tests, so the search goes on extending the step into the NaN region
    def obj_func_finite(x, t, d):
        f_new, g_new = obj_func(x, t, d)
        if not np.isfinite(f_new):
            f_new = float('inf')
        return f_new, g_new
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func_finite(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func_finite(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func_finite(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals

def linear_spsolve(H, r, verbose=False):
    #H on the GPU (torch sparse CSR tensor, AortaFEModel on the GPU): solved on the GPU (cuDSS)
    #scipy.sparse.csr_matrix: solved on the CPU (pypardiso)
    if torch.is_tensor(H) and H.is_cuda:
        return linear_spsolve_cuda(H, r, verbose)
    return linear_spsolve_cpu(H, r, verbose)

def linear_spsolve_cuda(H, r, verbose=False):
    #q=solve(H, r) with cuDSS (torch_fea/optimizer/cuDSS_linear_sparse_solver.py); like pypardiso, the solver keeps
    #the factorization: the same H again is only solved, a new H with the same sparsity pattern is only factorized
    rr=r.detach().reshape(-1).to(H.dtype)
    t0=time.time()
    q=cudss_spsolve(H, rr)
    if verbose == True:
        torch.cuda.synchronize(H.device)
        t1=time.time()
        print("done: linear_spsolve_cuda", t1-t0)
    q=q.to(r.dtype).reshape(-1) #shape (n,): spsolve squeezes 1 unknown to shape ()
    return q

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
        #print("r", r)
        #print("q", q)
    q=torch.tensor(q, dtype=r.dtype, device=r.device).reshape(-1) #shape (n,): spsolve squeezes 1 unknown to shape ()
    return q

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
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9). It also ends the zoom of the
            strong Wolfe line search, at bracket width times max abs(d) below it; a
            very small value (1e-20) makes the zoom shrink the bracket that far: a few
            more evaluations per step once the loss is at rounding level (max_ls 25).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe', 'backtracking' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None,
                 strong_wolfe={"t_max":None, "tolerance_change":1e-9, "verbose":False},
                 backtracking={"c":0.5, "t_list": [1, 0.5, 0.1, 0.05, 0.01],
                               "t_default":0.5, "t_default_init":"auto", "verbose":False}
                 ):
        defaults = dict(
            lr=lr,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
            backtracking=backtracking.copy(),
            strong_wolfe=strong_wolfe.copy())
        super(LBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
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

    def evaluate_loss(self, closure, t, d):
        closure = torch.no_grad()(closure)
        p_init = self._clone_param()
        self._add_grad(t, d)
        loss=float(closure())
        self._set_param(p_init)
        return loss

    def backtracking(self, closure, loss0, gd, d, t_list, t_default, c, verbose):
        closure = torch.no_grad()(closure)
        gd=float(gd)
        if gd > 0:
            gd=0 #just in case something is wrong
        p_init = self._clone_param()
        #-------------------
        #a loss that is NaN or inf is never taken as the best; with finite losses the choice is as before
        t_best=t_list[0]
        loss_best=np.inf
        evaluated=[] #(t, loss) of every evaluation
        for n in range(0, len(t_list)):
            t=t_list[n]
            self._add_grad(t, d)
            loss=float(closure())
            self._set_param(p_init)
            evaluated.append((t, loss))
            if np.isfinite(loss) and loss < loss_best:
                t_best=t
                loss_best=loss
                if loss_best < loss0+c*t*gd and gd < 0:
                    #print("break, n", n)
                    break
        if not np.isfinite(loss_best):
            loss_best=np.nan #every trial step gave NaN or inf
        #-------------------
        if verbose == True:
            if loss_best < loss0:
                print("backtracking(good): t_best", t_best, "loss_best", loss_best, '< loss0', loss0, 'gd', gd)
            elif loss_best == loss0:
                print("backtracking(ok): t_best", t_best, "loss_best", loss_best, '= loss0', loss0, 'gd', gd)
            else: #loss_best > loss0 or loss_best is nan
                print("backtracking(bad): t_best", t_best, "loss_best ", loss_best, "> loss0 ", loss0, 'gd', gd)
        #-------------------
        if loss_best < loss0:
            flag=1
        elif loss_best == loss0:
            flag=0
        else: #loss_best > loss0 or loss_best is nan
            flag=-1
            #no optimal t is found, make the best guess
            #small t is bad for Rmax
            t_best=t_default
            if any(tt == t_default and not np.isfinite(ll) for (tt, ll) in evaluated):
                #t_default itself gave NaN or inf: take the largest evaluated t with a finite loss
                finite=[tt for (tt, ll) in evaluated if np.isfinite(ll)]
                if len(finite) > 0:
                    t_best=max(finite)
            if verbose == True:
                print("backtracking(bad): set t_best to", t_best, "(t_default =", t_default, ")")
        #------------------
        return t_best, loss_best, flag

    def set_strong_wolfe(self, t_max=None, verbose=None):
        strong_wolfe = self.param_groups[0]['strong_wolfe']
        strong_wolfe['t_max']=t_max
        if verbose is not None:
            strong_wolfe['verbose']=verbose

    def set_backtracking(self, t_list=None, t_default=None, t_default_init=None, c=None, verbose=None):
        backtracking = self.param_groups[0]['backtracking']
        if t_list is not None:
            backtracking["t_list"]=t_list.copy()
        if t_default is not None:
            backtracking["t_default"]=t_default
        if t_default_init is not None:
            backtracking["t_default_init"]=t_default_init
        if c is not None:
            backtracking["c"]=c
        if verbose is not None:
            backtracking['verbose']=verbose

    def get_linesearch_flag(self):
        state = self.state[self._params[0]]
        return state['linesearch_flag']

    def get_linesearch_t(self):
        state = self.state[self._params[0]]
        return state['t']

    def set_linesearch_fn(self, line_search_fn):
        group = self.param_groups[0]
        group['line_search_fn']=line_search_fn

    def get_linesearch_fn(self):
        group = self.param_groups[0]
        line_search_fn = group['line_search_fn']
        return line_search_fn

    def set_lr(self, lr=None):
        group = self.param_groups[0]
        if lr is not None:
            group['lr']=lr

    def set_tolerance_change(self, tolerance_change):
        group = self.param_groups[0]
        group['tolerance_change']=tolerance_change

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
        state['d'] = None
        state['t'] = None
        state['s'] = None
        state['y'] = None
        state['ρ'] = None
        state['H_diag'] = None
        state['H0'] = None
        state['invH0_diag'] = None
        state['prev_flat_grad'] = None
        state['prev_loss'] = None
        state['linesearch_flag']=0

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        #closure_grad: closure is called with grad enabled
        #do loss.backward() inside closure only if loss.requires_grad: backtracking calls the closure with grad disabled
        closure_grad = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        tolerance_grad = group['tolerance_grad']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        backtracking = group['backtracking']
        strong_wolfe = group['strong_wolfe']
        if line_search_fn is not None:
            if line_search_fn not in ("strong_wolfe", "backtracking"):
                raise RuntimeError(line_search_fn+" not supported")
        
        state = self.state[self._params[0]]

        # evaluate initial f(x) and df/dx
        orig_loss = closure_grad()
        loss = float(orig_loss)
        if np.isnan(loss):
            print("abort: loss is nan, optimizer will return 'nan'")
            return "nan"
        if np.isinf(loss):
            print("abort: loss is inf, optimizer will return 'inf'")
            return "inf"

        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = bool(flat_grad.abs().max() <= tolerance_grad)

        # optimal condition
        if opt_cond:
            print("optimal condition1, break the current iteration")
            return opt_cond

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        s = state.get('s')
        y = state.get('y')
        ρ = state.get('ρ')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        H0 = state.get('H0')
        invH0_diag = state.get('invH0_diag')

        # keep track of nb of iterations
        state['n_iter'] += 1

        ############################################################
        # compute gradient descent direction
        ############################################################
        if state['n_iter'] == 1:
            d = flat_grad.neg()
            s = []
            y = []
            ρ = []
            H_diag = 1
            if H0 is not None:
                d=linear_spsolve(H0, d)
                t=1
            else:
                if invH0_diag is not None:
                    d*=invH0_diag
                    t=1
        else:
            # do lbfgs update (update memory)
            y_new = flat_grad.sub(prev_flat_grad)
            s_new = d.mul(t)
            ys = y_new.dot(s_new)  # y*s
            yy = y_new.dot(y_new)
            ss = s_new.dot(s_new)
            #curvature test relative to abs(s)*abs(y), so that it does not depend on the units of the loss and the parameters
            #(torch.optim.LBFGS tests ys > 1e-10; this file tested yy > 1e-10 as well)
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
                ρ.append(1. / ys)

                # update scale of initial Hessian approximation
                H_diag = ys / yy

            # compute the approximate (L-BFGS) inverse Hessian multiplied by the gradient
            # also with an empty memory, so that H0 (or the diagonal scaling) is applied to -g
            # the scaling is the current H_diag unless invH0_diag is set by set_invH0_diag
            d = flat_grad.neg()
            d = update_invHd(s, y, ρ, d, invH0_diag if invH0_diag is not None else H_diag, H0)

        if prev_flat_grad is None:
            prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
        else:
            prev_flat_grad.copy_(flat_grad)
        prev_loss = loss

        ############################################################
        # compute step length
        ############################################################
        # reset initial guess for step size
        if state['n_iter'] == 1:
            #with H0 or invH0_diag the first step is t=1 (set above)
            if H0 is None and invH0_diag is None:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
        else:
            t = lr

        # directional derivative
        gtd = flat_grad.dot(d)  # g * d
        if gtd > 0:
            #d is not a descent direction (e.g. the stiffness H0 is indefinite): search along -d
            d = -d
            gtd = -gtd

        #the slope test below is off: it stalls the optimization. When it fires, no step is taken and nothing changes,
        #so the next step() computes the same d and the test fires again, every time; tolerance_grad is never reached.
        #The test is absolute, and g.d is small near convergence just because g is small
        # directional derivative is below tolerance
        #if gtd > -tolerance_change:
        #    if verbose == True:
        #        print("gtd > -tolerance_change("+str(-tolerance_change)+"), break")
        #    #no step is taken: t=0 is saved (state['t'] below), so the next step() gets s=t*d=0 and the curvature
        #    #test rejects that pair; torch saves t=lr here, the pair of a step that was never taken
        #    t=0.
        #    state['linesearch_flag']=0
        #    break

        # optional line search: user function
        ls_func_evals = 0
        if line_search_fn is not None:
            # perform line search, using user function
            if line_search_fn == "strong_wolfe":
                x_init = self._clone_param()
                def obj_func(x, t, d):
                    return self._directional_evaluate(closure_grad, x, t, d)
                loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                    obj_func, x_init, t, d, loss, flat_grad, gtd, tolerance_change=strong_wolfe['tolerance_change'])
                t=float(t)
                if t > 1:
                    if strong_wolfe['verbose'] == True:
                        print("~~~~~~~~~~~~strong_wolfe: t", t)
                t_maximum=strong_wolfe['t_max']
                if t_maximum is not None and t > t_maximum:
                    #every t above t_max is capped, also when t_max < 1;
                    #the loss and the gradient at the capped t, for opt_cond and the next L-BFGS pair
                    t=float(t_maximum)
                    loss, flat_grad = self._directional_evaluate(closure_grad, x_init, t, d)
                    ls_func_evals += 1
                    if strong_wolfe['verbose'] == True:
                        print("~~~~~~~~~~~~reduce t to", t_maximum)
                self._add_grad(t, d)
                opt_cond = bool(flat_grad.abs().max() <= tolerance_grad)
                #---------------------
                if loss > prev_loss:
                    flag=-1
                elif loss < prev_loss:
                    flag=1
                else:
                    flag=0
                state['linesearch_flag']=flag
                state['t']=float(t)
            elif line_search_fn == "backtracking":
                t_default=backtracking["t_default"]
                if state['n_iter'] == 1:
                    t_default_init=backtracking["t_default_init"]
                    if isinstance(t_default_init, str):
                        if t_default_init == "auto":
                            t_default_init=float(min(1., 1. / flat_grad.abs().sum()))*lr
                            t_default_init=min(t_default_init, min(backtracking["t_list"]))
                            t_default=min(t_default_init, t_default)
                        else:
                            raise ValueError("unknown t_default_init:"+t_default_init)
                    else: # t_default_init should be float or int
                        t_default=float(t_default_init)
                t, loss, flag = self.backtracking(closure, loss, gtd, d,
                                                  t_list=backtracking["t_list"],
                                                  t_default=t_default,
                                                  c=backtracking["c"],
                                                  verbose=backtracking['verbose'])
                #------------------------
                #do not need this for backtracking
                #if t_maximum is not None:
                #    t=min(t, t_maximum)
                #    loss=self.evaluate_loss(closure, t, d)
                #----------------------------
                state['linesearch_flag']=flag
                state['t']=t
                self._add_grad(t, d)
                opt_cond = bool(flat_grad.abs().max() <= tolerance_grad)
            else:
                raise RuntimeError(line_search_fn+" not supported")
        else:
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t=lr*float(min(1., 1. / flat_grad.abs().sum()))
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

        state['d'] = d
        state['t'] = t
        state['s'] = s
        state['y'] = y
        state['ρ'] = ρ
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return opt_cond

    def load_state_dict(self, state_dict):
        #disabled: this optimizer is not saved and loaded (FEBio does not do that either), and torch's
        #load_state_dict cannot load a scipy H0 (set_H0) back: it rebuilds an iterable state value item by item
        raise NotImplementedError("load_state_dict is disabled for this optimizer")
