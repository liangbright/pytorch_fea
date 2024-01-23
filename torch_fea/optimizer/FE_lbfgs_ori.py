import torch
from torch import matmul

from functools import reduce
from torch.optim import Optimizer
from scipy.sparse import coo_matrix
from pypardiso import spsolve
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
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
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
        f_new, g_new = obj_func(x, t, d)
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
        f_new, g_new = obj_func(x, t, d)
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
    q=torch.tensor(q, dtype=r.dtype, device=r.device)
    return q

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
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 max_iter=1,
                 max_eval=None,
                 tolerance_grad=1e-20,
                 tolerance_change=1e-20,
                 history_size=100,
                 line_search_fn=None,
                 backtracking={"c":0.0001, "t_list": [1, 0.5, 0.1, 0.05, 0.01],
                               "t_default":0.5, "t_default_init":"auto", "verbose":False},
                 t_max=None):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
            backtracking=backtracking.copy(),
            t_max=t_max)
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
        for n in range(0, len(t_list)):
            t=t_list[n]
            self._add_grad(t, d)
            loss=float(closure())
            self._set_param(p_init)
            if n==0:
                t_best=t
                loss_best=loss
                if loss_best < loss0+c*t*gd and gd < 0:
                    #print("break, n", n)
                    break
            else:
                if loss < loss_best:
                    t_best=t
                    loss_best=loss
                    if loss_best < loss0+c*t*gd and gd < 0:
                        #print("break, n", n)
                        break
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
            if verbose == True:
                print("backtracking(bad): set t_best to t_default =", t_default)
        #------------------
        return t_best, loss_best, flag

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
        state['old_dirs'] = None
        state['old_stps'] = None
        state['ro'] = None
        state['H_diag'] = None
        state['H0'] = None
        state['prev_flat_grad'] = None
        state['prev_loss'] = None
        state['linesearch_flag']=0
        group = self.param_groups[0]
        state['al']=[None] * group['history_size']

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        backtracking= group['backtracking']
        t_maximum=group['t_max']

        state = self.state[self._params[0]]

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
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
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        H0 = state.get('H0')

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
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
                if H0 is not None:
                    d=linear_spsolve(H0, d)
                    t=1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                yy = y.dot(y)
                if ys > 1e-10 and yy > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / yy

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                #d = r = torch.mul(q, H_diag)

                if H0 is not None:
                    d=r=linear_spsolve(H0, q)
                else:
                    d=r=torch.mul(q, H_diag)

                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

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
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            #if gtd > -tolerance_change:
            #    print("gtd > -tolerance_change("+str(-tolerance_change)+"), break")
            #    break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn == "strong_wolfe":
                    x_init = self._clone_param()
                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)
                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                    t=float(t)
                    if t > 1:
                        print("~~~~~~~~~~~~strong_wolfe: t", t)
                        if t_maximum is not None:
                            if t > t_maximum:
                                t=min(t, t_maximum)
                                loss=self.evaluate_loss(closure, t, d)
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
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        loss = float(closure())
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
            #    break

            #if current_evals >= max_eval:
            #    break

            # optimal condition
            if opt_cond:
                print("optimal condition2, break the current iteration")
                break

            # lack of progress
            #if d.mul(t).abs().max() <= tolerance_change:
            #    break

            #if abs(loss - prev_loss) < tolerance_change:
            #    break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return opt_cond
