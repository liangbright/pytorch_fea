# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 22:15:19 2021

@author: liang
"""
import numpy as np
from copy import deepcopy

class AutoStepper:
    def __init__(self, t_start, t_end, Δt_init, Δt_min, Δt_max, alpha, beta, max_retry, t_list=[], random_seed=None):
        self.t_start=t_start
        self.t_end=t_end
        self.Δt_init=Δt_init
        self.Δt_min=Δt_min
        self.Δt_max=Δt_max
        self.alpha=alpha
        self.beta=beta
        t_list=deepcopy(t_list)#must points in FEBio
        t_list=np.array(t_list, dtype=np.float64)
        self.t_list=np.sort(t_list)
        self.Δt=min(max(Δt_init, Δt_min), Δt_max)
        self.t=t_start
        if random_seed == "none" or random_seed == "None":
            random_seed=None
        self.random_seed=random_seed
        self.rng=np.random.RandomState(random_seed)
        self.max_retry=max_retry
        self.initialize_retry()
        if Δt_init <= 0:
            self.rand_Δt()

        if alpha <= 1:
            raise ValueError("alpha <= 1")
        if beta <= 0 or beta >= 1:
            raise ValueError("beta <= 0 or beta >= 1")

    def step(self):
        t=self.t+self.Δt
        if len(self.t_list) > 0:
            temp=np.where(self.t_list>self.t)[0]
            if len(temp)>0:
                id=temp[0]
                t_id=self.t_list[id]
                if t_id < t:
                    t=t_id
        t=min(max(t, self.t_start), self.t_end)
        self.t=t

    def initialize_retry(self, max_retry=None):
        if max_retry is not None:
            self.max_retry=max_retry
        if self.Δt > self.Δt_min:
            self.Δt_list=np.logspace(np.log10(0.99*self.Δt), np.log10(self.Δt_min), self.max_retry).tolist()
        else:
            self.Δt_list=np.logspace(np.log10(self.Δt_max), np.log10(self.Δt_min), self.max_retry).tolist()

    def goback(self, t, Δt=None):
        t=min(max(t, self.t_start), self.t_end)
        self.t=t
        if Δt is None:
            pass
        elif isinstance(Δt, str):
            if Δt == "retry":
                self.retry_Δt()
            elif Δt == "rand":
                self.rand_Δt()
            elif Δt == "increase":
                self.increase_Δt()
            elif Δt == "rand_increase":
                self.rand_increase_Δt()
            elif Δt == "decrease":
                self.decrease_Δt()
            elif Δt == "rand_decrease":
                self.rand_decrease_Δt()
            else:
                raise ValueError("invalid Δt:"+Δt)
        else:
            self.Δt=min(max(Δt, self.Δt_min), self.Δt_max)

    def retry_Δt(self):
        self.Δt=self.Δt_list.pop(0)

    def rand_Δt(self):
        a=np.log10(self.Δt_min)
        b=np.log10(self.Δt_max)
        c=a+(b-a)*self.rng.rand()
        self.Δt=10**c
        #self.Δt=self.Δt_min+(self.Δt_max-self.Δt_min)*self.rng.rand()

    def increase_Δt(self):
        Δt=self.alpha*self.Δt
        self.Δt=min(max(Δt, self.Δt_min), self.Δt_max)

    def rand_increase_Δt(self):
        a=self.Δt
        b=self.alpha*self.Δt
        c=a+(b-a)*self.rng.rand()
        self.Δt=min(max(c, self.Δt_min), self.Δt_max)

    def decrease_Δt(self):
        Δt=self.beta*self.Δt
        self.Δt=min(max(Δt, self.Δt_min), self.Δt_max)

    def rand_decrease_Δt(self):
        a=self.Δt
        b=self.beta*self.Δt
        c=b+(a-b)*self.rng.rand()
        self.Δt=min(max(c, self.Δt_min), self.Δt_max)

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    diff_list=[]
    max_retry=100
    for seed in range(0, 100):
        rng=np.random.RandomState(seed)
        number=rng.rand(max_retry)
        number=np.sort(number)
        number=np.abs(np.diff(number))
        #plt.plot(number, np.zeros(20), '.')
        diff_list.append(np.std(number))
    seed_best=np.argmin(diff_list)
    #%%
    seed_best=66
    rng=np.random.RandomState(seed_best)
    number=rng.rand(max_retry)
    number=np.sort(number)
    plt.plot(number, np.zeros(max_retry), '.')
    plt.title("seed_best"+str(seed_best))
    #%%
    a=np.log10(1e-5)
    b=np.log10(0.01)
    c=a+(b-a)*np.random.rand(1000)
    dt=10**c
    plt.hist(dt, bins=100)


