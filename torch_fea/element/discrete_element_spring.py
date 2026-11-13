# 2-node spring. A discrete element of FEBio (FEDiscreteElementTraits, FEDiscreteElasticDomain, FESpringMaterial)
# The spring connects two nodes and carries one force, along the line between them. It has no shape functions and
#   one integration point. A node has 3 displacements and no rotation.
#   elongation dl = l - L (deformed length - undeformed length), axis e = (x1 - x0)/l
#   spring force F = f(dl), a function that belongs to the spring material. FEBio has
#       linear spring:               f = E*dl
#       tension-only linear spring:  f = E*(dl - slack_length) when dl >= slack_length, else 0
#       experimental spring:         f = sM*(1 - exp(-E*dl/sM)) when dl >= 0, else -sm*(1 - exp(E*dl/sm))
#   nodal force: -F*e on node 0 and F*e on node 1 (a stretched spring pulls its nodes together)
#   a spring has no mass and no volume, so there is no cal_nodal_force_from_body_force (FEBio has none for it either)
#   the strain energy of a spring is the integral of f from 0 to dl, a number that comes from the spring material;
#       there is nothing to integrate over the element, so this file has no cal_strain_energy
#   stiffness of the element in FEBio: A = (E - F/l)*e(x)e + F/l*I between the two nodes, with E = df/d(dl);
#       it is the derivative of the nodal force here
#   https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEDiscreteElasticDomain.cpp
#   https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESpringMaterial.cpp
#%%
import torch
#%%
def cal_length(h):
    #h.shape (M,2,3), h is x or X
    length=torch.norm(h[:,1]-h[:,0], p=2, dim=-1)
    return length #(M,)
#%% unit vector from node 0 to node 1
def cal_axis(x):
    #x.shape (M,2,3)
    #a spring of zero length has no axis: the axis is then zero, and so is its nodal force (as in FEBio)
    d=x[:,1]-x[:,0]
    length=torch.norm(d, p=2, dim=-1, keepdim=True)
    zero=length == 0
    axis=torch.where(zero, torch.zeros_like(d), d/torch.where(zero, torch.ones_like(length), length))
    return axis #(M,3)
#%%
def cal_elongation(x, X):
    #x.shape (M,2,3), X.shape (M,2,3)
    elongation=cal_length(x)-cal_length(X)
    return elongation #(M,)
#%%
def cal_nodal_force_from_spring_force(spring_force, x):
    #spring_force.shape (M,) or (M,1): the force f(dl) of each spring, positive when the spring is stretched
    #x.shape (M,2,3)
    M=x.shape[0]
    Fe=spring_force.view(M,1)*cal_axis(x)
    force=torch.stack([-Fe, Fe], dim=1)
    return force #(M,2,3)
