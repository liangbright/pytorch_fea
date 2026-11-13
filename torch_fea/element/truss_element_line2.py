# 2-node truss element. "truss2" (also "line2") in FEBio (FETrussElementTraits, FEElasticTrussDomain, FELinearTrussDomain)
# r[0] ~ r in FEBio: node 0 at r[0]=-1, node 1 at r[0]=1
# The truss carries one force, along its axis. A node has 3 displacements and no rotation.
#   stretch lam = l/L (deformed length / undeformed length), axis n = (x1 - x0)/l
#   FEBio takes the deformation gradient in the axes of the truss as F = diag(lam, lam^(-v), lam^(-v)),
#       with v = 0.5 unless set otherwise, so J = det(F) = lam^(1-2v)   (FEElasticTrussDomain::Update)
#   force along the axis: T = S11*a, with S11 the Cauchy stress along the axis and a = area*L*J/l the deformed
#       cross-section; with the Kirchhoff stress tau = J*S11 this is T = tau*area*L/l
#       FEBio's linear truss material has tau = E*log(lam)   (FELinearTrussMaterial)
#       with the 1PK stress P = tau/lam (force per undeformed area) this is T = P*area
#   material_dim, an input of the three force functions, is the dimension of the material that gives the stress:
#   material_dim=1 (the default), a 1D material law: the stress along the truss, shape (M,) or (M,1), and T as above.
#       This is FEBio's truss: FEBio makes a truss domain only for a 1D truss material (FELinearTrussDomain, FETrussMaterial)
#   material_dim=3, a 3D material: the stress of shape (M,1,3,3) for F=cal_F_tensor(x, X, v), in the axes of the truss.
#       The cross-section changes with lam, so S22 and S33 also do work, and T = area*dW/dlam = (S11-v*(S22+S33))*a with
#       the Cauchy stress S (the same with tau), and P11-v*lam^(-1-v)*(P22+P33) in place of P with the 1PK stress;
#       then the force is dE/dx of cal_strain_energy, the energy of W(F).
#       The truss is in uniaxial tension (S22 = S33 = 0) only when v is the lateral contraction of the material: exactly
#       for an incompressible isotropic material (NeoHookean, ...) and v = 0.5. A compressible isotropic material contracts
#       less (about its Poisson's ratio (3K-2mu)/(2(3K+mu))), so v = 0.5 gives too large a force and its 1D law is better.
#       A fibre model of a tissue wall (GOH, UFD) is not for a truss: a truss is thin and carries one axial force, so the
#       fibre directions across its section mean nothing. A fibrous truss (a collagen bundle) takes a 1D law of lam.
#       FEBio takes T = S11*a (FEElasticTrussDomain::ElementInternalForces), the same when S22 = S33 = 0. For an uncoupled
#       material FEBio replaces the pressure so that S22 = 0 (FEElasticTrussDomain::Update), which gives the T of this
#       file when v = 0.5 and S22 = S33; for other materials FEBio's T leaves out the work of S22 and S33.
#       FEBio uses that domain only when the input file asks for it (type="elastic-truss")
#   nodal force: -T*n on node 0 and T*n on node 1, the internal force dE/dx (E: strain energy), as for the solid elements.
#       The truss pulls each node with the opposite force: a stretched truss (T > 0) pulls node 0 towards node 1.
#       FEBio's ElementInternalForces has that sign, fe = T*n on node 0 and -T*n on node 1.
# integration rule: 2 Gauss points at +-1/sqrt(3), weight 1 (FETrussElementTraits). Stretch and stress are the same
#   at every point of the element, so the rule is only used for a load along the element (body force).
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FEElementTraits.cpp
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FEBioMech/FEElasticTrussDomain.cpp
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FEBioMech/FELinearTrussDomain.cpp
#%%
import torch
from math import sqrt
#%% integration point location
def get_integration_point_2i(dtype, device):
    a=1/sqrt(3)
    r=[[-a], [a]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (2, 1)
    return r
#%% gaussian integration weight
def get_integration_weight_2i(dtype, device):
    w=[1.0, 1.0]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (2,)
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,1); a list of the points when dtype or device is None
    if n_points == 2:
        return get_integration_point_2i(dtype, device)
    else:
        raise ValueError("n_points must be 2")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 2:
        return get_integration_weight_2i(dtype, device)
    else:
        raise ValueError("n_points must be 2")
#%%
def get_shape_integration_weight_2i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (2,2); a list of its numbers when dtype or device is None
    r=get_integration_point_2i(None, None)
    w=get_integration_weight_2i(None, None)
    weight=[sf0(r[0])*w[0], sf0(r[1])*w[1], sf1(r[0])*w[0], sf1(r[1])*w[1]]
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(2,2)
    return weight
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (2,n_points); a list of its numbers when dtype or device is None
    if n_points == 2:
        return get_shape_integration_weight_2i(dtype, device)
    else:
        raise ValueError("n_points must be 2")
#%% shape function 0, node 0 is at r[0]=-1
def sf0(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 0.5*(1-r[0])
#%%
def cal_d_sf0_dr(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: [dr0], the derivative with respect to r[0], each a number or a tensor with the shape of r[0]
    dr0=-0.5+0*r[0]
    dr=[dr0]
    return dr
#%% shape function 1, node 1 is at r[0]=1
def sf1(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 0.5*(1+r[0])
#%%
def cal_d_sf1_dr(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: [dr0], the derivative with respect to r[0], each a number or a tensor with the shape of r[0]
    dr0=0.5+0*r[0]
    dr=[dr0]
    return dr
#%%
def cal_d_sf_dr(r):
    #r.shape is (K,1), K is the number of integration points
    K=r.shape[0]
    r=r.permute(1,0)#shape is (1,K)
    r=r.view(1,K,1,1)
    d_sf0_dr=torch.cat(cal_d_sf0_dr(r), dim=1)#shape is (K,1,1)
    d_sf1_dr=torch.cat(cal_d_sf1_dr(r), dim=1)#shape is (K,1,1)
    d_sf_dr=torch.cat([d_sf0_dr, d_sf1_dr], dim=2)
    d_sf_dr=d_sf_dr.view(1,K,1,2)#2: sf0 to sf1
    return d_sf_dr #(1,K,1,2)
#%%
def interpolate(r, h):
    #r.shape (K,1), K is the number of integration points
    #h.shape (M,2,3), M elements, 1 elemnet has 2 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_i{sf_i(r)*h_i, i=0 to 1}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (1,K)
    r=r.view(1,K,1)
    w=torch.cat([sf0(r), sf1(r)], dim=1)
    w=w.view(1,K,2,1)
    M=h.shape[0]
    h=h.view(M,1,2,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_length(h):
    #h.shape (M,2,3), h is x or X
    length=torch.norm(h[:,1]-h[:,0], p=2, dim=-1)
    return length #(M,)
#%% unit vector from node 0 to node 1
def cal_axis(x):
    #x.shape (M,2,3)
    d=x[:,1]-x[:,0]
    axis=d/torch.norm(d, p=2, dim=-1, keepdim=True)
    return axis #(M,3)
#%%
def cal_stretch(x, X):
    #x.shape (M,2,3), X.shape (M,2,3)
    stretch=cal_length(x)/cal_length(X)
    return stretch #(M,)
#%% deformation gradient in the axes of the truss, the first axis is along the truss
def cal_F_tensor(x, X, v=0.5):
    #x.shape (M,2,3), X.shape (M,2,3)
    #v: the cross-section changes by lam^(-v) in each direction; v=0.5 keeps the volume (FEBio's default)
    #F=diag(lam, lam^(-v), lam^(-v)), one integration point
    lam=cal_stretch(x, X) #(M,)
    side=lam**(-v)
    F=torch.diag_embed(torch.stack([lam, side, side], dim=-1))
    F=F.view(-1,1,3,3)
    return F #(M,1,3,3)
#%% the stress along the truss, from the input of the three force functions below
def cal_axial_stress(S, M, material_dim=1, transverse=0):
    #material_dim=1: S.shape (M,) or (M,1), the stress along the truss, returned as it is (transverse is not used)
    #material_dim=3: S.shape (M,1,3,3), the stress of a 3D material for F=cal_F_tensor(x, X, v), in the axes of the truss;
    #    return: S11-transverse*(S22+S33), the stress that does the work of S as lam changes (see the top of this file)
    #    transverse: a number or a tensor of shape (M,); v for the Cauchy and the Kirchhoff stress, v*lam^(-1-v) for the 1PK stress
    if material_dim == 1:
        if S.numel() != M:
            raise ValueError("material_dim=1: the stress must have the shape (M,) or (M,1), the stress along the truss; "
                             +"S.shape is "+str(tuple(S.shape))+"; set material_dim=3 for the stress (M,1,3,3) of a 3D material")
        return S.reshape(M) #(M,)
    elif material_dim == 3:
        if tuple(S.shape) != (M,1,3,3):
            raise ValueError("material_dim=3: the stress must have the shape (M,1,3,3); S.shape is "+str(tuple(S.shape)))
        return S[:,0,0,0]-transverse*(S[:,0,1,1]+S[:,0,2,2]) #(M,)
    else:
        raise ValueError("material_dim must be 1 or 3")
#%%
def cal_nodal_force_from_cauchy_stress(S, x, X, area, material_dim=1, v=0.5):
    #S: Cauchy stress, shape (M,) or (M,1) for material_dim=1, (M,1,3,3) for material_dim=3, see cal_axial_stress
    #x.shape (M,2,3), X.shape (M,2,3)
    #area: cross-section area of the undeformed truss, a number or a tensor of shape (M,) or (M,1)
    #material_dim: 1, a 1D material law (the default), or 3, a 3D material; see the top of this file
    #v: the cross-section changes by lam^(-v) in each direction, J = lam^(1-2v); with material_dim=3, the v of cal_F_tensor
    #force along the axis T = S11*area*L*J/l (FEElasticTrussDomain::ElementInternalForces in FEBio),
    #with S11-v*(S22+S33) in place of S11 for material_dim=3
    M=x.shape[0]
    J=cal_stretch(x, X)**(1-2*v) #(M,)
    force=cal_nodal_force_from_kirchhoff_stress(J*cal_axial_stress(S, M, material_dim, v), x, X, area)
    return force #(M,2,3)
#%%
def cal_nodal_force_from_kirchhoff_stress(S, x, X, area, material_dim=1, v=0.5):
    #S: Kirchhoff stress, J * Cauchy stress, shape (M,) or (M,1) for material_dim=1, (M,1,3,3) for material_dim=3,
    #   see cal_axial_stress
    #x.shape (M,2,3), X.shape (M,2,3)
    #area: cross-section area of the undeformed truss, a number or a tensor of shape (M,) or (M,1)
    #material_dim: 1, a 1D material law (the default), or 3, a 3D material; see the top of this file
    #v: used only for material_dim=3, the v of cal_F_tensor
    #force along the axis T = S11*area*L/l (FELinearTrussDomain::ElementInternalForces in FEBio),
    #with S11-v*(S22+S33) in place of S11 for material_dim=3
    M=x.shape[0]
    if torch.is_tensor(area):
        area=area.view(M)
    T=cal_axial_stress(S, M, material_dim, v)*area*cal_length(X)/cal_length(x) #(M,)
    Tn=T.view(M,1)*cal_axis(x)
    force=torch.stack([-Tn, Tn], dim=1)
    return force #(M,2,3)
#%%
def cal_nodal_force_from_1pk_stress(S, x, X, area, material_dim=1, v=0.5):
    #S: 1PK stress, force per undeformed area, shape (M,) or (M,1) for material_dim=1, (M,1,3,3) for material_dim=3,
    #   see cal_axial_stress
    #x.shape (M,2,3); X.shape (M,2,3), used only for material_dim=3
    #area: cross-section area of the undeformed truss, a number or a tensor of shape (M,) or (M,1)
    #material_dim: 1, a 1D material law (the default), or 3, a 3D material; see the top of this file
    #v: used only for material_dim=3, the v of cal_F_tensor
    #force along the axis T = S11*area, with S11-v*lam^(-1-v)*(S22+S33) in place of S11 for material_dim=3.
    #S11 = tau/lam = J*(Cauchy stress)/lam and S22*lam^(-v) = tau22, so this is the same force as in
    #cal_nodal_force_from_kirchhoff_stress and cal_nodal_force_from_cauchy_stress
    M=x.shape[0]
    if torch.is_tensor(area):
        area=area.view(M)
    transverse=0
    if material_dim == 3:
        transverse=v*cal_stretch(x, X)**(-1-v)
    Tn=(cal_axial_stress(S, M, material_dim, transverse)*area).view(M,1)*cal_axis(x)
    force=torch.stack([-Tn, Tn], dim=1)
    return force #(M,2,3)
#%%
def cal_nodal_force_from_body_force(body_force, density, r, X, area):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed truss, a number or a tensor of shape (M,1)
    #r.shape (K,1), from get_integration_point(2, dtype, device); X.shape (M,2,3)
    #area: a number or a tensor of shape (M,) or (M,1)
    #force_i=integration(density*area*body_force*Ni along the truss), i=0 to 1
    M=X.shape[0]
    K=r.shape[0]
    if torch.is_tensor(area):
        area=area.view(M,1,1)
    if torch.is_tensor(density):
        density=density.view(M,1,1)
    q=density*area*(0.5*cal_length(X)).view(M,1,1)*body_force #(M,1,3) or (M,K,3)
    q=q.expand(M,K,3).reshape(M,1,K,3)
    weight=get_shape_integration_weight(K, X.dtype, X.device)
    force=(weight.view(1,2,K,1)*q).sum(dim=2)
    return force #(M,2,3)
#%% the strain energy of each element
def cal_strain_energy(W, X, area):
    #W.shape (M,) or (M,1): strain energy per undeformed volume
    #X.shape (M,2,3); area: a number or a tensor of shape (M,) or (M,1)
    #energy=W*area*L
    M=X.shape[0]
    if torch.is_tensor(area):
        area=area.view(M)
    energy=W.view(M)*area*cal_length(X)
    return energy #(M,)
