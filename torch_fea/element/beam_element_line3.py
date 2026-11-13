# 3-node beam element, line3 in FEBio; the middle node is the last one. Geometrically exact beam (Simo-Reissner) of FEBio (FEElasticBeamDomain, FEBeam3_, FEBeam3G2)
# r[0] ~ r in FEBio: node 0 at r[0]=-1, node 1 at r[0]=1, node 2 at r[0]=0
# A beam node has a position (3 numbers) and a rotation (3 numbers). A rotation is not added like a displacement:
#   the element keeps a rotation matrix R and a curvature vector k at every integration point, and every change of
#   the nodal rotations updates them (cal_rotation_and_curvature_update, IncrementalUpdate in FEBio).
#   At the start R is the identity and k is zero.
# FEBio takes the element as straight in the undeformed state: dS/dr = L0/2, L0 = distance between node 0 and node 1
# initial frame Q=[E1, E2, E3] (cal_initial_frame): E3 along the element from node 0 to node 1, E1 and E2 across it
# strain at an integration point (cal_strain), vectors in the global axes:
#   Gamma = R^T * dx/dS - E3    stretch along E3, shear along E1 and E2
#   Kappa = R^T * k             bending about E1 and E2, twist about E3
# force t and moment m in the cross-section, in the global axes (cal_force_and_moment_linear_elastic,
#   FEElasticBeamMaterial in FEBio): t = R*Q*[G*A1*g1, G*A2*g2, E*A*g3], m = R*Q*[E*I1*k1, E*I2*k2, G*(I1+I2)*k3],
#   with g = Q^T*Gamma and k = Q^T*Kappa
# nodal force and nodal moment (cal_nodal_force_from_force_and_moment):
#   force_i  = integration(dNi/dS * t)
#   moment_i = integration(dNi/dS * m - Ni * cross(dx/dS, t))
# body force (cal_nodal_force_from_body_force): force_i = integration(density*area*body_force*Ni), no nodal moment
# integration rules (the first one is what FEBio uses for this element):
#   2 points: 2 Gauss points at +-1/sqrt(3), weight 1 (FEBeam3G2)
# This file was written by element_update_tools/generate_structural_elements.py
#   https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEElementTraits.cpp
#   https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticBeamDomain.cpp
#   https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticBeamMaterial.cpp
#%%
import torch
from torch import cross, matmul
from math import sqrt, pi
#%% integration point location
def get_integration_point_2i(dtype, device):
    #2 Gauss points at +-1/sqrt(3), weight 1
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
def cal_shape_integration_weight(r, w, dtype, device):
    #r: list of the K integration points, w: list of the K integration weights
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (3,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2]
    weight=[]
    for i in range(0, 3):
        for j in range(0, len(w)):
            weight.append(sf[i](r[j])*w[j])
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(3,len(w))
    return weight
#%%
def get_shape_integration_weight_2i(dtype, device):
    #return: weight, weight.shape (3,2); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_2i(None, None), get_integration_weight_2i(None, None), dtype, device)
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (3,n_points); a list of its numbers when dtype or device is None
    if n_points == 2:
        return get_shape_integration_weight_2i(dtype, device)
    else:
        raise ValueError("n_points must be 2")
#%% shape function 0, node 0 is at r[0]=-1
def sf0(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*(r[0] - 1)/2
#%%
def cal_d_sf0_dr(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: [dr0], the derivative with respect to r[0], each a number or a tensor with the shape of r[0]
    dr0=(2*r[0] - 1)/2
    dr=[dr0]
    return dr
#%% shape function 1, node 1 is at r[0]=1
def sf1(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*(r[0] + 1)/2
#%%
def cal_d_sf1_dr(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: [dr0], the derivative with respect to r[0], each a number or a tensor with the shape of r[0]
    dr0=(2*r[0] + 1)/2
    dr=[dr0]
    return dr
#%% shape function 2, node 2 is at r[0]=0
def sf2(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] - 1)*(r[0] + 1)
#%%
def cal_d_sf2_dr(r):
    #r.shape (1,...): r[0] is a number or a tensor
    #return: [dr0], the derivative with respect to r[0], each a number or a tensor with the shape of r[0]
    dr0=-2*r[0]
    dr=[dr0]
    return dr
#%%
def cal_sf(r):
    #r.shape is (K,1), K is the number of integration points
    K=r.shape[0]
    r=r.permute(1,0)#shape is (1,K)
    r=r.view(1,K,1)
    sf=torch.cat([sf0(r), sf1(r), sf2(r)], dim=1)
    sf=sf.view(1,K,3)
    return sf #(1,K,3)
#%%
def cal_d_sf_dr(r):
    #r.shape is (K,1), K is the number of integration points
    K=r.shape[0]
    r=r.permute(1,0)#shape is (1,K)
    r=r.view(1,K,1,1)
    d_sf0_dr=torch.cat(cal_d_sf0_dr(r), dim=1)#shape is (K,1,1)
    d_sf1_dr=torch.cat(cal_d_sf1_dr(r), dim=1)#shape is (K,1,1)
    d_sf2_dr=torch.cat(cal_d_sf2_dr(r), dim=1)#shape is (K,1,1)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,1,3)#3: sf0 to sf2
    return d_sf_dr #(1,K,1,3)
#%%
def interpolate(r, h):
    #r.shape (K,1), K is the number of integration points
    #h.shape (M,3,3), M elements, 1 elemnet has 3 nodes, h is a position, a displacement or a rotation vector
    #hr=sum_i{sf_i(r)*h_i, i=0 to 2}
    hr=matmul(cal_sf(r), h)
    return hr #(M,K,3)
#%% the undeformed length that FEBio uses
def cal_length(h):
    #h.shape (M,3,3); the distance between node 0 and node 1, the two end nodes. FEBio uses it with the undeformed nodes X
    length=torch.norm(h[:,1]-h[:,0], p=2, dim=-1)
    return length #(M,)
#%% initial frame of the cross-section
def cal_initial_frame(X):
    #X.shape (M,3,3), the undeformed nodes
    #Q[:,:,0]=E1, Q[:,:,1]=E2, Q[:,:,2]=E3, E3 from node 0 to node 1 (FEElasticBeamDomain::Init in FEBio)
    #E2 = E3 x [0,1,0] when the x-part of E3 is larger than 0.5 in size, else E3 x [1,0,0]; E1 = E2 x E3
    E3=X[:,1]-X[:,0]
    E3=E3/torch.norm(E3, p=2, dim=-1, keepdim=True)
    ex=torch.tensor([1.0,0.0,0.0], dtype=X.dtype, device=X.device).expand_as(E3)
    ey=torch.tensor([0.0,1.0,0.0], dtype=X.dtype, device=X.device).expand_as(E3)
    E2=torch.where((E3[:,0:1].abs() > 0.5), cross(E3, ey, dim=-1), cross(E3, ex, dim=-1))
    E2=E2/torch.norm(E2, p=2, dim=-1, keepdim=True)
    E1=cross(E2, E3, dim=-1)
    Q=torch.stack([E1, E2, E3], dim=2)
    return Q #(M,3,3)
#%%
def cal_d_sf_dS(r, X):
    #r.shape (K,1); X.shape (M,3,3), the undeformed nodes
    #dNi/dS = dNi/dr / (L0/2), S is the length along the undeformed element
    d_sf_dr=cal_d_sf_dr(r) #(1,K,1,3)
    J=0.5*cal_length(X).view(-1,1,1)
    d_sf_dS=d_sf_dr[:,:,0,:]/J
    return d_sf_dS #(M,K,3)
#%%
def cal_dh_dS(r, h, X):
    #r.shape (K,1); X.shape (M,3,3), the undeformed nodes
    #h.shape (M,3,3), h is x, or the change of the nodal rotation vectors
    dh_dS=matmul(cal_d_sf_dS(r, X), h)
    return dh_dS #(M,K,3)
#%% rotation matrix of a rotation vector
def cal_rotation_matrix(rotation_vector):
    #rotation_vector.shape (...,3): direction = axis, norm = angle in radian
    #R = I + sin(a)/a * W + (1-cos(a))/a^2 * W*W, W is the cross-product matrix of the vector, a is its norm
    v=rotation_vector
    a=torch.norm(v, p=2, dim=-1, keepdim=True).unsqueeze(-1) #(...,1,1)
    zero=torch.zeros_like(v[...,0])
    W=torch.stack([zero, -v[...,2], v[...,1],
                   v[...,2], zero, -v[...,0],
                   -v[...,1], v[...,0], zero], dim=-1).view(v.shape[:-1]+(3,3))
    g1=torch.sinc(a/pi)            #sin(a)/a, 1 at a=0
    g2=torch.sinc(0.5*a/pi)        #sin(a/2)/(a/2), 1 at a=0
    I=torch.eye(3, dtype=v.dtype, device=v.device)
    R=I+g1*W+0.5*g2*g2*matmul(W, W)
    return R #(...,3,3)
#%% new rotation and curvature at the integration points after a change of the nodal rotations
def cal_rotation_and_curvature_update(r, X, d_rotation, R, k):
    #r.shape (K,1); X.shape (M,3,3), the undeformed nodes
    #d_rotation.shape (M,3,3): the change of the rotation of each node, as a rotation vector in the global axes
    #R.shape (M,K,3,3), k.shape (M,K,3): rotation matrix and curvature at the integration points before the change
    #FEElasticBeamDomain::IncrementalUpdate in FEBio:
    #  dr = change of rotation at the point (interpolated), dR = its rotation matrix, R_new = dR*R
    #  k_new = g1*dr' + (1-g1)*(e.dr')*e + 0.5*g2^2*cross(dr, dr') + dR*k
    #  with dr' = d(dr)/dS, a = norm of dr, e = dr/a, g1 = sin(a)/a, g2 = sin(a/2)/(a/2)
    dr=interpolate(r, d_rotation) #(M,K,3)
    dr_dS=cal_dh_dS(r, d_rotation, X) #(M,K,3)
    dR=cal_rotation_matrix(dr)
    R_new=matmul(dR, R)
    a=torch.norm(dr, p=2, dim=-1, keepdim=True)
    g1=torch.sinc(a/pi)
    g2=torch.sinc(0.5*a/pi)
    #(1-g1)/a^2, which is 1/6 at a=0
    small=a < 1e-4
    a_safe=torch.where(small, torch.ones_like(a), a)
    c=torch.where(small, 1/6-a*a/120, (1-g1)/(a_safe*a_safe))
    k_new=g1*dr_dS+c*(dr*dr_dS).sum(dim=-1, keepdim=True)*dr+0.5*g2*g2*cross(dr, dr_dS, dim=-1)
    k_new=k_new+matmul(dR, k.unsqueeze(-1)).squeeze(-1)
    return R_new, k_new #(M,K,3,3), (M,K,3)
#%%
def cal_strain(r, x, X, R, k):
    #r.shape (K,1); x.shape (M,3,3), the deformed nodes; X.shape (M,3,3), the undeformed nodes
    #R.shape (M,K,3,3), k.shape (M,K,3): rotation matrix and curvature at the integration points
    #Gamma = R^T * dx/dS - E3, Kappa = R^T * k
    dx_dS=cal_dh_dS(r, x, X) #(M,K,3)
    E3=X[:,1]-X[:,0]
    E3=E3/torch.norm(E3, p=2, dim=-1, keepdim=True)
    Rt=R.transpose(2,3)
    Gamma=matmul(Rt, dx_dS.unsqueeze(-1)).squeeze(-1)-E3.view(-1,1,3)
    Kappa=matmul(Rt, k.unsqueeze(-1)).squeeze(-1)
    return Gamma, Kappa #(M,K,3), (M,K,3)
#%% FEElasticBeamMaterial in FEBio
def cal_force_and_moment_linear_elastic(Gamma, Kappa, R, Q, E, G, A, A1, A2, I1, I2):
    #Gamma.shape (M,K,3), Kappa.shape (M,K,3), from cal_strain
    #R.shape (M,K,3,3); Q.shape (M,3,3), from cal_initial_frame
    #E, G, A, A1, A2, I1, I2: each a number or a tensor of shape (M,1)
    #E: Young's modulus, G: shear modulus, A: area, A1, A2: shear areas along E1 and E2
    #I1, I2: second moments of area about E1 and E2; the torsion constant is I1+I2
    M=Gamma.shape[0]
    Qt=Q.transpose(1,2).view(M,1,3,3)
    gamma=matmul(Qt, Gamma.unsqueeze(-1)).squeeze(-1) #(M,K,3), in the axes E1, E2, E3
    kappa=matmul(Qt, Kappa.unsqueeze(-1)).squeeze(-1)
    N=torch.stack([G*A1*gamma[...,0], G*A2*gamma[...,1], E*A*gamma[...,2]], dim=-1)
    Mo=torch.stack([E*I1*kappa[...,0], E*I2*kappa[...,1], G*(I1+I2)*kappa[...,2]], dim=-1)
    RQ=matmul(R, Q.view(M,1,3,3))
    t=matmul(RQ, N.unsqueeze(-1)).squeeze(-1)
    m=matmul(RQ, Mo.unsqueeze(-1)).squeeze(-1)
    return t, m #(M,K,3), (M,K,3), in the global axes
#%% K integration points, K = 2
def cal_nodal_force_from_force_and_moment(t, m, r, x, X):
    #t, m: force and moment in the cross-section at the integration points, in the global axes, shape (M,K,3)
    #r.shape (K,1); x.shape (M,3,3), the deformed nodes; X.shape (M,3,3), the undeformed nodes
    #force_i=integration(dNi/dS * t), moment_i=integration(dNi/dS * m - Ni * cross(dx/dS, t))
    M=x.shape[0]
    K=r.shape[0]
    weight=get_integration_weight(K, x.dtype, x.device)
    wJ=weight.view(1,K,1)*0.5*cal_length(X).view(M,1,1) #(M,K,1)
    d_sf_dS=cal_d_sf_dS(r, X) #(M,K,3)
    sf=cal_sf(r) #(1,K,3)
    dx_dS=matmul(d_sf_dS, x) #(M,K,3)
    force=matmul((d_sf_dS*wJ).transpose(1,2), t)
    moment=matmul((d_sf_dS*wJ).transpose(1,2), m)-matmul((sf*wJ).transpose(1,2), cross(dx_dS, t, dim=-1))
    return force, moment #(M,3,3), (M,3,3)
#%% K integration points, K = 2
def cal_nodal_force_from_body_force(body_force, density, r, X, area):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed beam, a number or a tensor of shape (M,1)
    #r.shape (K,1); X.shape (M,3,3), the undeformed nodes
    #area: cross-section area, a number or a tensor of shape (M,) or (M,1)
    #force_i=integration(density*area*body_force*Ni along the beam), i=0 to 2
    #the nodal moments of a body force are zero (FEElasticBeamDomain::ElementBodyForce in FEBio)
    M=X.shape[0]
    K=r.shape[0]
    if torch.is_tensor(area):
        area=area.view(M,1,1)
    if torch.is_tensor(density):
        density=density.view(M,1,1)
    weight=get_integration_weight(K, X.dtype, X.device)
    wJ=weight.view(1,K,1)*0.5*cal_length(X).view(M,1,1) #(M,K,1)
    q=density*area*wJ*body_force #(M,K,3)
    force=matmul(cal_sf(r).transpose(1,2), q)
    return force #(M,3,3)
#%% K integration points, K = 2, the strain energy of each element for the linear elastic law
def cal_strain_energy_linear_elastic(Gamma, Kappa, t, m, R, X):
    #Gamma.shape (M,K,3), Kappa.shape (M,K,3), from cal_strain
    #t.shape (M,K,3), m.shape (M,K,3), from cal_force_and_moment_linear_elastic
    #R.shape (M,K,3,3); X.shape (M,3,3), the undeformed nodes
    #energy=integration(0.5*(R^T*t).Gamma + 0.5*(R^T*m).Kappa)
    M=Gamma.shape[0]
    K=Gamma.shape[1]
    weight=get_integration_weight(K, Gamma.dtype, Gamma.device)
    wJ=weight.view(1,K)*0.5*cal_length(X).view(M,1)
    Rt=R.transpose(2,3)
    N=matmul(Rt, t.unsqueeze(-1)).squeeze(-1)
    Mo=matmul(Rt, m.unsqueeze(-1)).squeeze(-1)
    W=0.5*((N*Gamma).sum(dim=-1)+(Mo*Kappa).sum(dim=-1)) #(M,K)
    energy=(wJ*W).sum(dim=1)
    return energy #(M,)
