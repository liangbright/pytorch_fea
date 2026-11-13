# 3-node triangle shell element. "tri3" in a shell domain of FEBio (FEShellTri3_, FEShellTri3G6, FEShellTri3G9)
# FEBio's shell (FESSIShellDomain, FEElasticShellDomain): a shell node sits on the front face of the shell, and a
#   second point on the back face belongs to it: back = node - director. The position inside the shell is
#       x(r) = sum_i M_i(r[0],r[1]) * (x_i - (1-r[2])/2 * director_i)
#   so the shell is a solid that is linear through the thickness, with 6 points:
#       point 0 to 2: the shell nodes (front face, r[2]=1), in FEBio's node order
#       point 3 to 5: the back points (r[2]=-1), point 3+i belongs to shell node i
#   This file has the functions of a solid_element file for these 6 points:
#       h in the functions below has the shape (M,6,3); cal_front_and_back_node makes it from nodes and directors
#       a nodal force has the same shape: force[:,0:3] acts on the shell nodes (the displacement u in FEBio)
#       and force[:,3:6] on the back points (the shell displacement w in FEBio)
#   The stress comes from a 3D material with the deformation gradient F, as for a solid element.
# With 6 points this element gives the same results as solid_element_wedge6 with 6 points (shell node i is wedge6 node 3+i, its back point is wedge6 node i).
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio, [r[0], r[1]] = [u, v] in the surface elements
# location of the shell nodes in (r[0], r[1]), in FEBio's node order:
#   shell node 0: (0, 0)
#   shell node 1: (1, 0)
#   shell node 2: (0, 1)
# shape functions: sf_i = M_i*(1+r[2])/2 for shell node i, sf_(3+i) = M_i*(1-r[2])/2 for its back point
#   M_i: linear polynomials of u, v; 1 at its own node and 0 at the other nodes
#   M_i gives the same values as FEBio's (FEShellTri3_::shape_fnc and shape_deriv)
# integration rules, with the points in FEBio's order (the first one is FEBio's default):
#   6 points: the 3-point triangle rule, times 2 Gauss points through the thickness (FEShellTri3G6)
#   9 points: the 3-point triangle rule, times 3 Gauss points through the thickness (FEShellTri3G9)
# The 9-point rule differs from FEBio: FEShellTri3G9 has r[2] = -2/3, 0, 2/3 with the weights 5/9, 8/9, 5/9. The value
#   2/3 is the constant of the in-plane points, used for r[2] as well. These are not Gauss points: the rule gives
#   40/81 instead of 2/3 for the integral of r[2]^2, so the bending energy comes out 26% too low. This file has
#   the Gauss points -sqrt(3/5), 0, sqrt(3/5).
# pressure on the front face: cal_nodal_force_from_pressure of the surface_element file of the same shape, with the shell nodes
# This file was written by element_update_tools/generate_structural_elements.py
#   https://github.com/febiosoftware/FEBio/blob/develop/FECore/FEElementTraits.cpp
#   https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FESSIShellDomain.cpp
#   https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticShellDomain.cpp
#%%
import torch
import torch_fea.element.solid_element_shared as shared
from torch import cross
#%% integration point location
def get_integration_point_6i(dtype, device):
    #6 points: the 3-point triangle rule in (r[0], r[1]),
    #  times 2 Gauss points in r[2] at +-1/sqrt(3); the in-plane index changes fastest
    #the points and weights of FEBio FEShellTri3G6
    r=[[0.16666666666666666, 0.16666666666666666, -0.5773502691896258],
       [0.6666666666666666, 0.16666666666666666, -0.5773502691896258],
       [0.16666666666666666, 0.6666666666666666, -0.5773502691896258],
       [0.16666666666666666, 0.16666666666666666, 0.5773502691896258],
       [0.6666666666666666, 0.16666666666666666, 0.5773502691896258],
       [0.16666666666666666, 0.6666666666666666, 0.5773502691896258]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (6, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_6i(dtype, device):
    w=[0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666,
       0.16666666666666666, 0.16666666666666666]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (6,)
    return w
#%% integration point location
def get_integration_point_9i(dtype, device):
    #9 points: the 3-point triangle rule in (r[0], r[1]),
    #  times 3 Gauss points in r[2] at -b, 0, b with b=sqrt(3/5); the in-plane index changes fastest
    #FEBio FEShellTri3G9 has r[2]=-2/3, 0, 2/3 with the same weights: not the Gauss points, see the top of the file
    r=[[0.16666666666666666, 0.16666666666666666, -0.7745966692414834],
       [0.6666666666666666, 0.16666666666666666, -0.7745966692414834],
       [0.16666666666666666, 0.6666666666666666, -0.7745966692414834],
       [0.16666666666666666, 0.16666666666666666, 0.0],
       [0.6666666666666666, 0.16666666666666666, 0.0],
       [0.16666666666666666, 0.6666666666666666, 0.0],
       [0.16666666666666666, 0.16666666666666666, 0.7745966692414834],
       [0.6666666666666666, 0.16666666666666666, 0.7745966692414834],
       [0.16666666666666666, 0.6666666666666666, 0.7745966692414834]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (9, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_9i(dtype, device):
    w=[0.09259259259259259, 0.09259259259259259, 0.09259259259259259, 0.14814814814814814,
       0.14814814814814814, 0.14814814814814814, 0.09259259259259259, 0.09259259259259259,
       0.09259259259259259]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (9,)
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,3); a list of the points when dtype or device is None
    if n_points == 6:
        return get_integration_point_6i(dtype, device)
    elif n_points == 9:
        return get_integration_point_9i(dtype, device)
    else:
        raise ValueError("n_points must be 6 or 9")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 6:
        return get_integration_weight_6i(dtype, device)
    elif n_points == 9:
        return get_integration_weight_9i(dtype, device)
    else:
        raise ValueError("n_points must be 6 or 9")
#%% useful for 9.15c in the book
def cal_shape_integration_weight(r, w, dtype, device):
    #r: list of the K integration points, w: list of the K integration weights
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (6,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2, sf3, sf4, sf5]
    weight=[]
    for i in range(0, 6):
        for j in range(0, len(w)):
            weight.append(sf[i](r[j])*w[j])
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(6,len(w))
    return weight
#%%
def get_shape_integration_weight_6i(dtype, device):
    #return: weight, weight.shape (6,6); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_6i(None, None), get_integration_weight_6i(None, None), dtype, device)
#%%
def get_shape_integration_weight_9i(dtype, device):
    #return: weight, weight.shape (6,9); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_9i(None, None), get_integration_weight_9i(None, None), dtype, device)
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (6,n_points); a list of its numbers when dtype or device is None
    if n_points == 6:
        return get_shape_integration_weight_6i(dtype, device)
    elif n_points == 9:
        return get_shape_integration_weight_9i(dtype, device)
    else:
        raise ValueError("n_points must be 6 or 9")
#%% shape function 0: shell node 0, front face, at (0, 0, 1)
def sf0(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[2] + 1)*(r[0] + r[1] - 1)/2
#%%
def cal_d_sf0_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[2] + 1)/2
    dr1=-(r[2] + 1)/2
    dr2=-(r[0] + r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 1: shell node 1, front face, at (1, 0, 1)
def sf1(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*(r[2] + 1)/2
#%%
def cal_d_sf1_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[2] + 1)/2
    dr1=0+0*r[0]
    dr2=r[0]/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 2: shell node 2, front face, at (0, 1, 1)
def sf2(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[1]*(r[2] + 1)/2
#%%
def cal_d_sf2_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=(r[2] + 1)/2
    dr2=r[1]/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 3: back point of shell node 0, at (0, 0, -1)
def sf3(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[2] - 1)*(r[0] + r[1] - 1)/2
#%%
def cal_d_sf3_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[2] - 1)/2
    dr1=(r[2] - 1)/2
    dr2=(r[0] + r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 4: back point of shell node 1, at (1, 0, -1)
def sf4(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[0]*(r[2] - 1)/2
#%%
def cal_d_sf4_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[2] - 1)/2
    dr1=0+0*r[0]
    dr2=-r[0]/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 5: back point of shell node 2, at (0, 1, -1)
def sf5(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[1]*(r[2] - 1)/2
#%%
def cal_d_sf5_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=-(r[2] - 1)/2
    dr2=-r[1]/2
    dr=[dr0, dr1, dr2]
    return dr
#%%
def cal_d_sf_dr(r):
    #r.shape is (K,3), K is the number of integration points
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1,1)
    d_sf0_dr=torch.cat(cal_d_sf0_dr(r), dim=1)#shape is (K,3,1)
    d_sf1_dr=torch.cat(cal_d_sf1_dr(r), dim=1)#shape is (K,3,1)
    d_sf2_dr=torch.cat(cal_d_sf2_dr(r), dim=1)#shape is (K,3,1)
    d_sf3_dr=torch.cat(cal_d_sf3_dr(r), dim=1)#shape is (K,3,1)
    d_sf4_dr=torch.cat(cal_d_sf4_dr(r), dim=1)#shape is (K,3,1)
    d_sf5_dr=torch.cat(cal_d_sf5_dr(r), dim=1)#shape is (K,3,1)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,3,6)#6: sf0 to sf5
    return d_sf_dr #(1,K,3,6)
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,6,3), M elements, 3 shell nodes and then their 3 back points, h is x or X or u
    #hr=sum_i{sf_i(r)*h_i, i=0 to 5}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r)], dim=1)
    w=w.view(1,K,6,1)
    M=h.shape[0]
    h=h.view(M,1,6,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,6,3), M elements, 3 shell nodes and then their 3 back points, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,6)
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #shape (M,K,3,3)
#%%
def cal_d_sf_dh(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,6,3), M elements, 3 shell nodes and then their 3 back points, 1 node has a 3D position
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,6), K is the number of integration points
    d_sf_dh, dh_dr, det_dh_dr=shared.cal_d_sf_dh(h, d_sf_dr)
    #d_sf_dh.shape is (M,K,3,6), d_sf_dh[:,:,:,0] is d_sf0_dh
    #dh_dr.shape is (M,K,3,3), numerator_layout=False
    #det_dh_dr.shape is (M,K,1,1)
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,6,3)
    #X.shape is (M,6,3)
    F=shared.cal_F_tensor(x, X, cal_d_sf_dr(r))
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,6,3)
    #d_sf_dX.shape is (M,K,3,6), from cal_d_sf_dh
    F=shared.cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    return F #(M,K,3,3)
#%% K integration points, K = 6 or 9
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,6), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 5, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dx_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr, weight)
    return force #(M,6,3)
#%% K integration points, K = 6 or 9
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,6), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dX at r), i=0 to 5, #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr, weight)
    return force #(M,6,3)
#%% K integration points, K = 6 or 9
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,6), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(S*dNi_dX at r), i=0 to 5, #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight)
    return force #(M,6,3)
#%% K integration points, K = 6 or 9
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #det_dX_dr.shape is (M,K,1,1), det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(density*body_force*Ni at r), i=0 to 5, #(9.15c) in the book
    weight=get_shape_integration_weight(det_dX_dr.shape[1], det_dX_dr.dtype, det_dX_dr.device) #(6,K)
    force=shared.cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight)
    return force #(M,6,3)
#%% K integration points, K = 6 or 9, the strain energy of each element
def cal_strain_energy(W, det_dX_dr):
    #W.shape (M,K)
    #det_dX_dr.shape (M,K,1,1)
    #W=cal_strain_energy_density(F, Mat) on the integration points
    #energy=integration(W at r)
    weight=get_integration_weight(det_dX_dr.shape[1], W.dtype, torch.device("cpu"))
    energy=shared.cal_strain_energy(W, det_dX_dr, weight)
    return energy #(M,)
#%% the director of each node of each element: thickness times the unit normal
def cal_director(X, thickness, nodal_normal=True):
    #X.shape (M,3,3), the shell nodes of the undeformed elements
    #thickness: a number, or a tensor of shape (M,1) or (M,3)
    #nodal_normal True: the normal at node j is (b-a) x (c-a), with a = node j, b = the next node and c = the previous
    #    node in the node order (FEModel::InitShells in FEBio). FEBio then gives a node the average of the directors
    #    that the elements sharing the node have for it; that average over elements is not done here.
    #nodal_normal False: every node of the element has the normal of the element at r[0]=r[1]=0
    #    (FESSIShellDomain::InitShells in FEBio, when its nodal normals are switched off)
    M=X.shape[0]
    if nodal_normal == True:
        b=torch.roll(X, shifts=-1, dims=1)
        c=torch.roll(X, shifts=1, dims=1)
        normal=cross(b-X, c-X, dim=-1) #(M,3,3)
    else:
        d_sf_du=torch.tensor([-1.0, 1.0, 0.0], dtype=X.dtype, device=X.device)
        d_sf_dv=torch.tensor([-1.0, 0.0, 1.0], dtype=X.dtype, device=X.device)
        dX_du=(d_sf_du.view(1,3,1)*X).sum(dim=1) #(M,3)
        dX_dv=(d_sf_dv.view(1,3,1)*X).sum(dim=1) #(M,3)
        normal=cross(dX_du, dX_dv, dim=-1).view(M,1,3).expand(M,3,3)
    normal=normal/torch.norm(normal, p=2, dim=-1, keepdim=True)
    if torch.is_tensor(thickness):
        thickness=thickness.view(M,-1,1)
    director=thickness*normal
    return director #(M,3,3)
#%% the 6 points of the element: the shell nodes and then their back points
def cal_front_and_back_node(x, director):
    #x.shape (M,3,3), the shell nodes (front face)
    #director.shape (M,3,3), from the back point to the shell node: back = x - director
    #FEBio: the current director is D0 + u - w, with u the displacement of the node and w its shell displacement,
    #       so the back point is at X - D0 + w
    h=torch.cat([x, x-director], dim=1)
    return h #(M,6,3)
