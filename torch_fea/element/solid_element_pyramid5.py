# 5-node pyramid. PYRA5 in FEBio. The unit element is the cube [-1,1]^3 with its top face r[2]=1 collapsed into node 4
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio
# location of the nodes in the unit element, in FEBio's node order:
#   node 0: (-1, -1, -1)
#   node 1: (1, -1, -1)
#   node 2: (1, 1, -1)
#   node 3: (-1, 1, -1)
#   node 4: (0, 0, 1)
# shape functions: the function space is
#   hex8 with its top face (nodes 4 to 7) collapsed into node 4: node 4 has the sum of the four functions
#   each shape function is 1 at its own node and 0 at the other nodes
#   they give the same values as FEBio's (FEPyra5::shape_fnc and shape_deriv)
# integration rules, with the points in FEBio's order:
#   8 points: 2 x 2 x 2 Gauss points at +-1/sqrt(3), in the order of the corner nodes (FEPyra5G8)
# FEBio's source, at commit 49f0976 of the develop branch (2026-11-11):
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FEElementTraits.cpp
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FESolidElementShape.cpp
#%%
import torch
import torch_fea.element.solid_element_shared as shared
#%% integration point location
def get_integration_point_8i(dtype, device):
    #8 points: 2 x 2 x 2 Gauss points at +-1/sqrt(3), in the order of the corner nodes
    #the same numbers as FEBio FEPyra5G8
    r=[[-0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
       [0.5773502691896258, -0.5773502691896258, -0.5773502691896258],
       [0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
       [-0.5773502691896258, 0.5773502691896258, -0.5773502691896258],
       [-0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
       [0.5773502691896258, -0.5773502691896258, 0.5773502691896258],
       [0.5773502691896258, 0.5773502691896258, 0.5773502691896258],
       [-0.5773502691896258, 0.5773502691896258, 0.5773502691896258]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (8, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_8i(dtype, device):
    w=[1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (8,)
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,3); a list of the points when dtype or device is None
    if n_points == 8:
        return get_integration_point_8i(dtype, device)
    else:
        raise ValueError("n_points must be 8")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 8:
        return get_integration_weight_8i(dtype, device)
    else:
        raise ValueError("n_points must be 8")
#%% useful for 9.15c in the book
def cal_shape_integration_weight(r, w, dtype, device):
    #r: list of the K integration points, w: list of the K integration weights
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (5,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2, sf3, sf4]
    weight=[]
    for i in range(0, 5):
        for j in range(0, len(w)):
            weight.append(sf[i](r[j])*w[j])
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(5,len(w))
    return weight
#%%
def get_shape_integration_weight_8i(dtype, device):
    #return: weight, weight.shape (5,8); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_8i(None, None), get_integration_weight_8i(None, None), dtype, device)
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (5,n_points); a list of its numbers when dtype or device is None
    if n_points == 8:
        return get_shape_integration_weight_8i(dtype, device)
    else:
        raise ValueError("n_points must be 8")
#%% shape function 0, node 0 is at (-1, -1, -1)
def sf0(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] - 1)*(r[1] - 1)*(r[2] - 1)/8
#%%
def cal_d_sf0_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[1] - 1)*(r[2] - 1)/8
    dr1=-(r[0] - 1)*(r[2] - 1)/8
    dr2=-(r[0] - 1)*(r[1] - 1)/8
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 1, node 1 is at (1, -1, -1)
def sf1(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] + 1)*(r[1] - 1)*(r[2] - 1)/8
#%%
def cal_d_sf1_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[1] - 1)*(r[2] - 1)/8
    dr1=(r[0] + 1)*(r[2] - 1)/8
    dr2=(r[0] + 1)*(r[1] - 1)/8
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 2, node 2 is at (1, 1, -1)
def sf2(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] + 1)*(r[1] + 1)*(r[2] - 1)/8
#%%
def cal_d_sf2_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[1] + 1)*(r[2] - 1)/8
    dr1=-(r[0] + 1)*(r[2] - 1)/8
    dr2=-(r[0] + 1)*(r[1] + 1)/8
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 3, node 3 is at (-1, 1, -1)
def sf3(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] - 1)*(r[1] + 1)*(r[2] - 1)/8
#%%
def cal_d_sf3_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[1] + 1)*(r[2] - 1)/8
    dr1=(r[0] - 1)*(r[2] - 1)/8
    dr2=(r[0] - 1)*(r[1] + 1)/8
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 4, node 4 is at (0, 0, 1)
def sf4(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[2] + 1)/2
#%%
def cal_d_sf4_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=0+0*r[0]
    dr2=1/2+0*r[0]
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
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,3,5)#5: sf0 to sf4
    return d_sf_dr #(1,K,3,5)
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,5,3), M elements, 1 elemnet has 5 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_i{sf_i(r)*h_i, i=0 to 4}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r)], dim=1)
    w=w.view(1,K,5,1)
    M=h.shape[0]
    h=h.view(M,1,5,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,5,3), M elements, 1 elemnet has 5 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,5)
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #shape (M,K,3,3)
#%%
def cal_d_sf_dh(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,5,3), M elements, 1 elemnet has 5 nodes, 1 node has a 3D position
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,5), K is the number of integration points
    d_sf_dh, dh_dr, det_dh_dr=shared.cal_d_sf_dh(h, d_sf_dr)
    #d_sf_dh.shape is (M,K,3,5), d_sf_dh[:,:,:,0] is d_sf0_dh
    #dh_dr.shape is (M,K,3,3), numerator_layout=False
    #det_dh_dr.shape is (M,K,1,1)
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,5,3)
    #X.shape is (M,5,3)
    F=shared.cal_F_tensor(x, X, cal_d_sf_dr(r))
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,5,3)
    #d_sf_dX.shape is (M,K,3,5), from cal_d_sf_dh
    F=shared.cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    return F #(M,K,3,3)
#%% K integration points, K = 8
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,5), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 4, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dx_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr, weight)
    return force #(M,5,3)
#%% K integration points, K = 8
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,5), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dX at r), i=0 to 4, #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr, weight)
    return force #(M,5,3)
#%% K integration points, K = 8
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,5), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(S*dNi_dX at r), i=0 to 4, #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight)
    return force #(M,5,3)
#%% K integration points, K = 8
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #det_dX_dr.shape is (M,K,1,1), det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(density*body_force*Ni at r), i=0 to 4, #(9.15c) in the book
    weight=get_shape_integration_weight(det_dX_dr.shape[1], det_dX_dr.dtype, det_dX_dr.device) #(5,K)
    force=shared.cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight)
    return force #(M,5,3)
#%% K integration points, K = 8, the strain energy of each element
def cal_strain_energy(W, det_dX_dr):
    #W.shape (M,K)
    #det_dX_dr.shape (M,K,1,1)
    #W=cal_strain_energy_density(F, Mat) on the integration points
    #energy=integration(W at r)
    weight=get_integration_weight(det_dX_dr.shape[1], W.dtype, torch.device("cpu"))
    energy=shared.cal_strain_energy(W, det_dX_dr, weight)
    return energy #(M,)
