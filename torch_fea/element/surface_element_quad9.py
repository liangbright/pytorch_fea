# 9-node quadratic quadrilateral surface element. A face of solid_element_hex27. QUAD9 in FEBio
# [r[0], r[1]] = [u, v] ~ [r, s] in FEBio
# location of the nodes in the unit element, in FEBio's node order:
#   node 0: (-1, -1)
#   node 1: (1, -1)
#   node 2: (1, 1)
#   node 3: (-1, 1)
#   node 4: (0, -1)
#   node 5: (1, 0)
#   node 6: (0, 1)
#   node 7: (-1, 0)
#   node 8: (0, 0)
# shape functions: the function space is
#   products r[0]^a * r[1]^b with a, b <= 2
#   each shape function is 1 at its own node and 0 at the other nodes
#   they give the same values as FEBio's (FEQuad9::shape_fnc and shape_deriv)
# integration rules, with the points in FEBio's order:
#   9 points: 3 x 3 Gauss points at -a, 0, a with a=sqrt(3/5), r[0] changes fastest (FEQuad9G9)
# FEBio's source, at commit 49f0976 of the develop branch (2026-11-11):
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FEElementTraits.cpp
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FESurfaceElementShape.cpp
#%%
import torch
import torch_fea.element.surface_element_shared as shared
#%% integration point location
def get_integration_point_9i(dtype, device):
    #9 points: 3 x 3 Gauss points at -a, 0, a with a=sqrt(3/5), r[0] changes fastest
    #the same numbers as FEBio FEQuad9G9
    r=[[-0.7745966692414834, -0.7745966692414834],
       [0.0, -0.7745966692414834],
       [0.7745966692414834, -0.7745966692414834],
       [-0.7745966692414834, 0.0],
       [0.0, 0.0],
       [0.7745966692414834, 0.0],
       [-0.7745966692414834, 0.7745966692414834],
       [0.0, 0.7745966692414834],
       [0.7745966692414834, 0.7745966692414834]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (9, 2)
    return r
#%% gaussian integration weight
def get_integration_weight_9i(dtype, device):
    w=[0.308641975308642, 0.49382716049382713, 0.308641975308642, 0.49382716049382713,
       0.7901234567901234, 0.49382716049382713, 0.308641975308642, 0.49382716049382713,
       0.308641975308642]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (9,)
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,2); a list of the points when dtype or device is None
    if n_points == 9:
        return get_integration_point_9i(dtype, device)
    else:
        raise ValueError("n_points must be 9")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 9:
        return get_integration_weight_9i(dtype, device)
    else:
        raise ValueError("n_points must be 9")
#%%
def cal_shape_integration_weight(r, w, dtype, device):
    #r: list of the K integration points, w: list of the K integration weights
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (9,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8]
    weight=[]
    for i in range(0, 9):
        for j in range(0, len(w)):
            weight.append(sf[i](r[j])*w[j])
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(9,len(w))
    return weight
#%%
def get_shape_integration_weight_9i(dtype, device):
    #return: weight, weight.shape (9,9); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_9i(None, None), get_integration_weight_9i(None, None), dtype, device)
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (9,n_points); a list of its numbers when dtype or device is None
    if n_points == 9:
        return get_shape_integration_weight_9i(dtype, device)
    else:
        raise ValueError("n_points must be 9")
#%% shape function 0, node 0 is at (-1, -1)
def sf0(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*r[1]*(r[0] - 1)*(r[1] - 1)/4
#%%
def cal_d_sf0_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=r[1]*(2*r[0] - 1)*(r[1] - 1)/4
    dr1=r[0]*(r[0] - 1)*(2*r[1] - 1)/4
    dr=[dr0, dr1]
    return dr
#%% shape function 1, node 1 is at (1, -1)
def sf1(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*r[1]*(r[0] + 1)*(r[1] - 1)/4
#%%
def cal_d_sf1_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=r[1]*(2*r[0] + 1)*(r[1] - 1)/4
    dr1=r[0]*(r[0] + 1)*(2*r[1] - 1)/4
    dr=[dr0, dr1]
    return dr
#%% shape function 2, node 2 is at (1, 1)
def sf2(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*r[1]*(r[0] + 1)*(r[1] + 1)/4
#%%
def cal_d_sf2_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=r[1]*(2*r[0] + 1)*(r[1] + 1)/4
    dr1=r[0]*(r[0] + 1)*(2*r[1] + 1)/4
    dr=[dr0, dr1]
    return dr
#%% shape function 3, node 3 is at (-1, 1)
def sf3(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*r[1]*(r[0] - 1)*(r[1] + 1)/4
#%%
def cal_d_sf3_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=r[1]*(2*r[0] - 1)*(r[1] + 1)/4
    dr1=r[0]*(r[0] - 1)*(2*r[1] + 1)/4
    dr=[dr0, dr1]
    return dr
#%% shape function 4, node 4 is at (0, -1)
def sf4(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[1]*(r[0] - 1)*(r[0] + 1)*(r[1] - 1)/2
#%%
def cal_d_sf4_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-r[0]*r[1]*(r[1] - 1)
    dr1=-(r[0] - 1)*(r[0] + 1)*(2*r[1] - 1)/2
    dr=[dr0, dr1]
    return dr
#%% shape function 5, node 5 is at (1, 0)
def sf5(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[0]*(r[0] + 1)*(r[1] - 1)*(r[1] + 1)/2
#%%
def cal_d_sf5_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-(2*r[0] + 1)*(r[1] - 1)*(r[1] + 1)/2
    dr1=-r[0]*r[1]*(r[0] + 1)
    dr=[dr0, dr1]
    return dr
#%% shape function 6, node 6 is at (0, 1)
def sf6(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[1]*(r[0] - 1)*(r[0] + 1)*(r[1] + 1)/2
#%%
def cal_d_sf6_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-r[0]*r[1]*(r[1] + 1)
    dr1=-(r[0] - 1)*(r[0] + 1)*(2*r[1] + 1)/2
    dr=[dr0, dr1]
    return dr
#%% shape function 7, node 7 is at (-1, 0)
def sf7(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[0]*(r[0] - 1)*(r[1] - 1)*(r[1] + 1)/2
#%%
def cal_d_sf7_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-(2*r[0] - 1)*(r[1] - 1)*(r[1] + 1)/2
    dr1=-r[0]*r[1]*(r[0] - 1)
    dr=[dr0, dr1]
    return dr
#%% shape function 8, node 8 is at (0, 0)
def sf8(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] - 1)*(r[0] + 1)*(r[1] - 1)*(r[1] + 1)
#%%
def cal_d_sf8_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=2*r[0]*(r[1] - 1)*(r[1] + 1)
    dr1=2*r[1]*(r[0] - 1)*(r[0] + 1)
    dr=[dr0, dr1]
    return dr
#%%
def cal_d_sf_dr(r):
    #r.shape is (K,2), K is the number of integration points
    K=r.shape[0]
    r=r.permute(1,0)#shape is (2,K)
    r=r.view(2,K,1,1)
    d_sf0_dr=torch.cat(cal_d_sf0_dr(r), dim=1)#shape is (K,2,1)
    d_sf1_dr=torch.cat(cal_d_sf1_dr(r), dim=1)#shape is (K,2,1)
    d_sf2_dr=torch.cat(cal_d_sf2_dr(r), dim=1)#shape is (K,2,1)
    d_sf3_dr=torch.cat(cal_d_sf3_dr(r), dim=1)#shape is (K,2,1)
    d_sf4_dr=torch.cat(cal_d_sf4_dr(r), dim=1)#shape is (K,2,1)
    d_sf5_dr=torch.cat(cal_d_sf5_dr(r), dim=1)#shape is (K,2,1)
    d_sf6_dr=torch.cat(cal_d_sf6_dr(r), dim=1)#shape is (K,2,1)
    d_sf7_dr=torch.cat(cal_d_sf7_dr(r), dim=1)#shape is (K,2,1)
    d_sf8_dr=torch.cat(cal_d_sf8_dr(r), dim=1)#shape is (K,2,1)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr, d_sf6_dr, d_sf7_dr,
             d_sf8_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,2,9)#9: sf0 to sf8
    return d_sf_dr #(1,K,2,9)
#%%
def interpolate(r, h):
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,9,3), M elements, 1 elemnet has 9 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_i{sf_i(r)*h_i, i=0 to 8}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (2,K)
    r=r.view(2,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r), sf6(r), sf7(r),
                 sf8(r)], dim=1)
    w=w.view(1,K,9,1)
    M=h.shape[0]
    h=h.view(M,1,9,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,9,3), M elements, 1 elemnet has 9 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,2,9)
    #dh_dr when numerator_layout is True
    #  h=[a,b,c]
    #  da/du, da/dv
    #  db/du, db/dv
    #  dc/du, dc/dv
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #(M,K,3,2), or (M,K,2,3) when numerator_layout is False
#%%
def cal_dh_du_and_dh_dv(r, h):
    #r.shape (K,2), K is the number of integration points
    #h.shape is (M,9,3)
    dh_du, dh_dv=shared.cal_dh_du_and_dh_dv(h, cal_d_sf_dr(r))
    return dh_du, dh_dv #(M,K,3), (M,K,3)
#%%
def cal_normal(r, x):
    #r.shape (K,2), K is the number of integration points
    #x.shape is (M,9,3)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x)
    normal=shared.cal_normal(dx_du, dx_dv)
    return normal #(M,K,3)
#%% K integration points, K = 9
def cal_nodal_force_from_pressure(pressure, x, n_points):
    #x.shape (M,9,3)
    #pressure is scalar or tensor with shape (M, 1)
    #force_i=Integration(pressure*cross(dx_du, dx_dv)*shape_function_i at r), i=0 to 8
    #n_points: the number of integration points
    r=get_integration_point(n_points, x.dtype, x.device)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x) #(M,K,3), (M,K,3)
    weight=get_shape_integration_weight(n_points, x.dtype, x.device) #(9,K)
    force=shared.cal_nodal_force_from_pressure(pressure, dx_du, dx_dv, weight)
    return force #(M,9,3)
