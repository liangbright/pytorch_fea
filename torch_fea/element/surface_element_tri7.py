# 7-node triangle surface element: tri6 and a node at the centre. A face of solid_element_tet15. TRI7 in FEBio
# [r[0], r[1]] = [u, v] ~ [r, s] in FEBio
# location of the nodes in the unit element, in FEBio's node order:
#   node 0: (0, 0)
#   node 1: (1, 0)
#   node 2: (0, 1)
#   node 3: (1/2, 0)
#   node 4: (1/2, 1/2)
#   node 5: (0, 1/2)
#   node 6: (1/3, 1/3)
# shape functions: the function space is
#   quadratic polynomials and the bubble L0*L1*L2 (L: area coordinates)
#   each shape function is 1 at its own node and 0 at the other nodes
#   they give the same values as FEBio's (FETri7::shape_fnc and shape_deriv)
# integration rules, with the points in FEBio's order:
#   3 points: exact up to degree 2 (FETri7G3)
#   4 points: exact up to degree 3, the weight of the first point is negative (FETri7G4)
#   7 points: exact up to degree 5 (FETri7G7)
# FEBio's source, at commit 49f0976 of the develop branch (2026-11-11):
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FEElementTraits.cpp
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FESurfaceElementShape.cpp
#%%
import torch
import torch_fea.element.surface_element_shared as shared
#%% integration point location
def get_integration_point_3i(dtype, device):
    #3 points: exact up to degree 2
    #FEBio FETri7G3 has the same points with fewer digits (largest difference 3e-17)
    r=[[0.16666666666666666, 0.16666666666666666],
       [0.6666666666666666, 0.16666666666666666],
       [0.16666666666666666, 0.6666666666666666]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (3, 2)
    return r
#%% gaussian integration weight
def get_integration_weight_3i(dtype, device):
    w=[0.16666666666666663, 0.16666666666666669, 0.16666666666666669]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (3,)
    return w
#%% integration point location
def get_integration_point_4i(dtype, device):
    #4 points: exact up to degree 3, the weight of the first point is negative
    #FEBio FETri7G4 has the same points with fewer digits (largest difference 3e-17)
    r=[[0.3333333333333333, 0.3333333333333333],
       [0.6, 0.19999999999999998],
       [0.2, 0.2],
       [0.19999999999999998, 0.6]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (4, 2)
    return r
#%% gaussian integration weight
def get_integration_weight_4i(dtype, device):
    w=[-0.28125, 0.2604166666666667, 0.2604166666666667, 0.2604166666666667]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (4,)
    return w
#%% integration point location
def get_integration_point_7i(dtype, device):
    #7 points: exact up to degree 5
    #FEBio FETri7G7 has the same points with fewer digits (largest difference 3e-16)
    r=[[0.3333333333333333, 0.3333333333333333],
       [0.7974269853530873, 0.10128650732345634],
       [0.10128650732345634, 0.7974269853530873],
       [0.10128650732345634, 0.10128650732345634],
       [0.4701420641051151, 0.4701420641051151],
       [0.4701420641051151, 0.05971587178976982],
       [0.05971587178976982, 0.4701420641051151]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (7, 2)
    return r
#%% gaussian integration weight
def get_integration_weight_7i(dtype, device):
    w=[0.1125, 0.06296959027241357, 0.06296959027241357, 0.06296959027241357,
       0.0661970763942531, 0.0661970763942531, 0.0661970763942531]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (7,)
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,2); a list of the points when dtype or device is None
    if n_points == 3:
        return get_integration_point_3i(dtype, device)
    elif n_points == 4:
        return get_integration_point_4i(dtype, device)
    elif n_points == 7:
        return get_integration_point_7i(dtype, device)
    else:
        raise ValueError("n_points must be 3 or 4 or 7")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 3:
        return get_integration_weight_3i(dtype, device)
    elif n_points == 4:
        return get_integration_weight_4i(dtype, device)
    elif n_points == 7:
        return get_integration_weight_7i(dtype, device)
    else:
        raise ValueError("n_points must be 3 or 4 or 7")
#%%
def cal_shape_integration_weight(r, w, dtype, device):
    #r: list of the K integration points, w: list of the K integration weights
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (7,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2, sf3, sf4, sf5, sf6]
    weight=[]
    for i in range(0, 7):
        for j in range(0, len(w)):
            weight.append(sf[i](r[j])*w[j])
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(7,len(w))
    return weight
#%%
def get_shape_integration_weight_3i(dtype, device):
    #return: weight, weight.shape (7,3); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_3i(None, None), get_integration_weight_3i(None, None), dtype, device)
#%%
def get_shape_integration_weight_4i(dtype, device):
    #return: weight, weight.shape (7,4); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_4i(None, None), get_integration_weight_4i(None, None), dtype, device)
#%%
def get_shape_integration_weight_7i(dtype, device):
    #return: weight, weight.shape (7,7); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_7i(None, None), get_integration_weight_7i(None, None), dtype, device)
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (7,n_points); a list of its numbers when dtype or device is None
    if n_points == 3:
        return get_shape_integration_weight_3i(dtype, device)
    elif n_points == 4:
        return get_shape_integration_weight_4i(dtype, device)
    elif n_points == 7:
        return get_shape_integration_weight_7i(dtype, device)
    else:
        raise ValueError("n_points must be 3 or 4 or 7")
#%% shape function 0, node 0 is at (0, 0)
def sf0(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] + r[1] - 1)*(3*r[0]*r[1] - 2*r[0] - 2*r[1] + 1)
#%%
def cal_d_sf0_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-6*r[0]*r[1] + 4*r[0] - 3*r[1]**2 + 7*r[1] - 3
    dr1=-3*r[0]**2 - 6*r[0]*r[1] + 7*r[0] + 4*r[1] - 3
    dr=[dr0, dr1]
    return dr
#%% shape function 1, node 1 is at (1, 0)
def sf1(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[0]*(3*r[0]*r[1] - 2*r[0] + 3*r[1]**2 - 3*r[1] + 1)
#%%
def cal_d_sf1_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-6*r[0]*r[1] + 4*r[0] - 3*r[1]**2 + 3*r[1] - 1
    dr1=-3*r[0]*(r[0] + 2*r[1] - 1)
    dr=[dr0, dr1]
    return dr
#%% shape function 2, node 2 is at (0, 1)
def sf2(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[1]*(3*r[0]**2 + 3*r[0]*r[1] - 3*r[0] - 2*r[1] + 1)
#%%
def cal_d_sf2_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-3*r[1]*(2*r[0] + r[1] - 1)
    dr1=-3*r[0]**2 - 6*r[0]*r[1] + 3*r[0] + 4*r[1] - 1
    dr=[dr0, dr1]
    return dr
#%% shape function 3, node 3 is at (1/2, 0)
def sf3(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*r[0]*(3*r[1] - 1)*(r[0] + r[1] - 1)
#%%
def cal_d_sf3_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=4*(3*r[1] - 1)*(2*r[0] + r[1] - 1)
    dr1=4*r[0]*(3*r[0] + 6*r[1] - 4)
    dr=[dr0, dr1]
    return dr
#%% shape function 4, node 4 is at (1/2, 1/2)
def sf4(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*r[0]*r[1]*(3*r[0] + 3*r[1] - 2)
#%%
def cal_d_sf4_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=4*r[1]*(6*r[0] + 3*r[1] - 2)
    dr1=4*r[0]*(3*r[0] + 6*r[1] - 2)
    dr=[dr0, dr1]
    return dr
#%% shape function 5, node 5 is at (0, 1/2)
def sf5(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*r[1]*(3*r[0] - 1)*(r[0] + r[1] - 1)
#%%
def cal_d_sf5_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=4*r[1]*(6*r[0] + 3*r[1] - 4)
    dr1=4*(3*r[0] - 1)*(r[0] + 2*r[1] - 1)
    dr=[dr0, dr1]
    return dr
#%% shape function 6, node 6 is at (1/3, 1/3)
def sf6(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -27*r[0]*r[1]*(r[0] + r[1] - 1)
#%%
def cal_d_sf6_dr(r):
    #r.shape (2,...): r[0], r[1] are numbers or tensors of one shape
    #return: [dr0, dr1], the derivatives with respect to r[0], r[1], each a number or a tensor with the shape of r[0]
    dr0=-27*r[1]*(2*r[0] + r[1] - 1)
    dr1=-27*r[0]*(r[0] + 2*r[1] - 1)
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
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr, d_sf6_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,2,7)#7: sf0 to sf6
    return d_sf_dr #(1,K,2,7)
#%%
def interpolate(r, h):
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,7,3), M elements, 1 elemnet has 7 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_i{sf_i(r)*h_i, i=0 to 6}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (2,K)
    r=r.view(2,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r), sf6(r)], dim=1)
    w=w.view(1,K,7,1)
    M=h.shape[0]
    h=h.view(M,1,7,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,2), K is the number of integration points
    #h.shape (M,7,3), M elements, 1 elemnet has 7 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,2,7)
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
    #h.shape is (M,7,3)
    dh_du, dh_dv=shared.cal_dh_du_and_dh_dv(h, cal_d_sf_dr(r))
    return dh_du, dh_dv #(M,K,3), (M,K,3)
#%%
def cal_normal(r, x):
    #r.shape (K,2), K is the number of integration points
    #x.shape is (M,7,3)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x)
    normal=shared.cal_normal(dx_du, dx_dv)
    return normal #(M,K,3)
#%% K integration points, K = 3 or 4 or 7
def cal_nodal_force_from_pressure(pressure, x, n_points):
    #x.shape (M,7,3)
    #pressure is scalar or tensor with shape (M, 1)
    #force_i=Integration(pressure*cross(dx_du, dx_dv)*shape_function_i at r), i=0 to 6
    #n_points: the number of integration points
    r=get_integration_point(n_points, x.dtype, x.device)
    dx_du, dx_dv=cal_dh_du_and_dh_dv(r, x) #(M,K,3), (M,K,3)
    weight=get_shape_integration_weight(n_points, x.dtype, x.device) #(7,K)
    force=shared.cal_nodal_force_from_pressure(pressure, dx_du, dx_dv, weight)
    return force #(M,7,3)
