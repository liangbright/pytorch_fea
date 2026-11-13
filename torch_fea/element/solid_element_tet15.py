# 15-node tetrahedron: tet10, a node at the centre of each face and a node at the centre of the element. TET15 in FEBio
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio
# location of the nodes in the unit element, in FEBio's node order:
#   node 0: (0, 0, 0)
#   node 1: (1, 0, 0)
#   node 2: (0, 1, 0)
#   node 3: (0, 0, 1)
#   node 4: (1/2, 0, 0)
#   node 5: (1/2, 1/2, 0)
#   node 6: (0, 1/2, 0)
#   node 7: (0, 0, 1/2)
#   node 8: (1/2, 0, 1/2)
#   node 9: (0, 1/2, 1/2)
#   node 10: (1/3, 1/3, 0)
#   node 11: (1/3, 0, 1/3)
#   node 12: (1/3, 1/3, 1/3)
#   node 13: (0, 1/3, 1/3)
#   node 14: (1/4, 1/4, 1/4)
# shape functions: the function space is
#   quadratic polynomials, the four face bubbles Li*Lj*Lk and the bubble L0*L1*L2*L3 (L: volume coordinates)
#   each shape function is 1 at its own node and 0 at the other nodes
#   they give the same values as FEBio's (FETet15::shape_fnc and shape_deriv)
# integration rules, with the points in FEBio's order:
#   4 points: exact up to degree 2 (FETet15G4)
#   8 points: exact up to degree 3 (FETet15G8)
#   11 points: exact up to degree 4, the weight of the first point is negative (FETet15G11)
#   15 points: exact up to degree 5 (FETet15G15)
# FEBio's source, at commit 49f0976 of the develop branch (2026-11-11):
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FEElementTraits.cpp
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FESolidElementShape.cpp
#%%
import torch
import torch_fea.element.solid_element_shared as shared
#%% integration point location
def get_integration_point_4i(dtype, device):
    #4 points: exact up to degree 2
    #FEBio FETet15G4 has the same points with fewer digits (largest difference 3e-09)
    r=[[0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
       [0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
       [0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
       [0.1381966011250105, 0.1381966011250105, 0.1381966011250105]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (4, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_4i(dtype, device):
    w=[0.041666666666666664, 0.041666666666666664, 0.041666666666666664, 0.041666666666666664]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (4,)
    return w
#%% integration point location
def get_integration_point_8i(dtype, device):
    #8 points: exact up to degree 3
    #FEBio FETet15G8 has the same points with fewer digits (largest difference 3e-10)
    r=[[0.01583590986571992, 0.3280546967114267, 0.3280546967114267],
       [0.3280546967114267, 0.01583590986571992, 0.3280546967114267],
       [0.3280546967114267, 0.3280546967114267, 0.01583590986571992],
       [0.3280546967114267, 0.3280546967114267, 0.3280546967114267],
       [0.679143178201208, 0.10695227393293069, 0.10695227393293069],
       [0.10695227393293069, 0.679143178201208, 0.10695227393293069],
       [0.10695227393293069, 0.10695227393293069, 0.679143178201208],
       [0.10695227393293069, 0.10695227393293069, 0.10695227393293069]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (8, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_8i(dtype, device):
    w=[0.02308799441864369, 0.02308799441864369, 0.02308799441864369, 0.02308799441864369,
       0.018578672248022975, 0.018578672248022975, 0.018578672248022975, 0.018578672248022975]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (8,)
    return w
#%% integration point location
def get_integration_point_11i(dtype, device):
    #11 points: exact up to degree 4, the weight of the first point is negative
    #FEBio FETet15G11 has the same points with fewer digits (largest difference 4e-12)
    r=[[0.25, 0.25, 0.25],
       [0.07142857142857136, 0.07142857142857136, 0.07142857142857136],
       [0.7857142857142856, 0.07142857142857144, 0.07142857142857144],
       [0.07142857142857144, 0.7857142857142856, 0.07142857142857144],
       [0.07142857142857144, 0.07142857142857144, 0.7857142857142856],
       [0.3994035761667991, 0.10059642383320079, 0.10059642383320079],
       [0.10059642383320079, 0.3994035761667991, 0.10059642383320079],
       [0.10059642383320079, 0.10059642383320079, 0.3994035761667991],
       [0.39940357616679917, 0.39940357616679917, 0.10059642383320079],
       [0.39940357616679917, 0.10059642383320079, 0.39940357616679917],
       [0.10059642383320079, 0.39940357616679917, 0.39940357616679917]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (11, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_11i(dtype, device):
    w=[-0.013155555555555556, 0.007622222222222208, 0.007622222222222227, 0.007622222222222227,
       0.007622222222222227, 0.024888888888888884, 0.024888888888888884, 0.024888888888888884,
       0.024888888888888894, 0.024888888888888894, 0.024888888888888894]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (11,)
    return w
#%% integration point location
def get_integration_point_15i(dtype, device):
    #15 points: exact up to degree 5
    #FEBio FETet15G15 has the same points with fewer digits (largest difference 4e-16)
    r=[[0.25, 0.25, 0.25],
       [0.33333333333333326, 0.33333333333333326, 0.33333333333333326],
       [4.3336562608475714e-16, 0.33333333333333304, 0.33333333333333304],
       [0.33333333333333304, 4.3336562608475714e-16, 0.33333333333333304],
       [0.33333333333333304, 0.33333333333333304, 4.3336562608475714e-16],
       [0.09090909090909093, 0.09090909090909093, 0.09090909090909093],
       [0.7272727272727273, 0.09090909090909093, 0.09090909090909093],
       [0.09090909090909093, 0.7272727272727273, 0.09090909090909093],
       [0.09090909090909093, 0.09090909090909093, 0.7272727272727273],
       [0.4334498464263359, 0.0665501535736642, 0.0665501535736642],
       [0.0665501535736642, 0.4334498464263359, 0.0665501535736642],
       [0.0665501535736642, 0.0665501535736642, 0.4334498464263359],
       [0.06655015357366417, 0.43344984642633577, 0.43344984642633577],
       [0.43344984642633577, 0.06655015357366417, 0.43344984642633577],
       [0.43344984642633577, 0.43344984642633577, 0.06655015357366417]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (15, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_15i(dtype, device):
    w=[0.030283678097089133, 0.006026785714285748, 0.006026785714285749, 0.006026785714285749,
       0.006026785714285749, 0.011645249086028975, 0.011645249086028972, 0.011645249086028972,
       0.011645249086028972, 0.010949141561386436, 0.010949141561386436, 0.010949141561386436,
       0.010949141561386447, 0.010949141561386447, 0.010949141561386447]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (15,)
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,3); a list of the points when dtype or device is None
    if n_points == 4:
        return get_integration_point_4i(dtype, device)
    elif n_points == 8:
        return get_integration_point_8i(dtype, device)
    elif n_points == 11:
        return get_integration_point_11i(dtype, device)
    elif n_points == 15:
        return get_integration_point_15i(dtype, device)
    else:
        raise ValueError("n_points must be 4 or 8 or 11 or 15")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 4:
        return get_integration_weight_4i(dtype, device)
    elif n_points == 8:
        return get_integration_weight_8i(dtype, device)
    elif n_points == 11:
        return get_integration_weight_11i(dtype, device)
    elif n_points == 15:
        return get_integration_weight_15i(dtype, device)
    else:
        raise ValueError("n_points must be 4 or 8 or 11 or 15")
#%% useful for 9.15c in the book
def cal_shape_integration_weight(r, w, dtype, device):
    #r: list of the K integration points, w: list of the K integration weights
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (15,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9,
        sf10, sf11, sf12, sf13, sf14]
    weight=[]
    for i in range(0, 15):
        for j in range(0, len(w)):
            weight.append(sf[i](r[j])*w[j])
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(15,len(w))
    return weight
#%%
def get_shape_integration_weight_4i(dtype, device):
    #return: weight, weight.shape (15,4); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_4i(None, None), get_integration_weight_4i(None, None), dtype, device)
#%%
def get_shape_integration_weight_8i(dtype, device):
    #return: weight, weight.shape (15,8); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_8i(None, None), get_integration_weight_8i(None, None), dtype, device)
#%%
def get_shape_integration_weight_11i(dtype, device):
    #return: weight, weight.shape (15,11); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_11i(None, None), get_integration_weight_11i(None, None), dtype, device)
#%%
def get_shape_integration_weight_15i(dtype, device):
    #return: weight, weight.shape (15,15); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_15i(None, None), get_integration_weight_15i(None, None), dtype, device)
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (15,n_points); a list of its numbers when dtype or device is None
    if n_points == 4:
        return get_shape_integration_weight_4i(dtype, device)
    elif n_points == 8:
        return get_shape_integration_weight_8i(dtype, device)
    elif n_points == 11:
        return get_shape_integration_weight_11i(dtype, device)
    elif n_points == 15:
        return get_shape_integration_weight_15i(dtype, device)
    else:
        raise ValueError("n_points must be 4 or 8 or 11 or 15")
#%% shape function 0, node 0 is at (0, 0, 0)
def sf0(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] + r[1] + r[2] - 1)*(4*r[0]*r[1]*r[2] - 3*r[0]*r[1] - 3*r[0]*r[2] + 2*r[0] - 3*r[1]*r[2] + 2*r[1] + 2*r[2] - 1)
#%%
def cal_d_sf0_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=8*r[0]*r[1]*r[2] - 6*r[0]*r[1] - 6*r[0]*r[2] + 4*r[0] + 4*r[1]**2*r[2] - 3*r[1]**2 + 4*r[1]*r[2]**2 - 13*r[1]*r[2] + 7*r[1] - 3*r[2]**2 + 7*r[2] - 3
    dr1=4*r[0]**2*r[2] - 3*r[0]**2 + 8*r[0]*r[1]*r[2] - 6*r[0]*r[1] + 4*r[0]*r[2]**2 - 13*r[0]*r[2] + 7*r[0] - 6*r[1]*r[2] + 4*r[1] - 3*r[2]**2 + 7*r[2] - 3
    dr2=4*r[0]**2*r[1] - 3*r[0]**2 + 4*r[0]*r[1]**2 + 8*r[0]*r[1]*r[2] - 13*r[0]*r[1] - 6*r[0]*r[2] + 7*r[0] - 3*r[1]**2 - 6*r[1]*r[2] + 7*r[1] + 4*r[2] - 3
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 1, node 1 is at (1, 0, 0)
def sf1(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*(4*r[0]*r[1]*r[2] - 3*r[0]*r[1] - 3*r[0]*r[2] + 2*r[0] + 4*r[1]**2*r[2] - 3*r[1]**2 + 4*r[1]*r[2]**2 - 7*r[1]*r[2] + 3*r[1] - 3*r[2]**2 + 3*r[2] - 1)
#%%
def cal_d_sf1_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=8*r[0]*r[1]*r[2] - 6*r[0]*r[1] - 6*r[0]*r[2] + 4*r[0] + 4*r[1]**2*r[2] - 3*r[1]**2 + 4*r[1]*r[2]**2 - 7*r[1]*r[2] + 3*r[1] - 3*r[2]**2 + 3*r[2] - 1
    dr1=r[0]*(4*r[2] - 3)*(r[0] + 2*r[1] + r[2] - 1)
    dr2=r[0]*(4*r[1] - 3)*(r[0] + r[1] + 2*r[2] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 2, node 2 is at (0, 1, 0)
def sf2(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[1]*(4*r[0]**2*r[2] - 3*r[0]**2 + 4*r[0]*r[1]*r[2] - 3*r[0]*r[1] + 4*r[0]*r[2]**2 - 7*r[0]*r[2] + 3*r[0] - 3*r[1]*r[2] + 2*r[1] - 3*r[2]**2 + 3*r[2] - 1)
#%%
def cal_d_sf2_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=r[1]*(4*r[2] - 3)*(2*r[0] + r[1] + r[2] - 1)
    dr1=4*r[0]**2*r[2] - 3*r[0]**2 + 8*r[0]*r[1]*r[2] - 6*r[0]*r[1] + 4*r[0]*r[2]**2 - 7*r[0]*r[2] + 3*r[0] - 6*r[1]*r[2] + 4*r[1] - 3*r[2]**2 + 3*r[2] - 1
    dr2=r[1]*(4*r[0] - 3)*(r[0] + r[1] + 2*r[2] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 3, node 3 is at (0, 0, 1)
def sf3(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[2]*(4*r[0]**2*r[1] - 3*r[0]**2 + 4*r[0]*r[1]**2 + 4*r[0]*r[1]*r[2] - 7*r[0]*r[1] - 3*r[0]*r[2] + 3*r[0] - 3*r[1]**2 - 3*r[1]*r[2] + 3*r[1] + 2*r[2] - 1)
#%%
def cal_d_sf3_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=r[2]*(4*r[1] - 3)*(2*r[0] + r[1] + r[2] - 1)
    dr1=r[2]*(4*r[0] - 3)*(r[0] + 2*r[1] + r[2] - 1)
    dr2=4*r[0]**2*r[1] - 3*r[0]**2 + 4*r[0]*r[1]**2 + 8*r[0]*r[1]*r[2] - 7*r[0]*r[1] - 6*r[0]*r[2] + 3*r[0] - 3*r[1]**2 - 6*r[1]*r[2] + 3*r[1] + 4*r[2] - 1
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 4, node 4 is at (1/2, 0, 0)
def sf4(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -4*r[0]*(r[0] + r[1] + r[2] - 1)*(8*r[1]*r[2] - 3*r[1] - 3*r[2] + 1)
#%%
def cal_d_sf4_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-4*(2*r[0] + r[1] + r[2] - 1)*(8*r[1]*r[2] - 3*r[1] - 3*r[2] + 1)
    dr1=-4*r[0]*(8*r[0]*r[2] - 3*r[0] + 16*r[1]*r[2] - 6*r[1] + 8*r[2]**2 - 14*r[2] + 4)
    dr2=-4*r[0]*(8*r[0]*r[1] - 3*r[0] + 8*r[1]**2 + 16*r[1]*r[2] - 14*r[1] - 6*r[2] + 4)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 5, node 5 is at (1/2, 1/2, 0)
def sf5(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -4*r[0]*r[1]*(8*r[0]*r[2] - 3*r[0] + 8*r[1]*r[2] - 3*r[1] + 8*r[2]**2 - 8*r[2] + 2)
#%%
def cal_d_sf5_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-4*r[1]*(16*r[0]*r[2] - 6*r[0] + 8*r[1]*r[2] - 3*r[1] + 8*r[2]**2 - 8*r[2] + 2)
    dr1=-4*r[0]*(8*r[0]*r[2] - 3*r[0] + 16*r[1]*r[2] - 6*r[1] + 8*r[2]**2 - 8*r[2] + 2)
    dr2=-32*r[0]*r[1]*(r[0] + r[1] + 2*r[2] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 6, node 6 is at (0, 1/2, 0)
def sf6(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -4*r[1]*(r[0] + r[1] + r[2] - 1)*(8*r[0]*r[2] - 3*r[0] - 3*r[2] + 1)
#%%
def cal_d_sf6_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-4*r[1]*(16*r[0]*r[2] - 6*r[0] + 8*r[1]*r[2] - 3*r[1] + 8*r[2]**2 - 14*r[2] + 4)
    dr1=-4*(r[0] + 2*r[1] + r[2] - 1)*(8*r[0]*r[2] - 3*r[0] - 3*r[2] + 1)
    dr2=-4*r[1]*(8*r[0]**2 + 8*r[0]*r[1] + 16*r[0]*r[2] - 14*r[0] - 3*r[1] - 6*r[2] + 4)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 7, node 7 is at (0, 0, 1/2)
def sf7(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -4*r[2]*(r[0] + r[1] + r[2] - 1)*(8*r[0]*r[1] - 3*r[0] - 3*r[1] + 1)
#%%
def cal_d_sf7_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-4*r[2]*(16*r[0]*r[1] - 6*r[0] + 8*r[1]**2 + 8*r[1]*r[2] - 14*r[1] - 3*r[2] + 4)
    dr1=-4*r[2]*(8*r[0]**2 + 16*r[0]*r[1] + 8*r[0]*r[2] - 14*r[0] - 6*r[1] - 3*r[2] + 4)
    dr2=-4*(r[0] + r[1] + 2*r[2] - 1)*(8*r[0]*r[1] - 3*r[0] - 3*r[1] + 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 8, node 8 is at (1/2, 0, 1/2)
def sf8(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -4*r[0]*r[2]*(8*r[0]*r[1] - 3*r[0] + 8*r[1]**2 + 8*r[1]*r[2] - 8*r[1] - 3*r[2] + 2)
#%%
def cal_d_sf8_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-4*r[2]*(16*r[0]*r[1] - 6*r[0] + 8*r[1]**2 + 8*r[1]*r[2] - 8*r[1] - 3*r[2] + 2)
    dr1=-32*r[0]*r[2]*(r[0] + 2*r[1] + r[2] - 1)
    dr2=-4*r[0]*(8*r[0]*r[1] - 3*r[0] + 8*r[1]**2 + 16*r[1]*r[2] - 8*r[1] - 6*r[2] + 2)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 9, node 9 is at (0, 1/2, 1/2)
def sf9(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -4*r[1]*r[2]*(8*r[0]**2 + 8*r[0]*r[1] + 8*r[0]*r[2] - 8*r[0] - 3*r[1] - 3*r[2] + 2)
#%%
def cal_d_sf9_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-32*r[1]*r[2]*(2*r[0] + r[1] + r[2] - 1)
    dr1=-4*r[2]*(8*r[0]**2 + 16*r[0]*r[1] + 8*r[0]*r[2] - 8*r[0] - 6*r[1] - 3*r[2] + 2)
    dr2=-4*r[1]*(8*r[0]**2 + 8*r[0]*r[1] + 16*r[0]*r[2] - 8*r[0] - 3*r[1] - 6*r[2] + 2)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 10, node 10 is at (1/3, 1/3, 0)
def sf10(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 27*r[0]*r[1]*(4*r[2] - 1)*(r[0] + r[1] + r[2] - 1)
#%%
def cal_d_sf10_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=27*r[1]*(4*r[2] - 1)*(2*r[0] + r[1] + r[2] - 1)
    dr1=27*r[0]*(4*r[2] - 1)*(r[0] + 2*r[1] + r[2] - 1)
    dr2=27*r[0]*r[1]*(4*r[0] + 4*r[1] + 8*r[2] - 5)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 11, node 11 is at (1/3, 0, 1/3)
def sf11(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 27*r[0]*r[2]*(4*r[1] - 1)*(r[0] + r[1] + r[2] - 1)
#%%
def cal_d_sf11_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=27*r[2]*(4*r[1] - 1)*(2*r[0] + r[1] + r[2] - 1)
    dr1=27*r[0]*r[2]*(4*r[0] + 8*r[1] + 4*r[2] - 5)
    dr2=27*r[0]*(4*r[1] - 1)*(r[0] + r[1] + 2*r[2] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 12, node 12 is at (1/3, 1/3, 1/3)
def sf12(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 27*r[0]*r[1]*r[2]*(4*r[0] + 4*r[1] + 4*r[2] - 3)
#%%
def cal_d_sf12_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=27*r[1]*r[2]*(8*r[0] + 4*r[1] + 4*r[2] - 3)
    dr1=27*r[0]*r[2]*(4*r[0] + 8*r[1] + 4*r[2] - 3)
    dr2=27*r[0]*r[1]*(4*r[0] + 4*r[1] + 8*r[2] - 3)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 13, node 13 is at (0, 1/3, 1/3)
def sf13(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 27*r[1]*r[2]*(4*r[0] - 1)*(r[0] + r[1] + r[2] - 1)
#%%
def cal_d_sf13_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=27*r[1]*r[2]*(8*r[0] + 4*r[1] + 4*r[2] - 5)
    dr1=27*r[2]*(4*r[0] - 1)*(r[0] + 2*r[1] + r[2] - 1)
    dr2=27*r[1]*(4*r[0] - 1)*(r[0] + r[1] + 2*r[2] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 14, node 14 is at (1/4, 1/4, 1/4)
def sf14(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -256*r[0]*r[1]*r[2]*(r[0] + r[1] + r[2] - 1)
#%%
def cal_d_sf14_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-256*r[1]*r[2]*(2*r[0] + r[1] + r[2] - 1)
    dr1=-256*r[0]*r[2]*(r[0] + 2*r[1] + r[2] - 1)
    dr2=-256*r[0]*r[1]*(r[0] + r[1] + 2*r[2] - 1)
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
    d_sf6_dr=torch.cat(cal_d_sf6_dr(r), dim=1)#shape is (K,3,1)
    d_sf7_dr=torch.cat(cal_d_sf7_dr(r), dim=1)#shape is (K,3,1)
    d_sf8_dr=torch.cat(cal_d_sf8_dr(r), dim=1)#shape is (K,3,1)
    d_sf9_dr=torch.cat(cal_d_sf9_dr(r), dim=1)#shape is (K,3,1)
    d_sf10_dr=torch.cat(cal_d_sf10_dr(r), dim=1)#shape is (K,3,1)
    d_sf11_dr=torch.cat(cal_d_sf11_dr(r), dim=1)#shape is (K,3,1)
    d_sf12_dr=torch.cat(cal_d_sf12_dr(r), dim=1)#shape is (K,3,1)
    d_sf13_dr=torch.cat(cal_d_sf13_dr(r), dim=1)#shape is (K,3,1)
    d_sf14_dr=torch.cat(cal_d_sf14_dr(r), dim=1)#shape is (K,3,1)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr, d_sf6_dr, d_sf7_dr,
             d_sf8_dr, d_sf9_dr, d_sf10_dr, d_sf11_dr, d_sf12_dr, d_sf13_dr, d_sf14_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,3,15)#15: sf0 to sf14
    return d_sf_dr #(1,K,3,15)
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,15,3), M elements, 1 elemnet has 15 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_i{sf_i(r)*h_i, i=0 to 14}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r), sf6(r), sf7(r),
                 sf8(r), sf9(r), sf10(r), sf11(r), sf12(r), sf13(r), sf14(r)], dim=1)
    w=w.view(1,K,15,1)
    M=h.shape[0]
    h=h.view(M,1,15,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,15,3), M elements, 1 elemnet has 15 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,15)
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #shape (M,K,3,3)
#%%
def cal_d_sf_dh(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,15,3), M elements, 1 elemnet has 15 nodes, 1 node has a 3D position
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,15), K is the number of integration points
    d_sf_dh, dh_dr, det_dh_dr=shared.cal_d_sf_dh(h, d_sf_dr)
    #d_sf_dh.shape is (M,K,3,15), d_sf_dh[:,:,:,0] is d_sf0_dh
    #dh_dr.shape is (M,K,3,3), numerator_layout=False
    #det_dh_dr.shape is (M,K,1,1)
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,15,3)
    #X.shape is (M,15,3)
    F=shared.cal_F_tensor(x, X, cal_d_sf_dr(r))
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,15,3)
    #d_sf_dX.shape is (M,K,3,15), from cal_d_sf_dh
    F=shared.cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    return F #(M,K,3,3)
#%% K integration points, K = 4 or 8 or 11 or 15
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,15), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 14, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dx_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr, weight)
    return force #(M,15,3)
#%% K integration points, K = 4 or 8 or 11 or 15
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,15), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dX at r), i=0 to 14, #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr, weight)
    return force #(M,15,3)
#%% K integration points, K = 4 or 8 or 11 or 15
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,15), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(S*dNi_dX at r), i=0 to 14, #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight)
    return force #(M,15,3)
#%% K integration points, K = 4 or 8 or 11 or 15
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #det_dX_dr.shape is (M,K,1,1), det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(density*body_force*Ni at r), i=0 to 14, #(9.15c) in the book
    weight=get_shape_integration_weight(det_dX_dr.shape[1], det_dX_dr.dtype, det_dX_dr.device) #(15,K)
    force=shared.cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight)
    return force #(M,15,3)
#%% K integration points, K = 4 or 8 or 11 or 15, the strain energy of each element
def cal_strain_energy(W, det_dX_dr):
    #W.shape (M,K)
    #det_dX_dr.shape (M,K,1,1)
    #W=cal_strain_energy_density(F, Mat) on the integration points
    #energy=integration(W at r)
    weight=get_integration_weight(det_dX_dr.shape[1], W.dtype, torch.device("cpu"))
    energy=shared.cal_strain_energy(W, det_dX_dr, weight)
    return energy #(M,)
