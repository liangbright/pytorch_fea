# 20-node cubic tetrahedron: two nodes on each edge and a node at the centre of each face. TET20 in FEBio
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio
# location of the nodes in the unit element, in FEBio's node order:
#   node 0: (0, 0, 0)
#   node 1: (1, 0, 0)
#   node 2: (0, 1, 0)
#   node 3: (0, 0, 1)
#   node 4: (1/3, 0, 0)
#   node 5: (2/3, 0, 0)
#   node 6: (2/3, 1/3, 0)
#   node 7: (1/3, 2/3, 0)
#   node 8: (0, 1/3, 0)
#   node 9: (0, 2/3, 0)
#   node 10: (0, 0, 1/3)
#   node 11: (0, 0, 2/3)
#   node 12: (2/3, 0, 1/3)
#   node 13: (1/3, 0, 2/3)
#   node 14: (0, 2/3, 1/3)
#   node 15: (0, 1/3, 2/3)
#   node 16: (1/3, 0, 1/3)
#   node 17: (1/3, 1/3, 1/3)
#   node 18: (0, 1/3, 1/3)
#   node 19: (1/3, 1/3, 0)
# shape functions: the function space is
#   cubic polynomials
#   each shape function is 1 at its own node and 0 at the other nodes
#   they give the same values as FEBio's (FETet20::shape_fnc and shape_deriv)
# integration rules, with the points in FEBio's order:
#   15 points: exact up to degree 5 (FETet20G15)
# FEBio's source, at commit 49f0976 of the develop branch (2026-11-11):
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FEElementTraits.cpp
#   https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FESolidElementShape.cpp
#%%
import torch
import torch_fea.element.solid_element_shared as shared
#%% integration point location
def get_integration_point_15i(dtype, device):
    #15 points: exact up to degree 5
    #FEBio FETet20G15 has the same points with fewer digits (largest difference 4e-16)
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
    if n_points == 15:
        return get_integration_point_15i(dtype, device)
    else:
        raise ValueError("n_points must be 15")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 15:
        return get_integration_weight_15i(dtype, device)
    else:
        raise ValueError("n_points must be 15")
#%% useful for 9.15c in the book
def cal_shape_integration_weight(r, w, dtype, device):
    #r: list of the K integration points, w: list of the K integration weights
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (20,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9,
        sf10, sf11, sf12, sf13, sf14, sf15, sf16, sf17, sf18, sf19]
    weight=[]
    for i in range(0, 20):
        for j in range(0, len(w)):
            weight.append(sf[i](r[j])*w[j])
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(20,len(w))
    return weight
#%%
def get_shape_integration_weight_15i(dtype, device):
    #return: weight, weight.shape (20,15); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_15i(None, None), get_integration_weight_15i(None, None), dtype, device)
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (20,n_points); a list of its numbers when dtype or device is None
    if n_points == 15:
        return get_shape_integration_weight_15i(dtype, device)
    else:
        raise ValueError("n_points must be 15")
#%% shape function 0, node 0 is at (0, 0, 0)
def sf0(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] + r[1] + r[2] - 1)*(3*r[0] + 3*r[1] + 3*r[2] - 2)*(3*r[0] + 3*r[1] + 3*r[2] - 1)/2
#%%
def cal_d_sf0_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(27*r[0]**2 + 54*r[0]*r[1] + 54*r[0]*r[2] - 36*r[0] + 27*r[1]**2 + 54*r[1]*r[2] - 36*r[1] + 27*r[2]**2 - 36*r[2] + 11)/2
    dr1=-(27*r[0]**2 + 54*r[0]*r[1] + 54*r[0]*r[2] - 36*r[0] + 27*r[1]**2 + 54*r[1]*r[2] - 36*r[1] + 27*r[2]**2 - 36*r[2] + 11)/2
    dr2=-(27*r[0]**2 + 54*r[0]*r[1] + 54*r[0]*r[2] - 36*r[0] + 27*r[1]**2 + 54*r[1]*r[2] - 36*r[1] + 27*r[2]**2 - 36*r[2] + 11)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 1, node 1 is at (1, 0, 0)
def sf1(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*(3*r[0] - 2)*(3*r[0] - 1)/2
#%%
def cal_d_sf1_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(27*r[0]**2 - 18*r[0] + 2)/2
    dr1=0+0*r[0]
    dr2=0+0*r[0]
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 2, node 2 is at (0, 1, 0)
def sf2(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[1]*(3*r[1] - 2)*(3*r[1] - 1)/2
#%%
def cal_d_sf2_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=(27*r[1]**2 - 18*r[1] + 2)/2
    dr2=0+0*r[0]
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 3, node 3 is at (0, 0, 1)
def sf3(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[2]*(3*r[2] - 2)*(3*r[2] - 1)/2
#%%
def cal_d_sf3_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=0+0*r[0]
    dr2=(27*r[2]**2 - 18*r[2] + 2)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 4, node 4 is at (1/3, 0, 0)
def sf4(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 9*r[0]*(r[0] + r[1] + r[2] - 1)*(3*r[0] + 3*r[1] + 3*r[2] - 2)/2
#%%
def cal_d_sf4_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=9*(9*r[0]**2 + 12*r[0]*r[1] + 12*r[0]*r[2] - 10*r[0] + 3*r[1]**2 + 6*r[1]*r[2] - 5*r[1] + 3*r[2]**2 - 5*r[2] + 2)/2
    dr1=9*r[0]*(6*r[0] + 6*r[1] + 6*r[2] - 5)/2
    dr2=9*r[0]*(6*r[0] + 6*r[1] + 6*r[2] - 5)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 5, node 5 is at (2/3, 0, 0)
def sf5(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -9*r[0]*(3*r[0] - 1)*(r[0] + r[1] + r[2] - 1)/2
#%%
def cal_d_sf5_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-9*(9*r[0]**2 + 6*r[0]*r[1] + 6*r[0]*r[2] - 8*r[0] - r[1] - r[2] + 1)/2
    dr1=-9*r[0]*(3*r[0] - 1)/2
    dr2=-9*r[0]*(3*r[0] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 6, node 6 is at (2/3, 1/3, 0)
def sf6(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 9*r[0]*r[1]*(3*r[0] - 1)/2
#%%
def cal_d_sf6_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=9*r[1]*(6*r[0] - 1)/2
    dr1=9*r[0]*(3*r[0] - 1)/2
    dr2=0+0*r[0]
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 7, node 7 is at (1/3, 2/3, 0)
def sf7(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 9*r[0]*r[1]*(3*r[1] - 1)/2
#%%
def cal_d_sf7_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=9*r[1]*(3*r[1] - 1)/2
    dr1=9*r[0]*(6*r[1] - 1)/2
    dr2=0+0*r[0]
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 8, node 8 is at (0, 1/3, 0)
def sf8(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 9*r[1]*(r[0] + r[1] + r[2] - 1)*(3*r[0] + 3*r[1] + 3*r[2] - 2)/2
#%%
def cal_d_sf8_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=9*r[1]*(6*r[0] + 6*r[1] + 6*r[2] - 5)/2
    dr1=9*(3*r[0]**2 + 12*r[0]*r[1] + 6*r[0]*r[2] - 5*r[0] + 9*r[1]**2 + 12*r[1]*r[2] - 10*r[1] + 3*r[2]**2 - 5*r[2] + 2)/2
    dr2=9*r[1]*(6*r[0] + 6*r[1] + 6*r[2] - 5)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 9, node 9 is at (0, 2/3, 0)
def sf9(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -9*r[1]*(3*r[1] - 1)*(r[0] + r[1] + r[2] - 1)/2
#%%
def cal_d_sf9_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-9*r[1]*(3*r[1] - 1)/2
    dr1=-9*(6*r[0]*r[1] - r[0] + 9*r[1]**2 + 6*r[1]*r[2] - 8*r[1] - r[2] + 1)/2
    dr2=-9*r[1]*(3*r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 10, node 10 is at (0, 0, 1/3)
def sf10(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 9*r[2]*(r[0] + r[1] + r[2] - 1)*(3*r[0] + 3*r[1] + 3*r[2] - 2)/2
#%%
def cal_d_sf10_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=9*r[2]*(6*r[0] + 6*r[1] + 6*r[2] - 5)/2
    dr1=9*r[2]*(6*r[0] + 6*r[1] + 6*r[2] - 5)/2
    dr2=9*(3*r[0]**2 + 6*r[0]*r[1] + 12*r[0]*r[2] - 5*r[0] + 3*r[1]**2 + 12*r[1]*r[2] - 5*r[1] + 9*r[2]**2 - 10*r[2] + 2)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 11, node 11 is at (0, 0, 2/3)
def sf11(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -9*r[2]*(3*r[2] - 1)*(r[0] + r[1] + r[2] - 1)/2
#%%
def cal_d_sf11_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-9*r[2]*(3*r[2] - 1)/2
    dr1=-9*r[2]*(3*r[2] - 1)/2
    dr2=-9*(6*r[0]*r[2] - r[0] + 6*r[1]*r[2] - r[1] + 9*r[2]**2 - 8*r[2] + 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 12, node 12 is at (2/3, 0, 1/3)
def sf12(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 9*r[0]*r[2]*(3*r[0] - 1)/2
#%%
def cal_d_sf12_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=9*r[2]*(6*r[0] - 1)/2
    dr1=0+0*r[0]
    dr2=9*r[0]*(3*r[0] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 13, node 13 is at (1/3, 0, 2/3)
def sf13(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 9*r[0]*r[2]*(3*r[2] - 1)/2
#%%
def cal_d_sf13_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=9*r[2]*(3*r[2] - 1)/2
    dr1=0+0*r[0]
    dr2=9*r[0]*(6*r[2] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 14, node 14 is at (0, 2/3, 1/3)
def sf14(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 9*r[1]*r[2]*(3*r[1] - 1)/2
#%%
def cal_d_sf14_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=9*r[2]*(6*r[1] - 1)/2
    dr2=9*r[1]*(3*r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 15, node 15 is at (0, 1/3, 2/3)
def sf15(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 9*r[1]*r[2]*(3*r[2] - 1)/2
#%%
def cal_d_sf15_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=9*r[2]*(3*r[2] - 1)/2
    dr2=9*r[1]*(6*r[2] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 16, node 16 is at (1/3, 0, 1/3)
def sf16(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -27*r[0]*r[2]*(r[0] + r[1] + r[2] - 1)
#%%
def cal_d_sf16_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-27*r[2]*(2*r[0] + r[1] + r[2] - 1)
    dr1=-27*r[0]*r[2]
    dr2=-27*r[0]*(r[0] + r[1] + 2*r[2] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 17, node 17 is at (1/3, 1/3, 1/3)
def sf17(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 27*r[0]*r[1]*r[2]
#%%
def cal_d_sf17_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=27*r[1]*r[2]
    dr1=27*r[0]*r[2]
    dr2=27*r[0]*r[1]
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 18, node 18 is at (0, 1/3, 1/3)
def sf18(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -27*r[1]*r[2]*(r[0] + r[1] + r[2] - 1)
#%%
def cal_d_sf18_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-27*r[1]*r[2]
    dr1=-27*r[2]*(r[0] + 2*r[1] + r[2] - 1)
    dr2=-27*r[1]*(r[0] + r[1] + 2*r[2] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 19, node 19 is at (1/3, 1/3, 0)
def sf19(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -27*r[0]*r[1]*(r[0] + r[1] + r[2] - 1)
#%%
def cal_d_sf19_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-27*r[1]*(2*r[0] + r[1] + r[2] - 1)
    dr1=-27*r[0]*(r[0] + 2*r[1] + r[2] - 1)
    dr2=-27*r[0]*r[1]
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
    d_sf15_dr=torch.cat(cal_d_sf15_dr(r), dim=1)#shape is (K,3,1)
    d_sf16_dr=torch.cat(cal_d_sf16_dr(r), dim=1)#shape is (K,3,1)
    d_sf17_dr=torch.cat(cal_d_sf17_dr(r), dim=1)#shape is (K,3,1)
    d_sf18_dr=torch.cat(cal_d_sf18_dr(r), dim=1)#shape is (K,3,1)
    d_sf19_dr=torch.cat(cal_d_sf19_dr(r), dim=1)#shape is (K,3,1)
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr, d_sf6_dr, d_sf7_dr,
             d_sf8_dr, d_sf9_dr, d_sf10_dr, d_sf11_dr, d_sf12_dr, d_sf13_dr, d_sf14_dr, d_sf15_dr,
             d_sf16_dr, d_sf17_dr, d_sf18_dr, d_sf19_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,3,20)#20: sf0 to sf19
    return d_sf_dr #(1,K,3,20)
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,20,3), M elements, 1 elemnet has 20 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_i{sf_i(r)*h_i, i=0 to 19}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r), sf6(r), sf7(r),
                 sf8(r), sf9(r), sf10(r), sf11(r), sf12(r), sf13(r), sf14(r), sf15(r),
                 sf16(r), sf17(r), sf18(r), sf19(r)], dim=1)
    w=w.view(1,K,20,1)
    M=h.shape[0]
    h=h.view(M,1,20,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,20,3), M elements, 1 elemnet has 20 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,20)
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #shape (M,K,3,3)
#%%
def cal_d_sf_dh(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,20,3), M elements, 1 elemnet has 20 nodes, 1 node has a 3D position
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,20), K is the number of integration points
    d_sf_dh, dh_dr, det_dh_dr=shared.cal_d_sf_dh(h, d_sf_dr)
    #d_sf_dh.shape is (M,K,3,20), d_sf_dh[:,:,:,0] is d_sf0_dh
    #dh_dr.shape is (M,K,3,3), numerator_layout=False
    #det_dh_dr.shape is (M,K,1,1)
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,20,3)
    #X.shape is (M,20,3)
    F=shared.cal_F_tensor(x, X, cal_d_sf_dr(r))
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,20,3)
    #d_sf_dX.shape is (M,K,3,20), from cal_d_sf_dh
    F=shared.cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    return F #(M,K,3,3)
#%% K integration points, K = 15
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,20), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 19, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dx_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr, weight)
    return force #(M,20,3)
#%% K integration points, K = 15
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,20), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dX at r), i=0 to 19, #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr, weight)
    return force #(M,20,3)
#%% K integration points, K = 15
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,20), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(S*dNi_dX at r), i=0 to 19, #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight)
    return force #(M,20,3)
#%% K integration points, K = 15
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #det_dX_dr.shape is (M,K,1,1), det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(density*body_force*Ni at r), i=0 to 19, #(9.15c) in the book
    weight=get_shape_integration_weight(det_dX_dr.shape[1], det_dX_dr.dtype, det_dX_dr.device) #(20,K)
    force=shared.cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight)
    return force #(M,20,3)
#%% K integration points, K = 15, the strain energy of each element
def cal_strain_energy(W, det_dX_dr):
    #W.shape (M,K)
    #det_dX_dr.shape (M,K,1,1)
    #W=cal_strain_energy_density(F, Mat) on the integration points
    #energy=integration(W at r)
    weight=get_integration_weight(det_dX_dr.shape[1], W.dtype, torch.device("cpu"))
    energy=shared.cal_strain_energy(W, det_dX_dr, weight)
    return energy #(M,)
