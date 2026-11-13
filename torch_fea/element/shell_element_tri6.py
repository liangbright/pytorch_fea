# 6-node quadratic triangle shell element. "tri6" in a shell domain of FEBio (FEShellTri6_, FEShellTri6G14, FEShellTri6G21)
# FEBio's shell (FESSIShellDomain, FEElasticShellDomain): a shell node sits on the front face of the shell, and a
#   second point on the back face belongs to it: back = node - director. The position inside the shell is
#       x(r) = sum_i M_i(r[0],r[1]) * (x_i - (1-r[2])/2 * director_i)
#   so the shell is a solid that is linear through the thickness, with 12 points:
#       point 0 to 5: the shell nodes (front face, r[2]=1), in FEBio's node order
#       point 6 to 11: the back points (r[2]=-1), point 6+i belongs to shell node i
#   This file has the functions of a solid_element file for these 12 points:
#       h in the functions below has the shape (M,12,3); cal_front_and_back_node makes it from nodes and directors
#       a nodal force has the same shape: force[:,0:6] acts on the shell nodes (the displacement u in FEBio)
#       and force[:,6:12] on the back points (the shell displacement w in FEBio)
#   The stress comes from a 3D material with the deformation gradient F, as for a solid element.
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio, [r[0], r[1]] = [u, v] in the surface elements
# location of the shell nodes in (r[0], r[1]), in FEBio's node order:
#   shell node 0: (0, 0)
#   shell node 1: (1, 0)
#   shell node 2: (0, 1)
#   shell node 3: (0.5, 0)
#   shell node 4: (0.5, 0.5)
#   shell node 5: (0, 0.5)
# shape functions: sf_i = M_i*(1+r[2])/2 for shell node i, sf_(6+i) = M_i*(1-r[2])/2 for its back point
#   M_i: quadratic polynomials of u, v; 1 at its own node and 0 at the other nodes
#   M_i gives the same values as FEBio's (FEShellTri6_::shape_fnc and shape_deriv)
# integration rules, with the points in FEBio's order (the first one is FEBio's default):
#   14 points: the 7-point triangle rule, times 2 Gauss points through the thickness (FEShellTri6G14)
#   21 points: the 7-point triangle rule, times 3 Gauss points through the thickness (FEShellTri6G21)
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
def get_integration_point_14i(dtype, device):
    #14 points: the 7-point triangle rule in (r[0], r[1]),
    #  times 2 Gauss points in r[2] at +-1/sqrt(3); the in-plane index changes fastest
    #the points and weights of FEBio FEShellTri6G14
    r=[[0.3333333333333333, 0.3333333333333333, -0.5773502691896258],
       [0.7974269853530873, 0.10128650732345634, -0.5773502691896258],
       [0.10128650732345634, 0.7974269853530873, -0.5773502691896258],
       [0.10128650732345634, 0.10128650732345634, -0.5773502691896258],
       [0.4701420641051151, 0.4701420641051151, -0.5773502691896258],
       [0.4701420641051151, 0.05971587178976982, -0.5773502691896258],
       [0.05971587178976982, 0.4701420641051151, -0.5773502691896258],
       [0.3333333333333333, 0.3333333333333333, 0.5773502691896258],
       [0.7974269853530873, 0.10128650732345634, 0.5773502691896258],
       [0.10128650732345634, 0.7974269853530873, 0.5773502691896258],
       [0.10128650732345634, 0.10128650732345634, 0.5773502691896258],
       [0.4701420641051151, 0.4701420641051151, 0.5773502691896258],
       [0.4701420641051151, 0.05971587178976982, 0.5773502691896258],
       [0.05971587178976982, 0.4701420641051151, 0.5773502691896258]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (14, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_14i(dtype, device):
    w=[0.1125, 0.06296959027241357, 0.06296959027241357, 0.06296959027241357,
       0.0661970763942531, 0.0661970763942531, 0.0661970763942531, 0.1125,
       0.06296959027241357, 0.06296959027241357, 0.06296959027241357, 0.0661970763942531,
       0.0661970763942531, 0.0661970763942531]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (14,)
    return w
#%% integration point location
def get_integration_point_21i(dtype, device):
    #21 points: the 7-point triangle rule in (r[0], r[1]),
    #  times 3 Gauss points in r[2] at -b, 0, b with b=sqrt(3/5); the in-plane index changes fastest
    #the points and weights of FEBio FEShellTri6G21
    r=[[0.3333333333333333, 0.3333333333333333, -0.7745966692414834],
       [0.7974269853530873, 0.10128650732345634, -0.7745966692414834],
       [0.10128650732345634, 0.7974269853530873, -0.7745966692414834],
       [0.10128650732345634, 0.10128650732345634, -0.7745966692414834],
       [0.4701420641051151, 0.4701420641051151, -0.7745966692414834],
       [0.4701420641051151, 0.05971587178976982, -0.7745966692414834],
       [0.05971587178976982, 0.4701420641051151, -0.7745966692414834],
       [0.3333333333333333, 0.3333333333333333, 0.0],
       [0.7974269853530873, 0.10128650732345634, 0.0],
       [0.10128650732345634, 0.7974269853530873, 0.0],
       [0.10128650732345634, 0.10128650732345634, 0.0],
       [0.4701420641051151, 0.4701420641051151, 0.0],
       [0.4701420641051151, 0.05971587178976982, 0.0],
       [0.05971587178976982, 0.4701420641051151, 0.0],
       [0.3333333333333333, 0.3333333333333333, 0.7745966692414834],
       [0.7974269853530873, 0.10128650732345634, 0.7745966692414834],
       [0.10128650732345634, 0.7974269853530873, 0.7745966692414834],
       [0.10128650732345634, 0.10128650732345634, 0.7745966692414834],
       [0.4701420641051151, 0.4701420641051151, 0.7745966692414834],
       [0.4701420641051151, 0.05971587178976982, 0.7745966692414834],
       [0.05971587178976982, 0.4701420641051151, 0.7745966692414834]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (21, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_21i(dtype, device):
    w=[0.0625, 0.03498310570689643, 0.03498310570689643, 0.03498310570689643,
       0.036776153552362835, 0.036776153552362835, 0.036776153552362835, 0.09999999999999999,
       0.05597296913103428, 0.05597296913103428, 0.05597296913103428, 0.05884184568378053,
       0.05884184568378053, 0.05884184568378053, 0.0625, 0.03498310570689643,
       0.03498310570689643, 0.03498310570689643, 0.036776153552362835, 0.036776153552362835,
       0.036776153552362835]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (21,)
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,3); a list of the points when dtype or device is None
    if n_points == 14:
        return get_integration_point_14i(dtype, device)
    elif n_points == 21:
        return get_integration_point_21i(dtype, device)
    else:
        raise ValueError("n_points must be 14 or 21")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 14:
        return get_integration_weight_14i(dtype, device)
    elif n_points == 21:
        return get_integration_weight_21i(dtype, device)
    else:
        raise ValueError("n_points must be 14 or 21")
#%% useful for 9.15c in the book
def cal_shape_integration_weight(r, w, dtype, device):
    #r: list of the K integration points, w: list of the K integration weights
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (12,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9,
        sf10, sf11]
    weight=[]
    for i in range(0, 12):
        for j in range(0, len(w)):
            weight.append(sf[i](r[j])*w[j])
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(12,len(w))
    return weight
#%%
def get_shape_integration_weight_14i(dtype, device):
    #return: weight, weight.shape (12,14); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_14i(None, None), get_integration_weight_14i(None, None), dtype, device)
#%%
def get_shape_integration_weight_21i(dtype, device):
    #return: weight, weight.shape (12,21); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_21i(None, None), get_integration_weight_21i(None, None), dtype, device)
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (12,n_points); a list of its numbers when dtype or device is None
    if n_points == 14:
        return get_shape_integration_weight_14i(dtype, device)
    elif n_points == 21:
        return get_shape_integration_weight_21i(dtype, device)
    else:
        raise ValueError("n_points must be 14 or 21")
#%% shape function 0: shell node 0, front face, at (0, 0, 1)
def sf0(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[2] + 1)*(r[0] + r[1] - 1)*(2*r[0] + 2*r[1] - 1)/2
#%%
def cal_d_sf0_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[2] + 1)*(4*r[0] + 4*r[1] - 3)/2
    dr1=(r[2] + 1)*(4*r[0] + 4*r[1] - 3)/2
    dr2=(r[0] + r[1] - 1)*(2*r[0] + 2*r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 1: shell node 1, front face, at (1, 0, 1)
def sf1(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*(2*r[0] - 1)*(r[2] + 1)/2
#%%
def cal_d_sf1_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(4*r[0] - 1)*(r[2] + 1)/2
    dr1=0+0*r[0]
    dr2=r[0]*(2*r[0] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 2: shell node 2, front face, at (0, 1, 1)
def sf2(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[1]*(2*r[1] - 1)*(r[2] + 1)/2
#%%
def cal_d_sf2_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=(4*r[1] - 1)*(r[2] + 1)/2
    dr2=r[1]*(2*r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 3: shell node 3, front face, at (0.5, 0, 1)
def sf3(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -2*r[0]*(r[2] + 1)*(r[0] + r[1] - 1)
#%%
def cal_d_sf3_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-2*(r[2] + 1)*(2*r[0] + r[1] - 1)
    dr1=-2*r[0]*(r[2] + 1)
    dr2=-2*r[0]*(r[0] + r[1] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 4: shell node 4, front face, at (0.5, 0.5, 1)
def sf4(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 2*r[0]*r[1]*(r[2] + 1)
#%%
def cal_d_sf4_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=2*r[1]*(r[2] + 1)
    dr1=2*r[0]*(r[2] + 1)
    dr2=2*r[0]*r[1]
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 5: shell node 5, front face, at (0, 0.5, 1)
def sf5(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -2*r[1]*(r[2] + 1)*(r[0] + r[1] - 1)
#%%
def cal_d_sf5_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-2*r[1]*(r[2] + 1)
    dr1=-2*(r[2] + 1)*(r[0] + 2*r[1] - 1)
    dr2=-2*r[1]*(r[0] + r[1] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 6: back point of shell node 0, at (0, 0, -1)
def sf6(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[2] - 1)*(r[0] + r[1] - 1)*(2*r[0] + 2*r[1] - 1)/2
#%%
def cal_d_sf6_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[2] - 1)*(4*r[0] + 4*r[1] - 3)/2
    dr1=-(r[2] - 1)*(4*r[0] + 4*r[1] - 3)/2
    dr2=-(r[0] + r[1] - 1)*(2*r[0] + 2*r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 7: back point of shell node 1, at (1, 0, -1)
def sf7(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[0]*(2*r[0] - 1)*(r[2] - 1)/2
#%%
def cal_d_sf7_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(4*r[0] - 1)*(r[2] - 1)/2
    dr1=0+0*r[0]
    dr2=-r[0]*(2*r[0] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 8: back point of shell node 2, at (0, 1, -1)
def sf8(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -r[1]*(2*r[1] - 1)*(r[2] - 1)/2
#%%
def cal_d_sf8_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=-(4*r[1] - 1)*(r[2] - 1)/2
    dr2=-r[1]*(2*r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 9: back point of shell node 3, at (0.5, 0, -1)
def sf9(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 2*r[0]*(r[2] - 1)*(r[0] + r[1] - 1)
#%%
def cal_d_sf9_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=2*(r[2] - 1)*(2*r[0] + r[1] - 1)
    dr1=2*r[0]*(r[2] - 1)
    dr2=2*r[0]*(r[0] + r[1] - 1)
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 10: back point of shell node 4, at (0.5, 0.5, -1)
def sf10(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -2*r[0]*r[1]*(r[2] - 1)
#%%
def cal_d_sf10_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-2*r[1]*(r[2] - 1)
    dr1=-2*r[0]*(r[2] - 1)
    dr2=-2*r[0]*r[1]
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 11: back point of shell node 5, at (0, 0.5, -1)
def sf11(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 2*r[1]*(r[2] - 1)*(r[0] + r[1] - 1)
#%%
def cal_d_sf11_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=2*r[1]*(r[2] - 1)
    dr1=2*(r[2] - 1)*(r[0] + 2*r[1] - 1)
    dr2=2*r[1]*(r[0] + r[1] - 1)
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
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr, d_sf6_dr, d_sf7_dr,
             d_sf8_dr, d_sf9_dr, d_sf10_dr, d_sf11_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,3,12)#12: sf0 to sf11
    return d_sf_dr #(1,K,3,12)
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,12,3), M elements, 6 shell nodes and then their 6 back points, h is x or X or u
    #hr=sum_i{sf_i(r)*h_i, i=0 to 11}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r), sf6(r), sf7(r),
                 sf8(r), sf9(r), sf10(r), sf11(r)], dim=1)
    w=w.view(1,K,12,1)
    M=h.shape[0]
    h=h.view(M,1,12,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,12,3), M elements, 6 shell nodes and then their 6 back points, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,12)
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #shape (M,K,3,3)
#%%
def cal_d_sf_dh(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,12,3), M elements, 6 shell nodes and then their 6 back points, 1 node has a 3D position
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,12), K is the number of integration points
    d_sf_dh, dh_dr, det_dh_dr=shared.cal_d_sf_dh(h, d_sf_dr)
    #d_sf_dh.shape is (M,K,3,12), d_sf_dh[:,:,:,0] is d_sf0_dh
    #dh_dr.shape is (M,K,3,3), numerator_layout=False
    #det_dh_dr.shape is (M,K,1,1)
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,12,3)
    #X.shape is (M,12,3)
    F=shared.cal_F_tensor(x, X, cal_d_sf_dr(r))
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,12,3)
    #d_sf_dX.shape is (M,K,3,12), from cal_d_sf_dh
    F=shared.cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    return F #(M,K,3,3)
#%% K integration points, K = 14 or 21
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,12), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 11, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dx_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr, weight)
    return force #(M,12,3)
#%% K integration points, K = 14 or 21
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,12), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dX at r), i=0 to 11, #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr, weight)
    return force #(M,12,3)
#%% K integration points, K = 14 or 21
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,12), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(S*dNi_dX at r), i=0 to 11, #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight)
    return force #(M,12,3)
#%% K integration points, K = 14 or 21
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #det_dX_dr.shape is (M,K,1,1), det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(density*body_force*Ni at r), i=0 to 11, #(9.15c) in the book
    weight=get_shape_integration_weight(det_dX_dr.shape[1], det_dX_dr.dtype, det_dX_dr.device) #(12,K)
    force=shared.cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight)
    return force #(M,12,3)
#%% K integration points, K = 14 or 21, the strain energy of each element
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
    #X.shape (M,6,3), the shell nodes of the undeformed elements
    #thickness: a number, or a tensor of shape (M,1) or (M,6)
    #nodal_normal True: the normal at node j is (b-a) x (c-a), with a = node j, b = the next node and c = the previous
    #    node in the node order (FEModel::InitShells in FEBio). FEBio then gives a node the average of the directors
    #    that the elements sharing the node have for it; that average over elements is not done here.
    #nodal_normal False: every node of the element has the normal of the element at r[0]=r[1]=0
    #    (FESSIShellDomain::InitShells in FEBio, when its nodal normals are switched off)
    M=X.shape[0]
    if nodal_normal == True:
        b=torch.roll(X, shifts=-1, dims=1)
        c=torch.roll(X, shifts=1, dims=1)
        normal=cross(b-X, c-X, dim=-1) #(M,6,3)
    else:
        d_sf_du=torch.tensor([-3.0, -1.0, 0.0, 4.0, 0.0, 0.0], dtype=X.dtype, device=X.device)
        d_sf_dv=torch.tensor([-3.0, 0.0, -1.0, 0.0, 0.0, 4.0], dtype=X.dtype, device=X.device)
        dX_du=(d_sf_du.view(1,6,1)*X).sum(dim=1) #(M,3)
        dX_dv=(d_sf_dv.view(1,6,1)*X).sum(dim=1) #(M,3)
        normal=cross(dX_du, dX_dv, dim=-1).view(M,1,3).expand(M,6,3)
    normal=normal/torch.norm(normal, p=2, dim=-1, keepdim=True)
    if torch.is_tensor(thickness):
        thickness=thickness.view(M,-1,1)
    director=thickness*normal
    return director #(M,6,3)
#%% the 12 points of the element: the shell nodes and then their back points
def cal_front_and_back_node(x, director):
    #x.shape (M,6,3), the shell nodes (front face)
    #director.shape (M,6,3), from the back point to the shell node: back = x - director
    #FEBio: the current director is D0 + u - w, with u the displacement of the node and w its shell displacement,
    #       so the back point is at X - D0 + w
    h=torch.cat([x, x-director], dim=1)
    return h #(M,12,3)
