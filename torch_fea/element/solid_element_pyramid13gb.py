# 13-node quadratic pyramid with the rational shape functions of G. Bedrosian, "Shape functions and integration formulas
# for three-dimensional finite element analysis", Int. J. Numer. Methods Eng. 35 (1992) 95-108; the same functions as
# libMesh's PYRAMID13 (src/fe/fe_lagrange_shape_3D.C). pyramid13 (FEBio's PYRA13) is the collapsed hex20 instead.
# The unit element is the cube [-1,1]^3 with its top face r[2]=1 collapsed into node 4, as for pyramid13:
#   the point of the pyramid (base [-1,1]^2 at z=0, apex at (0,0,1)) is x=r[0]*(1-r[2])/2, y=r[1]*(1-r[2])/2, z=(1+r[2])/2,
#   and Bedrosian's functions of (x, y, z), rational with the denominator 1-z, are polynomials of r
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio
# location of the nodes in the unit element, in the node order of pyramid13 (FEBio's and libMesh's):
#   node 0: (-1, -1, -1)
#   node 1: (1, -1, -1)
#   node 2: (1, 1, -1)
#   node 3: (-1, 1, -1)
#   node 4: (0, 0, 1)
#   node 5: (0, -1, -1)
#   node 6: (1, 0, -1)
#   node 7: (0, 1, -1)
#   node 8: (-1, 0, -1)
#   node 9: (-1, -1, 0)
#   node 10: (1, -1, 0)
#   node 11: (1, 1, 0)
#   node 12: (-1, 1, 0)
# shape functions:
#   each shape function is 1 at its own node and 0 at the other nodes, and they reproduce every quadratic polynomial of (x, y, z)
#   nodes 4 and 9 to 12 have the same functions as pyramid13; the corners 0 to 3 and the base mid-edges 5 to 8 do not
# faces: the four triangular faces are exactly tri6 and the base (nodes 0 to 3 and 5 to 8) is exactly quad8, so a pressure
#   computed with surface_element_tri6 or surface_element_quad8 is the consistent load of this element, and a pyramid13gb
#   conforms with a tet10 on a triangular face and with a hex20 (or a wedge15) on the base
# integration rule:
#   27 points: 3 x 3 x 3 Gauss points at -a, 0, a with a=sqrt(3/5), r[0] changes fastest (the points of hex20's 27-point rule);
#   the stiffness of a pyramid with straight edges is a polynomial of degree 4 in each of r[0], r[1], r[2], so it is exact.
#   2 x 2 x 2 points are not offered: they leave one zero-energy mode besides the six rigid-body modes
#%%
import torch
import torch_fea.element.solid_element_shared as shared
#%% integration point location
def get_integration_point_27i(dtype, device):
    #27 points: 3 x 3 x 3 Gauss points at -a, 0, a with a=sqrt(3/5), r[0] changes fastest
    r=[[-0.7745966692414834, -0.7745966692414834, -0.7745966692414834],
       [0.0, -0.7745966692414834, -0.7745966692414834],
       [0.7745966692414834, -0.7745966692414834, -0.7745966692414834],
       [-0.7745966692414834, 0.0, -0.7745966692414834],
       [0.0, 0.0, -0.7745966692414834],
       [0.7745966692414834, 0.0, -0.7745966692414834],
       [-0.7745966692414834, 0.7745966692414834, -0.7745966692414834],
       [0.0, 0.7745966692414834, -0.7745966692414834],
       [0.7745966692414834, 0.7745966692414834, -0.7745966692414834],
       [-0.7745966692414834, -0.7745966692414834, 0.0],
       [0.0, -0.7745966692414834, 0.0],
       [0.7745966692414834, -0.7745966692414834, 0.0],
       [-0.7745966692414834, 0.0, 0.0],
       [0.0, 0.0, 0.0],
       [0.7745966692414834, 0.0, 0.0],
       [-0.7745966692414834, 0.7745966692414834, 0.0],
       [0.0, 0.7745966692414834, 0.0],
       [0.7745966692414834, 0.7745966692414834, 0.0],
       [-0.7745966692414834, -0.7745966692414834, 0.7745966692414834],
       [0.0, -0.7745966692414834, 0.7745966692414834],
       [0.7745966692414834, -0.7745966692414834, 0.7745966692414834],
       [-0.7745966692414834, 0.0, 0.7745966692414834],
       [0.0, 0.0, 0.7745966692414834],
       [0.7745966692414834, 0.0, 0.7745966692414834],
       [-0.7745966692414834, 0.7745966692414834, 0.7745966692414834],
       [0.0, 0.7745966692414834, 0.7745966692414834],
       [0.7745966692414834, 0.7745966692414834, 0.7745966692414834]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (27, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_27i(dtype, device):
    w=[0.1714677640603567, 0.27434842249657065, 0.1714677640603567, 0.27434842249657065,
       0.438957475994513, 0.27434842249657065, 0.1714677640603567, 0.27434842249657065,
       0.1714677640603567, 0.27434842249657065, 0.43895747599451296, 0.27434842249657065,
       0.43895747599451296, 0.7023319615912208, 0.43895747599451296, 0.27434842249657065,
       0.43895747599451296, 0.27434842249657065, 0.1714677640603567, 0.27434842249657065,
       0.1714677640603567, 0.27434842249657065, 0.438957475994513, 0.27434842249657065,
       0.1714677640603567, 0.27434842249657065, 0.1714677640603567]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (27,)
    return w
#%% integration point location
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,3); a list of the points when dtype or device is None
    if n_points == 27:
        return get_integration_point_27i(dtype, device)
    else:
        raise ValueError("n_points must be 27")
#%% gaussian integration weight
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 27:
        return get_integration_weight_27i(dtype, device)
    else:
        raise ValueError("n_points must be 27")
#%% useful for 9.15c in the book
def cal_shape_integration_weight(r, w, dtype, device):
    #r: list of the K integration points, w: list of the K integration weights
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (13,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9,
        sf10, sf11, sf12]
    weight=[]
    for i in range(0, 13):
        for j in range(0, len(w)):
            weight.append(sf[i](r[j])*w[j])
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(13,len(w))
    return weight
#%%
def get_shape_integration_weight_27i(dtype, device):
    #return: weight, weight.shape (13,27); a list of its numbers when dtype or device is None
    return cal_shape_integration_weight(get_integration_point_27i(None, None), get_integration_weight_27i(None, None), dtype, device)
#%%
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (13,n_points); a list of its numbers when dtype or device is None
    if n_points == 27:
        return get_shape_integration_weight_27i(dtype, device)
    else:
        raise ValueError("n_points must be 27")
#%% shape function 0, node 0 is at (-1, -1, -1)
def sf0(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] - 1)*(r[1] - 1)*(r[2] - 1)*(r[0]*r[2] - r[0] + r[1]*r[2] - r[1] - 2)/16
#%%
def cal_d_sf0_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[1] - 1)*(r[2] - 1)*(2*r[0]*r[2] - 2*r[0] + r[1]*r[2] - r[1] - r[2] - 1)/16
    dr1=-(r[0] - 1)*(r[2] - 1)*(r[0]*r[2] - r[0] + 2*r[1]*r[2] - 2*r[1] - r[2] - 1)/16
    dr2=-(r[0] - 1)*(r[1] - 1)*(r[0]*r[2] - r[0] + r[1]*r[2] - r[1] - 1)/8
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 1, node 1 is at (1, -1, -1)
def sf1(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] + 1)*(r[1] - 1)*(r[2] - 1)*(r[0]*r[2] - r[0] - r[1]*r[2] + r[1] + 2)/16
#%%
def cal_d_sf1_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[1] - 1)*(r[2] - 1)*(2*r[0]*r[2] - 2*r[0] - r[1]*r[2] + r[1] + r[2] + 1)/16
    dr1=-(r[0] + 1)*(r[2] - 1)*(r[0]*r[2] - r[0] - 2*r[1]*r[2] + 2*r[1] + r[2] + 1)/16
    dr2=-(r[0] + 1)*(r[1] - 1)*(r[0]*r[2] - r[0] - r[1]*r[2] + r[1] + 1)/8
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 2, node 2 is at (1, 1, -1)
def sf2(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] + 1)*(r[1] + 1)*(r[2] - 1)*(r[0]*r[2] - r[0] + r[1]*r[2] - r[1] + 2)/16
#%%
def cal_d_sf2_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[1] + 1)*(r[2] - 1)*(2*r[0]*r[2] - 2*r[0] + r[1]*r[2] - r[1] + r[2] + 1)/16
    dr1=(r[0] + 1)*(r[2] - 1)*(r[0]*r[2] - r[0] + 2*r[1]*r[2] - 2*r[1] + r[2] + 1)/16
    dr2=(r[0] + 1)*(r[1] + 1)*(r[0]*r[2] - r[0] + r[1]*r[2] - r[1] + 1)/8
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 3, node 3 is at (-1, 1, -1)
def sf3(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] - 1)*(r[1] + 1)*(r[2] - 1)*(r[0]*r[2] - r[0] - r[1]*r[2] + r[1] - 2)/16
#%%
def cal_d_sf3_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[1] + 1)*(r[2] - 1)*(2*r[0]*r[2] - 2*r[0] - r[1]*r[2] + r[1] - r[2] - 1)/16
    dr1=(r[0] - 1)*(r[2] - 1)*(r[0]*r[2] - r[0] - 2*r[1]*r[2] + 2*r[1] - r[2] - 1)/16
    dr2=(r[0] - 1)*(r[1] + 1)*(r[0]*r[2] - r[0] - r[1]*r[2] + r[1] - 1)/8
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 4, node 4 is at (0, 0, 1)
def sf4(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[2]*(r[2] + 1)/2
#%%
def cal_d_sf4_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0+0*r[0]
    dr1=0+0*r[0]
    dr2=(2*r[2] + 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 5, node 5 is at (0, -1, -1)
def sf5(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] - 1)*(r[0] + 1)*(r[1] - 1)*(r[2] - 1)**2/8
#%%
def cal_d_sf5_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=r[0]*(r[1] - 1)*(r[2] - 1)**2/4
    dr1=(r[0] - 1)*(r[0] + 1)*(r[2] - 1)**2/8
    dr2=(r[0] - 1)*(r[0] + 1)*(r[1] - 1)*(r[2] - 1)/4
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 6, node 6 is at (1, 0, -1)
def sf6(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] + 1)*(r[1] - 1)*(r[1] + 1)*(r[2] - 1)**2/8
#%%
def cal_d_sf6_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[1] - 1)*(r[1] + 1)*(r[2] - 1)**2/8
    dr1=-r[1]*(r[0] + 1)*(r[2] - 1)**2/4
    dr2=-(r[0] + 1)*(r[1] - 1)*(r[1] + 1)*(r[2] - 1)/4
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 7, node 7 is at (0, 1, -1)
def sf7(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] - 1)*(r[0] + 1)*(r[1] + 1)*(r[2] - 1)**2/8
#%%
def cal_d_sf7_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-r[0]*(r[1] + 1)*(r[2] - 1)**2/4
    dr1=-(r[0] - 1)*(r[0] + 1)*(r[2] - 1)**2/8
    dr2=-(r[0] - 1)*(r[0] + 1)*(r[1] + 1)*(r[2] - 1)/4
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 8, node 8 is at (-1, 0, -1)
def sf8(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] - 1)*(r[1] - 1)*(r[1] + 1)*(r[2] - 1)**2/8
#%%
def cal_d_sf8_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[1] - 1)*(r[1] + 1)*(r[2] - 1)**2/8
    dr1=r[1]*(r[0] - 1)*(r[2] - 1)**2/4
    dr2=(r[0] - 1)*(r[1] - 1)*(r[1] + 1)*(r[2] - 1)/4
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 9, node 9 is at (-1, -1, 0)
def sf9(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] - 1)*(r[1] - 1)*(r[2] - 1)*(r[2] + 1)/4
#%%
def cal_d_sf9_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[1] - 1)*(r[2] - 1)*(r[2] + 1)/4
    dr1=-(r[0] - 1)*(r[2] - 1)*(r[2] + 1)/4
    dr2=-r[2]*(r[0] - 1)*(r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 10, node 10 is at (1, -1, 0)
def sf10(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] + 1)*(r[1] - 1)*(r[2] - 1)*(r[2] + 1)/4
#%%
def cal_d_sf10_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[1] - 1)*(r[2] - 1)*(r[2] + 1)/4
    dr1=(r[0] + 1)*(r[2] - 1)*(r[2] + 1)/4
    dr2=r[2]*(r[0] + 1)*(r[1] - 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 11, node 11 is at (1, 1, 0)
def sf11(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return -(r[0] + 1)*(r[1] + 1)*(r[2] - 1)*(r[2] + 1)/4
#%%
def cal_d_sf11_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-(r[1] + 1)*(r[2] - 1)*(r[2] + 1)/4
    dr1=-(r[0] + 1)*(r[2] - 1)*(r[2] + 1)/4
    dr2=-r[2]*(r[0] + 1)*(r[1] + 1)/2
    dr=[dr0, dr1, dr2]
    return dr
#%% shape function 12, node 12 is at (-1, 1, 0)
def sf12(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return (r[0] - 1)*(r[1] + 1)*(r[2] - 1)*(r[2] + 1)/4
#%%
def cal_d_sf12_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=(r[1] + 1)*(r[2] - 1)*(r[2] + 1)/4
    dr1=(r[0] - 1)*(r[2] - 1)*(r[2] + 1)/4
    dr2=r[2]*(r[0] - 1)*(r[1] + 1)/2
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
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr, d_sf6_dr, d_sf7_dr,
             d_sf8_dr, d_sf9_dr, d_sf10_dr, d_sf11_dr, d_sf12_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,3,13)#13: sf0 to sf12
    return d_sf_dr #(1,K,3,13)
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,13,3), M elements, 1 elemnet has 13 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_i{sf_i(r)*h_i, i=0 to 12}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r), sf6(r), sf7(r),
                 sf8(r), sf9(r), sf10(r), sf11(r), sf12(r)], dim=1)
    w=w.view(1,K,13,1)
    M=h.shape[0]
    h=h.view(M,1,13,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,13,3), M elements, 1 elemnet has 13 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,13)
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #shape (M,K,3,3)
#%%
def cal_d_sf_dh(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,13,3), M elements, 1 elemnet has 13 nodes, 1 node has a 3D position
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,13), K is the number of integration points
    d_sf_dh, dh_dr, det_dh_dr=shared.cal_d_sf_dh(h, d_sf_dr)
    #d_sf_dh.shape is (M,K,3,13), d_sf_dh[:,:,:,0] is d_sf0_dh
    #dh_dr.shape is (M,K,3,3), numerator_layout=False
    #det_dh_dr.shape is (M,K,1,1)
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,13,3)
    #X.shape is (M,13,3)
    F=shared.cal_F_tensor(x, X, cal_d_sf_dr(r))
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,13,3)
    #d_sf_dX.shape is (M,K,3,13), from cal_d_sf_dh
    F=shared.cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    return F #(M,K,3,3)
#%% K integration points, K = 27
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,13), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 12, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dx_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr, weight)
    return force #(M,13,3)
#%% K integration points, K = 27
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,13), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dX at r), i=0 to 12, #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr, weight)
    return force #(M,13,3)
#%% K integration points, K = 27
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,13), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(S*dNi_dX at r), i=0 to 12, #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight)
    return force #(M,13,3)
#%% K integration points, K = 27
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #det_dX_dr.shape is (M,K,1,1), det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(density*body_force*Ni at r), i=0 to 12, #(9.15c) in the book
    weight=get_shape_integration_weight(det_dX_dr.shape[1], det_dX_dr.dtype, det_dX_dr.device) #(13,K)
    force=shared.cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight)
    return force #(M,13,3)
#%% K integration points, K = 27, the strain energy of each element
def cal_strain_energy(W, det_dX_dr):
    #W.shape (M,K)
    #det_dX_dr.shape (M,K,1,1)
    #W=cal_strain_energy_density(F, Mat) on the integration points
    #energy=integration(W at r)
    weight=get_integration_weight(det_dX_dr.shape[1], W.dtype, torch.device("cpu"))
    energy=shared.cal_strain_energy(W, det_dX_dr, weight)
    return energy #(M,)
