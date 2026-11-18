# C3D10 in Abaqus, TET10 in FEBio
#
# Integration rules: 1 point, 4 points (exact up to degree 2), 8 points (exact up to degree 3), 14 points (exact up to
# degree 5) and 35 points (exact up to degree 7); the aorta example uses 8. The numbers are given here to full double
# precision, the same as in solid_element_tet5.py and solid_element_tet15.py:
#   4 points: (a,b,b), (b,a,b), (b,b,a) and (b,b,b), each with the weight 1/24, where a=(5+3*sqrt(5))/20 and
#     b=(5-sqrt(5))/20.
#   8 points: (a,b,b), (b,a,b), (b,b,a), (b,b,b) with the weight w1 and (c,d,d), (d,c,d), (d,d,c), (d,d,d) with the
#     weight w2. The rules of this form that are exact up to degree 3 make a family with one free parameter (any b
#     near 0.328 gives one). The numbers here, from solid_element_tet15.py, are the one closest to FEBio's numbers
#     (within 3e-10).
#   14 and 35 points: both fully symmetric (unchanged by the 24 permutations of the volume coordinates), with all
#     weights positive and all points strictly inside the element. 14 points (orbits 2 x S31 + S22) are exact for the
#     mass of a straight-sided tet10 (degree 4) and for the body force of a curved one (degree 5); 35 points (orbits
#     S4 + S31 + S22 + 2 x S211) are exact for the mass of a curved tet10 (degree 7). They were computed on 2026-11-17
#     in the implicit_dynamic project by solving the moment equations; they are not from FEBio or from a published
#     table. Their numbers are in get_integration_point_14i, get_integration_weight_14i, get_integration_point_35i
#     and get_integration_weight_35i below.
# Until 2026-11-12 this file had FEBio's numbers (FECore/FEElementTraits.cpp), with 8 to 10 digits:
#   4 points (as FETet4G4): a=0.58541020, b=0.13819660;
#   8 points (FETet10G8): a=0.0158359099, b=0.3280546970, c=0.6791431780, d=0.1069522740,
#     w1=0.138527967/6, w2=0.111472033/6.
# With them the rules are exact only to about 1e-9: up to 7.5e-9 relative error for the 4-point rule at degree 2,
# and 5e-10 to 3.1e-9 for the 8-point rule up to degree 3, where even a linear function was not integrated exactly
# (a+3b-1 = 9e-10). Results now differ from FEBio's by about that much.
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio
# integration point:
#     https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FEElementTraits.cpp
# shape function:
#    https://github.com/febiosoftware/FEBio/blob/49f0976c640a8f0d0a36ff3251b6cba045f8c613/FECore/FESolidElementShape.cpp
#%%
import itertools
import torch
import torch_fea.element.solid_element_shared as shared
from torch import matmul
from torch.linalg import det, solve
#%% integration point location
def get_integration_point_1i(dtype, device):
    r=[[0.25,0.25,0.25]] #a list of 1 point, as the other rules
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (1, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_1i(dtype, device):
    w=[1/6] #a list of 1 weight, as the other rules
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (1,)
    return w
#%% integration point location
def get_integration_point_4i(dtype, device):
    a=0.5854101966249685 #(5+3*sqrt(5))/20
    b=0.1381966011250105 #(5-sqrt(5))/20
    r0=[a,b,b]
    r1=[b,a,b]
    r2=[b,b,a]
    r3=[b,b,b]
    r=[r0, r1, r2, r3]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (4, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_4i(dtype, device):
    w=[1/24, 1/24, 1/24, 1/24]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (4,)
    return w
#%% integration point location
def get_integration_point_8i(dtype, device):
    a=0.01583590986571992
    b=0.3280546967114267
    c=0.679143178201208
    d=0.10695227393293069
    r0=[a,b,b]
    r1=[b,a,b]
    r2=[b,b,a]
    r3=[b,b,b]
    r4=[c,d,d]
    r5=[d,c,d]
    r6=[d,d,c]
    r7=[d,d,d]
    r=[r0, r1, r2, r3, r4, r5, r6, r7]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (8, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_8i(dtype, device):
    w1=0.02308799441864369
    w2=0.018578672248022975
    w=[w1, w1, w1, w1, w2, w2, w2, w2]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (8,)
    return w
#%% useful for 9.15c in the book
def get_shape_integration_weight_1i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1/6 for j=0
    #return: weight, weight.shape (10,1); a list of its numbers when dtype or device is None
    r0=get_integration_point_1i(None, None)[0]
    a=1/6
    weight=[]
    for i in range(0, 10):
        wi0=eval("sf"+str(i)+"(r0)")*a
        weight.append(wi0)
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(10,1)
    return weight
#%%
def get_shape_integration_weight_4i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is 1/24 for j=0,1,2,3,4,5,6,7
    #return: weight, weight.shape (10,4); a list of its numbers when dtype or device is None
    r0, r1, r2, r3=get_integration_point_4i(None, None)
    a=1/24
    weight=[]
    for i in range(0, 10):
        for j in range(0, 4):
             wij=eval("sf"+str(i)+"(r"+str(j)+")")*a
             weight.append(wij)
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(10,4)
    return weight
#%% useful for 9.15c in the book
def get_shape_integration_weight_8i(dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #integration_weight_at_j is a[j]
    #return: weight, weight.shape (10,8); a list of its numbers when dtype or device is None
    r0, r1, r2, r3, r4, r5, r6, r7=get_integration_point_8i(None, None)
    a=get_integration_weight_8i(None, None)
    weight=[]
    for i in range(0, 10):
        for j in range(0, 8):
             wij=eval("sf"+str(i)+"(r"+str(j)+")")*a[j]
             weight.append(wij)
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(10,8)
    return weight
#%% exact rules for the tet10 mass and body force: fully symmetric (invariant under the 24 permutations of the
#volume coordinates), all weights positive, all points inside the element
#  14 points: exact up to degree 5 (enough for the mass of a straight-sided tet10, degree 4, and for the body force of
#             a curved one, degree 5)
#  35 points: exact up to degree 7 (enough for the mass of a curved tet10, degree 7)
#get_integration_point_14i and _35i: one point of each orbit, by its volume coordinates (L0, L1, L2, L3); the orbit's
#points are the distinct permutations of the four coordinates. r = (L1, L2, L3), as sf0 uses L0 = 1-r[0]-r[1]-r[2].
#get_integration_weight_14i and _35i: the weight per point of each orbit, as a fraction of the element volume, divided
#by 6 for the unit reference tetrahedron (volume 1/6).
#Computed 2026-11-17 by solving the moment equations (implicit_dynamic/tet10_15_integration_point.md).
def expand_orbits(orbits):
    #orbits: a list of the volume coordinates (L0, L1, L2, L3) of one point of each orbit
    #return: r, a list of [r0, r1, r2]: the points of each orbit in turn, the distinct permutations of its coordinates
    r=[]
    for L in orbits:
        for p in sorted(set(itertools.permutations(L))):
            r.append([p[1], p[2], p[3]])
    return r
#%% integration point location
def get_integration_point_14i(dtype, device):
    #orbits S31, S31 and S22: 4, 4 and 6 points
    orbits=[(0.09273525031089122, 0.09273525031089122, 0.09273525031089122, 0.7217942490673264),
            (0.3108859192633006, 0.3108859192633006, 0.3108859192633006, 0.06734224221009817),
            (0.04550370412564965, 0.04550370412564965, 0.45449629587435036, 0.45449629587435036)]
    r=expand_orbits(orbits)
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (14, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_14i(dtype, device):
    #the weight per point of each orbit of get_integration_point_14i, in the same order
    w1=0.07349304311636196/6 #orbit S31, 4 points
    w2=0.11268792571801585/6 #orbit S31, 4 points
    w3=0.042546020777081466/6 #orbit S22, 6 points
    w=[w1, w1, w1, w1, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (14,)
    return w
#%% integration point location
def get_integration_point_35i(dtype, device):
    #orbits S4, S31, S22, S211 and S211: 1, 4, 6, 12 and 12 points
    orbits=[(0.25, 0.25, 0.25, 0.25),
            (0.3157011497782028, 0.3157011497782028, 0.3157011497782028, 0.0528965506653916),
            (0.05048982259839637, 0.05048982259839637, 0.44951017740160365, 0.44951017740160365),
            (0.18883383102600104, 0.18883383102600104, 0.047160700360997884, 0.5751716375870001),
            (0.021265472541483248, 0.021265472541483248, 0.14663881381848495, 0.8108302410985485)]
    r=expand_orbits(orbits)
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (35, 3)
    return r
#%% gaussian integration weight
def get_integration_weight_35i(dtype, device):
    #the weight per point of each orbit of get_integration_point_35i, in the same order
    w1=0.09548528946413085/6 #orbit S4, 1 point
    w2=0.04232958120996703/6 #orbit S31, 4 points
    w3=0.03189692783285758/6 #orbit S22, 6 points
    w4=0.03720713072833462/6 #orbit S211, 12 points
    w5=0.008110770829903342/6 #orbit S211, 12 points
    w=[w1,
       w2, w2, w2, w2,
       w3, w3, w3, w3, w3, w3,
       w4, w4, w4, w4, w4, w4, w4, w4, w4, w4, w4, w4,
       w5, w5, w5, w5, w5, w5, w5, w5, w5, w5, w5, w5]
    if dtype is not None and device is not None:
        w=torch.tensor(w, dtype=dtype, device=device) #w.shape (35,)
    return w
#%%
def get_shape_integration_weight_from_rule(r, a, dtype, device):
    #wij: shape_function_i_at_integration_point_j * integration_weight_at_j
    #return: weight, weight.shape (10,K); a list of its numbers when dtype or device is None
    sf=[sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7, sf8, sf9]
    K=len(r)
    weight=[sf[i](r[j])*a[j] for i in range(10) for j in range(K)]
    if dtype is not None and device is not None:
        weight=torch.tensor(weight, dtype=dtype, device=device)
        weight=weight.view(10,K)
    return weight
def get_shape_integration_weight_14i(dtype, device):
    return get_shape_integration_weight_from_rule(get_integration_point_14i(None, None), get_integration_weight_14i(None, None), dtype, device)
def get_shape_integration_weight_35i(dtype, device):
    return get_shape_integration_weight_from_rule(get_integration_point_35i(None, None), get_integration_weight_35i(None, None), dtype, device)
#%% the three dispatchers, replacing the existing ones
def get_integration_point(n_points, dtype, device):
    #return: r, r.shape (n_points,3); a list of the points when dtype or device is None
    if n_points==1:
        return get_integration_point_1i(dtype, device)
    elif n_points==4:
        return get_integration_point_4i(dtype, device)
    elif n_points==8:
        return get_integration_point_8i(dtype, device)
    elif n_points==14:
        return get_integration_point_14i(dtype, device)
    elif n_points==35:
        return get_integration_point_35i(dtype, device)
    else:
        raise ValueError("n_points must be 1, 4, 8, 14 or 35")
def get_integration_weight(n_points, dtype, device):
    #return: w, w.shape (n_points,); a list of the weights when dtype or device is None
    if n_points == 1:
        return get_integration_weight_1i(dtype, device)
    elif n_points == 4:
        return get_integration_weight_4i(dtype, device)
    elif n_points == 8:
        return get_integration_weight_8i(dtype, device)
    elif n_points == 14:
        return get_integration_weight_14i(dtype, device)
    elif n_points == 35:
        return get_integration_weight_35i(dtype, device)
    else:
        raise ValueError("n_points must be 1, 4, 8, 14 or 35")
def get_shape_integration_weight(n_points, dtype, device):
    #return: weight, weight.shape (10,n_points); a list of its numbers when dtype or device is None
    if n_points==1:
        return get_shape_integration_weight_1i(dtype, device)
    elif n_points==4:
        return get_shape_integration_weight_4i(dtype, device)
    elif n_points==8:
        return get_shape_integration_weight_8i(dtype, device)
    elif n_points==14:
        return get_shape_integration_weight_14i(dtype, device)
    elif n_points==35:
        return get_shape_integration_weight_35i(dtype, device)
    else:
        raise ValueError("n_points must be 1, 4, 8, 14 or 35")
#%% shape function 0
def sf0(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    a=1-r[0]-r[1]-r[2]
    return a*(2*a-1)
#%%
def cal_d_sf0_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    a=1-r[0]-r[1]-r[2]
    da=4*a-1
    dr0=-da
    dr1=-da
    dr2=-da
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf1(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[0]*(2*r[0]-1)
#%%
def cal_d_sf1_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=4*r[0]-1
    dr1=0*r[1]
    dr2=0*r[2]
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf2(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[1]*(2*r[1]-1)
#%%
def cal_d_sf2_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0*r[0]
    dr1=4*r[1]-1
    dr2=0*r[2]
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf3(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return r[2]*(2*r[2]-1)
#%%
def cal_d_sf3_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0*r[0]
    dr1=0*r[1]
    dr2=4*r[2]-1
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf4(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*(1-r[0]-r[1]-r[2])*r[0]
#%%
def cal_d_sf4_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-8*r[0]+4*(1-r[1]-r[2])
    dr1=-4*r[0]
    dr2=-4*r[0]
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf5(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*r[0]*r[1]
#%%
def cal_d_sf5_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=4*r[1]
    dr1=4*r[0]
    dr2=0*r[2]
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf6(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*(1-r[0]-r[1]-r[2])*r[1]
#%%
def cal_d_sf6_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-4*r[1]
    dr1=-8*r[1]+4*(1-r[0]-r[2])
    dr2=-4*r[1]
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf7(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*(1-r[0]-r[1]-r[2])*r[2]
#%%
def cal_d_sf7_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=-4*r[2]
    dr1=-4*r[2]
    dr2=-8*r[2]+4*(1-r[0]-r[1])
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf8(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*r[0]*r[2]
#%%
def cal_d_sf8_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=4*r[2]
    dr1=0*r[1]
    dr2=4*r[0]
    dr=[dr0, dr1, dr2]
    return dr
#%%
def sf9(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: the value of the shape function, a number or a tensor with the shape of r[0]
    return 4*r[1]*r[2]
#%%
def cal_d_sf9_dr(r):
    #r.shape (3,...): r[0], r[1], r[2] are numbers or tensors of one shape
    #return: [dr0, dr1, dr2], the derivatives with respect to r[0], r[1], r[2], each a number or a tensor with the shape of r[0]
    dr0=0*r[0]
    dr1=4*r[2]
    dr2=4*r[1]
    dr=[dr0, dr1, dr2]
    return dr
#%%
def cal_d_sf_dr(r):
    #r.shape is (K,3), K is the number of integration points
    #return: d_sf_dr, d_sf_dr.shape (1,K,3,10)
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
    d_sf_dr=[d_sf0_dr, d_sf1_dr, d_sf2_dr, d_sf3_dr, d_sf4_dr, d_sf5_dr, d_sf6_dr, d_sf7_dr, d_sf8_dr, d_sf9_dr]
    d_sf_dr=torch.cat(d_sf_dr, dim=2)
    d_sf_dr=d_sf_dr.view(1,K,3,10)#10: sf0 to sf9
    return d_sf_dr
#%%
def interpolate(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,10,3), M elements, 1 elemnet has 10 nodes, 1 node has a 3D position, h is x or X or u
    #hr=sum_k{sf_i(r)*h_i, k=0 to 9}
    K=r.shape[0]
    r=r.permute(1,0)#shape is (3,K)
    r=r.view(3,K,1)
    w=torch.cat([sf0(r), sf1(r), sf2(r), sf3(r), sf4(r), sf5(r), sf6(r), sf7(r), sf8(r), sf9(r)], dim=1)
    w=w.view(1,K,10,1)
    M=h.shape[0]
    h=h.view(M,1,10,3)
    hr=(w*h).sum(dim=2)#shape (M,K,3)
    return hr
#%%
def cal_dh_dr(r, h, d_sf_dr=None, numerator_layout=True):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,10,3), M elements, 1 elemnet has 10 nodes, 1 node has a 3D position
    #d_sf_dr.shape (1,K,3,10)
    if d_sf_dr is None:
        d_sf_dr=cal_d_sf_dr(r)
    dh_dr=shared.cal_dh_dr(h, d_sf_dr, numerator_layout)
    return dh_dr #shape (M,K,3,3)
#%%
def cal_d_sf_dh(r, h):
    #r.shape (K,3), K is the number of integration points
    #h.shape (M,10,3), M elements, 1 elemnet has 10 nodes, 1 node has a 3D position
    #h is x, X or u=x-X
    #(page264) in the book dNi/dX at r, Ni is the shape function of node-i
    #(9.15b) in the book dNi/dx at r, Ni is the shape function of node-i
    d_sf_dr=cal_d_sf_dr(r)
    #d_sf_dr.shape (1,K,3,10), K is the number of integration points
    d_sf_dh, dh_dr, det_dh_dr=shared.cal_d_sf_dh(h, d_sf_dr)
    #d_sf_dh.shape is (M,K,3,10), d_sf_dh[:,:,:,0] is d_sf0_dh
    #dh_dr.shape is (M,K,3,3), numerator_layout=False
    #det_dh_dr.shape is (M,K,1,1)
    return d_sf_dh, dh_dr, det_dh_dr
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3), K is the number of integration points
    #x.shape is (M,10,3)
    #X.shape is (M,10,3)
    F=shared.cal_F_tensor(x, X, cal_d_sf_dr(r))
    #F.shape is (M,K,3,3)
    return F
#%%
def cal_F_tensor_with_d_sf_dX(x, d_sf_dX):
    #x.shape is (M,10,3)
    #d_sf_dX.shape is (M,K,3,10), from cal_d_sf_dh
    F=shared.cal_F_tensor_with_d_sf_dX(x, d_sf_dX)
    return F #(M,K,3,3)
#%% K integration points, K = 1, 4, 8, 14 or 35
def cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_Cauchy_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dx, dx_dr, det_dx_dr=cal_d_sf_dh(r, x)
    #d_sf_dx.shape is (M,K,3,10), det_dx_dr.shape is (M,K,1,1)
    #det_dx_dr should be > 0
    #----------------------------------------------------------------------
    #force_i=integration(S*dNi_dx at r), i=0 to 9, #(9.15b) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dx_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_cauchy_stress(S, d_sf_dx, det_dx_dr, weight)
    return force #(M,10,3)
#%% K integration points, K = 1, 4, 8, 14 or 35
def cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_2pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,10), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(F*S*dNi_dX at r), i=0 to 9, #(page264) in the book
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_2pk_stress(F, S, d_sf_dX, det_dX_dr, weight)
    return force #(M,10,3)
#%% K integration points, K = 1, 4, 8, 14 or 35
def cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr):
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #F=cal_F_tensor(...) on the integration points, F.shape is (M,K,3,3)
    #S=cal_1pk_stress(F, Mat) on the integration points, S.shape is (M,K,3,3)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #d_sf_dX.shape is (M,K,3,10), det_dX_dr.shape is (M,K,1,1)
    #det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(S*dNi_dX at r), i=0 to 9, #(page264) in the book
    #S_1pk=F*S_2pk
    #shape_function_weight is not used for stress-force calculation
    weight=get_integration_weight(det_dX_dr.shape[1], S.dtype, torch.device("cpu"))
    force=shared.cal_nodal_force_from_1pk_stress(S, d_sf_dX, det_dX_dr, weight)
    return force #(M,10,3)
#%% K integration points, K = 1, 4, 8, 14 or 35
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #det_dX_dr.shape is (M,K,1,1), det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(density*body_force*Ni at r), i=0 to 9, #(9.15c) in the book
    weight=get_shape_integration_weight(det_dX_dr.shape[1], det_dX_dr.dtype, det_dX_dr.device) #(10,K)
    force=shared.cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight)
    return force #(M,10,3)
#%% K integration points, K = 1, 4, 8, 14 or 35, the strain energy of each element
def cal_strain_energy(W, det_dX_dr):
    #W.shape (M,K)
    #det_dX_dr.shape (M,K,1,1)
    #W=cal_strain_energy_density(F, Mat) on the integration points
    #energy=integration(W at r)
    weight=get_integration_weight(det_dX_dr.shape[1], W.dtype, torch.device("cpu"))
    energy=shared.cal_strain_energy(W, det_dX_dr, weight)
    return energy #(M,)
