# 4-node quadrilateral shell element with assumed natural strains. "q4ans" in FEBio (FEElasticANSShellDomain)
# The nodes, the 8 points (4 shell nodes and their 4 back points), the shape functions and the integration rules
#   are those of shell_element_quad4.py (FEBio uses FEShellQuad4G8, 8 points, for q4ans).
# What differs is the strain. The Green-Lagrange strain is taken in covariant components
#       E_ij = (g_i.g_j - G_i.G_j)/2,   g_i = dx/dr[i],   G_i = dX/dr[i]
#   and three of them are not taken at the integration point but interpolated from fixed points in the middle
#   surface r[2]=0 (FEElasticANSShellDomain::CollocationStrainsANS and EvaluateANS):
#       E13 from A=(0,-1) and C=(0,1):  ((1-v)*E13_A + (1+v)*E13_C)/2
#       E23 from D=(-1,0) and B=(1,0):  ((1-u)*E23_D + (1+u)*E23_B)/2
#       E33 from the corners E=(-1,-1), F=(1,-1), G=(1,1), H=(-1,1), with the four bilinear functions
#   with u=r[0], v=r[1] of the integration point. E11, E22, E12 stay as they are.
#   This removes the transverse shear locking and the curvature thickness locking of shell_element_quad4.
# The strain tensor is E = sum_ij E_ij * G^i (x) G^j, with G^i the contravariant base vectors (G^i.G_j = 1 if i=j, else 0).
# The material gets this strain, not the deformation gradient of the element (cal_F_tensor below), and the nodal
#   force comes from the 2PK stress S and the derivative of the assumed strain:
#       force_a = integration(sum_ij S^ij * dE_ij/dx_a),   S^ij = G^i.S.G^j
# So the force functions of a solid_element file (stress times dN/dx) do not apply to this element. The functions
#   cal_nodal_force_from_cauchy_stress, cal_nodal_force_from_2pk_stress and cal_nodal_force_from_1pk_stress below are
#   in the same place as in a solid_element file but take F, the stress, r and the points of the element
#   (F before the stress, as in cal_nodal_force_from_2pk_stress(F, S, ...) of a solid_element file).
# cal_nodal_force_from_body_force and cal_strain_energy do not depend on the strain: they are those of
#   shell_element_quad4.py.
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio
#   https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticANSShellDomain.cpp
#%%
import torch
import torch_fea.element.solid_element_shared as shared
from torch_fea.utility.small_matmul import matmul #torch.matmul on the CPU, elementwise for small matrices on the GPU
from torch.linalg import det, inv, solve, cholesky
from torch_fea.element.shell_element_quad4 import get_integration_point_8i, get_integration_weight_8i
from torch_fea.element.shell_element_quad4 import get_integration_point_12i, get_integration_weight_12i
from torch_fea.element.shell_element_quad4 import get_integration_point, get_integration_weight
from torch_fea.element.shell_element_quad4 import get_shape_integration_weight_8i, get_shape_integration_weight_12i
from torch_fea.element.shell_element_quad4 import get_shape_integration_weight
from torch_fea.element.shell_element_quad4 import sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7
from torch_fea.element.shell_element_quad4 import cal_d_sf_dr, interpolate, cal_dh_dr, cal_d_sf_dh
from torch_fea.element.shell_element_quad4 import cal_director, cal_front_and_back_node
#%% the points where the assumed strains are taken, all in the middle surface r[2]=0
def get_collocation_point(dtype, device):
    #A, B, C, D: the middle of the four edges; E, F, G, H: the four corners
    r=[[0,-1,0], [1,0,0], [0,1,0], [-1,0,0], [-1,-1,0], [1,-1,0], [1,1,0], [-1,1,0]]
    if dtype is not None and device is not None:
        r=torch.tensor(r, dtype=dtype, device=device) #r.shape (8, 3)
    return r
#%% the weight of each collocation point in the assumed strain at r
def cal_collocation_weight(r):
    #r.shape (K,3), K is the number of integration points
    #weight[:,0], weight[:,2]: points A and C, for E13
    #weight[:,1], weight[:,3]: points B and D, for E23
    #weight[:,4:8]: points E, F, G, H, for E33
    u=r[:,0]
    v=r[:,1]
    weight=[(1-v)/2, (1+u)/2, (1+v)/2, (1-u)/2,
            (1-u)*(1-v)/4, (1+u)*(1-v)/4, (1+u)*(1+v)/4, (1-u)*(1+v)/4]
    weight=torch.stack(weight, dim=1)
    return weight #(K,8)
#%%
def cal_covariant_base_vector(r, h):
    #r.shape (K,3); h.shape (M,8,3), 4 shell nodes and then their 4 back points, h is x or X
    #g[:,:,i] is the base vector dh/dr[i]
    g=cal_dh_dr(r, h, None, False)
    return g #(M,K,3,3)
#%%
def cal_contravariant_base_vector(r, X):
    #r.shape (K,3); X.shape (M,8,3)
    #G_cnt[:,:,i] is the base vector G^i of the undeformed element: G^i.G_j = 1 if i=j, else 0
    G_cov=cal_covariant_base_vector(r, X)
    G_cnt=inv(G_cov).transpose(2,3)
    return G_cnt #(M,K,3,3)
#%% Green-Lagrange strain of the element as it is, in covariant components
def cal_covariant_strain(r, x, X):
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3)
    #E[:,:,i,j] = (g_i.g_j - G_i.G_j)/2
    g=cal_covariant_base_vector(r, x)
    G=cal_covariant_base_vector(r, X)
    E=0.5*(matmul(g, g.transpose(2,3))-matmul(G, G.transpose(2,3)))
    return E #(M,K,3,3)
#%% the assumed strain, in covariant components
def cal_assumed_covariant_strain(r, x, X):
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3)
    #E11, E22, E12 at r; E13, E23, E33 interpolated from the collocation points
    E=cal_covariant_strain(r, x, X) #(M,K,3,3)
    rc=get_collocation_point(x.dtype, x.device)
    Ec=cal_covariant_strain(rc, x, X) #(M,8,3,3)
    w=cal_collocation_weight(r) #(K,8)
    E13=matmul(Ec[:,[0,2],0,2], w[:,[0,2]].t()) #(M,K)
    E23=matmul(Ec[:,[1,3],1,2], w[:,[1,3]].t()) #(M,K)
    E33=matmul(Ec[:,4:8,2,2], w[:,4:8].t()) #(M,K)
    row0=torch.stack([E[:,:,0,0], E[:,:,0,1], E13], dim=-1)
    row1=torch.stack([E[:,:,0,1], E[:,:,1,1], E23], dim=-1)
    row2=torch.stack([E13, E23, E33], dim=-1)
    E_ans=torch.stack([row0, row1, row2], dim=-2)
    return E_ans #(M,K,3,3)
#%% from covariant components to the tensor in the global axes
def cal_tensor_from_covariant_component(E_cov, r, X):
    #E_cov.shape (M,K,3,3), covariant components at the K points; r.shape (K,3); X.shape (M,8,3)
    #E = sum_ij E_ij * G^i (x) G^j
    G_cnt=cal_contravariant_base_vector(r, X)
    E=matmul(matmul(G_cnt.transpose(2,3), E_cov), G_cnt)
    return E #(M,K,3,3)
#%% the strain that the material gets
def cal_strain(r, x, X):
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3)
    #Green-Lagrange strain with the assumed E13, E23, E33, as a tensor in the global axes
    E=cal_tensor_from_covariant_component(cal_assumed_covariant_strain(r, x, X), r, X)
    return E #(M,K,3,3)
#%% a deformation gradient that has the strain E
def cal_F_tensor_from_strain(E):
    #E.shape (M,K,3,3), Green-Lagrange strain
    #F is upper triangular with F^T*F = 2*E + I (Cholesky), so (F^T*F - I)/2 = E.
    #FEBio gives the material the symmetric F = sqrt(2*E+I) instead. The two differ by a rotation, so the strain
    #energy and the 2PK stress of a material are the same; the Cholesky factor has a derivative everywhere,
    #also at E=0 where the square root computed from eigenvalues has none.
    I=torch.eye(3, dtype=E.dtype, device=E.device)
    F=cholesky(2*E+I).transpose(2,3)
    return F #(M,K,3,3)
#%%
def cal_F_tensor(r, x, X):
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3)
    #not the deformation gradient of the element: the F that has the assumed strain, for the material
    #S_1pk, W = cal_1pk_stress(F, Mat, ...) and then cal_nodal_force_from_1pk_stress(F, S_1pk, r, x, X)
    F=cal_F_tensor_from_strain(cal_strain(r, x, X))
    return F #(M,K,3,3)
#%%
def cal_2pk_stress_from_cauchy_stress(F, S):
    #F.shape (M,K,3,3); S.shape (M,K,3,3), Cauchy stress
    #S_2pk = det(F) * F^-1 * S * F^-T
    M=F.shape[0]
    K=F.shape[1]
    S_2pk=solve(F, solve(F, S.transpose(2,3)).transpose(2,3))
    S_2pk=det(F).view(M,K,1,1)*S_2pk
    return S_2pk #(M,K,3,3)
#%% K integration points, K = 8 or 12
def cal_nodal_force_from_cauchy_stress(F, S, r, x, X):
    #F.shape (M,K,3,3), from cal_F_tensor; S.shape (M,K,3,3), Cauchy stress of the material at F
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3)
    #S, W = cal_cauchy_stress(F, Mat, ...) and then this function
    force=cal_nodal_force_from_2pk_stress(cal_2pk_stress_from_cauchy_stress(F, S), r, x, X)
    return force #(M,8,3)
#%% K integration points, K = 8 or 12
def cal_nodal_force_from_2pk_stress(S, r, x, X):
    #S.shape (M,K,3,3), 2PK stress at the integration points for the strain from cal_strain
    #r=get_integration_point(K, dtype, device), r.shape is (K,3)
    #x.shape (M,8,3), X.shape (M,8,3): 4 shell nodes and then their 4 back points
    #force_a=integration(sum_ij S^ij * dE_ij/dx_a), a=0 to 7, E_ij the assumed strain, S^ij = G^i.S.G^j
    M=x.shape[0]
    K=r.shape[0]
    d_sf_dr=cal_d_sf_dr(r) #(1,K,3,8)
    g=matmul(d_sf_dr, x.view(M,1,8,3)) #(M,K,3,3), g[:,:,i] is g_i
    G=matmul(d_sf_dr, X.view(M,1,8,3)) #(M,K,3,3)
    G_cnt=inv(G).transpose(2,3)
    Sc=matmul(matmul(G_cnt, S), G_cnt.transpose(2,3)) #(M,K,3,3), S^ij
    weight=get_integration_weight(K, x.dtype, x.device)
    q=weight.view(1,K)*det(G) #(M,K), weight times volume
    #E11, E22, E12 at the integration point: dE_ij/dx_a = (dNa/dr_i*g_j + dNa/dr_j*g_i)/2
    Sg=matmul(Sc[:,:,0:2,0:2], g[:,:,0:2,:]) #(M,K,2,3), sum_j S^ij*g_j
    force=matmul(d_sf_dr[:,:,0:2,:].transpose(2,3), Sg) #(M,K,8,3)
    force=(q.view(M,K,1,1)*force).sum(dim=1) #(M,8,3)
    #E13, E23, E33 at the collocation points
    rc=get_collocation_point(x.dtype, x.device)
    wc=cal_collocation_weight(r) #(K,8)
    dN=cal_d_sf_dr(rc)[0] #(8,3,8), dN[c,i,a] = dNa/dr_i at the point c
    gc=matmul(dN, x.view(M,1,8,3)) #(M,8,3,3)
    T13=matmul(q*Sc[:,:,0,2], wc[:,[0,2]]) #(M,2), points A and C
    T23=matmul(q*Sc[:,:,1,2], wc[:,[1,3]]) #(M,2), points B and D
    T33=matmul(q*Sc[:,:,2,2], wc[:,4:8])   #(M,4), points E, F, G, H
    #d(2*E13)/dx_a = dNa/dr_0*g_2 + dNa/dr_2*g_0
    c=[0,2]
    B13=dN[c,0,:].view(1,2,8,1)*gc[:,c,2,:].view(M,2,1,3)+dN[c,2,:].view(1,2,8,1)*gc[:,c,0,:].view(M,2,1,3)
    force=force+(T13.view(M,2,1,1)*B13).sum(dim=1)
    #d(2*E23)/dx_a = dNa/dr_1*g_2 + dNa/dr_2*g_1
    c=[1,3]
    B23=dN[c,1,:].view(1,2,8,1)*gc[:,c,2,:].view(M,2,1,3)+dN[c,2,:].view(1,2,8,1)*gc[:,c,1,:].view(M,2,1,3)
    force=force+(T23.view(M,2,1,1)*B23).sum(dim=1)
    #d(E33)/dx_a = dNa/dr_2*g_2
    B33=dN[4:8,2,:].view(1,4,8,1)*gc[:,4:8,2,:].view(M,4,1,3)
    force=force+(T33.view(M,4,1,1)*B33).sum(dim=1)
    return force #(M,8,3)
#%% K integration points, K = 8 or 12
def cal_nodal_force_from_1pk_stress(F, S, r, x, X):
    #F.shape (M,K,3,3), from cal_F_tensor; S.shape (M,K,3,3), 1PK stress of the material at F
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3)
    #S_2pk = F^-1 * S_1pk
    force=cal_nodal_force_from_2pk_stress(solve(F, S), r, x, X)
    return force #(M,8,3)
#%% K integration points, K = 8 or 12
def cal_nodal_force_from_body_force(body_force, density, det_dX_dr):
    #body_force: force per unit mass, shape (3,) or (M,K,3) at the integration points
    #density: density of the undeformed element, a number or a tensor of shape (M,1)
    #d_sf_dX, dX_dr, det_dX_dr=cal_d_sf_dh(r, X)
    #det_dX_dr.shape is (M,K,1,1), det_dX_dr should be > 0
    #----------------------------------------------------------------
    #force_i=integration(density*body_force*Ni at r), i=0 to 7: the 4 shell nodes and then their 4 back points
    weight=get_shape_integration_weight(det_dX_dr.shape[1], det_dX_dr.dtype, det_dX_dr.device) #(8,K)
    force=shared.cal_nodal_force_from_body_force(body_force, density, det_dX_dr, weight)
    return force #(M,8,3)
#%% K integration points, K = 8 or 12, the strain energy of each element
def cal_strain_energy(W, det_dX_dr):
    #W.shape (M,K)
    #det_dX_dr.shape (M,K,1,1)
    #W=cal_strain_energy_density(F, Mat) on the integration points
    #energy=integration(W at r)
    weight=get_integration_weight(det_dX_dr.shape[1], W.dtype, torch.device("cpu"))
    energy=shared.cal_strain_energy(W, det_dX_dr, weight)
    return energy #(M,)
