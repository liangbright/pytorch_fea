# 4-node quadrilateral shell element with assumed and enhanced strains. "q4eas" in FEBio (FEElasticEASShellDomain)
# The nodes, the 8 points (4 shell nodes and their 4 back points), the shape functions and the integration rules
#   are those of shell_element_quad4.py (FEBio uses FEShellQuad4G8, 8 points, for q4eas).
# The strain is the assumed strain of shell_element_q4ans.py plus an enhanced strain with 7 numbers alpha per element:
#       E = E_ans(x) + E_eas(alpha)
#   in covariant components, as the list [E11, E22, E33, 2*E12, 2*E23, 2*E13]:
#       E_eas = Gm * alpha,   Gm = J(centre)/J(r) * T0 * N(r)        (FEElasticEASShellDomain::GenerateGMatrix)
#   N(r): the 7 modes in the base vectors of the element centre, with u=r[0], v=r[1], t=r[2]:
#       alpha[0]: E11 = u        alpha[1]: E22 = v        alpha[2]: E33 = t      alpha[3]: E33 = u*t
#       alpha[4]: E33 = v*t      alpha[5]: 2*E12 = u      alpha[6]: 2*E12 = v
#   T0 takes the components from the base vectors of the centre to those of the point r, J is the volume factor
#       det(dX/dr). The integral of each mode over the element is zero for a flat element.
#   alpha[2:5] let the thickness strain vary through the thickness, which removes the stiffening that
#       shell_element_quad4 and shell_element_q4ans show in bending when Poisson's ratio is not zero.
# alpha has no nodes. It is found from the condition that its force is zero (cal_eas_force_from_2pk_stress):
#       eas_force = integration(Gm^T * [S^11, S^22, S^33, S^12, S^23, S^13]) = 0
#   FEBio removes alpha from the equations of the element (cal_condensed_nodal_force) and updates it after each
#   iteration from the change of the nodes (cal_alpha_increment).
# The nodal force from the stress is the one of shell_element_q4ans.py: E_eas does not depend on the nodes.
#   cal_nodal_force_from_cauchy_stress, cal_nodal_force_from_2pk_stress, cal_nodal_force_from_1pk_stress,
#   cal_nodal_force_from_body_force and cal_strain_energy are in the same place as in a solid_element file.
# [r[0], r[1], r[2]] ~ [r, s, t] in FEBio
#   https://github.com/febiosoftware/FEBio/blob/develop/FEBioMech/FEElasticEASShellDomain.cpp
#%%
import torch
import torch_fea.element.solid_element_shared as shared
from torch_fea.utility.small_matmul import matmul #torch.matmul on the CPU, elementwise for small matrices on the GPU
from torch.linalg import det, inv, solve
from torch_fea.element.shell_element_quad4 import get_integration_point_8i, get_integration_weight_8i
from torch_fea.element.shell_element_quad4 import get_integration_point_12i, get_integration_weight_12i
from torch_fea.element.shell_element_quad4 import get_integration_point, get_integration_weight
from torch_fea.element.shell_element_quad4 import get_shape_integration_weight_8i, get_shape_integration_weight_12i
from torch_fea.element.shell_element_quad4 import get_shape_integration_weight
from torch_fea.element.shell_element_quad4 import sf0, sf1, sf2, sf3, sf4, sf5, sf6, sf7
from torch_fea.element.shell_element_quad4 import cal_d_sf_dr, interpolate, cal_dh_dr, cal_d_sf_dh
from torch_fea.element.shell_element_quad4 import cal_director, cal_front_and_back_node
from torch_fea.element.shell_element_q4ans import get_collocation_point, cal_collocation_weight
from torch_fea.element.shell_element_q4ans import cal_covariant_base_vector, cal_contravariant_base_vector
from torch_fea.element.shell_element_q4ans import cal_covariant_strain, cal_assumed_covariant_strain
from torch_fea.element.shell_element_q4ans import cal_tensor_from_covariant_component, cal_F_tensor_from_strain
from torch_fea.element.shell_element_q4ans import cal_2pk_stress_from_cauchy_stress
import torch_fea.element.shell_element_q4ans as q4ans
#%% number of enhanced strain parameters of an element
n_alpha=7
#%%
def cal_eas_matrix_from_base_vector(G_cov, G_cnt_center, J_ratio, r):
    #G_cov.shape (M,K,3,3): covariant base vectors of the undeformed element at the K points, G_cov[:,:,i] is G_i
    #G_cnt_center.shape (M,3,3): contravariant base vectors at the element centre, G_cnt_center[:,i] is G^i
    #J_ratio.shape (M,K): det(dX/dr) at the centre divided by det(dX/dr) at the point
    #r.shape (K,3)
    #Gm[:,:,:,n]: the strain of mode n as [E11, E22, E33, 2*E12, 2*E23, 2*E13], covariant components at the point
    M=G_cov.shape[0]
    K=G_cov.shape[1]
    g=matmul(G_cov, G_cnt_center.view(M,1,3,3).transpose(2,3)) #(M,K,3,3), g[:,:,i,j] = G_i(point).G^j(centre)
    pair=[(0,0), (1,1), (2,2), (0,1), (1,2), (0,2)]
    #T0 column 0, 1, 2: the components 11, 22, 33 of the centre; column 3: the component 12 of the centre
    T0=[]
    for (i, k) in pair:
        row=[]
        for (j, l) in pair[0:4]:
            if i == k and j == l:
                row.append(g[:,:,i,j]*g[:,:,i,j])
            elif i == k:
                row.append(g[:,:,i,j]*g[:,:,i,l])
            elif j == l:
                row.append(2*g[:,:,i,j]*g[:,:,k,j])
            else:
                row.append(g[:,:,i,j]*g[:,:,k,l]+g[:,:,i,l]*g[:,:,k,j])
        T0.append(torch.stack(row, dim=-1))
    T0=torch.stack(T0, dim=-2) #(M,K,6,4)
    u=r[:,0].view(1,K,1)
    v=r[:,1].view(1,K,1)
    t=r[:,2].view(1,K,1)
    Gm=[u*T0[...,0], v*T0[...,1], t*T0[...,2], u*t*T0[...,2], v*t*T0[...,2], u*T0[...,3], v*T0[...,3]]
    Gm=torch.stack(Gm, dim=-1)*J_ratio.view(M,K,1,1)
    return Gm #(M,K,6,7)
#%%
def cal_eas_matrix(r, X):
    #r.shape (K,3); X.shape (M,8,3), 4 shell nodes and then their 4 back points, undeformed
    G_cov=cal_covariant_base_vector(r, X) #(M,K,3,3)
    r_center=torch.zeros((1,3), dtype=X.dtype, device=X.device)
    G_cov_center=cal_covariant_base_vector(r_center, X)[:,0] #(M,3,3)
    G_cnt_center=inv(G_cov_center).transpose(1,2)
    J_ratio=det(G_cov_center).view(-1,1)/det(G_cov)
    Gm=cal_eas_matrix_from_base_vector(G_cov, G_cnt_center, J_ratio, r)
    return Gm #(M,K,6,7)
#%% the enhanced strain, in covariant components
def cal_enhanced_covariant_strain(r, X, alpha):
    #r.shape (K,3); X.shape (M,8,3); alpha.shape (M,7)
    M=X.shape[0]
    e=matmul(cal_eas_matrix(r, X), alpha.view(M,1,7,1)).squeeze(-1) #(M,K,6): E11, E22, E33, 2*E12, 2*E23, 2*E13
    row0=torch.stack([e[...,0], 0.5*e[...,3], 0.5*e[...,5]], dim=-1)
    row1=torch.stack([0.5*e[...,3], e[...,1], 0.5*e[...,4]], dim=-1)
    row2=torch.stack([0.5*e[...,5], 0.5*e[...,4], e[...,2]], dim=-1)
    E_eas=torch.stack([row0, row1, row2], dim=-2)
    return E_eas #(M,K,3,3)
#%% the strain that the material gets
def cal_strain(r, x, X, alpha):
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3); alpha.shape (M,7)
    #Green-Lagrange strain: assumed strain plus enhanced strain, as a tensor in the global axes
    E_cov=cal_assumed_covariant_strain(r, x, X)+cal_enhanced_covariant_strain(r, X, alpha)
    E=cal_tensor_from_covariant_component(E_cov, r, X)
    return E #(M,K,3,3)
#%%
def cal_F_tensor(r, x, X, alpha):
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3); alpha.shape (M,7)
    #not the deformation gradient of the element: the F that has the strain of cal_strain, for the material
    #S_1pk, W = cal_1pk_stress(F, Mat, ...), then cal_nodal_force_from_1pk_stress(F, S_1pk, r, x, X)
    #and cal_eas_force_from_1pk_stress(F, S_1pk, r, X)
    F=cal_F_tensor_from_strain(cal_strain(r, x, X, alpha))
    return F #(M,K,3,3)
#%% K integration points, K = 8 or 12
def cal_nodal_force_from_cauchy_stress(F, S, r, x, X):
    #F.shape (M,K,3,3), from cal_F_tensor; S.shape (M,K,3,3), Cauchy stress of the material at F
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3)
    #the nodal force of shell_element_q4ans.py: the enhanced strain does not depend on the nodes
    force=q4ans.cal_nodal_force_from_cauchy_stress(F, S, r, x, X)
    return force #(M,8,3)
#%% K integration points, K = 8 or 12
def cal_nodal_force_from_2pk_stress(S, r, x, X):
    #S.shape (M,K,3,3), 2PK stress at the integration points for the strain from cal_strain
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3)
    #the nodal force of shell_element_q4ans.py: the enhanced strain does not depend on the nodes
    force=q4ans.cal_nodal_force_from_2pk_stress(S, r, x, X)
    return force #(M,8,3)
#%% K integration points, K = 8 or 12
def cal_nodal_force_from_1pk_stress(F, S, r, x, X):
    #F.shape (M,K,3,3), from cal_F_tensor; S.shape (M,K,3,3), 1PK stress of the material at F
    #r.shape (K,3); x.shape (M,8,3), X.shape (M,8,3)
    #the nodal force of shell_element_q4ans.py: the enhanced strain does not depend on the nodes
    force=q4ans.cal_nodal_force_from_1pk_stress(F, S, r, x, X)
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
#%% K integration points, K = 8 or 12
def cal_eas_force_from_cauchy_stress(F, S, r, X):
    #F.shape (M,K,3,3), from cal_F_tensor; S.shape (M,K,3,3), Cauchy stress of the material at F
    #r.shape (K,3); X.shape (M,8,3)
    eas_force=cal_eas_force_from_2pk_stress(cal_2pk_stress_from_cauchy_stress(F, S), r, X)
    return eas_force #(M,7)
#%% K integration points, K = 8 or 12
def cal_eas_force_from_2pk_stress(S, r, X):
    #S.shape (M,K,3,3), 2PK stress at the integration points for the strain from cal_strain
    #r.shape (K,3); X.shape (M,8,3)
    #eas_force=integration(Gm^T * [S^11, S^22, S^33, S^12, S^23, S^13]): the derivative of the strain energy
    #of the element with respect to alpha (m_fa in FEBio). It is zero when alpha has its right value.
    M=X.shape[0]
    K=r.shape[0]
    G=cal_covariant_base_vector(r, X)
    G_cnt=inv(G).transpose(2,3)
    Sc=matmul(matmul(G_cnt, S), G_cnt.transpose(2,3)) #(M,K,3,3), S^ij
    Sv=torch.stack([Sc[:,:,0,0], Sc[:,:,1,1], Sc[:,:,2,2], Sc[:,:,0,1], Sc[:,:,1,2], Sc[:,:,0,2]], dim=-1) #(M,K,6)
    weight=get_integration_weight(K, X.dtype, X.device)
    q=weight.view(1,K)*det(G) #(M,K)
    Gm=cal_eas_matrix(r, X) #(M,K,6,7)
    eas_force=(q.view(M,K,1)*matmul(Gm.transpose(2,3), Sv.unsqueeze(-1)).squeeze(-1)).sum(dim=1)
    return eas_force #(M,7)
#%% K integration points, K = 8 or 12
def cal_eas_force_from_1pk_stress(F, S, r, X):
    #F.shape (M,K,3,3), from cal_F_tensor; S.shape (M,K,3,3), 1PK stress of the material at F
    #r.shape (K,3); X.shape (M,8,3)
    eas_force=cal_eas_force_from_2pk_stress(solve(F, S), r, X)
    return eas_force #(M,7)
#%% nodal force with alpha removed from the equations of the element
def cal_condensed_nodal_force(force, eas_force, K_ua, K_aa):
    #force.shape (M,8,3), eas_force.shape (M,7)
    #K_ua.shape (M,24,7): derivative of force.view(M,24) with respect to alpha
    #K_aa.shape (M,7,7): derivative of eas_force with respect to alpha
    #force - K_ua * K_aa^-1 * eas_force   (FEBio: fe += Kua*Kaa^-1*fa, fe being minus the nodal force)
    M=force.shape[0]
    out=force.reshape(M,24,1)-matmul(K_ua, solve(K_aa, eas_force.view(M,7,1)))
    return out.view(M,8,3) #(M,8,3)
#%% change of alpha that belongs to a change of the nodes
def cal_alpha_increment(eas_force, K_ua, K_aa, dx):
    #eas_force.shape (M,7), K_ua.shape (M,24,7), K_aa.shape (M,7,7), as in cal_condensed_nodal_force
    #dx.shape (M,8,3): change of the 8 points of the element in this iteration
    #d_alpha = -K_aa^-1 * (eas_force + K_ua^T * dx)   (FEElasticEASShellDomain::UpdateEAS in FEBio)
    M=dx.shape[0]
    d_alpha=-solve(K_aa, eas_force.view(M,7,1)+matmul(K_ua.transpose(1,2), dx.reshape(M,24,1)))
    return d_alpha.view(M,7) #(M,7)
