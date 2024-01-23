#%%
import sys
sys.path.append("../../../pytorch_fea")
sys.path.append("../../../pytorch_fea/examples/aorta")
sys.path.append("../../../pytorch_fea/data/aorta")
sys.path.append("../../../mesh")
#%%
import torch
from TetrahedronMesh import TetrahedronMesh
from PolyhedronMesh import PolyhedronMesh
from aorta_element_orientation import cal_element_orientation_hex8
#%%
path="../../../pytorch_fea/data/aorta"
aorta_tet4=TetrahedronMesh()
aorta_tet4.load_from_torch(path+"/p0_171_solid_tet4.pt")
#%%
aorta_tet4.build_edge()
edge_tet4=aorta_tet4.edge
#%%
node_tet4=aorta_tet4.node
node_mid=0.5*(node_tet4[edge_tet4[:,0]]+node_tet4[edge_tet4[:,1]])
#%%
element_tet4=aorta_tet4.element
element_tet10=[]
for m in range(0, element_tet4.shape[0]):
    p0=int(element_tet4[m,0])
    p1=int(element_tet4[m,1])
    p2=int(element_tet4[m,2])
    p3=int(element_tet4[m,3])
    #find p4 in node_mid
    if p0 < p1:
        idx=torch.where((edge_tet4[:,0]==p0)&(edge_tet4[:,1]==p1))
    else:
        idx=torch.where((edge_tet4[:,0]==p1)&(edge_tet4[:,1]==p0))
    p4=int(idx[0])+node_tet4.shape[0]
    #find p5 in node_mid
    if p1 < p2:
        idx=torch.where((edge_tet4[:,0]==p1)&(edge_tet4[:,1]==p2))
    else:
        idx=torch.where((edge_tet4[:,0]==p2)&(edge_tet4[:,1]==p1))
    p5=int(idx[0])+node_tet4.shape[0]
    #find p6 in node_mid
    if p0 < p2:
        idx=torch.where((edge_tet4[:,0]==p0)&(edge_tet4[:,1]==p2))
    else:
        idx=torch.where((edge_tet4[:,0]==p2)&(edge_tet4[:,1]==p0))
    p6=int(idx[0])+node_tet4.shape[0]
    #find p7 in node_mid
    if p0 < p3:
        idx=torch.where((edge_tet4[:,0]==p0)&(edge_tet4[:,1]==p3))
    else:
        idx=torch.where((edge_tet4[:,0]==p3)&(edge_tet4[:,1]==p0))
    p7=int(idx[0])+node_tet4.shape[0]
    #find p8 in node_mid
    if p1 < p3:
        idx=torch.where((edge_tet4[:,0]==p1)&(edge_tet4[:,1]==p3))
    else:
        idx=torch.where((edge_tet4[:,0]==p3)&(edge_tet4[:,1]==p1))
    p8=int(idx[0])+node_tet4.shape[0]
    #find p9 in node_mid
    if p2 < p3:
        idx=torch.where((edge_tet4[:,0]==p2)&(edge_tet4[:,1]==p3))
    else:
        idx=torch.where((edge_tet4[:,0]==p3)&(edge_tet4[:,1]==p2))
    p9=int(idx[0])+node_tet4.shape[0]
    #---
    element_tet10.append([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9])
    print("m", m)
#%%
surface_tet4=aorta_tet4.element_set['Element_surface_pressure']
surface_tet10=[]
for m in range(0, surface_tet4.shape[0]):
    p0=int(surface_tet4[m,0])
    p1=int(surface_tet4[m,1])
    p2=int(surface_tet4[m,2])
    #find p3 in node_mid
    if p0 < p1:
        idx=torch.where((edge_tet4[:,0]==p0)&(edge_tet4[:,1]==p1))
    else:
        idx=torch.where((edge_tet4[:,0]==p1)&(edge_tet4[:,1]==p0))
    p3=int(idx[0])+node_tet4.shape[0]
    #find p4 in node_mid
    if p1 < p2:
        idx=torch.where((edge_tet4[:,0]==p1)&(edge_tet4[:,1]==p2))
    else:
        idx=torch.where((edge_tet4[:,0]==p2)&(edge_tet4[:,1]==p1))
    p4=int(idx[0])+node_tet4.shape[0]
    #find p5 in node_mid
    if p0 < p2:
        idx=torch.where((edge_tet4[:,0]==p0)&(edge_tet4[:,1]==p2))
    else:
        idx=torch.where((edge_tet4[:,0]==p2)&(edge_tet4[:,1]==p0))
    p5=int(idx[0])+node_tet4.shape[0]
    surface_tet10.append([p0,p1,p2,p3,p4,p5])
#%%
boundary0_tet4=aorta_tet4.node_set['boundary0']
boundary0A_tet4=boundary0_tet4[0:50]
boundary0B_tet4=boundary0_tet4[50:]
boundary0_tet10=[]
for n in range(0,len(boundary0A_tet4)):
    pn=int(boundary0A_tet4[n])
    if n < len(boundary0A_tet4)-1:
        pn1=int(boundary0A_tet4[n+1])
    else:
        pn1=int(boundary0A_tet4[0])
    if pn < pn1:
        idx=torch.where((edge_tet4[:,0]==pn)&(edge_tet4[:,1]==pn1))
    else:
        idx=torch.where((edge_tet4[:,0]==pn1)&(edge_tet4[:,1]==pn))
    if len(idx[0]) > 0:
        p_mid=int(idx[0])+node_tet4.shape[0]
        boundary0_tet10.append(pn)
        boundary0_tet10.append(p_mid)
for n in range(0,len(boundary0B_tet4)):
    pn=int(boundary0B_tet4[n])
    if n < len(boundary0B_tet4)-1:
        pn1=int(boundary0B_tet4[n+1])
    else:
        pn1=int(boundary0B_tet4[0])
    if pn < pn1:
        idx=torch.where((edge_tet4[:,0]==pn)&(edge_tet4[:,1]==pn1))
    else:
        idx=torch.where((edge_tet4[:,0]==pn1)&(edge_tet4[:,1]==pn))
    if len(idx[0]) > 0:
        p_mid=int(idx[0])+node_tet4.shape[0]
        boundary0_tet10.append(pn)
        boundary0_tet10.append(p_mid)
for i in range(0,len(boundary0A_tet4)):
    pi=int(boundary0A_tet4[i])
    for j in range(0,len(boundary0B_tet4)):
        pj=int(boundary0B_tet4[j])
        if pi < pj:
            idx=torch.where((edge_tet4[:,0]==pi)&(edge_tet4[:,1]==pj))
        else:
            idx=torch.where((edge_tet4[:,0]==pj)&(edge_tet4[:,1]==pi))
        if len(idx[0]) > 0:
            p_mid=int(idx[0])+node_tet4.shape[0]
            boundary0_tet10.append(p_mid)
#%%
boundary1_tet4=aorta_tet4.node_set['boundary1']
boundary1A_tet4=boundary1_tet4[0:50]
boundary1B_tet4=boundary1_tet4[50:]
boundary1_tet10=[]
for n in range(0,len(boundary1A_tet4)):
    pn=int(boundary1A_tet4[n])
    if n < len(boundary1A_tet4)-1:
        pn1=int(boundary1A_tet4[n+1])
    else:
        pn1=int(boundary1A_tet4[0])
    if pn < pn1:
        idx=torch.where((edge_tet4[:,0]==pn)&(edge_tet4[:,1]==pn1))
    else:
        idx=torch.where((edge_tet4[:,0]==pn1)&(edge_tet4[:,1]==pn))
    if len(idx[0]) > 0:
        p_mid=int(idx[0])+node_tet4.shape[0]
        boundary1_tet10.append(pn)
        boundary1_tet10.append(p_mid)
for n in range(0,len(boundary1B_tet4)):
    pn=int(boundary1B_tet4[n])
    if n < len(boundary1B_tet4)-1:
        pn1=int(boundary1B_tet4[n+1])
    else:
        pn1=int(boundary1B_tet4[0])
    if pn < pn1:
        idx=torch.where((edge_tet4[:,0]==pn)&(edge_tet4[:,1]==pn1))
    else:
        idx=torch.where((edge_tet4[:,0]==pn1)&(edge_tet4[:,1]==pn))
    if len(idx[0]) > 0:
        p_mid=int(idx[0])+node_tet4.shape[0]
        boundary1_tet10.append(pn)
        boundary1_tet10.append(p_mid)
for i in range(0,len(boundary1A_tet4)):
    pi=int(boundary1A_tet4[i])
    for j in range(0,len(boundary1B_tet4)):
        pj=int(boundary1B_tet4[j])
        if pi < pj:
            idx=torch.where((edge_tet4[:,0]==pi)&(edge_tet4[:,1]==pj))
        else:
            idx=torch.where((edge_tet4[:,0]==pj)&(edge_tet4[:,1]==pi))
        if len(idx[0]) > 0:
            p_mid=int(idx[0])+node_tet4.shape[0]
            boundary1_tet10.append(p_mid)
#%%
boundary0_tet10=torch.tensor(boundary0_tet10, dtype=torch.int64)
boundary1_tet10=torch.tensor(boundary1_tet10, dtype=torch.int64)
surface_tet10=torch.tensor(surface_tet10, dtype=torch.int64)
element_tet10=torch.tensor(element_tet10, dtype=torch.int64)
aorta_tet10=PolyhedronMesh()
aorta_tet10.element=element_tet10
aorta_tet10.node=torch.cat([node_tet4, node_mid], dim=0)
aorta_tet10.element_set['Element_surface_pressure']=surface_tet10
aorta_tet10.element_data['orientation']=aorta_tet4.element_data['orientation']
aorta_tet10.node_set['boundary0']=boundary0_tet10
aorta_tet10.node_set['boundary1']=boundary1_tet10
aorta_tet10.save_by_torch(path+"/p0_171_solid_tet10.pt")
aorta_tet10.save_by_vtk(path+"/p0_171_solid_tet10.vtk")