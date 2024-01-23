#%%
import sys
sys.path.append("../../../pytorch_fea")
sys.path.append("../../../pytorch_fea/examples/aorta")
sys.path.append("../../../pytorch_fea/data/aorta")
sys.path.append("../../../mesh")
#%%
import torch
from PolyhedronMesh import PolyhedronMesh
from aorta_element_orientation import cal_element_orientation_hex8
#%%
def hex8_to_tet4(element_hex8):
    ori_tet4=[]
    element_tet4=[]
    Element_surface_pressure=[]
    for m in range(0, element_hex8.shape[0]):
        p0=int(element_hex8[m,0])
        p1=int(element_hex8[m,1])
        p2=int(element_hex8[m,2])
        p3=int(element_hex8[m,3])
        p4=int(element_hex8[m,4])
        p5=int(element_hex8[m,5])
        p6=int(element_hex8[m,6])
        p7=int(element_hex8[m,7])
        #-----------------------
        element_tet4.append([p1,p2,p3,p5])
        element_tet4.append([p4,p7,p5,p3])
        element_tet4.append([p0,p4,p5,p3])
        element_tet4.append([p2,p5,p6,p3])
        element_tet4.append([p5,p7,p6,p3])
        element_tet4.append([p0,p1,p3,p5])
        Element_surface_pressure.append([p0,p1,p3])
        Element_surface_pressure.append([p1,p2,p3])
        ori_tet4.append(ori_hex8[m].view(1,3,3))
        ori_tet4.append(ori_hex8[m].view(1,3,3))
        ori_tet4.append(ori_hex8[m].view(1,3,3))
        ori_tet4.append(ori_hex8[m].view(1,3,3))
        ori_tet4.append(ori_hex8[m].view(1,3,3))
        ori_tet4.append(ori_hex8[m].view(1,3,3))
    ori_tet4=torch.cat(ori_tet4, dim=0)
    element_tet4=torch.tensor(element_tet4, dtype=torch.int64)
    Element_surface_pressure=torch.tensor(Element_surface_pressure, dtype=torch.int64)
    return element_tet4, Element_surface_pressure, ori_tet4
#%%
if __name__ == '__main__':
    path="../../../pytorch_fea/data/aorta"
    aorta=PolyhedronMesh()
    aorta.load_from_torch(path+"/p0_171_solid.pt")
    ori_hex8=cal_element_orientation_hex8(aorta.node, aorta.element)

    element_tet4, Element_surface_pressure, ori_tet4=hex8_to_tet4(aorta.element)
    aorta.element=element_tet4
    aorta.element_data['orientation']=ori_tet4.view(-1,9)
    aorta.element_set['Element_surface_pressure']=Element_surface_pressure
    aorta.save_by_torch(path+"/p0_171_solid_tet4.pt")
    aorta.save_by_vtk(path+"/p0_171_solid_tet4.vtk")