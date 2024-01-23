import torch

def cal_element_orientation(node, element):
    if element.shape[1] == 4: #tet4
        raise ValueError("only support hex8")
    elif element.shape[1] == 10: #tet10
        raise ValueError("only support hex8")
    elif element.shape[1] == 6: #wedge6
        raise ValueError("only support hex8")
    elif element.shape[1] == 8:
        ori=cal_element_orientation_hex8(node, element)
    else:
        raise ValueError("unsupported element type")
    return ori

def cal_element_orientation_hex8(node, element):
    x0=node[element[:,0]]
    x1=node[element[:,1]]
    x2=node[element[:,2]]
    x3=node[element[:,3]]
    x4=node[element[:,4]]
    x5=node[element[:,5]]
    x6=node[element[:,6]]
    x7=node[element[:,7]]
    a=(x1+x2+x5+x6)-(x0+x3+x4+x7)
    d=(x2+x3+x6+x7)-(x0+x1+x4+x5)
    e1=a/torch.norm(a, p=2, dim=1, keepdim=True)
    e3=torch.cross(a, d)
    e3=e3/torch.norm(e3, p=2, dim=1, keepdim=True)
    e2=torch.cross(e3, e1)
    e2=e2/torch.norm(e2, p=2, dim=1, keepdim=True)
    e1=e1.view(-1,3,1)
    e2=e2.view(-1,3,1)
    e3=e3.view(-1,3,1)
    orientation=torch.cat([e1, e2, e3], dim=2)
    return orientation
