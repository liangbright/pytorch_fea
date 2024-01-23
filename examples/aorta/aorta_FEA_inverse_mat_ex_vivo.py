#based on aorta_FEA_C3D8_SRI_inverse_mat_ex_vivo.py
import sys
sys.path.append("../../../pytorch_fea")
sys.path.append("../../../pytorch_fea/examples/aorta")
sys.path.append("../../../mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch_fea.optimizer.FE_lbfgs_ori import LBFGS
from PolyhedronMesh import PolyhedronMesh
import time
#%%
#attention:
#    run aorta_FEA_forward_inflation.py to generate mesh_px
#    mat_model for inverse_mat should be the same as the mat_model for inflation
all_mat=torch.load('../../../pytorch_fea/data/aorta/125mat.pt')['mat_str']
matMean=torch.load('../../../pytorch_fea/data/aorta/125mat.pt')['mean_mat_str']
mat_true=matMean
mat_name='matMean'
px_pressure=20
mat_model='GOH_Jv'
shape_id='171' #[24,150,168,171,174,192,318]
element_type='tet10'
mesh_p0_str='../../../pytorch_fea/data/aorta/p0_'+str(shape_id)+'_solid_'+element_type
mesh_px_str=('../../../pytorch_fea/examples/aorta/result/inflation/'
             +'p0_'+str(shape_id)+'_solid_'+element_type+'_'+mat_model+'_'+mat_name+'_p'+str(px_pressure))
folder_result='../../../pytorch_fea/examples/aorta/result/inverse_mat_ex_vivo'
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--folder_result', default=folder_result, type=str)
parser.add_argument('--mat_model', default=mat_model, type=str)
parser.add_argument('--mat_true', default=mat_true, type=str)#
parser.add_argument('--mesh_p0', default=mesh_p0_str, type=str)
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--pressure', default=px_pressure, type=float)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--max_iter', default=500, type=int)
arg = parser.parse_args()
print(arg)
#%%
if arg.cuda >=0:
    device=torch.device("cuda:"+str(arg.cuda))
else:
    device=torch.device("cpu")
if arg.dtype == "float64":
    dtype=torch.float64
elif arg.dtype == "float32":
    dtype=torch.float32
else:
    raise ValueError("unkown dtype:"+arg.dtype)
#%%
Mesh_X=PolyhedronMesh()
Mesh_X.load_from_torch(arg.mesh_p0+".pt")
Node_X=Mesh_X.node.to(dtype).to(device)
Element=Mesh_X.element.to(device)
#%%
boundary0=Mesh_X.node_set['boundary0']
boundary1=Mesh_X.node_set['boundary1']
Element_surface_pressure=Mesh_X.element_set['Element_surface_pressure']
try:
    ElementOrientation=Mesh_X.element_data['orientation'].view(-1,3,3)
except:
    ElementOrientation=None
#%%
Mesh_x=PolyhedronMesh()
Mesh_x.load_from_torch(arg.mesh_px+".pt")
Node_x=Mesh_x.node.to(dtype).to(device)
print("loss1", Mesh_x.mesh_data['loss1'][-1])
#%%
if "SRI" in arg.mat_model:
    from AortaFEModel_SRI import AortaFEModel
else:
    from AortaFEModel import AortaFEModel
#------------------
if arg.mat_model == "GOH_SRI":
    from torch_fea.material.Mat_GOH_SRI import cal_1pk_stress
elif arg.mat_model == "GOH":
    from torch_fea.material.Mat_GOH import cal_1pk_stress
elif arg.mat_model == "GOH_Fbar":
    from torch_fea.material.Mat_GOH_Fbar import cal_1pk_stress
elif arg.mat_model == "GOH_Jv":
    from torch_fea.material.Mat_GOH_Jv import cal_1pk_stress
elif arg.mat_model == "GOH_3Field":
    from torch_fea.material.Mat_GOH_3Field import cal_1pk_stress
else:
    raise ValueError("this file does not supports :"+arg.mat_model)
#%%
m0_min=0;    m0_max=120
m1_min=0;    m1_max=6000
m2_min=0.1; m2_max=60
m0=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
m1=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
m2=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
m3=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
m4=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
def get_Mat(m0, m1, m2, m3, m4):
    Mat=torch.zeros((1,6),dtype=dtype, device=device)
    Mat[0,0]=m0_min+(m0_max-m0_min)*torch.sigmoid(m0)
    Mat[0,1]=m1_min+(m1_max-m1_min)*torch.sigmoid(m1)
    Mat[0,2]=m2_min+(m2_max-m2_min)*torch.sigmoid(m2)
    Mat[0,3]=(1/3)*torch.sigmoid(m3)
    Mat[0,4]=(np.pi/2)*torch.sigmoid(m4)
    Mat[0,5]=1e5
    return Mat
#%%
Mat_init=get_Mat(m0, m1, m2, m3, m4)
aorta_model=AortaFEModel(Node_x, Element, Node_X, boundary0, boundary1, Element_surface_pressure,
                         Mat_init, ElementOrientation, cal_1pk_stress, dtype, device, mode='inverse_mat')
pressure=arg.pressure
#%%
def loss_function(A, B, reduction):
    Res=A-B
    if reduction == "MSE":
        loss=(Res**2).mean()
    elif reduction == "RMSE":
        loss=(Res**2).mean().sqrt()
    elif reduction == "MAE":
        loss=Res.abs().mean()
    elif reduction == "SSE":
        loss=(Res**2).sum()
    elif reduction == "SAE":
        loss=Res.abs().sum()
    return loss
#%%
loss_list=[]
time_list=[]
Mat_list=[Mat_init.detach().cpu().numpy().tolist()]
t0=time.time()
#%%
optimizer = LBFGS([m0, m1, m2, m3, m4], lr=arg.lr, line_search_fn="strong_wolfe",
                  tolerance_grad =1e-5, tolerance_change=1e-10, history_size =100, max_iter=1)
optimizer.set_backtracking(t_list=[1, 0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-10], c=0.0001, verbose=False)
#%%
for iter1 in range(0, arg.max_iter):
    def closure(loss_fn="SSE"):
        Mat=get_Mat(m0, m1, m2, m3, m4)
        aorta_model.set_material(Mat)
        out =aorta_model.cal_energy_and_force(pressure)
        force_int=out['force_int']
        force_ext=out['force_ext']
        loss=loss_function(force_int, force_ext, loss_fn)
        if loss.requires_grad==True:
            optimizer.zero_grad()
            loss.backward()
        return loss
    opt_cond=optimizer.step(closure)

    loss=float(closure(loss_fn="MAE"))
    loss_list.append(loss)
    t1=time.time()
    time_list.append(t1-t0)

    Mat=get_Mat(m0, m1, m2, m3, m4)
    Mat_list.append(Mat.detach().cpu().numpy().tolist())
    if len(Mat_list) > 2:
        Mat0=np.array(Mat_list[-2])
        Mat1=np.array(Mat_list[-1])
        a=np.abs(Mat0-Mat1).max()
        if a < 1e-5:
            opt_cond=True

    if iter1%100 == 0 or opt_cond==True:
        print(iter1, loss, t1-t0)
        print("Mat:", Mat_list[-1])
        display.clear_output(wait=False)
        fig, ax = plt.subplots()
        ax.plot(np.array(loss_list)/max(loss_list), 'r')
        ax.set_ylim(0, 1)
        ax.grid(True)
        display.display(fig)
        plt.close(fig)

    if opt_cond == True:
        print(iter1, loss, t1-t0)
        print("break: opt_cond is True")
        break
#%%
Mat_true=[float(m) for m in arg.mat_true.split(",")]
Mat_true[4]=np.pi*(Mat_true[4]/180)
Mat_true=np.array(Mat_true)
Mat_pred=np.array(Mat_list[-1]).reshape(-1)
error=np.abs(Mat_pred-Mat_true)/Mat_true
print("error", error)
#%%
import os
if os.path.exists(arg.folder_result) == False:
    os.makedirs(arg.folder_result)
#%%
#'''
filename=arg.folder_result+'/'+arg.mesh_px.split('/')[-1]+"_ex_vivo_mat"
torch.save({"arg":arg,
            "Mat":Mat_list,
            'Mat_ture':Mat_true,
            'Mat_pred':Mat_pred,
            "time":time_list},
           filename+".pt")
print("save", filename)
#'''