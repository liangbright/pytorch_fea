#based on aorta_FEA_C3D8_SRI_inverse_mat_ex_vivo.py
#%%
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
import sys
sys.path.append('../../../pytorch_fea')
sys.path.append('../../../pytorch_fea/example/aorta')
sys.path.append('../../../pytorch_fea_mesh')
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch_fea.optimizer.FE_lbfgs_loss import LBFGS
from torch_fea_mesh.processing.polyhedron_mesh_processing import TetrahedronMesh, Tet10Mesh, HexahedronMesh
import time
#%%
#attention:
#    run aorta_FEA_forward_inflation.py to generate mesh_px
#    mat_model for inverse_mat should be the same as the mat_model for inflation
all_mat=torch.load('../../../pytorch_fea/data/aorta/125mat.pt', weights_only=False)['mat_str']
matMean=torch.load('../../../pytorch_fea/data/aorta/125mat.pt', weights_only=False)['mean_mat_str']
px_pressure=16
mat_model='GOH' #'GOH', 'UFD', or 'NeoHookean'
if mat_model == 'GOH':
    mat_true=matMean
    mat_name='matMean'
elif mat_model == 'UFD':
    #UFD [c0, k1, k2, zeta, k] in kPa, the representative set of aorta_FEA_QN_forward_inflation.py
    mat_true='160, 440, 30, 0.55, 1e5'
    mat_name='matUFD'
elif mat_model == 'NeoHookean':
    #NeoHookean [C10, D1] in kPa and 1/kPa, the representative set of aorta_FEA_QN_forward_inflation.py
    mat_true='210, 2e-5'
    mat_name='matNH'
else:
    raise ValueError('unknown mat_model: '+mat_model)
shape_id='171' #[24,150,168,171,174,192,318]
element_type_str='hex8'
mesh_p0_str='../../../pytorch_fea/data/aorta/p0_'+str(shape_id)+'_solid'
if element_type_str != 'hex8':
    mesh_p0_str=mesh_p0_str+'_'+element_type_str
mesh_px_str=('../../../pytorch_fea/example/aorta/result/inflation/'
             +'p0_'+str(shape_id)+'_solid_'+element_type_str+'_'+mat_model+'_'+mat_name+'_p'+str(px_pressure))
folder_result='../../../pytorch_fea/example/aorta/result/inverse_mat_ex_vivo'
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=2, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--folder_result', default=folder_result, type=str)
parser.add_argument('--mat_model', default=mat_model, type=str)
parser.add_argument('--mat_true', default=mat_true, type=str)#
parser.add_argument('--mesh_p0', default=mesh_p0_str, type=str)
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--element_type', default=element_type_str, type=str)# hex8, tet4, tet10
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
if 'hex8' in arg.element_type:
    PolyhedronMesh=HexahedronMesh
elif 'tet4' in arg.element_type:
    PolyhedronMesh=TetrahedronMesh
elif 'tet10' in arg.element_type:
    PolyhedronMesh=Tet10Mesh
else:
    raise ValueError('unsupported element_type: '+arg.element_type)      
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
try:
    print("loss1", Mesh_x.mesh_data['loss1'][-1])
except:
    pass
#%%
from AortaFEModel import AortaFEModel
#------------------
#material_GOH, material_UFD, material_NeoHookean: the model passes V (the volume of each integration point) to the material functions,
#and the volumetric part is evaluated at the element mean dilatation (no volumetric locking)
if arg.mat_model == "GOH":
    from torch_fea.material.material_GOH import cal_1pk_stress
elif arg.mat_model == "UFD":
    from torch_fea.material.material_UFD import cal_1pk_stress
elif arg.mat_model == "NeoHookean":
    #material_NeoHookean is isotropic and takes no orientation: drop the orientation that the model passes
    from torch_fea.material import material_NeoHookean
    def cal_1pk_stress(F, Mat, Orientation, create_graph, return_W, V=None):
        return material_NeoHookean.cal_1pk_stress(F, Mat, create_graph, return_W, V=V)
else:
    raise ValueError("this file does not supports :"+arg.mat_model)
#%%
#each parameter is min+(max-min)*sigmoid(m), m=0 at the start (the middle of the range); k (D1 for NeoHookean) is fixed
m0=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
m1=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
m2=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
m3=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
m4=torch.tensor(0,dtype=dtype, device=device, requires_grad=True)
if arg.mat_model == "GOH":
    m0_min=0;    m0_max=120
    m1_min=0;    m1_max=6000
    m2_min=0.1; m2_max=60
    m_list=[m0, m1, m2, m3, m4]
    def get_Mat(m0, m1, m2, m3, m4):
        Mat=torch.zeros((1,6),dtype=dtype, device=device)
        Mat[0,0]=m0_min+(m0_max-m0_min)*torch.sigmoid(m0)
        Mat[0,1]=m1_min+(m1_max-m1_min)*torch.sigmoid(m1)
        Mat[0,2]=m2_min+(m2_max-m2_min)*torch.sigmoid(m2)
        Mat[0,3]=(1/3)*torch.sigmoid(m3)
        Mat[0,4]=(np.pi/2)*torch.sigmoid(m4)
        Mat[0,5]=1e5
        return Mat
elif arg.mat_model == "UFD":
    #Mat = [c0, k1, k2, zeta, k]: four parameters are fitted, zeta in [0, 1]
    m0_min=0;    m0_max=400
    m1_min=0;    m1_max=2000
    m2_min=0.1; m2_max=100
    m_list=[m0, m1, m2, m3]
    def get_Mat(m0, m1, m2, m3):
        Mat=torch.zeros((1,5),dtype=dtype, device=device)
        Mat[0,0]=m0_min+(m0_max-m0_min)*torch.sigmoid(m0)
        Mat[0,1]=m1_min+(m1_max-m1_min)*torch.sigmoid(m1)
        Mat[0,2]=m2_min+(m2_max-m2_min)*torch.sigmoid(m2)
        Mat[0,3]=torch.sigmoid(m3)
        Mat[0,4]=1e5
        return Mat
elif arg.mat_model == "NeoHookean":
    #Mat = [C10, D1]: C10 is fitted, D1 is fixed
    m0_min=0;    m0_max=600
    m_list=[m0]
    def get_Mat(m0):
        Mat=torch.zeros((1,2),dtype=dtype, device=device)
        Mat[0,0]=m0_min+(m0_max-m0_min)*torch.sigmoid(m0)
        Mat[0,1]=2e-5
        return Mat
#%%
Mat_init=get_Mat(*m_list)
#check --mat_true before the fit: it is used only at the end
Mat_true=[float(m) for m in arg.mat_true.split(",")]
if len(Mat_true) != Mat_init.shape[1]:
    raise ValueError("--mat_true has "+str(len(Mat_true))+" values, but "+arg.mat_model+" has "+str(Mat_init.shape[1])
                     +": give --mat_true for "+arg.mat_model+" (the default is for mat_model='"+mat_model+"' at the top)")
if arg.mat_model == "GOH":
    Mat_true[4]=np.pi*(Mat_true[4]/180) #UFD has no fiber angle
Mat_true=np.array(Mat_true)
aorta_model=AortaFEModel(arg.element_type, Node_x, Element, Node_X, boundary0, boundary1, Element_surface_pressure,
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
optimizer = LBFGS(m_list, lr=arg.lr, line_search_fn="strong_wolfe",
                  tolerance_grad =1e-5, tolerance_change=1e-10, history_size =100)
optimizer.set_backtracking(t_list=[1, 0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-10], c=0.0001, verbose=False)
#%%
#run_end records how the loop ended (as run_end in the forward, GPA and inverse p0 scripts) and sets the file name:
#   'opt_cond':   the optimizer's test, max abs gradient <= tolerance_grad          -> the final name
#   'Mat_change': the material parameters changed by less than 1e-5 in one iteration -> the final name
#   'max_iter':   arg.max_iter iterations without either                            -> suffix _stopped
#   'nan', 'inf': the loss is NaN or inf (the optimizer then returns 'nan' or 'inf' and does not move)
#                 -> suffix _aborted, and Mat_pred is the last parameters with a finite loss
run_end='max_iter'
for iter1 in range(0, arg.max_iter):
    def closure(loss_fn="SSE"):
        Mat=get_Mat(*m_list)
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
    if isinstance(opt_cond, str):
        #'nan' or 'inf': the loss at the current parameters is not finite, and the optimizer did not move
        run_end=opt_cond
        print("abort: loss is", opt_cond, ", iter1 =", iter1)
        break

    loss=float(closure(loss_fn="MAE"))
    loss_list.append(loss)
    t1=time.time()
    time_list.append(t1-t0)

    Mat=get_Mat(*m_list)
    Mat_list.append(Mat.detach().cpu().numpy().tolist())
    if not np.isfinite(loss):
        run_end='nan' if np.isnan(loss) else 'inf'
        print("abort: loss is", run_end, ", iter1 =", iter1)
        break

    Mat_change=False
    if len(Mat_list) > 2:
        Mat0=np.array(Mat_list[-2])
        Mat1=np.array(Mat_list[-1])
        a=np.abs(Mat0-Mat1).max()
        if a < 1e-5:
            Mat_change=True

    if iter1%100 == 0 or opt_cond==True or Mat_change==True:
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
        run_end='opt_cond'
        print(iter1, loss, t1-t0)
        print("break: opt_cond is True")
        break
    if Mat_change == True:
        run_end='Mat_change'
        print(iter1, loss, t1-t0)
        print("break: Mat change < 1e-5")
        break
if run_end == 'max_iter':
    print("stop_at_max_iter:", arg.max_iter, "iterations without convergence, loss =",
          loss_list[-1] if len(loss_list) > 0 else None)
print("run_end:", run_end)
#%%
if run_end in ('nan', 'inf'):
    #Mat_list[i+1] goes with loss_list[i]; Mat_list[0] is the initial guess
    finite=[i for i, l in enumerate(loss_list) if np.isfinite(l)]
    Mat_pred=np.array(Mat_list[finite[-1]+1] if len(finite) > 0 else Mat_list[0]).reshape(-1)
else:
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
if run_end == 'max_iter':
    filename=filename+'_stopped'
elif run_end in ('nan', 'inf'):
    filename=filename+'_aborted'
#loss: the MAE of force_int - force_ext after each iteration; run_end: how the loop ended (see above the loop)
torch.save({"arg":arg,
            "Mat":Mat_list,
            'Mat_true':Mat_true,
            'Mat_pred':Mat_pred,
            "time":time_list,
            "loss":loss_list,
            "run_end":run_end},
           filename+".pt")
print("save", filename, ", run_end:", run_end)
#'''