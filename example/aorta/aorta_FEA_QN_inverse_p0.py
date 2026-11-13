#based on aorta_FEA_C3D8_SRI_quasi_newton_inverse_p0_R1.py, QN=quasi_newton
#this method works well for small deformation between mesh_X and and mesh_x
#it is good for stress analysis based on statical determinacy
#
#the step size after a failed load step: auto_stepper.goback(t_good, 'rand') draws the next step size at random,
#uniformly on a log scale between --min_step_size and --max_step_size (AutoStepper.rand_Δt), so every decade between
#the two is equally likely: --min_step_size is not only a floor, it also sets where most retries land.
#with the defaults 1e-10 and 0.1 (9 decades) the median draw is 3e-6, and 78% of the draws after a failed step of 0.03
#are below 1e-3; at the growth of 1.5 per converged load step, 3e-6 needs about 20 load steps to get back to 0.01.
#two ways to keep the random draw and narrow it, neither is better in general:
#  (1) a higher floor: e.g. --min_step_size 1e-5 (the value of the forward example) or 1e-4; only an option, no code change
#  (2) a floor relative to the failed load step: draw between Δt_failed/100 and --max_step_size, so a retry is at most
#      two decades below the failed step size and can still be larger than it; not implemented, it needs a change of
#      the call goback(t_good, 'rand') below
#with --tangent exact a load step rarely fails, so the draw is rarely used
#%%
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
import sys
#note: change the following three lines if necessary
sys.path.append('../../../pytorch_fea')
sys.path.append('../../../pytorch_fea/example/aorta')
sys.path.append('../../../pytorch_fea_mesh')
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch.linalg import det
from torch_fea.utility.function import cal_attribute_on_node, cal_von_mises_stress
from torch_fea_mesh.processing.polyhedron_mesh_processing import TetrahedronMesh, Tet10Mesh, HexahedronMesh
import time
#%%
try:
    all_mat=torch.load('../../../pytorch_fea/data/aorta/125mat.pt', weights_only=False)['mat_str']
    matMean=torch.load('../../../pytorch_fea/data/aorta/125mat.pt', weights_only=False)['mean_mat_str']
except:
    pass
px_pressure=16
mat_model='GOH' #'GOH', 'UFD', or 'NeoHookean'
if mat_model == 'GOH':
    mat_str=matMean; mat_name='matMean'
    #----------------------------------------------------------
    #use this stiff mat for statical determinacy approach
    #mat_str='10000, 0, 1, 0.3333, 0, 1e5'; mat_name='1e4'
    #----------------------------------------------------------
elif mat_model == 'UFD':
    #UFD [c0, k1, k2, zeta, k] in kPa, the representative set of aorta_FEA_QN_forward_inflation.py
    mat_str='160, 440, 30, 0.55, 1e5'; mat_name='matUFD'
elif mat_model == 'NeoHookean':
    #NeoHookean [C10, D1] in kPa and 1/kPa, the representative set of aorta_FEA_QN_forward_inflation.py
    mat_str='210, 2e-5'; mat_name='matNH'
else:
    raise ValueError('unknown mat_model: '+mat_model)
shape_id='171' #[24,150,168,171,174,192,318]
element_type_str='hex8'  #'hex8', 'tet10', 'tet4'
mesh_p0_str='../../../pytorch_fea/data/aorta/p0_'+str(shape_id)+'_solid'
if element_type_str != 'hex8':
    mesh_p0_str=mesh_p0_str+'_'+element_type_str
mesh_px_str=('../../../pytorch_fea/example/aorta/result/inflation/'
             +'p0_'+str(shape_id)+'_solid_'+element_type_str+'_'+mat_model+'_'+mat_name+'_p'+str(px_pressure))
mesh_inverse_p0_str=mesh_px_str+'_QN_inverse_p0'
mesh_inverse_p0_str=mesh_inverse_p0_str.replace('inflation', 'inverse_p0')
#%%
def get_must_points(delta):
    #pressure=20, delta=0.05: 20 must points
    t=0
    n=0
    t_str='0'
    while t<1:
        n=n+1
        t=delta*n
        ts='{:.2f}'.format(t)
        t_str=t_str+','+ts
    return t_str
#pressure=20, delta=0.05: 20 must points
must_points=get_must_points(0.05)
must_points=''
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=2, type=int)
parser.add_argument('--dtype', default='float64', type=str)
parser.add_argument('--element_type', default=element_type_str, type=str)# hex8, tet4, tet10
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--mesh_p0', default=mesh_inverse_p0_str, type=str)
parser.add_argument('--mesh_p0_init', default='', type=str)
parser.add_argument('--t_start_init', default=-1, type=float)
parser.add_argument('--mat', default=mat_str, type=str)
parser.add_argument('--mat_model', default=mat_model, type=str)
parser.add_argument('--pressure', default=px_pressure, type=float)
parser.add_argument('--optimizer', default='FE_lbfgs_residual', type=str)#FE_lbfgs_loss, FE_lbfgs_residual
#tangent: the stiffness matrix given to the optimizer
#  exact:   the derivative of the residual with respect to the unknown displacement (Newton's method)
#  forward: the stiffness matrix of the forward problem (the method of this file before --tangent was added)
parser.add_argument('--tangent', default='exact', type=str)#exact, forward
parser.add_argument('--init_step_size', default=0.01, type=float)
parser.add_argument('--min_step_size', default=1e-10, type=float)
parser.add_argument('--max_step_size', default=0.1, type=float)
#the defaults of max_t_linesearch, min_t_linesearch and reform_stiffness_interval depend on --tangent
#  exact:   max_t_linesearch 1,   min_t_linesearch 1e-3,  reform_stiffness_interval 1
#  forward: max_t_linesearch 0.5, min_t_linesearch 1e-10, reform_stiffness_interval 10
parser.add_argument('--max_t_linesearch', default=None, type=float)
parser.add_argument('--min_t_linesearch', default=None, type=float)
parser.add_argument('--default_t_linesearch', default=1e-3, type=float)
parser.add_argument('--reform_stiffness_interval', default=None, type=int)
parser.add_argument('--max_iter1', default=int(1e10), type=int)
parser.add_argument('--max_iter2', default=1000, type=int)
#max_stall_iter2: a load step ends as failed when Rmax has not dropped by 10% (below 0.9 times its value at the last
#                 such drop) within this number of iterations (a diverged load step would otherwise run to max_iter2)
parser.add_argument('--max_stall_iter2', default=50, type=int)
parser.add_argument('--U_ratio', default=1e-2, type=float)
parser.add_argument('--loss1', default=0.01, type=float)
parser.add_argument('--Rmax', default=0.005, type=float)
parser.add_argument('--max_retry', default=100, type=int)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--must_points', default=must_points, type=str)
parser.add_argument('--save_all_t', default='False', type=str, choices=['True', 'False']) #True: save all (converged) time points
parser.add_argument('--save_by_vtk', default='True', type=str, choices=['True', 'False'])#True: save *.vtk file in addition to *.pt file
parser.add_argument('--plot', default='False', type=str, choices=['True', 'False'])#plot loss1, TPE, etc vs sim_t
arg = parser.parse_args()
if not arg.pressure > 0: #with no load, Rmax (relative to the forces) is rounding over rounding and never converges
    raise ValueError("--pressure must be > 0, got "+str(arg.pressure))
if arg.tangent == 'exact':
    default_of_tangent={'max_t_linesearch':1.0, 'min_t_linesearch':1e-3, 'reform_stiffness_interval':1}
elif arg.tangent == 'forward':
    default_of_tangent={'max_t_linesearch':0.5, 'min_t_linesearch':1e-10, 'reform_stiffness_interval':10}
else:
    raise ValueError('unknown tangent:'+arg.tangent)
for name, value in default_of_tangent.items():
    if getattr(arg, name) is None:
        setattr(arg, name, value)
arg.save_all_t = True if arg.save_all_t == 'True' else False
arg.save_by_vtk = True if arg.save_by_vtk == 'True' else False
arg.plot = True if arg.plot == 'True' else False
print(arg)
#%% create the folder of the result before the solve, so the result is not lost at the save
folder_result=os.path.dirname(arg.mesh_p0)
if len(folder_result) > 0:
    os.makedirs(folder_result, exist_ok=True)
#%%
if arg.cuda >=0:
    device=torch.device('cuda:'+str(arg.cuda))
else:
    device=torch.device('cpu')
if arg.dtype == 'float64':
    dtype=torch.float64
elif arg.dtype == 'float32':
    dtype=torch.float32
else:
    raise ValueError('unkown dtype:'+arg.dtype)
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
Mesh_x=PolyhedronMesh()
Mesh_x.load_from_torch(arg.mesh_px+'.pt')
Node_x=Mesh_x.node.to(dtype).to(device)
Element=Mesh_x.element.to(device)
#%%
boundary0=Mesh_x.node_set['boundary0']
boundary1=Mesh_x.node_set['boundary1']
Element_surface_pressure=Mesh_x.element_set['Element_surface_pressure']
#%%
from AortaFEModel import AortaFEModel, cal_potential_energy
#------------------
#material_GOH, material_UFD, material_NeoHookean: the model passes V (the volume of each integration point) to the material functions,
#and the volumetric part is evaluated at the element mean dilatation (no volumetric locking)
if arg.mat_model == 'GOH':
    from torch_fea.material.material_GOH import cal_1pk_stress
elif arg.mat_model == 'UFD':
    from torch_fea.material.material_UFD import cal_1pk_stress
elif arg.mat_model == 'NeoHookean':
    #material_NeoHookean is isotropic and takes no orientation: drop the orientation that the model passes
    from torch_fea.material import material_NeoHookean
    def cal_1pk_stress(F, Mat, Orientation, create_graph, return_W, V=None):
        return material_NeoHookean.cal_1pk_stress(F, Mat, create_graph, return_W, V=V)
else:
    raise ValueError('this file does not support:'+arg.mat_model)
#------------------
if 'GOH' in arg.mat_model:
    if 'distribution' in arg.mat:
        Mat=eval(arg.mat) # run generate_mat_distribution(?)
        Mat=Mat.clone()
        Mat[:,4]=np.pi*(Mat[:,4]/180)
        Mat=Mat.to(dtype=dtype, device=device)
    else:
        Mat=[float(m) for m in arg.mat.split(',')]
        Mat[4]=np.pi*(Mat[4]/180)
        Mat=torch.tensor([Mat], dtype=dtype, device=device)
elif arg.mat_model == 'UFD':
    #Mat = [c0, k1, k2, zeta, k]: no fiber angle, so no conversion from degrees
    if 'distribution' in arg.mat:
        Mat=eval(arg.mat) # run generate_mat_distribution(?)
        Mat=Mat.clone().to(dtype=dtype, device=device)
    else:
        Mat=[float(m) for m in arg.mat.split(',')]
        Mat=torch.tensor([Mat], dtype=dtype, device=device)
    if Mat.shape[1] != 5:
        raise ValueError('UFD needs 5 parameters: c0, k1, k2, zeta, k; got '+str(Mat.shape[1]))
    if ((Mat[:,0]<0).any() or (Mat[:,1]<0).any() or (Mat[:,2]<0).any()
        or (Mat[:,3]<0).any() or (Mat[:,3]>1).any() or (Mat[:,4]<=0).any()):
        raise ValueError('UFD needs c0>=0, k1>=0, k2>=0, 0<=zeta<=1, k>0')
elif arg.mat_model == 'NeoHookean':
    #Mat = [C10, D1]: initial shear modulus 2*C10, initial bulk modulus 2/D1
    if 'distribution' in arg.mat:
        Mat=eval(arg.mat) # run generate_mat_distribution(?)
        Mat=Mat.clone().to(dtype=dtype, device=device)
    else:
        Mat=[float(m) for m in arg.mat.split(',')]
        Mat=torch.tensor([Mat], dtype=dtype, device=device)
    if Mat.shape[1] != 2:
        raise ValueError('NeoHookean needs 2 parameters: C10, D1; got '+str(Mat.shape[1]))
    if (Mat[:,0]<=0).any() or (Mat[:,1]<=0).any():
        raise ValueError('NeoHookean needs C10>0, D1>0')
#------------------
aorta_model=AortaFEModel(arg.element_type, Node_x, Element, None, boundary0, boundary1, Element_surface_pressure,
                         Mat, None, cal_1pk_stress, dtype, device, mode='inverse_p0')
pressure=arg.pressure
#%%
def loss1_function(force_int, force_ext):
    loss=((force_int-force_ext)**2).sum(dim=1).sqrt().mean()
    return loss
#%%
t_start=0
if len(arg.mesh_p0_init) > 0:
    t0=time.time()
    Mesh_X_init=PolyhedronMesh()
    Mesh_X_init.load_from_torch(arg.mesh_p0_init+'.pt')
    Node_X_init=Mesh_X_init.node.to(dtype).to(device)
    u_field_init=Node_x-Node_X_init
    u_field_init=u_field_init[aorta_model.state['free_node']]
    print('initialize u_field with u_field_init')
    aorta_model.state['u_field'].data=u_field_init.to(dtype).to(device)
    out=aorta_model.cal_energy_and_force(pressure)
    TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
    force_int=out['force_int']; force_ext=out['force_ext']
    F=out['F']; u_field=out['u_field']
    loss1=loss1_function(force_int, force_ext)
    print('Target TPE', float(TPE2), 'loss1', loss1, 'at pressure', pressure)
    del out, loss1, TPE1, TPE2, SE, force_int, force_ext, F, u_field
    del Mesh_X_init, Node_X_init, u_field_init
    if 0 <= arg.t_start_init <= 1:
        t_start=arg.t_start_init
    else:
        print('search for the optimal t_start of AutoStepper')
        t_start_list=np.linspace(0, 1, int(np.ceil(1/arg.max_step_size))+1) #from 0 to 1, spacing at most max_step_size
        loss_list=[]
        for τ in t_start_list:
            out=aorta_model.cal_energy_and_force(pressure*τ)
            TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
            force_int=out['force_int']; force_ext=out['force_ext']
            F=out['F']; u_field=out['u_field']
            loss=loss1_function(force_int, force_ext)
            loss_list.append(float(loss))
        idx_min=np.argmin(loss_list)
        loss_min=loss_list[idx_min]
        t_start=t_start_list[idx_min]
        t_start=t_start-arg.init_step_size
        t_start=max(min(t_start, 1), 0)
        t1=time.time()
        print('t_start', t_start, 'loss', loss_min, ', time cost', t1-t0)
        del out, TPE1, TPE2, SE, force_int, force_ext, F, u_field
        del t_start_list, loss, loss_list, t1, t0
#%%
Rmax_list=[]
loss1_list=[]
loss2_list=[]
TPE_list=[]
sim_t_list=[]
time_list=[]
flag_list=[]
rst_list=[]
t0=time.time()
#%%
state={}
state['idx_save']=-1
state['t_save']=[]
def save(is_final_result=False):
    if is_final_result == False:
        state['idx_save']+=1
        state['t_save'].append(t_good)
    u_field=aorta_model.get_u_field()
    Node_X=Node_x-u_field
    S=aorta_model.cal_stress('cauchy', create_graph=False, return_W=False)
    S_element=S.mean(dim=1)
    VM_element=cal_von_mises_stress(S_element)
    S_node=cal_attribute_on_node(Node_x.shape[0], Element, S_element)
    VM_node=cal_von_mises_stress(S_node)
    Mesh_X=PolyhedronMesh()
    Mesh_X.element=Element.detach().cpu()
    Mesh_X.node=Node_X.detach().cpu()
    #for convenience, store stress and F in Mesh_X
    Mesh_X.element_data['S']=S_element.view(-1,9).detach().cpu()
    Mesh_X.element_data['VM']=VM_element.view(-1,1).detach().cpu()
    Mesh_X.node_data['S']=S_node.view(-1,9).detach().cpu()
    Mesh_X.node_data['VM']=VM_node.view(-1,1).detach().cpu()
    Mesh_X.mesh_data['arg']=arg
    Mesh_X.mesh_data['t_good']=t_good
    Mesh_X.mesh_data['TPE']=TPE_list
    Mesh_X.mesh_data['loss1']=loss1_list
    Mesh_X.mesh_data['Rmax']=Rmax_list
    Mesh_X.mesh_data['rst']=rst_list
    Mesh_X.mesh_data['t']=sim_t_list
    Mesh_X.mesh_data['time']=time_list
    Mesh_X.mesh_data['t_save']=state['t_save']
    F=aorta_model.cal_F_tensor()
    Mesh_X.mesh_data['F']=F.detach().cpu()
    #the node sets and element sets of mesh_px, so that the result can be the input of the other examples
    Mesh_X.node_set=Mesh_x.node_set
    Mesh_X.element_set=Mesh_x.element_set
    filename_save=arg.mesh_p0
    if is_final_result == False:
        filename_save+='_i'+str(state['idx_save'])
    #the final result holds the last converged state (t_good); its name and run_end say how the run ended
    if is_final_result == True:
        Mesh_X.mesh_data['run_end']=run_end
    if is_final_result == True and run_end == 'max_retry':
        filename_save+='_aborted' #the retries ran out: the load is applied up to t_good only
    if is_final_result == True and run_end == 'max_iter1':
        filename_save+='_stopped' #stopped by --max_iter1: the load is applied up to t_good only
    if arg.save_by_vtk == True:
        Mesh_X.save_as_vtk(filename_save+'.vtk')
    Mesh_X.save_as_torch(filename_save+'.pt')
    print('save: t', t_good, 'name', filename_save)
#%%
if arg.optimizer == 'FE_lbfgs_loss':
    from torch_fea.optimizer.FE_lbfgs_loss import LBFGS
elif arg.optimizer == 'FE_lbfgs_residual':
    from torch_fea.optimizer.FE_lbfgs_residual import LBFGS
else:
    raise ValueError('unkown optimizer '+arg.optimizer)
optimizer = LBFGS([aorta_model.state['u_field']], lr=1, line_search_fn='backtracking',
                  tolerance_grad =1e-10, history_size=10+arg.reform_stiffness_interval)
optimizer.set_backtracking(t_list=np.logspace(np.log10(arg.max_t_linesearch), np.log10(arg.min_t_linesearch), 10).tolist())
optimizer.set_backtracking(t_default=arg.default_t_linesearch, t_default_init=arg.default_t_linesearch) #the fallback t of a failed line search, in every iteration
optimizer.set_backtracking(c=0.5, verbose=False)
#%%
from torch_fea.optimizer.AutoStepper import AutoStepper
must_points=[]
if len(arg.must_points) > 0:
    must_points=[float(m) for m in arg.must_points.split(',')]
auto_stepper=AutoStepper(t_start=t_start, t_end=1,
                         Δt_init=arg.init_step_size,
                         Δt_min=arg.min_step_size,
                         Δt_max=arg.max_step_size,
                         alpha=1.5, beta=0.5,
                         max_retry=arg.max_retry,
                         random_seed=arg.random_seed,
                         t_list=must_points)
#%%
def reform_stiffness(pressure_t):
    ta=time.time()
    out=aorta_model.cal_energy_and_force(pressure_t, return_stiffness='sparse', inverse_p0_tangent=arg.tangent)
    H=out['H']
    if torch.is_tensor(H):
        H=H.to(torch.float64) #the model on the GPU: torch sparse CSR on the GPU (cuDSS)
    else:
        H=H.astype('float64')
    #print('type(H)', type(H))
    #print('H (mean, max)', np.mean(np.abs(H.data)), np.max(np.abs(H.data)), 'nnz', H.nnz)
    optimizer.reset_state()
    optimizer.set_H0(H)
    tb=time.time()
    #print('reform stiffness done:', tb-ta)
    return tb-ta
#%%
def plot_result():
    if len(loss1_list) == 0:
        return #nothing to plot yet: the first load step ended at its first iteration
    display.clear_output(wait=False)
    fig, ax = plt.subplots()
    ax.plot(sim_t_list, np.array(loss1_list)/max(loss1_list), 'm', linewidth=0.5)
    ax.plot(sim_t_list, loss1_list, 'r-', linewidth=0.5)
    TPE_=-np.array(TPE_list)
    ax.plot(sim_t_list, TPE_/TPE_.max(),  'g')
    ax.plot(sim_t_list, -np.array(flag_list), 'b.')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title('t '+ str(auto_stepper.t)+' iter2 '+str(iter2)+' rsi ' +str(reform_stiffness_interval)
                 +' time '+str(int(time_list[-1])))
    display.display(fig)
    plt.close(fig)
#%%
t_good=t_start
Ugood=aorta_model.state['u_field'].clone().detach()
#save mesh at t=0
if arg.save_all_t == True or len(must_points) !=0:
    save(False)
#%%
closure_opt={'output':'TPE', 'loss1_fn':loss1_function}
max_iter1=arg.max_iter1
max_iter2=arg.max_iter2
reform_stiffness_interval=arg.reform_stiffness_interval
tab=0
tab_total=0
t_R_total=0
retry_counter=0
iter1=0
run_end='max_iter1' #how the outer loop ends: 't_end', 'max_retry' (abort), or 'max_iter1' (the while condition)
#%%
while iter1 <= max_iter1:
    iter1+=1
    auto_stepper.step()
    pressure_t=auto_stepper.t*pressure
    print('t', auto_stepper.t, ', Δt', auto_stepper.Δt, ', rsi', reform_stiffness_interval)
    opt_cond=False
    reform_stiffness_counter=0
    reform_stiffness_flag=0
    reform_stiffness_iter2=0
    fatal_error_iter2=None
    minor_error_iter2=None
    Rmax_best=None #Rmax at the last drop by 10% in the load step, for the check of progress
    Rmax_best_iter2=None #the iteration of that drop
    iter2=0
    while iter2 <= max_iter2:
        iter2+=1
        if iter2==1 or (reform_stiffness_interval > 0 and (iter2-1)%reform_stiffness_interval==0) or reform_stiffness_flag == 1: #interval 0: re-form only at iter2 1 and after an error
            try:
                rs_time_cost=reform_stiffness(pressure_t)
            except Exception as e: #not KeyboardInterrupt (Ctrl-C) or SystemExit, which stop the script
                opt_cond=False
                print('break1: except at iter2 =', iter2, ':', type(e).__name__, e)
                break
            tab=0.5*(tab+rs_time_cost)
            tab_total+=rs_time_cost
            rst_list.append(rs_time_cost)
            reform_stiffness_counter+=1
            reform_stiffness_flag=0
            reform_stiffness_iter2=iter2
            #print('reform_stiffness done')
        #-------------------------------------------------
        def closure(output=closure_opt['output'], loss1_fn=closure_opt['loss1_fn']):
            out=aorta_model.cal_energy_and_force(pressure_t)
            TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
            force_int=out['force_int']; force_ext=out['force_ext']
            F=out['F']; u_field=out['u_field']
            loss1=loss1_fn(force_int, force_ext)
            loss2=cal_potential_energy(force_int-force_ext, u_field)
            if output == 'TPE':
                loss=loss2
            elif output == 'loss1':
                loss=loss1
            elif output == 'R':
                R=force_int-force_ext
                R=R[aorta_model.state['free_node']]
                return R
            else: #output is 'loss2' or 'all'
                loss=loss2
            if loss.requires_grad==True:
                optimizer.zero_grad()
                loss.backward()
            if output == 'TPE':
                return TPE1
            elif output == 'loss1':
                return loss1
            elif output == 'loss2':
                return loss2
            else:
                loss1=float(loss1)
                loss2=float(loss2)
                TPE1=float(TPE1)
                F=F.detach()
                force_int=force_int.detach()
                force_ext=force_ext.detach()
                force_int_at_element=out['force_int_at_element'].detach()
                return loss1, loss2, TPE1, F, force_int, force_ext, force_int_at_element
        #-------------------------------------------------
        try:
            U1=aorta_model.state['u_field'].clone().detach()
            opt_cond=optimizer.step(closure)
            U2=aorta_model.state['u_field'].clone().detach()
            t_linesearch=optimizer.get_linesearch_t()
            flag_linesearch=optimizer.get_linesearch_flag()
            output=torch.no_grad()(closure)(output='all', loss1_fn=loss1_function)
            loss1, loss2, TPE, F, force_int, force_ext, force_int_at_element=output
            J=det(F)
            force_avg=(force_int_at_element**2).sum(dim=2).sqrt().mean()
            force_res=((force_int-force_ext)**2).sum(dim=1).sqrt()
            R=force_res/(force_avg+1e-10)
            Rmean=R.mean().item()
            Rmax=R.max().item()
        except Exception as e: #not KeyboardInterrupt (Ctrl-C) or SystemExit, which stop the script
            opt_cond=False
            print('break2: except at iter2 =', iter2, ':', type(e).__name__, e)
            break
        #--------------------------------------------------------
        #check for error
        minor_error_flag=0
        fatal_error_flag=0
        if opt_cond == 'nan' or opt_cond == 'inf':
            opt_cond=False
            fatal_error_flag=1
            print('iter2', iter2, 'error: loss is nan or inf')

        if (np.isnan(TPE) == True or np.isinf(TPE) == True
            # or TPE > 0 this is possible
            or np.isnan(loss1) == True or np.isinf(loss1) == True
            or np.isnan(loss2) == True or np.isinf(loss2) == True):
            opt_cond=False
            fatal_error_flag=1
            print('iter2', iter2, 'errorA')

        J_min=J.min().item()
        J_max=J.max().item()
        J_error=(J-1).abs().max().item()

        if J_error > 0.5:
            opt_cond=False
            fatal_error_flag=1
            print('iter2', iter2, 'errorB')

        if 0.1 <= J_error <= 0.5:
            opt_cond=False
            minor_error_flag=1
            print('iter2', iter2, 'errorC')

        #'''
        if TPE > 0:
            opt_cond=False
            minor_error_flag=1
            print('iter2', iter2, 'errorD')
        #'''
        if fatal_error_flag == 1 or minor_error_flag == 1:
            print('  TPE', TPE, 'Rmax', Rmax, 'loss1', loss1, 'loss2', loss2)
            print('  J_min', J_min, 'J_max', J_max, 'flag_linesearch', flag_linesearch, 't_linesearch', t_linesearch)

        #'''
        if fatal_error_flag == 1:
            print('break3: error at iter2 =', iter2)
            break
        else:
            pass
            #if minor_error_flag == 1:
            #    reform_stiffness_flag=1
            #t_default_init is set to 'auto'
            # set reform_stiffness_flag=1 then t in optimizer may be always t_default_init too small
            #the code below may be better
        '''
        if fatal_error_flag == 1:
            if iter2 == 1:
                print('break3: error at iter2 =', iter2)
                break
            if fatal_error_iter2 is None:
                fatal_error_iter2=iter2
                reform_stiffness_flag=1
                aorta_model.state['u_field'].data=U1
                continue
            else:
                fatal_error_iter2_old=fatal_error_iter2
                fatal_error_iter2=iter2
                if (fatal_error_iter2 - fatal_error_iter2_old) >  reform_stiffness_interval//2:
                    reform_stiffness_flag=1
                    aorta_model.state['u_field'].data=U1
                    continue
                else:
                    print('break4: error at iter2 =', iter2)
                    break
        #'''
        if minor_error_flag == 1 and fatal_error_flag == 0:
            if minor_error_iter2 is None:
                minor_error_iter2=iter2
                reform_stiffness_flag=1
            else:
                minor_error_iter2_old=minor_error_iter2
                minor_error_iter2=iter2
                if (minor_error_iter2 - minor_error_iter2_old) >  reform_stiffness_interval//2:
                    reform_stiffness_flag=1
        #'''
        #--------------------------------------------------------
        flag_list.append(flag_linesearch)
        Rmax_list.append(Rmax)
        loss1_list.append(loss1)
        loss2_list.append(loss2)
        TPE_list.append(TPE)
        t1=time.time()
        time_list.append(t1-t0)
        sim_t_list.append(auto_stepper.t)
        #--------------------------------------------------------
        #check for convergence
        if iter2==1:
            loss1_ref=loss1
            loss2_ref=abs(loss2)
        f_ratio=loss1_list[-1]/(1e-10+loss1_ref)
        E_ratio=abs(loss2_list[-1])/(1e-10+loss2_ref)

        if len(TPE_list) == 1:
            TPE_ratio=1
        else:
            TPE_ratio=abs(TPE_list[-1]-TPE_list[-2])/(1e-10+abs(TPE_list[-1]))

        U_ratio=((U2-U1)**2).sum().sqrt()/(1e-10+(U2**2).sum().sqrt())
        U_ratio=float(U_ratio)

        opt_cond=False #opt_cond may be set to True by optimizer, so set it False
        if U_ratio < arg.U_ratio and loss1 < arg.loss1 and Rmax < arg.Rmax:
            opt_cond=True

        if iter2==1 or iter2%100==0 or opt_cond is True:            
            print('iter2', iter2, 'Rmean', Rmean, 'Rmax', Rmax, 'force_avg', float(force_avg), 'U_ratio', U_ratio)
            print('  loss1', loss1, 'loss2', loss2, 'TPE', TPE, 'max|J-1|', float((J-1).abs().max()))
            print('  flag_linesearch', flag_linesearch, 't_linesearch', t_linesearch)

        if opt_cond is True:
            print('opt_cond', opt_cond)
            print('  t', auto_stepper.t, 'iter2', iter2, 'pressure_t', pressure_t, 'time', time_list[-1])
            print('  reform_stiffness_counter', reform_stiffness_counter, 'tab', tab, 'tab_total', tab_total)
            break

        #check for progress: the minor errors (errorC, errorD) do not end the load step, and they also occur
        #in states that are correct, so a load step that diverges is found by its residual instead
        if Rmax_best is None or Rmax < 0.9*Rmax_best:
            Rmax_best=Rmax
            Rmax_best_iter2=iter2
        elif iter2-Rmax_best_iter2 >= arg.max_stall_iter2:
            print('break5: no progress at iter2 =', iter2, ', Rmax', Rmax, ', Rmax at the last drop', Rmax_best, 'at iter2 =', Rmax_best_iter2)
            break

        #flag_linesearch < 0 very often
        #it is possible that TPE increases and loss1 (Rmax) decreases, so flag < 0
        #if flag_linesearch < 0:
        #    reform_stiffness_flag=1
    #--------------------------------------------------------
    #end of while True (iter2)
    if arg.plot == True:
        plot_result()

    if opt_cond is True:
        t_good=auto_stepper.t
        Ugood=aorta_model.state['u_field'].clone().detach()
        if retry_counter == 0:
            auto_stepper.increase_Δt()
        retry_counter=0
        if arg.save_all_t == True or t_good in must_points:
            save(False)
    else:
        print('opt_cond is False, try to adjust Δt, retry_counter', retry_counter)
        if retry_counter==0:
            auto_stepper.initialize_retry()
        retry_counter+=1
        if retry_counter > arg.max_retry:
            print('abort: retry_counter > max_retry')
            aorta_model.state['u_field'].data=Ugood.clone().detach() #the last converged state (t_good) is saved
            run_end='max_retry'
            break
        print('go back to the previous t', t_good, 'Δt', auto_stepper.Δt)
        auto_stepper.goback(t_good, 'rand')
        aorta_model.state['u_field'].data=Ugood.clone().detach()

    if auto_stepper.t == auto_stepper.t_end and opt_cond == True:
        print('t is t_end and opt_cond is True, break')
        run_end='t_end'
        break
if run_end == 'max_iter1':
    print('stop_at_max_iter1: iter1 =', iter1, ', t_good =', t_good)
#%%
#save the final result
save(True)
