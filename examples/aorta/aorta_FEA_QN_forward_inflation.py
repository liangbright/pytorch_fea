#based on aorta_FEA_C3D8_SRI_quasi_newton_R1.py, QN=quasi_newton
#%%
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
sys.path.append("D:/MLFEA/code/pytorch_fea")
sys.path.append("D:/MLFEA/code/pytorch_fea/examples/aorta")
sys.path.append("D:/MLFEA/code/mesh")
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch.linalg import det
from torch_fea.utils.functions import cal_attribute_on_node, cal_von_mises_stress
from PolyhedronMesh import PolyhedronMesh
import time
#%%
mat_str='1e8, 0, 1, 0, 0, 1e5'; mat_name=mat_str.split(',')[0]
mat_model='GOH'
px_pressure=16
mesh_p0_str='E:/TAA_dataset10/paper/postR_sress_analysis/data/P10t3_solid_L1'
mesh_px_str='E:/TAA_dataset10/paper/postR_sress_analysis/data/P10t3_solid_L1_p16_'+mat_name

'''
all_mat=torch.load('../../../pytorch_fea/data/aorta/125mat.pt')['mat_str']
matMean=torch.load('../../../pytorch_fea/data/aorta/125mat.pt')['mean_mat_str']
px_pressure=20
mat_model='GOH_Jv'
#mat_str="1e5, 0, 1, 0, 0, 1e5"; mat_name='1e5'
mat_str=matMean; mat_name='matMean'
shape_id='171' #[24,150,168,171,174,192,318]
element_type='tet4'
mesh_p0_str='../../../pytorch_fea/data/aorta/p0_'+str(shape_id)+'_solid' #'_'+element_type
mesh_px_str=('../../../pytorch_fea/examples/aorta/result/inflation/'
             +'p0_'+str(shape_id)+'_solid_'+element_type+'_'+mat_model+'_'+mat_name+'_p'+str(px_pressure))
'''
#%%
def get_must_points(delta):
    t=0
    n=0
    t_str="0"
    while t<1:
        n=n+1
        t=delta*n
        ts='{:.2f}'.format(t)
        t_str=t_str+","+ts
    return t_str
#pressure=20, delta=0.05: 20 must points
must_points=get_must_points(0.05)
must_points=''
#%%
import argparse
parser = argparse.ArgumentParser(description='Input Parameters:')
parser.add_argument('--cuda', default=1, type=int)
parser.add_argument('--dtype', default="float64", type=str)
parser.add_argument('--mesh_p0', default=mesh_p0_str, type=str)
parser.add_argument('--mesh_px', default=mesh_px_str, type=str)
parser.add_argument('--mesh_px_init', default='', type=str)
parser.add_argument('--t_start_init', default=-1, type=float)
parser.add_argument('--mat', default=mat_str, type=str)
parser.add_argument('--mat_model', default=mat_model, type=str)
parser.add_argument('--pressure', default=px_pressure, type=float)
parser.add_argument('--optimizer', default='FE_lbfgs_residual', type=str)#FE_lbfgs, FE_lbfgs_residual
parser.add_argument('--init_step_size', default=0.1, type=float)
parser.add_argument('--min_step_size', default=1e-5, type=float)
parser.add_argument('--max_step_size', default=0.1, type=float)
parser.add_argument('--reform_stiffness_interval', default=20, type=int)
parser.add_argument('--max_iter1', default=1e10, type=int)
parser.add_argument('--max_iter2', default=200, type=int)
parser.add_argument('--U_ratio', default=1e-2, type=float)
parser.add_argument('--loss1', default=0.01, type=float)
parser.add_argument('--Rmax', default=0.005, type=float)
parser.add_argument('--max_retry', default=100, type=int)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--must_points', default=must_points, type=str)
parser.add_argument('--save_all_t', default='False', type=str) #True: save all (converged) time points
parser.add_argument('--save_by_vtk', default='True', type=str)#True: save *.vtk file in addition to *.pt file
parser.add_argument('--plot', default='False', type=str)#plot loss1, TPE, etc vs sim_t
arg = parser.parse_args()
arg.save_all_t = True if arg.save_all_t == 'True' else False
arg.save_by_vtk = True if arg.save_by_vtk == 'True' else False
arg.plot = True if arg.plot == 'True' else False
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
if "SRI" in arg.mat_model:
    from AortaFEModel_SRI import AortaFEModel, cal_potential_energy
else:
    from AortaFEModel import AortaFEModel, cal_potential_energy
#------------------
#Jv, Fbar, and 3Field are the same mathematically, but have numerical differences
if arg.mat_model == "GOH": #no SRI, volumetric locking
    from torch_fea.material.Mat_GOH import cal_1pk_stress
elif arg.mat_model == "GOH_SRI": # SRI (selective-reduced-integration) using one additional integration point
    from torch_fea.material.Mat_GOH_SRI import cal_1pk_stress
elif arg.mat_model == "GOH_Fbar": # SRI used in some papers
    from torch_fea.material.Mat_GOH_Fbar import cal_1pk_stress
elif arg.mat_model == "GOH_Jv": # SRI used in Abaqus according to its document
    from torch_fea.material.Mat_GOH_Jv import cal_1pk_stress
elif arg.mat_model == "GOH_3Field": # SRI used in FEBio according to its document and code
    from torch_fea.material.Mat_GOH_3Field import cal_1pk_stress
elif arg.mat_model == "MooneyRivlin_SRI":
    from torch_fea.material.Mat_MooneyRivlin_SRI import cal_1pk_stress_ori as cal_1pk_stress
    Mat=[float(m) for m in arg.mat.split(",")]
    Mat=torch.tensor([Mat], dtype=dtype, device=device)
else:
    raise ValueError("this file does not support:"+arg.mat_model)
#------------------
if 'GOH' in arg.mat_model:
    if 'distribution' in arg.mat:
        Mat=eval(arg.mat) # run generate_mat_distribution(?)
        Mat=Mat.clone()
        Mat[:,4]=np.pi*(Mat[:,4]/180)
        Mat=Mat.to(dtype=dtype, device=device)
    else:
        Mat=[float(m) for m in arg.mat.split(",")]
        Mat[4]=np.pi*(Mat[4]/180)
        Mat=torch.tensor([Mat], dtype=dtype, device=device)
#------------------
aorta_model=AortaFEModel(None, Element, Node_X, boundary0, boundary1, Element_surface_pressure,
                         Mat, ElementOrientation, cal_1pk_stress, dtype, device, mode='inflation')
pressure=arg.pressure
#%%
def loss1_function(force_int, force_ext):
    loss=((force_int-force_ext)**2).sum(dim=1).sqrt().mean()
    return loss
#%%
t_start=0
if len(arg.mesh_px_init) > 0:
    t0=time.time()
    Mesh_x_init=PolyhedronMesh()
    Mesh_x_init.load_from_torch(arg.mesh_px_init+".pt")
    Node_x_init=Mesh_x_init.node.to(dtype).to(device)
    u_field_init=Node_x_init-Node_X
    u_field_init=u_field_init[aorta_model.state['free_node']]
    print("initialize u_field with u_field_init")
    aorta_model.state['u_field'].data=u_field_init.to(dtype).to(device)
    out=aorta_model.cal_energy_and_force(pressure)
    TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
    force_int=out['force_int']; force_ext=out['force_ext']
    F=out['F']; u_field=out['u_field']
    loss1=loss1_function(force_int, force_ext)
    print("Target TPE", float(TPE2), "loss1", loss1, "at pressure", pressure)
    del out, loss1, TPE1, TPE2, SE, force_int, force_ext, F, u_field
    del Mesh_x_init, Node_x_init, u_field_init
    if 0 <= arg.t_start_init <= 1:
        t_start=arg.t_start_init
    else:
        print("search for the optimal t_start of AutoStepper")
        t_start_list=np.linspace(0, 1, int(1/arg.max_step_size))
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
        print("t_start", t_start, "loss", loss_min, ", time cost", t1-t0)
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
    Node_x=Node_X+u_field
    S=aorta_model.cal_stress('cauchy', create_graph=False, return_W=False)
    S_element=S.mean(dim=1)
    VM_element=cal_von_mises_stress(S_element)
    S_node=cal_attribute_on_node(Node_x.shape[0], Element, S_element)
    VM_node=cal_von_mises_stress(S_node)
    Mesh_x=PolyhedronMesh()
    Mesh_x.element=Element.detach().cpu()
    Mesh_x.node=Node_x.detach().cpu()
    Mesh_x.element_data['S']=S_element.view(-1,9).detach().cpu()
    Mesh_x.element_data['VM']=VM_element.view(-1,1).detach().cpu()
    Mesh_x.node_data['S']=S_node.view(-1,9).detach().cpu()
    Mesh_x.node_data['VM']=VM_node.view(-1,1).detach().cpu()
    Mesh_x.mesh_data['arg']=arg
    Mesh_x.mesh_data['t_good']=t_good
    Mesh_x.mesh_data['TPE']=TPE_list
    Mesh_x.mesh_data['loss1']=loss1_list
    Mesh_x.mesh_data['Rmax']=Rmax_list
    Mesh_x.mesh_data['rst']=rst_list
    Mesh_x.mesh_data['t']=sim_t_list
    Mesh_x.mesh_data['time']=time_list
    Mesh_x.mesh_data['t_save']=state['t_save']
    filename_save=arg.mesh_px
    if is_final_result == False:
        filename_save+="_i"+str(state['idx_save'])
    if is_final_result == True and opt_cond == False:
        filename_save+="_error"
    if arg.save_by_vtk == True:
        Mesh_x.save_as_vtk(filename_save+".vtk")
    Mesh_x.save_as_torch(filename_save+".pt")
    print("save: t", t_good, "name", filename_save)
#%%
if arg.optimizer == 'FE_lbfgs':
    from torch_fea.optimizer.FE_lbfgs import LBFGS
elif arg.optimizer == 'FE_lbfgs_residual':
    from torch_fea.optimizer.FE_lbfgs_residual import LBFGS
else:
    raise ValueError('unkown optimizer '+arg.optimizer)
optimizer = LBFGS([aorta_model.state['u_field']], lr=1, line_search_fn="backtracking",
                  tolerance_grad =1e-10, tolerance_change=0, history_size=10+arg.reform_stiffness_interval, max_iter=1)
optimizer.set_backtracking(t_list=[1, 0.5, 0.3, 0.1, 0.05, 0.01], t_default=0.5, t_default_init=0.5)
optimizer.set_backtracking(c=0.5, verbose=False)
#%%
from torch_fea.optimizer.AutoStepper import AutoStepper
must_points=[]
if len(arg.must_points) > 0:
    must_points=[float(m) for m in arg.must_points.split(",")]
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
    out=aorta_model.cal_energy_and_force(pressure_t, return_stiffness="sparse")
    H=out['H']
    H=H.astype("float64")
    #print("type(H)", type(H))
    #print('H (mean, max)', np.mean(np.abs(H.data)), np.max(np.abs(H.data)), 'nnz', H.nnz)
    optimizer.reset_state()
    optimizer.set_H0(H)
    tb=time.time()
    #print("reform stiffness done:", tb-ta)
    return tb-ta
#%%
def plot_result():
    display.clear_output(wait=False)
    fig, ax = plt.subplots()
    ax.plot(sim_t_list, np.array(loss1_list)/max(loss1_list), 'm', linewidth=0.5)
    ax.plot(sim_t_list, loss1_list, 'r-', linewidth=0.5)
    TPE_=-np.array(TPE_list)
    ax.plot(sim_t_list, TPE_/TPE_.max(),  'g')
    ax.plot(sim_t_list, -np.array(flag_list), 'b.')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_title("t "+ str(auto_stepper.t)+" iter2 "+str(iter2)+" rsi " +str(reform_stiffness_interval)
                 +" time "+str(int(time_list[-1])))
    display.display(fig)
    plt.close(fig)
#%%
t_good=t_start
Ugood=aorta_model.state['u_field'].clone().detach()
#save mesh at t=0
if arg.save_all_t == True or len(must_points) !=0:
    save(False)
#%%
max_iter1=arg.max_iter1
max_iter2=arg.max_iter2
reform_stiffness_interval=arg.reform_stiffness_interval
tab=0
tab_total=0
t_R_total=0
retry_counter=0
iter1=0
while iter1 <= max_iter1:
    iter1+=1
    auto_stepper.step()
    pressure_t=auto_stepper.t*pressure
    closure_opt={"output":"TPE", "loss1_fn":loss1_function}
    print("t", auto_stepper.t, ", Δt", auto_stepper.Δt, ", rsi", reform_stiffness_interval)
    opt_cond=False
    reform_stiffness_counter=0
    reform_stiffness_flag=0
    reform_stiffness_iter2=0
    error_iter2=None
    iter2=0
    while iter2 <= max_iter2:
        iter2+=1
        if iter2==1 or (iter2-1)%reform_stiffness_interval==0 or reform_stiffness_flag == 1:
            rs_time_cost=reform_stiffness(pressure_t)
            tab=0.5*(tab+rs_time_cost)
            tab_total+=rs_time_cost
            rst_list.append(rs_time_cost)
            reform_stiffness_counter+=1
            reform_stiffness_flag=0
            reform_stiffness_iter2=iter2
            #print("reform_stiffness done")
        #-------------------------------------------------
        def closure(output=closure_opt["output"], loss1_fn=closure_opt["loss1_fn"]):
            out=aorta_model.cal_energy_and_force(pressure_t)
            TPE1=out['TPE1']; TPE2=out['TPE2']; SE=out['SE']
            force_int=out['force_int']; force_ext=out['force_ext']
            F=out['F']; u_field=out['u_field']
            loss1=loss1_fn(force_int, force_ext)
            loss2=cal_potential_energy(force_int-force_ext, u_field)
            if output == "TPE":
                loss=loss2
            elif output == "loss1":
                loss=loss1
            elif output == "R":
                R=force_int-force_ext
                R=R[aorta_model.state['free_node']]
                return R
            else: #output is "loss2" or "all"
                loss=loss2
            if loss.requires_grad==True:
                optimizer.zero_grad()
                loss.backward()
            if output == "TPE":
                return TPE1
            elif output == "loss1":
                return loss1
            elif output == "loss2":
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
        U1=aorta_model.state['u_field'].clone().detach()
        opt_cond=optimizer.step(closure)
        U2=aorta_model.state['u_field'].clone().detach()
        t_linesearch=optimizer.get_linesearch_t()
        flag_linesearch=optimizer.get_linesearch_flag()

        loss1, loss2, TPE, F, force_int, force_ext, force_int_at_element=torch.no_grad()(closure)(output='all')
        J=det(F)
        #--------------------------------------------------------
        #check for error
        error_flag=0
        if opt_cond == "nan" or opt_cond == "inf":
            opt_cond=False
            error_flag=1
            print("iter2", iter2, "error: loss is nan or inf")

        if (np.isnan(TPE) == True or np.isinf(TPE) == True or TPE > 0
            or np.isnan(loss1) == True or np.isinf(loss1) == True
            or np.isnan(loss2) == True or np.isinf(loss2) == True):
            opt_cond=False
            error_flag=1
            print("iter2", iter2, "error: TPE", TPE, "loss1", loss1, "loss2", loss2)
            print("iter2", iter2, "error: min(J)", float(J.min()), "max(J)", float(J.max()))

        J_error_counter=(J<=0).sum().item()
        if J_error_counter > 0:
            opt_cond=False
            error_flag=1
            print("iter2", iter2, "error: sum(J<=0) =", J_error_counter)

        if error_flag==1:
            if iter2 == 1:
                print("break: error at iter2 =", iter2)
                break
            if error_iter2 is None:
                error_iter2=iter2
                reform_stiffness_flag=1
                aorta_model.state['u_field'].data=U1
                continue
            else:
                error_iter2_old=error_iter2
                error_iter2=iter2
                if (error_iter2 - error_iter2_old) >  reform_stiffness_interval//2:
                    reform_stiffness_flag=1
                    aorta_model.state['u_field'].data=U1
                    continue
                else:
                    print("break: error at iter2 =", iter2)
                    break
        #--------------------------------------------------------
        flag_list.append(flag_linesearch)
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

        U_ratio=((U2-U1)**2).sum().sqrt()/(1e-10+(U2**2).sum().sqrt())
        U_ratio=float(U_ratio)

        force_avg=(force_int_at_element**2).sum(dim=2).sqrt().mean()
        force_res=((force_int-force_ext)**2).sum(dim=1).sqrt()
        R=force_res/(force_avg+1e-10)
        Rmean=R.mean().item()
        Rmax=R.max().item()
        Rmax_list.append(Rmax)

        opt_cond=False #opt_cond may be set to True by optimizer, so set it False
        if U_ratio < arg.U_ratio and loss1 < arg.loss1 and Rmax < arg.Rmax:
            opt_cond=True

        #if len(TPE_list) > 1:
        #    print("iter2", iter2, "flag", flag_linesearch, "t", t_linesearch,
        #          "Rmax", Rmax, "TPE_diff", TPE_list[-1]-TPE_list[-2])

        if iter2==1 or opt_cond is True:
            print('iter2', iter2, 'Rmean', Rmean, 'Rmax', Rmax, "force_avg", float(force_avg), "U_ratio", U_ratio, )
            print("  loss1", loss1, "loss2", loss2, "TPE", TPE, "max|J-1|", float((J-1).abs().max()))
            print("  flag_linesearch", flag_linesearch, "t_linesearch", t_linesearch)

        if opt_cond is True:
            print("opt_cond", opt_cond)
            print("  t", auto_stepper.t, "iter2", iter2, "pressure_t", pressure_t, "time", time_list[-1])
            print("  reform_stiffness_counter", reform_stiffness_counter, "tab", tab, "tab_total", tab_total)
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
        print("opt_cond is False, try to adjust Δt, retry_counter", retry_counter)
        if retry_counter >= arg.max_retry:
            print("abort: retry_counter >= max_retry")
            break
        if retry_counter==0:
            auto_stepper.initialize_retry()
        auto_stepper.goback(t_good, "rand")
        aorta_model.state['u_field'].data=Ugood.clone().detach()
        retry_counter+=1
        print("go back to the previous t", t_good, "with Δt", auto_stepper.Δt)

    if auto_stepper.t == auto_stepper.t_end and opt_cond == True:
        print("t is t_end and opt_cond is True, break")
        break
#%%
#save the final result
save(True)
