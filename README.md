# PyTorch FEA

The repo has the refactored code of our paper published at Computer Methods and Programs in Biomedicine, titled "**PyTorch-FEA: Autograd-enabled Finite Element Analysis Methods with Applications for Biomechanical Analysis of Human Aorta**" at https://doi.org/10.1016/j.cmpb.2023.107616

I am working to make it useful for more applications.

The orignal code of the paper is available at https://github.com/liangbright/pytorch_fea_paper

The preprint of our paper is available at https://www.biorxiv.org/content/10.1101/2023.03.27.533816v1

PyTorch-FEA needs the mesh library at https://github.com/liangbright/pytorch_fea_mesh

Example data: https://drive.google.com/file/d/1ByOjc9RVFEexLXB-u6Qd1SMAS-BKvW3g/view?usp=sharing

Try those examples:

forward analysis:
   > aorta_FEA_QN_forward_inflation.py to obtain pressurized geometry given material parameters and unpressurized geometry. 

inverse analysis:
> (1) aorta_FEA_inverse_mat_ex_vivo.py to obtain material parameters given pressurized and unpressurized geometries. \
> (2) aorta_FEA_QN_inverse_p0.py to obtain unpressurized geometry given material parameters and pressurized geometry. \
> (3) aorta_FEA_QN_GPA_prestress.py to obtain stress and strain of pressurized geometry given material parameters. \
> note: residual stress/strain is not considered

Dependencies:
- PyTorch
- NumPy and SciPy
- torch-scatter and torch-sparse, built for your PyTorch and CUDA versions (wheels at https://data.pyg.org/whl/); PyTorch Geometric itself is not needed
- PyPardiso (https://github.com/haasad/PyPardiso): the sparse linear solver on the CPU (it needs Intel MKL)
- the mesh library pytorch_fea_mesh (see above), with its own dependencies
- optional, the sparse linear solver on the GPU (cuDSS, torch_fea/optimizer/cuDSS_linear_sparse_solver.py): nvmath-python and cuDSS, e.g. pip install nvmath-python "nvidia-cudss-cu13==0.8.*"
- for the examples: Matplotlib and IPython

Tested with Python 3.14.10, PyTorch 2.12.1 (CUDA 13.2), NumPy 2.5.4, SciPy 1.18.1, torch-scatter 2.1.2, torch-sparse 0.6.18, PyPardiso 0.4.7,
nvmath-python 1.0.0 with nvidia-cudss-cu13 0.8.0.10 and cuda-core 1.2.2, Matplotlib 3.11.2, IPython 9.17.1
