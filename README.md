# PyTorch FEA

The repo has the refactored code of our paper published at Computer Methods and Programs in Biomedicine, titled "**PyTorch-FEA: Autograd-enabled Finite Element Analysis Methods with Applications for Biomechanical Analysis of Human Aorta**" at https://doi.org/10.1016/j.cmpb.2023.107616

I am working to make it useful for more applications.

The orignal code of the paper is available at https://github.com/liangbright/pytorch_fea_paper

The preprint of our paper is available at https://www.biorxiv.org/content/10.1101/2023.03.27.533816v1

PyTorch-FEA needs the mesh library at https://github.com/liangbright/mesh

Example data: https://drive.google.com/file/d/1ByOjc9RVFEexLXB-u6Qd1SMAS-BKvW3g/view?usp=sharing

Try those two examples:

forward analysis: aorta_FEA_QN_forward_inflation.py

inverse analysis: aorta_FEA_inverse_mat_ex_vivo.py

Dependency: PyTorch, PyTorch Geometric, and PyPardiso
