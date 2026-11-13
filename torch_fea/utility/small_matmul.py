# matmul for the many small matrices of FEA, e.g. the 3x3 tensors F, C, S at every integration point.
# On the GPU, torch.matmul sends a batch of tiny matrices to cuBLAS batched GEMM, which is slow for them:
#     RTX 6000 Ada, float64: ~1 ms for 39600 3x3 products (hex8 aorta mesh, 4950 elements x 8 points),
#     ~0.05 ms elementwise; A800: ~0.3 ms
# The stiffness matrix differentiates through these products 8-24 times, and the force once per solver iteration.
# Elementwise, the derivatives of a product are again elementwise products and sums, never bmm.
# Measured with material_GOH.py and solid_element_shared.py (hex8 aorta mesh):
#     RTX 6000 Ada: force 4.5 -> 2.8 ms, force and stiffness 127 -> 22 ms; A800: 5.0 -> 2.8 ms, 76 -> 42 ms
# On the CPU, torch.matmul is faster (force 20 ms, elementwise 32 ms), so it is used there.
#%%
import torch

def matmul(a, b):
    #the same as torch.matmul(a, b). On the GPU, a batch of small matrices (inner size <= 10, e.g. the 3x3
    #tensors of the integration points) is multiplied elementwise: torch.matmul sends it to cuBLAS batched GEMM,
    #which is slow for tiny matrices (RTX 6000 Ada, float64: ~1 ms for 39600 3x3 products, ~0.05 ms elementwise)
    #torch.matmul is used on the CPU, for 1-D inputs (which it treats specially), and where it raises an error
    #(other dtype, inner sizes that differ): the elementwise product would broadcast a size 1 instead
    #inner size <= 10: the temporary below is that many times the size of the result, and for larger matrices
    #cuBLAS is faster (16x16 products: 3 times slower elementwise)
    if (a.is_cuda and a.dim() >= 2 and b.dim() >= 2 and a.dtype == b.dtype
        and a.shape[-1] == b.shape[-2] and a.shape[-1] <= 10):
        #(..., p, k, 1) * (..., 1, k, q) summed over k: a temporary k times the size of the result
        return (a.unsqueeze(-1)*b.unsqueeze(-3)).sum(dim=-2)
    return torch.matmul(a, b)
