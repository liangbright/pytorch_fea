# The stiffness matrix of nodal forces, by automatic differentiation.
# It does not depend on the material or on the kind of element, so every model file and model_surface.py use it.
#     x=node_x[element], shape (M,A,3), requires grad: M elements, A nodes per element
#     force_element: the nodal forces of each element, shape (M,A,3), calculated from x
#     element.shape (M,A), the node indices of the elements; n_nodes: N, the number of nodes of the mesh
# The matrix is d(force)/d(node_x) with the element parts added up:
#     row 3*a+i is component i of the force at node a, column 3*b+j is component j of the position of node b
#
# The first part of this file is the original code, plain and easy to read: cal_dense_stiffness_matrix_slow and
# cal_sparse_stiffness_matrix_slow, kept for reference, and cal_diagonal_stiffness_matrix.
# The second part is the new code of cal_dense_stiffness_matrix and cal_sparse_stiffness_matrix, which the models use:
# the same matrices as the _slow functions, made faster (fewer backward passes on the GPU, index_add_ instead of
# coalesce).
#%%
import torch
import time
import warnings
#%% the original code
def cal_dense_stiffness_matrix_slow(n_nodes, element, force_element, x):
    #the same matrix as cal_dense_stiffness_matrix, written out plainly, kept for reference: slow
    #(3*A backward passes, 9*A*A index_put_) and a dense (3N,3N) matrix, for small meshes only
    #force_element.shape is (M,A,3)
    #x.shape (M,A,3)
    #for every element e, with a=3*element[e,n] and b=3*element[e,m]:
    #    H[a+i, b+j] += d force_element[e,n,i] / d x[e,m,j]
    #index_put_ with accumulate=True adds up the parts of the elements that give the same (a, b);
    #H[a, b]+=g would keep only one of them (n==m always has such elements: a node is in more than one element)
    H=torch.zeros((3*n_nodes,3*n_nodes), dtype=x[0].dtype, device=x[0].device)
    for n in range(0, x.shape[1]):
        a=3*element[:,n]
        gn0=torch.autograd.grad(force_element[:,n,0].sum(), x, retain_graph=True)[0]
        gn1=torch.autograd.grad(force_element[:,n,1].sum(), x, retain_graph=True)[0]
        gn2=torch.autograd.grad(force_element[:,n,2].sum(), x, retain_graph=True)[0]
        for m in range(0, x.shape[1]):
            g0=gn0[:,m].detach()
            g1=gn1[:,m].detach()
            g2=gn2[:,m].detach()
            b=3*element[:,m]
            H.index_put_((a,   b),   g0[:,0], accumulate=True)
            H.index_put_((a,   b+1), g0[:,1], accumulate=True)
            H.index_put_((a,   b+2), g0[:,2], accumulate=True)
            H.index_put_((a+1, b),   g1[:,0], accumulate=True)
            H.index_put_((a+1, b+1), g1[:,1], accumulate=True)
            H.index_put_((a+1, b+2), g1[:,2], accumulate=True)
            H.index_put_((a+2, b),   g2[:,0], accumulate=True)
            H.index_put_((a+2, b+1), g2[:,1], accumulate=True)
            H.index_put_((a+2, b+2), g2[:,2], accumulate=True)
    H=H.detach()
    return H

def cal_sparse_stiffness_matrix_slow(n_nodes, element, force_element, x):
    #the same matrix as cal_sparse_stiffness_matrix, written out plainly, kept for reference: slow
    #(3*A backward passes, and coalesce sorts all 9*A*A*M entries at every call: much more memory on the GPU
    #and time on the CPU)
    #force_element.shape is (M,A,3)
    #x.shape (M,A,3)
    #for every element e, with a=3*element[e,n] and b=3*element[e,m]:
    #    H[a+i, b+j] += d force_element[e,n,i] / d x[e,m,j]
    #coalesce adds up the entries of the elements that have the same (row, col)
    RowIndex=[]
    ColIndex=[]
    Value=[]
    for n in range(0, x.shape[1]):
        a=3*element[:,n]
        gn0=torch.autograd.grad(force_element[:,n,0].sum(), x, retain_graph=True)[0]
        gn1=torch.autograd.grad(force_element[:,n,1].sum(), x, retain_graph=True)[0]
        gn2=torch.autograd.grad(force_element[:,n,2].sum(), x, retain_graph=True)[0]
        for m in range(0, x.shape[1]):
            g0=gn0[:,m].detach()
            g1=gn1[:,m].detach()
            g2=gn2[:,m].detach()
            b=3*element[:,m]
            RowIndex.append(a);   ColIndex.append(b);   Value.append(g0[:,0])
            RowIndex.append(a);   ColIndex.append(b+1); Value.append(g0[:,1])
            RowIndex.append(a);   ColIndex.append(b+2); Value.append(g0[:,2])
            RowIndex.append(a+1); ColIndex.append(b);   Value.append(g1[:,0])
            RowIndex.append(a+1); ColIndex.append(b+1); Value.append(g1[:,1])
            RowIndex.append(a+1); ColIndex.append(b+2); Value.append(g1[:,2])
            RowIndex.append(a+2); ColIndex.append(b);   Value.append(g2[:,0])
            RowIndex.append(a+2); ColIndex.append(b+1); Value.append(g2[:,1])
            RowIndex.append(a+2); ColIndex.append(b+2); Value.append(g2[:,2])
    with torch.no_grad():
        RowIndex=torch.cat(RowIndex, dim=0).view(1,-1)
        ColIndex=torch.cat(ColIndex, dim=0).view(1,-1)
        Index=torch.cat([RowIndex, ColIndex], dim=0).to(torch.int64)
        Value=torch.cat(Value, dim=0)
        H=torch.sparse_coo_tensor(Index, Value, (3*n_nodes,3*n_nodes), check_invariants=False)
        H=H.coalesce()
    return H

def cal_diagonal_stiffness_matrix(n_nodes, element, force_element, x):
    #force_element.shape is  (M,A,3)
    #x.shape (M,A,3)
    H=torch.zeros((3*n_nodes,), dtype=x[0].dtype, device=x[0].device)
    for n in range(0, x.shape[1]):
        g0=torch.autograd.grad(force_element[:,n,0].sum(), x, retain_graph=True)[0].detach()
        g1=torch.autograd.grad(force_element[:,n,1].sum(), x, retain_graph=True)[0].detach()
        g2=torch.autograd.grad(force_element[:,n,2].sum(), x, retain_graph=True)[0].detach()
        a=3*element[:,n]
        #accumulate=True adds the part of every element that has the node; H[a]=g (or H[a]+=g) keeps one part
        #where a node occurs more than once in a, e.g. when it is local node n of several elements
        H.index_put_((a,),   g0[:,n,0], accumulate=True)
        H.index_put_((a+1,), g1[:,n,1], accumulate=True)
        H.index_put_((a+2,), g2[:,n,2], accumulate=True)
    H=H.detach()
    return H
#%% the new code
def cal_dense_stiffness_matrix(n_nodes, element, force_element, x):
    #force_element.shape is (M,A,3)
    #x.shape (M,A,3)
    #the sparse matrix adds up the parts of all elements at each (row, col);
    #H[a, b]+=g on a dense matrix would not: where two elements give the same (a, b), only one part is kept
    H=cal_sparse_stiffness_matrix(n_nodes, element, force_element, x).to_dense()
    return H

def get_sparse_pattern_slow(n_nodes, element):
    #the same pattern as get_sparse_pattern, made plainly, kept for reference: one sort of all the 9*A*A*M
    #(row, col) keys (tet10 aorta mesh, 29700 elements: 16 ms and a peak of 1.2 GB on the GPU, 0.4 s on the CPU)
    with torch.no_grad():
        #int64: with int32, the key row*(3*n_nodes)+col overflows for more than about 15000 nodes
        element=element.detach().to(torch.int64)
        M, A=element.shape
        r3=torch.arange(3, dtype=element.dtype, device=element.device)
        row=(3*element.t()).view(A,1,M,1,1)+r3.view(1,3,1,1,1)
        col=(3*element).view(1,1,M,A,1)+r3.view(1,1,1,1,3)
        #one sort: the unique (row, col) keys and, for each value, the position of its key
        key, inverse=torch.unique((row*(3*n_nodes)+col).reshape(-1), return_inverse=True)
        index=torch.stack([key//(3*n_nodes), key%(3*n_nodes)], dim=0)
    return {'index':index, 'inverse':inverse}

def get_sparse_pattern(n_nodes, element):
    #the pattern of the matrix of cal_sparse_stiffness_matrix, made at every call of it, a dict:
    #  'index' (2, nnz): the (row, col) of the nonzero entries, sorted by row and then col, as after coalesce
    #  'inverse' (9*A*A*M,): for each value of the element parts, the position of its (row, col) in 'index'
    #the values are in the order of cal_sparse_stiffness_matrix: local node n, i, element e, local node m, j,
    #    at row 3*element[e,n]+i and col 3*element[e,m]+j; it is the layout of the gradients (_element_jacobian),
    #    so the values need no copy
    #keeping the pattern for many calls on one mesh was tested and saved little (1-3% of the time on the GPU)
    #
    #The nonzero entries come in 3x3 blocks, one per pair of nodes (a, b) that are in one element, so only the
    #A*A*M node pairs are sorted: 9 times fewer keys than get_sparse_pattern_slow, which sorts every (row, col).
    #The place of each entry then follows from the sorted pairs:
    #    node a is the first node of cnt[a] pairs (a, b), which start at start[a] in the sorted pairs;
    #    pair p=(a,b) is the r-th of them, r=p-start[a]
    #    in the sorted entries, node a has 9*cnt[a] entries from 9*start[a] on: the 3 rows 3a+i one after the other,
    #    each with 3*cnt[a] entries, the cols 3b+j in the order of b and j
    #    so the entry (3a+i, 3b+j) of pair p is at 9*start[a] + 3*cnt[a]*i + 3*r + j
    #                                            = 6*start[a] + 3*p + 3*cnt[a]*i + j
    with torch.no_grad():
        #int64: with int32, the key a*n_nodes+b overflows for more than about 46000 nodes
        element=element.detach().to(torch.int64)
        M, A=element.shape
        r3=torch.arange(3, dtype=element.dtype, device=element.device)
        #the node pairs (a, b) of the values, in the order local node n, element e, local node m
        a=element.t().reshape(A,M,1)
        b=element.reshape(1,M,A)
        #one sort of the node pairs: the unique pairs and, for each (n, e, m), the position of its pair
        pair, pair_inverse=torch.unique((a*n_nodes+b).reshape(-1), return_inverse=True)
        pair_a=pair//n_nodes
        pair_b=pair%n_nodes
        cnt=torch.bincount(pair_a, minlength=n_nodes)
        start=torch.cumsum(cnt, dim=0)-cnt
        #for pair p: the place of its entry (3a, 3b), and how much further the next row 3a+1 is
        p=torch.arange(pair.shape[0], dtype=element.dtype, device=element.device)
        place00=6*start[pair_a]+3*p
        row_step=3*cnt[pair_a]
        #index: the (row, col) of each entry, put at its place
        place=place00.view(-1,1,1)+row_step.view(-1,1,1)*r3.view(1,3,1)+r3.view(1,1,3) #(pair, i, j)
        index=torch.empty((2, 9*pair.shape[0]), dtype=element.dtype, device=element.device)
        index[0, place.reshape(-1)]=(3*pair_a.view(-1,1,1)+r3.view(1,3,1)).expand(-1,3,3).reshape(-1)
        index[1, place.reshape(-1)]=(3*pair_b.view(-1,1,1)+r3.view(1,1,3)).expand(-1,3,3).reshape(-1)
        del place
        #inverse: the value (n, i, e, m, j) is the entry (i, j) of the pair of (n, e, m)
        place00=place00[pair_inverse].view(A,1,M,A,1)
        row_step=row_step[pair_inverse].view(A,1,M,A,1)
        inverse=(place00+row_step*r3.view(1,3,1,1,1)+r3.view(1,1,1,1,3)).reshape(-1)
    return {'index':index, 'inverse':inverse}

def _element_jacobian(force_element, x):
    #force_element.shape is (M,A,3), x.shape (M,A,3)
    #return: G, shape (A,3,M,A,3), G[n,i,e,m,j] = d force_element[e,n,i] / d x[e,m,j], detached
    #force_element[e] depends only on x[e], so the gradient of the sum over e of force_element[e,n,i] holds them all
    M, A=x.shape[0], x.shape[1]
    if force_element.is_cuda:
        #GPU: one backward pass per local node n, with its 3 components i batched in it (is_grads_batched, i.e.
        #vmap over the backward graph): A passes instead of 3*A. Batching more nodes per pass was not faster
        #and needs much more memory. On the CPU the batched passes were slower than the loop below.
        eye=torch.eye(3*A, dtype=force_element.dtype, device=force_element.device).view(3*A,1,A,3)
        try:
            G=[torch.autograd.grad(force_element, x, grad_outputs=eye[3*n:3*n+3].expand(3,M,A,3),
                                   retain_graph=True, is_grads_batched=True)[0].detach()
               for n in range(0, A)]
            return torch.stack(G, dim=0).view(A,3,M,A,3)
        except RuntimeError as err:
            #an op of the graph that vmap cannot batch (e.g. a custom autograd.Function without vmap support)
            warnings.warn("cal_sparse_stiffness_matrix: batched backward failed, one pass per force component "
                          "is used: "+str(err).split('\n')[0])
    G=[torch.autograd.grad(force_element[:,n,i].sum(), x, retain_graph=True)[0].detach()
       for n in range(0, A) for i in range(0, 3)]
    return torch.stack(G, dim=0).view(A,3,M,A,3)

def cal_sparse_stiffness_matrix(n_nodes, element, force_element, x):
    #force_element.shape is (M,A,3)
    #x.shape (M,A,3)
    #return: a coalesced sparse COO tensor of shape (3*n_nodes, 3*n_nodes)
    #the gradients first: the temporary memory of the backward passes is freed before the pattern is made
    #(tet10 aorta mesh: peak 0.79 GB this way, 0.93 GB with the pattern first)
    G=_element_jacobian(force_element, x)
    pattern=get_sparse_pattern(n_nodes, element)
    with torch.no_grad():
        #the parts of all elements added up at each (row, col) of the pattern with index_add_:
        #no coalesce of the 9*A*A*M entries, which needs much more memory (GPU) or time (CPU)
        value=torch.zeros(pattern['index'].shape[1], dtype=G.dtype, device=G.device)
        value.index_add_(0, pattern['inverse'], G.reshape(-1))
        H=torch.sparse_coo_tensor(pattern['index'], value, (3*n_nodes,3*n_nodes),
                                  check_invariants=False, is_coalesced=True)
    return H
