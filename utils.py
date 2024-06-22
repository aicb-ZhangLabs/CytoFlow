import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import cvxpy as cp


class coo_init():
    def __init__(self):
        self.row = []
        self.col = []
        self.data = []


def load_raw_expr(celltype='EXC'):
    net = np.loadtxt('../ProcessedData/Correlation/CON_' + celltype + '.csv')
    genes = np.loadtxt('../ProcessedData/Genes/hgene_' + celltype + '.txt', dtype=str)
    return net, genes


def make_adj_list(mat):
    """
    Input: 2D numpy array
    Output: Sparse representation in data frame, removing edge ij when i>j
    """
    mat = coo_matrix(mat)
    edges = pd.DataFrame({ 'row': mat.row, 'col': mat.col, 'data': mat.data }).astype({
        'row': int, 'col': int, 'data': float })
    edges = edges[edges.row < edges.col]
    return edges


def random_network(nG=300, density=0.1, seed=None):
    np.random.seed(seed)
    net = np.random.beta(1, 2, (nG,nG)) * np.random.binomial(1, density, (nG,nG))
    return net


def random_expr(nG=300, seed=None):
    np.random.seed(seed)
    expr = np.random.gamma(2, 1, nG)
    return expr


def get_confusion_matrix(net_genes, DE_genes, all_genes):
    # designed for DEG prediction task
    y_true = np.array([(1 if g in DE_genes else 0) for g in all_genes])
    y_pred = np.array([(1 if g in net_genes else 0) for g in all_genes])
    return confusion_matrix(y_true, y_pred)


def plot_par_grid(mtx, name, ax, x_range=[0,0.45], y_range=None):
    x_tick_pos = np.linspace(0, mtx.shape[1], num=4)
    y_tick_pos = np.linspace(0, mtx.shape[0], num=4)
    if y_range==None:
        y_range = x_range
    x_tick_lab = np.linspace(x_range[0], x_range[1], num=4)
    y_tick_lab = np.linspace(y_range[0], y_range[1], num=4)
    im = ax.imshow(mtx)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(x_tick_pos, x_tick_lab)
    ax.set_yticks(y_tick_pos, y_tick_lab)
    ax.set_title(name)
    ax.set(xlabel='lambda-edge', ylabel='lambda-node')
    return None


def plot_pairwise_par_grid(mtx, name, ax, Rs, TFs):
    y_tick_pos = np.arange(len(Rs))
    x_tick_pos = np.arange(len(TFs))
    im = ax.imshow(mtx, vmin=0)
    ax.figure.colorbar(im, ax=ax)
    ax.set_yticks(y_tick_pos, Rs)
    ax.set_xticks(x_tick_pos, TFs)
    ax.set_title(name)
    ax.set(ylabel='Receptors', xlabel='TFs')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    return None


def filter_input_network(adj, nG, thresh=0.5, N=50):
    net_out = np.zeros((nG, nG))
    for i in range(nG):
        if adj[i].shape[0] == 0:
            continue
        order = np.argsort(adj[i][:,2])[::-1]
        for j in range(min(N, len(order))):
            if adj[i][order[j],2] >= thresh:
                tgt_idx = int(adj[i][order[j],0])
                net_out[i, tgt_idx] = adj[i][order[j],2]
                net_out[tgt_idx, i] = adj[i][order[j],2]
    net_out = coo_matrix(net_out)
    edges = pd.DataFrame({ 'row': net_out.row, 'col': net_out.col, 'data': net_out.data }).astype({
        'row': int, 'col': int, 'data': float })
    edges = edges[ edges.row < edges.col ]
    return edges


def pairwise_optimize_wrapper(lmd_n, lmd_e, expr, edges, genes, nR, nK, nTF, returns=['obj', 'flow', 'pen', 'deg'], save_network=False):
    from stn_cvx import build_nf_mixed_penalty_model
    
    if len(returns) == 0:
        raise ValueError('Please specify output fields')
    out_dict = {}
    returns = list(returns)
    if 'obj' in returns:
        returns.remove('obj')
        out_dict['obj'] = np.zeros((nR, nTF))
    if 'flow' in returns:
        returns.remove('flow')
        out_dict['flow'] = np.zeros((nR, nTF))
    if 'pen' in returns:
        returns.remove('pen')
        out_dict['pen'] = np.zeros((nR, nTF))
    if 'deg' in returns:
        returns.remove('deg')
        out_dict['deg'] = np.zeros((nR, nTF))
    
    if len(returns) != 0:
        raise ValueError(f'Invalid output arguments {returns}')
    
    edges_full = coo_matrix((edges.data, (edges.row, edges.col)), shape=(len(genes), len(genes))).tocsr()
    for r in range(nR):
        for t in range(nTF):
            idx = np.array([r, *[k+nR for k in range(nK)], t+nR+nK])
            genes_sub = genes[idx]
            expr_sub = expr[idx]
            edges_sub = coo_matrix(edges_full[idx,:][:,idx])
            edges_sub = pd.DataFrame({ 'row': edges_sub.row, 'col': edges_sub.col, 'data': edges_sub.data }).astype({'row': int, 'col': int, 'data': float })
            c, A_ub, b_ub, A_eq, b_eq, bounds = build_nf_mixed_penalty_model(
                lmd_n, lmd_e, expr_sub, edges_sub, 1, nK, 1)
            
            try:
                x = cp.Variable(len(c))
                prob = cp.Problem(cp.Minimize(c.T @ x), [A_ub @ x <= b_ub, A_eq @ x == b_eq, x >= bounds[:,0], x <= bounds[:,1]])
                prob.solve(solver='MOSEK')
            except cp.error.SolverError:
                print(f'Inaccurate result: {genes_sub[0]} to {genes_sub[-1]}')
                prob.solve(solver='MOSEK', accept_unknown=True)
            
            # process results
            edges1 = np.array(edges_sub)
            edges2 = edges1[:, [1,0,2]]
            edges2[:,2] = -edges2[:,2]
            diedges = np.concatenate( (edges1, edges2), axis=0 )
            diedges = pd.DataFrame({ 'row': diedges[:,0], 'col': diedges[:,1], 'data': diedges[:,2] }).astype({
                'row': int, 'col': int, 'data': float })

            nG = len(genes_sub)
            nE = diedges.shape[0]
            edge_weights = x.value[:nE]
            node_weights = x.value[nE:(nE+nG)]
            out_net = coo_matrix( (edge_weights, (np.array(diedges.row), np.array(diedges.col))), shape = (nG, nG) )
            out_net = out_net.tocsr()

            if 'obj' in out_dict.keys():
                out_dict['obj'][r,t] = - prob.value.copy()
            if 'flow' in out_dict.keys():
                out_dict['flow'][r,t] = out_net.tocsr()[:, -1].sum()
            if 'pen' in out_dict.keys():
                out_dict['pen'][r,t] = out_dict['flow'][r,t] - out_dict['obj'][r,t]
            if 'deg' in out_dict.keys():
                out_dict['deg'][r,t] = np.sum( np.abs(edge_weights)>1e-4 ) / nG
                
            #### temporary output writer
            if save_network == True:
                out_net = out_net.tocoo()
                out_net = pd.DataFrame({ 'row': out_net.row, 'col': out_net.col, 'data': out_net.data }).astype({'row': int, 'col': int, 'data': float })
                out_net = out_net[out_net.data > 1e-5]
                edge_save = pd.DataFrame({'Source': genes_sub[np.array(out_net.row)],
                                          'Target': genes_sub[np.array(out_net.col)],
                                          'Weight': np.array(out_net.data)})
                edge_save.to_csv('Results/NF/Networks/'+genes_sub[0]+'_'+genes_sub[-1]+'.csv')
            
    return out_dict


def ranking_wrapper(lmd_n, lmd_e, expr, edges, genes, nR, nK, nTF):
    from stn_cvx import build_nf_mixed_penalty_model
    
    edges_full = coo_matrix((edges.data, (edges.row, edges.col)), shape=(len(genes), len(genes))).tocsr()
    flows_from_R = np.zeros((nR, nTF))
    flows_from_TF = np.zeros((nTF, nR))
    
    for r in range(nR):
        idx = np.array([r, *[k+nR for k in range(nK)], *[t+nR+nK for t in range(nTF)]])
        genes_sub = genes[idx]
        expr_sub = expr[idx]
        edges_sub = coo_matrix(edges_full[idx,:][:,idx])
        edges_sub = pd.DataFrame({ 'row': edges_sub.row, 'col': edges_sub.col, 'data': edges_sub.data }).astype({'row': int, 'col': int, 'data': float })
        c, A_ub, b_ub, A_eq, b_eq, bounds = build_nf_mixed_penalty_model(
            lmd_n, lmd_e, expr_sub, edges_sub, 1, nK, nTF)

        try:
            x = cp.Variable(len(c))
            prob = cp.Problem(cp.Minimize(c.T @ x), [A_ub @ x <= b_ub, A_eq @ x == b_eq, x >= bounds[:,0], x <= bounds[:,1]])
            prob.solve(solver='MOSEK')
        except cp.error.SolverError:
            print(f'Inaccurate result: from receptor {genes_sub[0]}')
            prob.solve(solver='MOSEK', accept_unknown=True)
        
        # process results
        edges1 = np.array(edges_sub)
        edges2 = edges1[:, [1,0,2]]
        edges2[:,2] = -edges2[:,2]
        diedges = np.concatenate( (edges1, edges2), axis=0 )
        diedges = pd.DataFrame({ 'row': diedges[:,0], 'col': diedges[:,1], 'data': diedges[:,2] }).astype({
            'row': int, 'col': int, 'data': float })
        
        nG = len(genes_sub)
        nE = diedges.shape[0]
        edge_weights = x.value[:nE]
        edge_weights = edge_weights * (edge_weights > 0)
        node_weights = x.value[nE:(nE+nG)]
        out_net = coo_matrix( (edge_weights, (np.array(diedges.row), np.array(diedges.col))), shape = (nG, nG) )
        out_net = out_net.tocsr()
        
        flows_from_R[r,] = out_net[:,-nTF:].sum(axis=0)
    
    for tf in range(nTF):
        idx = np.array([*[r for r in range(nR)], *[k+nR for k in range(nK)], t+nR+nK])
        genes_sub = genes[idx]
        expr_sub = expr[idx]
        edges_sub = coo_matrix(edges_full[idx,:][:,idx])
        edges_sub = pd.DataFrame({ 'row': edges_sub.row, 'col': edges_sub.col, 'data': edges_sub.data }).astype({'row': int, 'col': int, 'data': float })
        c, A_ub, b_ub, A_eq, b_eq, bounds = build_nf_mixed_penalty_model(
            lmd_n, lmd_e, expr_sub, edges_sub, nR, nK, 1)

        try:
            x = cp.Variable(len(c))
            prob = cp.Problem(cp.Minimize(c.T @ x), [A_ub @ x <= b_ub, A_eq @ x == b_eq, x >= bounds[:,0], x <= bounds[:,1]])
            prob.solve(solver='MOSEK')
        except cp.error.SolverError:
            print(f'Inaccurate result: from receptor {genes_sub[0]}')
            prob.solve(solver='MOSEK', accept_unknown=True)
        
        # process results
        edges1 = np.array(edges_sub)
        edges2 = edges1[:, [1,0,2]]
        edges2[:,2] = -edges2[:,2]
        diedges = np.concatenate( (edges1, edges2), axis=0 )
        diedges = pd.DataFrame({ 'row': diedges[:,0], 'col': diedges[:,1], 'data': diedges[:,2] }).astype({
            'row': int, 'col': int, 'data': float })
        
        nG = len(genes_sub)
        nE = diedges.shape[0]
        edge_weights = x.value[:nE]
        edge_weights = edge_weights * (edge_weights > 0)
        node_weights = x.value[nE:(nE+nG)]
        out_net = coo_matrix( (edge_weights, (np.array(diedges.row), np.array(diedges.col))), shape = (nG, nG) )
        out_net = out_net.tocsr()
        
        flows_from_TF[r,] = out_net[:nR,].sum(axis=1)
    
    return flows_from_R, flows_from_TF

