import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.sparse import coo_matrix, load_npz
from scipy.sparse.csgraph import connected_components
import sys
sys.path.append('/home/ydai12/zhanglab/hier-net/Models/')
from utils import coo_init

def load_data(celltype = 'EXC', cor_thresh = 0, kegg = None, fc = False, input_R = None, input_TF = None, dataset = 'brain'):
    if dataset == 'brain':
        ppi_dir = '../ProcessedData/PPI_filtered/CON_' + celltype + '_full.npz'
        genes_dir = ('../ProcessedData/Genes/hgene_' + celltype + '_top3k.txt' if fc
                     else '../ProcessedData/Genes/ppi_' + celltype + '_full.txt')
        R_dir = '../PublicData/Processed/final_receptor.txt'
        TF_dir = '../PublicData/Processed/final_tf.txt'
        weights_dir = ('../ProcessedData/Correlation/CON_' + celltype + '.csv' if fc
                       else '../ProcessedData/Correlation/CON_' + celltype + '_full_filt.csv')
    if dataset == 'mouse':
        ppi_dir = '../Mouse/Data_ppi/ppi_matrix.npz'
        genes_dir = '../Mouse/Data_expr/high_genes.txt' ####
        R_dir = '../Mouse/Data_gene/coding_receptors.txt'
        TF_dir = '../Mouse/Data_gene/coding_tf.txt'
        weights_dir = '../Mouse/Data_expr/correlation.txt' ####
    if dataset == 'pbmc':
        if fc == False:
            raise NotImplementedError
        genes_dir = '../PBMC/WGCNA/Data/Genes/hgene_' + celltype + '_top500.txt'
        R_dir = '../PublicData/Processed/final_receptor.txt'
        TF_dir = '../PublicData/Processed/final_tf.txt'
        weights_dir = '../PBMC/WGCNA/Data/Correlation/' + celltype + '_top500.csv'
    if dataset == 'yeast':
        if fc == False:
            genes_dir = '../Yeast/Data_dip/genes_ppi_w+.txt'
            weights_dir = '../Yeast/Data_dip/ppi_w_dense+.txt'
        else:
            genes_dir = '../Yeast/Data_dip/genes_GSE8895+.txt'
            weights_dir = '../Yeast/Data_dip/ppi_GSE8895_dense+.txt'
        R_dir = '../Yeast/Data_dip/kegg_receptors.txt'
        TF_dir = '../Yeast/Data_dip/kegg_tfs.txt'
    
    # load network weights
    #weights = np.loadtxt('../ProcessedData/Weights/CON_' + celltype + '_expr*exy.csv')
    weights = np.loadtxt(weights_dir)
    if fc == False:
        if dataset == 'yeast':
            ppi = np.ones(weights.shape)
        else:
            ppi = load_npz(ppi_dir).toarray()
    genes_unordered = np.loadtxt(genes_dir, dtype=str)
    if kegg != None:
        net = pd.read_csv('../PublicData/Raw/KEGG_download/hsa' + str(kegg) + '.csv', sep='\t')
        genes_kegg = np.union1d(net.genesymbol_source, net.genesymbol_target)
    receptors = np.loadtxt(R_dir, dtype=str)
    tf = np.loadtxt(TF_dir, dtype=str)
#    if fc == False:
#        kinase = np.loadtxt('../PublicData/Processed/final_kinase.txt', dtype=str)
#    else:
    kinase = np.array([g for g in genes_unordered if g not in receptors and g not in tf])
    if input_R is not None:
        receptors = input_R
    if input_TF is not None:
        tf = input_TF
    if kegg != None:
        receptors = np.intersect1d(receptors, genes_kegg)
        kinase = np.intersect1d(kinase, genes_kegg)
        tf = np.intersect1d(tf, genes_kegg)
    
    # reorder the genes
    R_id = np.array([ np.argwhere(genes_unordered == r)[0,0] for r in receptors if r in genes_unordered ])
    K_id = np.array([ np.argwhere(genes_unordered == k)[0,0] for k in kinase if k in genes_unordered ])
    T_id = np.array([ np.argwhere(genes_unordered == t)[0,0] for t in tf if t in genes_unordered ])
    order = np.concatenate( (R_id,K_id,T_id), axis=0 ).astype(int)
    genes = genes_unordered[order]
    nG = len(order)
    nR = len(R_id)
    nK = len(K_id)
    nTF = len(T_id)
    
    if fc == False:
        ppi = ppi[order,:][:,order]
        weights = weights[order,:][:,order]
        weights = weights * ppi
    else:
        weights = weights[order,:][:,order]
    
    thresh = np.quantile(weights[weights>0], cor_thresh)
    weights = weights * (weights > thresh)
    
    # filter connected components ####
    if fc == False:
        conn = connected_components(weights.astype(bool).astype(int))
        conn_ids = []
        for i in range(conn[0]):
            if (conn[1] == i).sum() > 3:
                conn_ids.append(i)
        idx = np.array([ x for x in range(len(genes)) if conn[1][x] in conn_ids ])
        if len(idx) == 0:
            print('No connected components with size >= 4')
            return None, None, genes, nR, nK, nTF
        ppi = ppi[idx,:][:,idx]
        weights = weights[idx,:][:,idx]
        genes = genes[idx]

        # update gene numbers
        nR = (idx < nR).sum()
        nTF = (idx >= nG-nTF).sum()
        nG = len(idx)
        nK = nG-nR-nTF
    
    if nR * nK * nTF == 0:
        print('All receptors or TFs has been removed after connection filter')
        return None, None, genes, nR, nK, nTF
    
    weights = coo_matrix(weights)
    weights = np.array([ weights.row, weights.col, weights.data ]).transpose()
    
    # filter: row_id<col_id -- to cound each undirected edge only once
    edges = pd.DataFrame({ 'row': weights[:,0], 'col': weights[:,1], 'data': weights[:,2] }).astype({
        'row': int, 'col': int, 'data': float })
    edges = edges[ edges.row < edges.col ]
    
    
    # create adjacency list: adj[i] == Dx2 dframe, 1st col == adj node, 2nd col == edge idx
    adj = []
    for i in range(len(genes)):
        adj.append([])
    for ii in range(edges.shape[0]):
        adj[ edges.iloc[ii,0] ].append([ edges.iloc[ii,1], ii, edges.iloc[ii,2] ])
        adj[ edges.iloc[ii,1] ].append([ edges.iloc[ii,0], ii, edges.iloc[ii,2] ])
    for i in range(len(genes)):
        adj[i] = np.array(adj[i])
    
    # return
    return edges, adj, genes, nR, nK, nTF


def load_data_kegg_only(celltype='EXC', kegg=None):
    # load kegg network and gene
    net = pd.read_csv('../PublicData/Raw/KEGG_download/hsa' + str(kegg) + '.csv', sep='\t')
    net = net[['genesymbol_source','genesymbol_target']].to_numpy()
    genes_high = np.loadtxt('../ProcessedData/Genes/hgene_' + celltype + '_top3k.txt', dtype=str)
    genes_kegg = np.union1d(net[:,0], net[:,1])
    genes = np.intersect1d(genes_high, genes_kegg)
    
    # load receptors and tfs
    receptors = np.loadtxt('../PublicData/Processed/final_receptor.txt', dtype=str)
    tf = np.loadtxt('../PublicData/Processed/final_tf.txt', dtype=str)
    receptors = np.intersect1d(receptors, genes_kegg)
    tf = np.intersect1d(tf, genes_kegg)
    kinase = np.array([g for g in genes if g not in receptors and g not in tf])
    # reorder the genes
    R_id = np.array([ i for i in range(len(genes)) if genes[i] in receptors ])
    K_id = np.array([ i for i in range(len(genes)) if genes[i] in kinase ])
    T_id = np.array([ i for i in range(len(genes)) if genes[i] in tf ])
    order = np.concatenate( (R_id,K_id,T_id), axis=0 ).astype(int)
    genes = genes[order]
    nG = len(order)
    nR = len(R_id)
    nK = len(K_id)
    nTF = len(T_id)
    if nR * nK * nTF == 0:
        return None, None, None, None, None, None
    
    # process kegg network
    kegg = np.zeros((genes.shape[0], genes.shape[0]))
    for i in range(net.shape[0]):
        idx1 = np.argwhere(genes == net[i,0])
        idx2 = np.argwhere(genes == net[i,1])
        if len(idx1) * len(idx2) > 0:
            idx1 = idx1[0,0]
            idx2 = idx2[0,0]
            kegg[idx1,idx2] = 1
            kegg[idx2,idx1] = 1
    
    # get edge weights
    weights = np.loadtxt('../ProcessedData/Correlation/CON_' + celltype + '_full_filt.csv')
    order_all = np.array([ i for i in range(len(genes)) if genes[i] in genes_high ])
    weights = weights[order_all,:][:,order_all]
    weights = weights * kegg
    
    # filter connected components ####
    conn = connected_components(weights.astype(bool).astype(int))
    conn_ids = []
    for i in range(conn[0]):
        if (conn[1] == i).sum() > 3:
            conn_ids.append(i)
    idx = np.array([ x for x in range(len(genes)) if conn[1][x] in conn_ids ])
    if len(idx) == 0:
        return None, None, None, None, None, None
    kegg = kegg[idx,:][:,idx]
    weights = weights[idx,:][:,idx]
    genes = genes[idx]

    # update gene numbers
    nR = (idx < nR).sum()
    nTF = (idx >= nG-nTF).sum()
    nG = len(idx)
    nK = nG-nR-nTF
    
    if nR * nK * nTF == 0:
        return None, None, None, None, None, None
    
    weights = coo_matrix(weights)
    weights = np.array([ weights.row, weights.col, weights.data ]).transpose()
    
    # filter: row_id<col_id -- to cound each undirected edge only once
    edges = pd.DataFrame({ 'row': weights[:,0], 'col': weights[:,1], 'data': weights[:,2] }).astype({
        'row': int, 'col': int, 'data': float })
    edges = edges[ edges.row < edges.col ]
    
    
    # create adjacency list: adj[i] == Dx2 dframe, 1st col == adj node, 2nd col == edge idx
    adj = []
    for i in range(len(genes)):
        adj.append([])
    for ii in range(edges.shape[0]):
        adj[ edges.iloc[ii,0] ].append([ edges.iloc[ii,1], ii, edges.iloc[ii,2] ])
        adj[ edges.iloc[ii,1] ].append([ edges.iloc[ii,0], ii, edges.iloc[ii,2] ])
    for i in range(len(genes)):
        adj[i] = np.array(adj[i])
    
    # return
    return edges, adj, genes, nR, nK, nTF


def load_expr(genes, celltype='EXC', fc=False, dataset='brain', load_all=False):
    if dataset == 'brain':
        if fc == False:
            genes_unordered = np.loadtxt('../ProcessedData/Genes/ppi_' + celltype + '_full.txt', dtype=str)
            expr = np.loadtxt('../ProcessedData/Expression/CON_' + celltype + '_full_filt.csv')
        else:
            genes_unordered = np.loadtxt('../ProcessedData/Genes/hgene_' + celltype + '_top3k.txt', dtype=str)
            expr = np.loadtxt('../ProcessedData/Expression/CON_' + celltype + '.csv')
    if dataset == 'mouse':
        genes_unordered = np.loadtxt('../Mouse/Data_expr/high_genes.txt', dtype=str) ####
        expr = np.loadtxt('../Mouse/Data_expr/expression.txt') ####
    if dataset == 'pbmc':
        if load_all:
            genes_unordered = np.loadtxt('../PBMC/WGCNA/genes_all.txt', dtype=str)
        else:
            genes_unordered = np.loadtxt('../PBMC/WGCNA/Data/Genes/hgene_' + celltype + '_top500.txt', dtype=str)
        expr = np.loadtxt('../PBMC/WGCNA/Data/Expression/' + celltype + ('_all.txt' if load_all else '.txt'))
    idx = np.array([ np.argwhere(genes_unordered==g)[0,0] for g in genes ])
    expr = expr[idx]
    return expr


def build_nf_model(C, edges, nR, nK, nTF):
    # transform edge list into double-directioned
    # if edge id of ij is x, then edge id of ji is nE/2+x
    edges1 = np.array(edges)
    edges2 = edges1[:, [1,0,2]]
    edges = np.concatenate( (edges1, edges2), axis=0 )
    edges = pd.DataFrame({ 'row': edges[:,0], 'col': edges[:,1], 'data': edges[:,2] }).astype({
        'row': int, 'col': int, 'data': float })
    
    ## parameters: f_ij, f_ji, p_i
    nE = edges.shape[0]
    nG = nR+nK+nTF
    
    # create adjacency list: adj_in for incoming edges, adj_out for outgoing edges
    adj_in = []
    adj_out = []
    for i in range(nG):
        adj_in.append([])
        adj_out.append([])
    for e in range(nE):
        adj_in[ edges.col.iloc[e] ].append([ edges.row.iloc[e], e ])
        adj_out[ edges.row.iloc[e] ].append([ edges.col.iloc[e], e ])
    for i in range(nG):
        adj_in[i] = np.array(adj_in[i])
        adj_out[i] = np.array(adj_out[i])
    
    ## objective: min \sum_t\sum_j f_jt
    c = np.zeros(nE+nG)
    for t in range(nTF):
        c[ adj_in[nR+nK+t][:,1] ] = -1
    
    ## constraints
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    # f_ij - w_ij*p_i <= 0
    # -f_ij - w_ij*p_i <= 0
    # f_ij - w_ij*p_j <= 0
    # -f_ij - w_ij*p_j <= 0
    for e in range( int(nE/2) ):
        tmpA = np.zeros(nE+nG)
        tmpA[e] = 1
        tmpA[ nE+edges.row.iloc[e] ] = -edges.data.iloc[e]
        A_ub.append(tmpA.copy())
        b_ub.append(0)
        tmpA[e] = -1
        A_ub.append(tmpA.copy())
        b_ub.append(0)
        tmpA = np.zeros(nE+nG)
        tmpA[e] = 1
        tmpA[ nE+edges.col.iloc[e] ] = -edges.data.iloc[e]
        A_ub.append(tmpA.copy())
        b_ub.append(0)
        tmpA[e] = -1
        A_ub.append(tmpA.copy())
        b_ub.append(0)
    
    # \sum_j f_ij == 0, except r,t
    for k in range(nK):
        tmpA = np.zeros(nE+nG)
        tmpA[ adj_in[nR+k][:,1] ] = 1
        A_eq.append(tmpA.copy())
        b_eq.append(0)
    
    # f_ij + f_ji == 0
    for e in range( int(nE/2) ):
        tmpA = np.zeros(nE+nG)
        tmpA[e] = 1
        tmpA[ int(e+nE/2) ] = 1
        A_eq.append(tmpA.copy())
        b_eq.append(0)
    
    # don't add any penalty/constraint for now
    ## \sum_i p_i <= c
    #tmpA = np.zeros(nE+nG)
    #tmpA[nE:] = 1
    #A_ub.append(tmpA.copy())
    #b_ub.append(C)
    
    ## bounds:
    bounds = []
    
    # flows: due to constraints and that weights are in [0,1], bounds will be [-expr.max(), expr.max()]
    for e in range(nE):
        bounds.append([-expr.max(), expr.max()])
    
    # participations: [0,1]
    for i in range(nG):
        bounds.append([0, 1])
    
    ## pack and return
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    bounds = np.array(bounds)
    
    A_ub = coo_matrix(A_ub)
    A_eq = coo_matrix(A_eq)
    
    return c, A_ub, b_ub, A_eq, b_eq, bounds


def build_nf_node_split_model(expr, edges, nR, nK, nTF):
    """
    Requires that the first column of input network is R, the last column is TF
    """
    # transform edge list into double-directioned
    # if edge id of ij is x, then edge id of ji is nE/2+x
    edges1 = np.array(edges)
    edges2 = edges1[:, [1,0,2]]
    edges = np.concatenate( (edges1, edges2), axis=0 )
    edges = pd.DataFrame({ 'row': edges[:,0], 'col': edges[:,1], 'data': edges[:,2] }).astype({
        'row': int, 'col': int, 'data': float })
    
    ## parameters: f_ij, f_ji, p_ij, p_ji
    nE = edges.shape[0]
    nG = nR+nK+nTF
    
    # create adjacency list: adj_in for incoming edges, adj_out for outgoing edges
    adj_in = []
    adj_out = []
    for i in range(nG):
        adj_in.append([])
        adj_out.append([])
    for e in range(nE):
        adj_in[ edges.col.iloc[e] ].append([ edges.row.iloc[e], e ])
        adj_out[ edges.row.iloc[e] ].append([ edges.col.iloc[e], e ])
    for i in range(nG):
        adj_in[i] = np.array(adj_in[i])
        adj_out[i] = np.array(adj_out[i])
    
    ## objective: min \sum_t\sum_j f_jt
    c = np.zeros(nE+nE)
    for t in range(nTF):
        c[ adj_in[nR+nK+t][:,1] ] = -1
    
    ## constraints
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    # f_ij - w_ij*p_ij <= 0
    # -f_ij - w_ij*p_ij <= 0
    # f_ij - w_ij*p_ji <= 0
    # -f_ij - w_ij*p_ji <= 0
    for e in range( int(nE/2) ):
        tmpA = np.zeros(nE+nE)
        tmpA[e] = 1
        tmpA[ nE+e ] = -edges.data.iloc[e]
        A_ub.append(tmpA.copy())
        b_ub.append(0)
        tmpA[e] = -1
        A_ub.append(tmpA.copy())
        b_ub.append(0)
        tmpA = np.zeros(nE+nE)
        tmpA[e] = 1
        tmpA[ nE+int(nE/2)+e ] = -edges.data.iloc[e]
        A_ub.append(tmpA.copy())
        b_ub.append(0)
        tmpA[e] = -1
        A_ub.append(tmpA.copy())
        b_ub.append(0)
    
    # \sum_j f_ij == 0, except r,t
    for k in range(nK):
        tmpA = np.zeros(nE+nE)
        tmpA[ adj_in[nR+k][:,1] ] = 1
        A_eq.append(tmpA.copy())
        b_eq.append(0)
    
    # f_ij + f_ji == 0
    for e in range( int(nE/2) ):
        tmpA = np.zeros(nE+nE)
        tmpA[e] = 1
        tmpA[ int(e+nE/2) ] = 1
        A_eq.append(tmpA.copy())
        b_eq.append(0)
    
    # \sum_j p_ij <= expr_i
    for g in range(nG):
        tmpA = np.zeros(nE+nE)
        tmpA[ nE + adj_out[g][:,1] ] = 1
        A_ub.append(tmpA.copy())
        b_ub.append(expr[g])
    
    ## bounds:
    # only bound: p >= 0
    # will be included in cvx problem def
    
    ## pack and return
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    A_ub = coo_matrix(A_ub)
    A_eq = coo_matrix(A_eq)
    
    return c, A_ub, b_ub, A_eq, b_eq


def build_nf_mixed_penalty_model(lmd_n, lmd_e, expr, edges, nR, nK, nTF, known_ids=np.array([]), tf_terminal=False):
    if len(known_ids) == 0:
        r_ids = np.array([])
        k_ids = np.array([])
        t_ids = np.array([])
    else:
        r_ids = known_ids[known_ids<nR]
        k_ids = known_ids[(known_ids>=nR) * (known_ids<nR+nK)]
        t_ids = known_ids[known_ids>=nR+nK]
    # transform edge list into double-directioned
    # if edge id of ij is x, then edge id of ji is nE/2+x
    edges1 = np.array(edges)
    edges2 = edges1[:, [1,0,2]]
    edges = np.concatenate( (edges1, edges2), axis=0 )
    edges = pd.DataFrame({ 'row': edges[:,0], 'col': edges[:,1], 'data': edges[:,2] }).astype({
        'row': int, 'col': int, 'data': float })
    
    ## parameters: f_ij, f_ji, p_i, |f_ij|
    nE = int( edges.shape[0]/2 )
    nG = nR+nK+nTF
    
    # create adjacency list: adj_in for incoming edges, adj_out for outgoing edges
    adj_in = []
    adj_out = []
    for i in range(nG):
        adj_in.append([])
        adj_out.append([])
    for e in range(nE*2):
        adj_in[ edges.col.iloc[e] ].append([ edges.row.iloc[e], e ])
        adj_out[ edges.row.iloc[e] ].append([ edges.col.iloc[e], e ])
    for i in range(nG):
        adj_in[i] = np.array(adj_in[i])
        adj_out[i] = np.array(adj_out[i])
    
    ## objective: min - \sum_j f_jt + lambda * \sum_ij |f_ij|
    c = np.zeros(nE*3+nG)
    for t in range(nTF):
        if len(adj_in[nR+nK+t]) == 0:
            continue
        c[ adj_in[nR+nK+t][:,1] ] = -1
    c[(nE*2):(nE*2+nG)] = lmd_n
    if len(known_ids) > 0:
        c[nE*2+known_ids] = 0
    c[-nE:] = lmd_e
    
    ## constraints
    A_ub = coo_init()
    b_ub = []
    A_eq = coo_init()
    b_eq = []
    cnt_ub = 0
    cnt_eq = 0
    
    # f_ij - w_ij*p_i <= 0
    # -f_ij - w_ij*p_i <= 0
    # f_ij - w_ij*p_j <= 0
    # -f_ij - w_ij*p_j <= 0
    for e in range(nE):
        A_ub.row.append(cnt_ub)
        A_ub.col.append(nE*2+edges.row.iloc[e])
        A_ub.data.append(-edges.data.iloc[e])
        A_ub.row.append(cnt_ub)
        A_ub.col.append(e)
        A_ub.data.append(1)
        b_ub.append(0)
        cnt_ub = cnt_ub + 1
        A_ub.row.append(cnt_ub)
        A_ub.col.append(nE*2+edges.row.iloc[e])
        A_ub.data.append(-edges.data.iloc[e])
        A_ub.row.append(cnt_ub)
        A_ub.col.append(e)
        A_ub.data.append(-1)
        b_ub.append(0)
        cnt_ub = cnt_ub + 1
        A_ub.row.append(cnt_ub)
        A_ub.col.append(nE*2+edges.col.iloc[e])
        A_ub.data.append(-edges.data.iloc[e])
        A_ub.row.append(cnt_ub)
        A_ub.col.append(e)
        A_ub.data.append(1)
        b_ub.append(0)
        cnt_ub = cnt_ub + 1
        A_ub.row.append(cnt_ub)
        A_ub.col.append(nE*2+edges.col.iloc[e])
        A_ub.data.append(-edges.data.iloc[e])
        A_ub.row.append(cnt_ub)
        A_ub.col.append(e)
        A_ub.data.append(-1)
        b_ub.append(0)
        cnt_ub = cnt_ub + 1
    
    # f_ij - |f_ij| <= 0
    # -f_ij - |f_ij| <= 0
    for e in range(nE):
        A_ub.row.append(cnt_ub)
        A_ub.col.append(e)
        A_ub.data.append(1)
        A_ub.row.append(cnt_ub)
        A_ub.col.append(nE*2+nG+e)
        A_ub.data.append(-1)
        b_ub.append(0)
        cnt_ub = cnt_ub + 1
        A_ub.row.append(cnt_ub)
        A_ub.col.append(e)
        A_ub.data.append(-1)
        A_ub.row.append(cnt_ub)
        A_ub.col.append(nE*2+nG+e)
        A_ub.data.append(-1)
        b_ub.append(0)
        cnt_ub = cnt_ub + 1
    
    # \sum_j f_ij == 0, for k
    for i in range(nK):
        if len(adj_in[nR+i].shape) == 1:
            continue
        for ii in adj_in[nR+i][:,1]:
            A_eq.row.append(cnt_eq)
            A_eq.col.append(ii)
            A_eq.data.append(1)
        b_eq.append(0)
        cnt_eq = cnt_eq + 1
    
    # f_ij + f_ji == 0
    for e in range(nE):
        A_eq.row.append(cnt_eq)
        A_eq.col.append(e)
        A_eq.data.append(1)
        A_eq.row.append(cnt_eq)
        A_eq.col.append(e+nE)
        A_eq.data.append(1)
        b_eq.append(0)
        cnt_eq = cnt_eq + 1
    
    if tf_terminal == True:
        for i in range(nTF):
            if len(adj_in[nR+nK+i]) > 0:
                for ii in adj_in[nR+nK+i][:,1]:
                    A_ub.row.append(cnt_ub)
                    A_ub.col.append(ii)
                    A_ub.data.append(-1)
                    b_ub.append(0)
                    cnt_ub = cnt_ub + 1
        
    # p_i == expr_i, for known i
    for i in known_ids:
        A_eq.row.append(cnt_eq)
        A_eq.col.append(nE*2+i)
        A_eq.data.append(1)
        b_eq.append(expr[i])
        cnt_eq = cnt_eq + 1
    
    ## bounds:
    bounds = []
    
    # flows: due to constraints and that weights are in [0,1], bounds will be [-expr.max(), expr.max()]
    for e in range(nE*2):
        bounds.append([-expr.max(), expr.max()])
    
    # participations: [0,expr]
    for i in range(nG):
        bounds.append([0, expr[i]])
    
    # absolute flows
    for e in range(nE):
        bounds.append([0, expr.max()])
    
    ## pack and return
    #A_ub = coo_matrix(np.array(A_ub))
    A_ub = coo_matrix( (A_ub.data, (A_ub.row,A_ub.col)), shape=(cnt_ub,len(c)) )
    b_ub = np.array(b_ub)
    #A_eq = coo_matrix(np.array(A_eq))
    A_eq = coo_matrix( (A_eq.data, (A_eq.row,A_eq.col)), shape=(cnt_eq,len(c)) )
    b_eq = np.array(b_eq)
    bounds = np.array(bounds)
    
    return c, A_ub, b_ub, A_eq, b_eq, bounds


if __name__ == '__main__':
    import sys
    mode = sys.argv[1]
    celltype = sys.argv[2]
    print(celltype)
    
    import tracemalloc
    
    if mode == 'run':
        fc = eval(sys.argv[3])
        
        tracemalloc.start()
    
        #lmds_n = {"EXC": 0.24, "INH": 0.001, "OLI": 0.24, "OPC": 0.18, "END": 0.18, "AST": 0.001, "MIC": 0.001}
        #lmds_e = {"EXC": 0.24, "INH": 0.33, "OLI": 0.27, "OPC": 0.24, "END": 0.30, "AST": 0.33, "MIC": 0.33}
        keggs = np.array(['04722', '04064', '04151', '04010', '04020'])
        flows = np.zeros(len(keggs))
        for ii, kegg in enumerate(keggs):
            print(kegg)
            #kegg = None if sys.argv[4]=='None' else sys.argv[4]
            kegg_print = kegg #sys.argv[4]

            kegg_only = eval(sys.argv[4])
            if kegg_only:
                kegg_print = kegg_print + 'ONLY'
            
            if kegg_only:
                edges, _, genes, nR, nK, nTF = load_data_kegg_only(celltype, kegg=kegg)
            else:
                edges, _, genes, nR, nK, nTF = load_data(celltype, kegg=kegg, fc=fc, cor_thresh=(0.9 if fc else 0))
            if edges is None:
                print('Skipped')
                continue
                #raise FileNotFoundError(f'Network invalid for celltype {celltype} pathway {kegg_print}')
        #    edges, _, genes, nK = load_data_1R_1TF(273, 806)
        #    nR, nTF = 1, 1
            if kegg_only:
                expr = load_expr(genes, celltype, fc=True)
            else:
                expr = load_expr(genes, celltype, fc=fc)
            #c, A_ub, b_ub, A_eq, b_eq, bounds = build_nf_mixed_penalty_model(lmds_n[celltype], lmds_e[celltype], expr, edges, nR, nK, nTF)
            #c, A_ub, b_ub, A_eq, b_eq, bounds = build_nf_mixed_penalty_model(0.01, 0.01, expr, edges, nR, nK, nTF)
            try:
                lmd = np.loadtxt('Results/NF/Parameters/' +celltype+('T' if fc else 'F')+kegg_print+ '.txt')
                c, A_ub, b_ub, A_eq, b_eq, bounds = build_nf_mixed_penalty_model(lmd[0], lmd[1], expr, edges, nR, nK, nTF)
            except FileNotFoundError:
                continue
                #raise FileNotFoundError(f'No parameter available for celltype {celltype} pathway {kegg_print}')
            nE = edges.shape[0]
            try:
                x = cp.Variable(len(c))
                prob = cp.Problem(cp.Minimize(c.T @ x), [A_ub @ x <= b_ub, A_eq @ x == b_eq, x >= bounds[:,0], x <= bounds[:,1]])
            #    prob = cp.Problem(cp.Minimize(c.T @ x), [A_ub @ x <= b_ub, A_eq @ x == b_eq, x[-nE:] >= 0])
                prob.solve(solver='MOSEK')
            except cp.error.SolverError:
                prob.solve(solver='MOSEK', accept_unknown=True)

            # process results
            edges1 = np.array(edges)
            edges2 = edges1[:, [1,0,2]]
            diedges = np.concatenate( (edges1, edges2), axis=0 )
            diedges = pd.DataFrame({ 'row': diedges[:,0], 'col': diedges[:,1], 'data': diedges[:,2] }).astype({
                'row': int, 'col': int, 'data': float })

            nG = len(genes)
            nE = diedges.shape[0]
            edge_weights = x.value[:nE]
            node_weights = x.value[nE:(nE+nG)]
            in_net = coo_matrix( (np.array(edges.data), (np.array(edges.row), np.array(edges.col))), shape = (nG, nG) )
            in_net = in_net.tocsr()
            out_net = coo_matrix( (edge_weights, (np.array(diedges.row), np.array(diedges.col))), shape = (nG, nG) )
            out_net = out_net.tocsr()
            out_net = out_net.multiply(out_net > 0)

            # plot weight distributions
            import matplotlib.pyplot as plt
            plt.hist(np.log10(np.abs(edge_weights)+1e-16))
            plt.show()
        #    fig, ax = plt.subplots(1,3, figsize=(10,3))
        #    ax[0].hist(np.abs(node_weights[:nR]))
        #    ax[0].set_title('Receptor')
        #    ax[1].hist(np.abs(node_weights[nR:-nTF]))
        #    ax[1].set_title('Kinase')
        #    ax[2].hist(np.abs(node_weights[-nTF:]))
        #    ax[2].set_title('TF')
            plt.hist(np.log10(node_weights+1e-16))
            plt.show()

            # print absolute flow
            flow = out_net.sum(axis=0).A1[-nTF:].sum()
            print(celltype + '\t' + kegg_print + '\t' + str(np.round(flow, 3)))
            flows[ii] = flow

            # plot edge heatmap
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages('Results/NF/Figures/out_'+celltype+('T' if fc else 'F')+kegg_print+'.pdf') as pdf:
                plt.imshow(out_net.toarray())
                pdf.savefig()

            """
            # show flow distribution
            flow_distr = []
            for t in range(nTF):
                flow_distr.append( np.log10(out_net[:,nR+nK+t].sum()+1e-16) )
            flow_distr = np.array(flow_distr)
            plt.hist(flow_distr)
            plt.show()
            """

            # save output network
            node_list = np.arange(nG)#[node_weights > 1e-5]
            if len(node_list) == 0:
                continue
            nr, nk, ntf = (node_list<nR).sum(), ((node_list>=nR)*(node_list<nR+nK)).sum(), (node_list>=nR+nK).sum()
            node_type_list = np.array(['Receptor', 'Kinase', 'TF'])
            node_types = np.array([*[0 for i in range(nr)], *[1 for i in range(nk)], *[2 for i in range(ntf)]])
            out_net = out_net[node_list,:][:,node_list].tocoo()
            out_net = pd.DataFrame({ 'row': out_net.row, 'col': out_net.col, 'data': out_net.data }).astype({
                'row': int, 'col': int, 'data': float })
            out_net = out_net[out_net.data > 1e-5]
            node_save = pd.DataFrame({'Name': genes[node_list],
                                      'Type': node_type_list[node_types],
                                      'Weight': node_weights[node_list],
                                      'Expression': expr[node_list]})
            edge_save = pd.DataFrame({'Source': genes[np.array(out_net.row)],
                                      'Target': genes[np.array(out_net.col)],
                                      'Weight': np.array(out_net.data)})
            
            #with pd.ExcelWriter('Results/NF/Networks/'+celltype+('T' if fc else 'F')+kegg_print+'.xlsx') as writer:
            #    node_save.to_excel(writer, sheet_name='Node')
            #    edge_save.to_excel(writer, sheet_name='Edge')
            
            node_save.to_csv('Results/NF/Networks/node_'+celltype+('T' if fc else 'F')+kegg_print+'.csv')
            edge_save.to_csv('Results/NF/Networks/edge_'+celltype+('T' if fc else 'F')+kegg_print+'.csv')
            
            # save input network
            edge_raw = pd.DataFrame({'Source': genes[np.array(edges.row)],
                                     'Target': genes[np.array(edges.col)],
                                     'Weight': np.array(edges.data)})
            edge_raw.to_csv('Results/NF/Networks/edge_raw_'+celltype+('T' if fc else 'F')+kegg_print+'.csv')
        
        np.savetxt('Results/NF/Flows/' +celltype+'_'+('T' if fc else 'F')+ '.txt', flows)
        
        print(tracemalloc.get_traced_memory())
        tracemalloc.stop()
    
    if mode == 'tune':
        fc = eval(sys.argv[3])
        fc_print = ('T' if fc else 'F')
        job_id = sys.argv[4]
        lmd_n_lo = float(sys.argv[5])
        lmd_n_hi = float(sys.argv[6])
        lmd_n_num = int(sys.argv[7])
        lmd_e_lo = float(sys.argv[8])
        lmd_e_hi = float(sys.argv[9])
        lmd_e_num = int(sys.argv[10])
        kegg = (None if sys.argv[11] == 'None' else sys.argv[11])
        kegg_print = sys.argv[11]
        kegg_only = eval(sys.argv[12])
        if kegg_only:
            kegg_print = kegg_print + 'ONLY'
        
        ## parameter tuning
        lmds_n = np.linspace(lmd_n_lo, lmd_n_hi, lmd_n_num, endpoint=False)
        lmds_n[lmds_n==0.0] = 1e-3
        lmds_e = np.linspace(lmd_e_lo, lmd_e_hi, lmd_e_num, endpoint=False)
        # objective, penalty, unscaled penalty, flow, avg degree, active tf
        objs = np.zeros((len(lmds_n), len(lmds_e)))
        pens = np.zeros((len(lmds_n), len(lmds_e)))
        unss = np.zeros((len(lmds_n), len(lmds_e)))
        flows = np.zeros((len(lmds_n), len(lmds_e)))
        degs = np.zeros((len(lmds_n), len(lmds_e)))
        nrs = np.zeros((len(lmds_n), len(lmds_e)))
        nks = np.zeros((len(lmds_n), len(lmds_e)))
        ntfs = np.zeros((len(lmds_n), len(lmds_e)))
        nns = np.zeros((len(lmds_n), len(lmds_e)))
        errs = np.zeros((len(lmds_n), len(lmds_e)))
        erks = np.zeros((len(lmds_n), len(lmds_e)))
        erts = np.zeros((len(lmds_n), len(lmds_e)))
        ekrs = np.zeros((len(lmds_n), len(lmds_e)))
        ekks = np.zeros((len(lmds_n), len(lmds_e)))
        ekts = np.zeros((len(lmds_n), len(lmds_e)))
        etrs = np.zeros((len(lmds_n), len(lmds_e)))
        etks = np.zeros((len(lmds_n), len(lmds_e)))
        etts = np.zeros((len(lmds_n), len(lmds_e)))

        if kegg_only:
            edges, _, genes, nR, nK, nTF = load_data_kegg_only(celltype, kegg=kegg)
        else:
            edges, _, genes, nR, nK, nTF = load_data(celltype, kegg=kegg, fc=fc, cor_thresh=(0.9 if fc else 0))
        if edges is None:
            print("Skipped")
            sys.exit()
        if kegg_only:
            expr = load_expr(genes, celltype, fc=True)
        else:
            expr = load_expr(genes, celltype, fc=fc)
        for i in range(len(lmds_n)):
            for j in range(len(lmds_e)):
                tracemalloc.start()
                
                c, A_ub, b_ub, A_eq, b_eq, bounds = build_nf_mixed_penalty_model(lmds_n[i], lmds_e[j], expr, edges, nR, nK, nTF)
                nE = edges.shape[0]
                x = cp.Variable(len(c))
                prob = cp.Problem(cp.Minimize(c.T @ x), [A_ub @ x <= b_ub, A_eq @ x == b_eq, x >= bounds[:,0], x <= bounds[:,1]])
                try:
                    prob.solve(solver='MOSEK')
                except cp.error.SolverError:
                    prob.solve(solver='MOSEK', accept_unknown=True)

                # process results
                edges1 = np.array(edges)
                edges2 = edges1[:, [1,0,2]]
                edges2[:,2] = -edges2[:,2]
                diedges = np.concatenate( (edges1, edges2), axis=0 )
                diedges = pd.DataFrame({ 'row': diedges[:,0], 'col': diedges[:,1], 'data': diedges[:,2] }).astype({
                    'row': int, 'col': int, 'data': float })

                nG = len(genes)
                nE = diedges.shape[0]
                edge_weights = x.value[:nE]
                node_weights = x.value[nE:(nE+nG)]
                out_net = coo_matrix( (edge_weights, (np.array(diedges.row), np.array(diedges.col))), shape = (nG, nG) )
                out_net = out_net.tocsr()

                objs[i,j] = - prob.value.copy()
                flows[i,j] = out_net.tocsr()[:, (nR+nK):].sum()
                #for t in range(nTF):
                #    flow = flow + out_net.tocsr()[:,nR+nK+t].sum()
                pens[i,j] = flows[i,j] - objs[i,j]
                unss[i,j] = x.value[nE:].sum()
                degs[i,j] = np.sum( np.abs(edge_weights)>1e-4 ) / nG
                
                nrs[i,j] = (node_weights[:nR] > 1e-5).sum()
                nks[i,j] = (node_weights[nR:-nTF] > 1e-5).sum()
                ntfs[i,j] = (node_weights[-nTF:] > 1e-5).sum()
                nns[i,j] = (node_weights > 1e-5).sum()
                
                errs[i,j] = (out_net[:nR,:nR]>1e-5).sum()
                erks[i,j] = (out_net[:nR,nR:-nTF]>1e-5).sum()
                erts[i,j] = (out_net[:nR,-nTF]>1e-5).sum()
                ekrs[i,j] = (out_net[nR:-nTF,:nR]>1e-5).sum()
                ekks[i,j] = (out_net[nR:-nTF,nR:-nTF]>1e-5).sum()
                ekts[i,j] = (out_net[nR:-nTF,-nTF:]>1e-5).sum()
                etrs[i,j] = (out_net[-nTF:,nR:]>1e-5).sum()
                etks[i,j] = (out_net[-nTF:,nR:-nTF]>1e-5).sum()
                etts[i,j] = (out_net[-nTF:,-nTF:]>1e-5).sum()
                
                print(tracemalloc.get_traced_memory())
                tracemalloc.stop()
                
        np.savetxt('Results/NF/Tuning/mixed_objs_'+celltype+job_id+fc_print+kegg_print+'.csv', objs)
        np.savetxt('Results/NF/Tuning/mixed_flows_'+celltype+job_id+fc_print+kegg_print+'.csv', flows)
        np.savetxt('Results/NF/Tuning/mixed_pens_'+celltype+job_id+fc_print+kegg_print+'.csv', pens)
        np.savetxt('Results/NF/Tuning/mixed_unss_'+celltype+job_id+fc_print+kegg_print+'.csv', unss)
        np.savetxt('Results/NF/Tuning/mixed_degs_'+celltype+job_id+fc_print+kegg_print+'.csv', degs)
        np.savetxt('Results/NF/Tuning/mixed_nrs_'+celltype+job_id+fc_print+kegg_print+'.csv', nrs)
        np.savetxt('Results/NF/Tuning/mixed_nks_'+celltype+job_id+fc_print+kegg_print+'.csv', nks)
        np.savetxt('Results/NF/Tuning/mixed_ntfs_'+celltype+job_id+fc_print+kegg_print+'.csv', ntfs)
        np.savetxt('Results/NF/Tuning/mixed_nns_'+celltype+job_id+fc_print+kegg_print+'.csv', nns)
        np.savetxt('Results/NF/Tuning/mixed_errs_'+celltype+job_id+fc_print+kegg_print+'.csv', errs)
        np.savetxt('Results/NF/Tuning/mixed_erks_'+celltype+job_id+fc_print+kegg_print+'.csv', erks)
        np.savetxt('Results/NF/Tuning/mixed_erts_'+celltype+job_id+fc_print+kegg_print+'.csv', erts)
        np.savetxt('Results/NF/Tuning/mixed_ekrs_'+celltype+job_id+fc_print+kegg_print+'.csv', ekrs)
        np.savetxt('Results/NF/Tuning/mixed_ekks_'+celltype+job_id+fc_print+kegg_print+'.csv', ekks)
        np.savetxt('Results/NF/Tuning/mixed_ekts_'+celltype+job_id+fc_print+kegg_print+'.csv', ekts)
        np.savetxt('Results/NF/Tuning/mixed_etrs_'+celltype+job_id+fc_print+kegg_print+'.csv', etrs)
        np.savetxt('Results/NF/Tuning/mixed_etks_'+celltype+job_id+fc_print+kegg_print+'.csv', etks)
        np.savetxt('Results/NF/Tuning/mixed_etts_'+celltype+job_id+fc_print+kegg_print+'.csv', etts)
        
        #fig, ax = plt.subplots()
        #ax.plot(penalties, objs, 'b-', label='objective')
        #ax.plot(penalties, flows, 'g-', label='pure flow')
        #ax.plot(penalties, pens, 'r-', label='penalty')
        #ax.plot(penalties, degs, 'y-', label='avg degree')
        #plt.legend()
        #plt.xlabel('lambda')
        #plt.show()
    
    if mode == 'plot':
        fc = sys.argv[3]
        kegg = (None if sys.argv[4] == 'None' else sys.argv[4])
        kegg_print = sys.argv[4]
        fc_print = ('T' if fc=='True' else 'F')
        kegg_only = eval(sys.argv[5])
        if kegg_only:
            kegg_print = kegg_print + 'ONLY'
        
        print(kegg)
        
        import os
        if 'mixed_objs_'+celltype+'0'+fc_print+kegg_print+'.csv' not in os.listdir('Results/NF/Tuning'):
            print('Skipped')
            sys.exit()
        
        objs = np.zeros((15,15))
        flows = np.zeros((15,15))
        pens = np.zeros((15,15))
        unss = np.zeros((15,15))
        degs = np.zeros((15,15))
        nrs = np.zeros((15,15))
        nks = np.zeros((15,15))
        ntfs = np.zeros((15,15))
        nns = np.zeros((15,15))
        errs = np.zeros((15,15))
        erks = np.zeros((15,15))
        erts = np.zeros((15,15))
        ekrs = np.zeros((15,15))
        ekks = np.zeros((15,15))
        ekts = np.zeros((15,15))
        etrs = np.zeros((15,15))
        etks = np.zeros((15,15))
        etts = np.zeros((15,15))
        for id1 in range(3):
            for id2 in range(3):
                job_id = 3*id1+id2
                obj = np.loadtxt('Results/NF/Tuning/mixed_objs_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                objs[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = obj.copy()
                flow = np.loadtxt('Results/NF/Tuning/mixed_flows_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                flows[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = flow.copy()
                pen = np.loadtxt('Results/NF/Tuning/mixed_pens_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                pens[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = pen.copy()
                uns = np.loadtxt('Results/NF/Tuning/mixed_unss_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                unss[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = uns.copy()
                deg = np.loadtxt('Results/NF/Tuning/mixed_degs_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                degs[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = deg.copy()
                nr = np.loadtxt('Results/NF/Tuning/mixed_nrs_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                nrs[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = nr.copy()
                nk = np.loadtxt('Results/NF/Tuning/mixed_nks_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                nks[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = nk.copy()
                ntf = np.loadtxt('Results/NF/Tuning/mixed_ntfs_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                ntfs[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = ntf.copy()
                nn = np.loadtxt('Results/NF/Tuning/mixed_nns_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                nns[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = nn.copy()
                err = np.loadtxt('Results/NF/Tuning/mixed_errs_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                errs[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = err.copy()
                erk = np.loadtxt('Results/NF/Tuning/mixed_erks_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                erks[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = erk.copy()
                ert = np.loadtxt('Results/NF/Tuning/mixed_erts_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                erts[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = ert.copy()
                ekr = np.loadtxt('Results/NF/Tuning/mixed_ekrs_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                ekrs[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = ekr.copy()
                ekk = np.loadtxt('Results/NF/Tuning/mixed_ekks_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                ekks[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = ekk.copy()
                ekt = np.loadtxt('Results/NF/Tuning/mixed_ekts_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                ekts[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = ekt.copy()
                etr = np.loadtxt('Results/NF/Tuning/mixed_etrs_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                etrs[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = etr.copy()
                etk = np.loadtxt('Results/NF/Tuning/mixed_etks_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                etks[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = etk.copy()
                ett = np.loadtxt('Results/NF/Tuning/mixed_etts_'+celltype+str(job_id)+fc_print+kegg_print+'.csv')
                etts[ (id2*5):(id2*5+5), (id1*5):(id1*5+5) ] = ett.copy()
        
        lmd_n_lo, lmd_n_hi, lmd_e_lo, lmd_e_hi = 0.0, 0.45, 0.0, 0.45
        lmd_n_num, lmd_e_num = 15, 15
        lmds_n = np.linspace(lmd_n_lo, lmd_n_hi, lmd_n_num, endpoint=False)
        lmds_n[lmds_n==0.0] = 1e-3
        lmds_e = np.linspace(lmd_e_lo, lmd_e_hi, lmd_e_num, endpoint=False)
        
        good_par_ids = np.argwhere(pens >= pens.max() * 0.9)
        good_pens = np.array([ pens[tuple(ii)] for ii in good_par_ids ])
        good_flows = np.array([ flows[tuple(ii)] for ii in good_par_ids ])
        best_par_id = good_par_ids[good_flows==good_flows.max(), ]
        print("lmd_n", lmds_n[best_par_id[0,0]], "\t lmd_e", lmds_e[best_par_id[0,1]])
        np.savetxt('Results/NF/Parameters/'+celltype+fc_print+kegg_print+'.txt', [lmds_n[best_par_id[0,0]], lmds_e[best_par_id[0,1]]])
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from utils import plot_par_grid
        
        with PdfPages('Results/NF/Figures/tuning_' + celltype + fc_print + kegg_print + '.pdf') as pdf:
            fig, axes = plt.subplots(2,3,figsize=(12,8))
            plot_par_grid(objs, 'Objectives', axes[0,0], x_range=[lmd_n_lo, lmd_n_hi])
            plot_par_grid(pens, 'Penalties', axes[0,1], x_range=[lmd_n_lo, lmd_n_hi])
            plot_par_grid(flows, 'Pure flows', axes[0,2], x_range=[lmd_n_lo, lmd_n_hi])
            plot_par_grid(nns, 'Num nodes', axes[1,0], x_range=[lmd_n_lo, lmd_n_hi])
            plot_par_grid(ntfs, 'Num TFs', axes[1,1], x_range=[lmd_n_lo, lmd_n_hi])
            plot_par_grid(degs, 'Avg degree', axes[1,2], x_range=[lmd_n_lo, lmd_n_hi])
            pdf.savefig()
            

    if mode == 'rank':
        from utils import ranking_wrapper as ranking
        fc = eval(sys.argv[3])
        fc_print = 'True' if fc else 'False'
        
        lmd_n, lmd_e = 0.01, 0.01
        
        edges, _, genes, nR, nK, nTF = load_data(celltype, fc=fc, cor_thresh=(0.9 if fc else 0))
        if edge is None:
            print('Skipped')
            quit()
        expr = load_expr(genes, celltype, fc=fc)
        print('Data loading complete')
        
        flows_from_R, flows_to_TF = ranking(lmd_n, lmd_e, expr, edges, genes, nR, nK, nTF)
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages('Results/NF/Figures/ranking_R_' + fc_print + celltype + '.pdf') as pdf:
            fig, ax = plt.subplots(figsize=(12,12))
            im = ax.imshow(flows_from_R)
            ax.figure.colorbar(im, ax=ax)
            ax.set_yticks(np.arange(nR), genes[:nR])
            ax.set_xticks(np.arange(nTF), genes[-nTF:])
            ax.set_title('R to TF')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.set(xlabel='TFs', ylabel='Receptors')
            plt.savefig()
        
        with PdfPages('Results/NF/Figures/ranking_TF_' + fc_print + celltype + '.pdf') as pdf:
            fig, ax = plt.subplots(figsize=(12,12))
            im = ax.imshow(flows_to_TF)
            ax.figure.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(nR), genes[:nR])
            ax.set_yticks(np.arange(nTF), genes[-nTF:])
            ax.set_title('TF from R')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            ax.set(ylabel='TFs', xlabel='Receptors')
            plt.savefig()

