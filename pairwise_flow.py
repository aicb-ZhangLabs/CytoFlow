import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, load_npz
from scipy.sparse.csgraph import connected_components
from utils import coo_init, filter_input_network, plot_pairwise_par_grid
from utils import pairwise_optimize_wrapper as pairwise

from stn_cvx import load_data, load_data_kegg_only, load_expr

if __name__ == '__main__':
    import sys
    celltype = sys.argv[1]
    print(celltype)
    fc = eval(sys.argv[2])
    fc_print = sys.argv[2]
    kegg_only = eval(sys.argv[3])
    R_ct = sys.argv[4]
    TF_ct = sys.argv[5]
    filt = True
    
    ########
    #import os
    #if os.path.isfile('Results/NF/Pairwise/flow_True'+celltype+R_ct+TF_ct+'.txt'):
    #    quit()
    ########
    
    #### DE R/TF PAIRWISE FLOW ####
    CellTypes = ['EXC','INH','OLI','OPC','END','AST','MIC']
    input_R = np.loadtxt('../ProcessedData/Genes/top_r_' + R_ct + '.txt', dtype=str)
    input_TF = np.loadtxt('../ProcessedData/Genes/top_tf_' + TF_ct + '.txt', dtype=str)
    if filt:
        edges, adj, genes, nR, nK, nTF = load_data(celltype, fc=fc, cor_thresh=(0.5 if fc else 0), input_R=input_R, input_TF=input_TF)
        if edges is None:
            print('Skipped')
            quit()
        edges = filter_input_network(adj, nR+nK+nTF, thresh=0.5, N=50)
    else:
        edges, adj, genes, nR, nK, nTF = load_data(celltype, fc=fc, cor_thresh=(0.9 if fc else 0), input_R=input_R, input_TF=input_TF)
        if edges is None:
            print('Skipped')
            quit()
    #    raise ValueError(f'Connectivity issue for parameters celltype: {celltype}; use_corr: {fc}; cor_thresh: {0.5 if fc else -1}; input_R: {input_R}; input_TF: {input_TF}')
    expr = load_expr(genes, celltype, fc=fc)
    print('data loading complete')
    #try:
    #    lmd = np.loadtxt('Results/NF/Parameters/' +celltype+('T' if fc else 'F')+kegg_print+ '.txt')
    #except FileNotFoundError:
    #    print('No parameter available')
    #    continue

    #out_dict = pairwise(lmd[0], lmd[1], expr, edges, genes, nR, nK, nTF)
    out_dict = pairwise(0.01, 0.01, expr, edges, genes, nR, nK, nTF, save_network=True)
    print('model construction complete')

    from matplotlib.backends.backend_pdf import PdfPages
    Rs, TFs = genes[:nR], genes[-nTF:]
    
    #with PdfPages('Results/NF/Figures/pairwise_' + celltype + fc_print + '_DE.pdf') as pdf:
    #    fig, axes = plt.subplots(2,2,figsize=(8,8))
    #    plot_pairwise_par_grid(out_dict['obj'], 'Objectives', axes[0,0], Rs, TFs)
    #    plot_pairwise_par_grid(out_dict['flow'], 'Pure flows', axes[1,0], Rs, TFs)
    #    plot_pairwise_par_grid(out_dict['pen'], 'Penalty', axes[0,1], Rs, TFs)
    #    plot_pairwise_par_grid(out_dict['deg'], 'Avg degree', axes[1,1], Rs, TFs)
    #    pdf.savefig()
    
    #with PdfPages('Results/NF/Figures/pairwise_' + celltype + fc_print + '_DE_allg.pdf') as pdf:
    #    fig, ax = plt.subplots(figsize=(12,12))
    #    plot_pairwise_par_grid(out_dict['flow'], 'Pairwise flows', ax, Rs, TFs)
    #    pdf.savefig()
    
    flow = np.zeros((5,5))
    mean_expr = np.zeros((5,5))
    idx_R = [i for i in range(5) if input_R[i] in Rs]
    idx_TF = [i for i in range(5) if input_TF[i] in TFs]
    for i,ir in enumerate(idx_R):
        for j,itf in enumerate(idx_TF):
            flow[ir,itf] = out_dict['flow'][i,j]
            mean_expr[ir,itf] = np.sqrt(expr[i]*expr[-nTF+j])
    np.savetxt('Results/NF/Pairwise/flow_'+('filt' if filt else '')+fc_print+celltype+R_ct+TF_ct+'.txt', flow)
    np.savetxt('Results/NF/Pairwise/expr_'+('filt' if filt else '')+fc_print+celltype+R_ct+TF_ct+'.txt', mean_expr)
    
    
    #nG = len(genes)
    #edge_raw = pd.DataFrame({'Source': genes[np.array(edges.row)],
    #                         'Target': genes[np.array(edges.col)],
    #                         'Weight': np.array(edges.data)})
    #edge_raw.to_csv('Results/NF/Networks/edge_raw_'+celltype+('T' if fc else 'F')+'_DE.csv')
    
    """
    #### KEGG PAIRWISE FLOW ####
    keggs = np.array(['04722', '04064', '04151', '04010', '04020'])
    out_dicts = []
    for ii, kegg in enumerate(keggs):
        print(kegg)
        kegg_print = kegg

        if kegg_only:
            kegg_print = kegg_print + 'ONLY'

        if kegg_only:
            edges, _, genes, nR, nK, nTF = load_data_kegg_only(celltype, kegg=kegg)
        else:
            edges, _, genes, nR, nK, nTF = load_data(celltype, kegg=kegg, fc=fc, cor_thresh=(0.9 if fc else 0))
        if edges is None:
            print('Skipped')
            continue
        if kegg_only:
            expr = load_expr(genes, celltype, fc=True)
        else:
            expr = load_expr(genes, celltype, fc=fc)
        try:
            lmd = np.loadtxt('Results/NF/Parameters/' +celltype+('T' if fc else 'F')+kegg_print+ '.txt')
        except FileNotFoundError:
            print('No parameter available')
            continue
        
        #out_dict = pairwise(lmd[0], lmd[1], expr, edges, genes, nR, nK, nTF)
        out_dict = pairwise(0.1, 0.1, expr, edges, genes, nR, nK, nTF)
        
        from matplotlib.backends.backend_pdf import PdfPages
        Rs, TFs = genes[:nR], genes[-nTF:]
        with PdfPages('Results/NF/Figures/pairwise_' + celltype + fc_print + kegg_print + '.pdf') as pdf:
            fig, axes = plt.subplots(2,2,figsize=(8,8))
            plot_pairwise_par_grid(out_dict['obj'], 'Objectives', axes[0,0], Rs, TFs)
            plot_pairwise_par_grid(out_dict['flow'], 'Pure flows', axes[1,0], Rs, TFs)
            plot_pairwise_par_grid(out_dict['pen'], 'Penalty', axes[0,1], Rs, TFs)
            plot_pairwise_par_grid(out_dict['deg'], 'Avg degree', axes[1,1], Rs, TFs)
            pdf.savefig()
    """
    

