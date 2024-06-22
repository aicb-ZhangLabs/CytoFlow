import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys


suffix = sys.argv[1]
CellTypes = ['EXC','INH','OLI','OPC','END','AST','MIC']


Rs = []
TFs = []
for ct in CellTypes:
    Rs = Rs + list(np.loadtxt('../ProcessedData/Genes/top_r_'+ct+'.txt', dtype=str))
    TFs = TFs + list(np.loadtxt('../ProcessedData/Genes/top_tf_'+ct+'.txt', dtype=str))

flow_max = np.zeros(7)
for ct in range(len(CellTypes)):
    flows = np.zeros((35,35))
    for r_ct in range(len(CellTypes)):
        for t_ct in range(len(CellTypes)):
            try:
                flow_sub = np.loadtxt('Results/NF/Pairwise/flow_'+suffix+CellTypes[ct]+CellTypes[r_ct]+CellTypes[t_ct]+'.txt')
            except FileNotFoundError:
                continue
            flows[ r_ct*5:(r_ct+1)*5, t_ct*5:(t_ct+1)*5 ] = flow_sub
    for r_ct in range(len(CellTypes)):
        flow_max[r_ct] = max(flow_max[r_ct], flows[r_ct*5:(r_ct+1)*5, r_ct*5:(r_ct+1)*5].max())
    
    with PdfPages('Results/NF/Figures/pairwise_full_'+suffix+CellTypes[ct]+'.pdf') as pdf:
        y_tick_pos = np.arange(len(Rs))
        x_tick_pos = np.arange(len(TFs))
        fig, ax = plt.subplots(figsize=(12,12))
        im = ax.imshow(flows)
        ax.figure.colorbar(im, ax=ax)
        ax.set_yticks(y_tick_pos, Rs)
        ax.set_xticks(x_tick_pos, TFs)
        ax.set_title(CellTypes[ct])
        ax.set(ylabel='Receptors', xlabel='TFs')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        pdf.savefig()

# 7x7 subplots for all cell types
with PdfPages('Results/NF/Figures/pairwise_full_'+suffix+'all.pdf') as pdf:
    fig, axes = plt.subplots(7,8,figsize=(15,15), width_ratios=[10,10,10,10,10,10,10,1])#, sharey=True)
    fig.subplots_adjust(right=0.95, hspace=0.05, wspace=0.01)
    for r_ct in range(len(CellTypes)):
        for ct in range(len(CellTypes)):
            try:
                flows = np.loadtxt('Results/NF/Pairwise/flow_'+suffix+CellTypes[ct]+CellTypes[r_ct]+CellTypes[r_ct]+'.txt')
            except FileNotFoundError:
                flows = np.zeros((5,5))
            im = axes[r_ct,ct].imshow(flows, vmin=0, vmax=flow_max[r_ct])
            #if ct == 0:
            #    axes[r_ct,ct].set_yticks(np.arange(5), Rs[r_ct*5:(r_ct+1)*5])
            #else:
            #    axes[r_ct,ct].tick_params(axis='y', left=False, labelleft=False)
            #axes[r_ct,ct].set_xticks(np.arange(5), TFs[r_ct*5:(r_ct+1)*5])
            axes[r_ct,ct].tick_params(axis='x', bottom=False, labelbottom=False)
            axes[r_ct,ct].tick_params(axis='y', left=False, labelleft=False)
            #plt.setp(axes[r_ct,ct].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            if ct == 0:
                axes[r_ct,ct].set_ylabel(CellTypes[r_ct], fontsize=30)
                #print(axes[r_ct,ct].get_position().get_points())
            if r_ct == 6:
                axes[r_ct,ct].set_xlabel(CellTypes[ct], fontsize=30)
        cbar = fig.colorbar(im, cax=axes[r_ct,7])
        cbar.ax.tick_params(labelsize=15)
    fig.supylabel('Receptors; R-TF pairs specific in cell type', fontsize=35, x=0.05)
    fig.supxlabel('TFs; network in cell type', fontsize=35, y=0.04)
    pdf.savefig()

with PdfPages('Results/NF/Figures/pairwise_full_'+suffix+'all_expr.pdf') as pdf:
    fig, axes = plt.subplots(7,8,figsize=(15,15), width_ratios=[10,10,10,10,10,10,10,1])#, sharey=True)
    fig.subplots_adjust(right=0.95, hspace=0.05, wspace=0.01)
    for r_ct in range(len(CellTypes)):
        vmax = 0
        for ct in range(len(CellTypes)):
            try:
                expr_tmp = np.loadtxt('Results/NF/Pairwise/expr_'+suffix+CellTypes[ct]+CellTypes[r_ct]+CellTypes[r_ct]+'.txt')
            except FileNotFoundError:
                expr_tmp = np.zeros((5,5))
            vmax = max(vmax, expr_tmp.max())
        for ct in range(len(CellTypes)):
            try:
                exprs = np.loadtxt('Results/NF/Pairwise/expr_'+suffix+CellTypes[ct]+CellTypes[r_ct]+CellTypes[r_ct]+'.txt')
            except FileNotFoundError:
                exprs = np.zeros((5,5))
            im = axes[r_ct,ct].imshow(exprs, vmin=0, vmax=vmax) ####
            #if ct == 0:
            #    axes[r_ct,ct].set_yticks(np.arange(5), Rs[r_ct*5:(r_ct+1)*5])
            #else:
            #    axes[r_ct,ct].tick_params(axis='y', left=False, labelleft=False)
            #axes[r_ct,ct].set_xticks(np.arange(5), TFs[r_ct*5:(r_ct+1)*5])
            #plt.setp(axes[r_ct,ct].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            axes[r_ct,ct].tick_params(axis='x', bottom=False, labelbottom=False)
            axes[r_ct,ct].tick_params(axis='y', left=False, labelleft=False)
            if ct == 0:
                axes[r_ct,ct].set_ylabel(CellTypes[r_ct], fontsize=30)
                #print(axes[r_ct,ct].get_position().get_points())
            if r_ct == 6:
                axes[r_ct,ct].set_xlabel(CellTypes[ct], fontsize=30)
        cbar = fig.colorbar(im, cax=axes[r_ct,7])
        cbar.ax.tick_params(labelsize=15)
    fig.supylabel('Receptors; R-TF pairs specific in cell type', fontsize=35, x=0.05)
    fig.supxlabel('TFs; network in cell type', fontsize=35, y=0.04)
    pdf.savefig()

