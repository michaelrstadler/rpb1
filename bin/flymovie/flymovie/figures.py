#!/usr/bin/env python

"""
Functions used to make figures for simulation paper. These are at present
not as well-maintained 

"""
__version__ = '1.0.0'
__author__ = 'Michael Stadler'
__copyright__   = "Copyright 2022, California, USA"

import flymovie as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#-----------------------------------------------------------------------
# Figure W4.

plt.rcParams.update({'font.weight':'heavy'})

def make_df(filenames):
    """Make a dataframe of parameters from collection of list of simulation
    filenames."""
    p = []
    for f in filenames:
        p.append(f.split('_')[1:-1])

    p = np.array(p)
    p = p.astype(float) 
    df = pd.DataFrame(p, columns=['ntotal', 'HLB diam','HLB nmols', 'n clusters', 'Cluster diam mean', 'Cluster diam sd','Cluster nmols mean','Cluster nmols sd','Noise sigma'])
    return df[['ntotal', 'HLB diam','HLB nmols', 'n clusters', 'Cluster diam mean', 'Cluster diam sd','Cluster nmols mean','Cluster nmols sd','Noise sigma']]

def embeddings_match_names(em, names, pattern):
    """Match names to a pattern, select corresponding fows of embedding."""
    if len(names) != em.shape[0]:
        raise ValueError('Sizes of embedding and names object do not match.')

    match_rows = pd.Series(names).str.contains(pattern).to_numpy()
    return em[match_rows], np.array(names)[match_rows]

def dist_to_center(ref_arr, arr):
    """Get the euclidean distance between each row of arr to center (mean) of ref_arr."""
    center = ref_arr.mean(axis=0)
    sqdist = np.sum((arr - center) ** 2, axis=1)
    return np.sqrt(sqdist)

def extract_params_df(em_ref, em_sims, names_sims, cutoff, ncols=4):
    """Extract the parameters of simulations that embed within a cutoff
    distance of the center of a reference set."""
    dists = dist_to_center(em_ref, em_sims)
    names = np.array(names_sims)[dists <= cutoff]
    df = make_df(names)
    if ncols == 4:
        return df[['n clusters',  'Cluster diam sd','Cluster nmols mean','Cluster nmols sd']]
    elif ncols == 6:
        return df[['HLB diam','HLB nmols', 'n clusters', 'Cluster diam sd','Cluster nmols mean','Cluster nmols sd']]
    elif ncols == 7:
        return df[['ntotal', 'HLB diam','HLB nmols', 'n clusters', 'Cluster diam sd','Cluster nmols mean','Cluster nmols sd','Noise sigma']]
    else:
        raise ValueError('ncols must be 4, 6, or 7.')

def plot_hist(embedding_real, names_real, embedding_sims, pattern=''):
    """Plot histograms of embeddings of simulations and selected (via pattern-matching)
    real images."""
    embedding_ref, _ = embeddings_match_names(embedding_real, names_real, pattern)
    dists_self = dist_to_center(embedding_ref, embedding_ref)
    dists_sims = dist_to_center(embedding_ref, embedding_sims)
    fig, ax = plt.subplots()
    ax.hist(dists_self, bins=25, alpha=0.5, density=True);
    ax.hist(dists_sims, bins=25, alpha=0.5, density=True);
    xtick_locs = np.arange(0,ax.get_xlim()[1],0.5)
    ytick_locs = np.arange(0,ax.get_ylim()[1],0.5)
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_locs, fontsize=16)
    ax.set_yticks(ytick_locs)
    ax.set_yticklabels(ytick_locs, fontsize=16)

    ax.set_xlabel('Embedding Distance', size=20, fontweight='bold')
    ax.set_ylabel('Density', size=20, fontweight='bold')
    plt.tight_layout()
    

def make_pairplot(embedding_real, names_real, embedding_sims, names_sims, pattern='', cutoff=np.inf, ncols=4,
    diag_kind='kde'):
    """Make a pairplot of the parameters of the sims whose embeddings are within a distance 
    cutoff of the center of the real embeddings."""
    #sns.set(font_scale=1)
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams.update({'font.weight':'heavy'})
    #plt.rcParams["xtick.fontweight"] = 'bold'
    embedding_ref, _ = embeddings_match_names(embedding_real, names_real, pattern)

    df1 = extract_params_df(embedding_ref, embedding_sims, names_sims, cutoff=cutoff, ncols=ncols)
    df2 = extract_params_df(embedding_ref, embedding_sims, names_sims, cutoff=np.inf, ncols=ncols)
    df1['sample'] = 'select'
    df2['sample'] = 'all'
    df3_ = df2.iloc[np.random.choice(np.arange(df2.shape[0]), size=np.min((10_000, df2.shape[0])), replace=False)]
    df3  = pd.concat([df3_, df1], ignore_index=True)
    print(df1.shape[0])
    g = sns.pairplot(data=df3, diag_kind=diag_kind, kind='scatter', hue='sample', plot_kws=dict(alpha=1), 
            diag_kws=dict(common_norm=False), height=2, corner=True)
    g._legend.remove()
    return df3

def make_pairplot_multdfs(dfs, labels, diag_kind='kde', include_bg=False, palette='dark', alpha=1):
    """Make a pairplot from 2 dataframes
    
    Args:
        dfs: iterable
            List of dataframes to plot from
        labels: iterable
            Length must match dfs, labels for selected items in 
            each df
        diag_kind: 'kde' or 'hist'
            seaborn pairplot parameter, changes type of plot for
            diagonal cells
        include_bg: bool
            If true, the background "all" samples are randomly sampled
            (10_000) and plotted with the "selected" data
    """
    if len(dfs) != len(labels):
        raise ValueError('dfs and labels must be the same length')
    dfs_select = []
    dfs_all = []

    # Collect and relabel dataframes.
    for i in range(len(dfs)):
        df = dfs[i].copy()
        df.loc[df['sample'] == 'select', 'sample'] = labels[i]
        dfs_select.append(df.loc[df['sample'] == labels[i]])
        dfs_all.append(df.loc[df['sample'] == 'all'])
    
    # Combine dfs, sample from background if required.
    if include_bg:
        df_all = pd.concat(dfs_all, ignore_index=True)
        df_all = df_all.iloc[np.random.choice(np.arange(df_all.shape[0]), size=np.min((10_000, df_all.shape[0])), replace=False)]
        dfs = [df_all] + dfs_select
        hue_order = ['all'] + labels
    else:
        dfs = dfs_select
        hue_order = labels
    df = pd.concat(dfs, ignore_index=True)
    g = sns.pairplot(data=df, diag_kind=diag_kind, kind='scatter', hue='sample', plot_kws=dict(alpha=alpha), 
            diag_kws=dict(common_norm=False), height=2, corner=True, hue_order=hue_order, palette=palette)
    g._legend.remove()
    
