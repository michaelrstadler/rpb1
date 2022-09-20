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
import os

#-----------------------------------------------------------------------
# Figure W2.
def match_names_and_distcutoff(real_embedding_pkl, sims_embedding_pkl, cutoff_lower, cutoff_upper, savestem=None, pattern=''):
    """Extract the names of real images that match a supplied pattern and of
    simulations that embed within a cutoff distance of the center of a 
    this real set.
    
    Args:
        real_embedding_pkl: str
            Path to embedding pkl object for real nucs
        sims_embedding_pkl: str
            Path to embedding pkl object for simulated nucs
        cutoff_lower: float
            Lower limit for embedding distance
        cutoff_upper: float
            Upper limit for embedding distance
        savestem: None or str
            If not None, filenames are saved in text files
        pattern: str
            Pattern to match to names of real sim files

    Returns:
        names_ref: list
            Names of reference image files (generally real images that
            match supplied pattern)
        names_withindist: list
            Names of simulated images that fall within distance cutoffs
            of reference images
    """

    def embeddings_match_names(em, names, pattern):
        """Match names to a pattern, select corresponding rows of embedding."""
        if len(names) != em.shape[0]:
            raise ValueError('Sizes of embedding and names object do not match.')

        match_rows = pd.Series(names).str.contains(pattern).to_numpy()
        return em[match_rows], np.array(names)[match_rows]

    def dist_to_center(ref_arr, arr):
        """Get the euclidean distance between each row of arr to center (mean) of ref_arr."""
        center = ref_arr.mean(axis=0)
        sqdist = np.sum((arr - center) ** 2, axis=1)
        return np.sqrt(sqdist)

    em_real, names_real = fm.load_pickle(real_embedding_pkl)
    em_sims, names_sims = fm.load_pickle(sims_embedding_pkl)

    # Extract embedding rows and names of real nuclei that match pattern.
    em_ref, names_ref = embeddings_match_names(em_real, names_real, pattern)

    # Get names of sim files that are within distance cutoffs of selected real
    # nucs in embedding space.
    dists = dist_to_center(em_ref, em_sims)
    names_withindist = np.array(names_sims)[(dists <= cutoff_upper) & (dists > cutoff_lower)]

    # Optional: save the filenames as text files for separate extraction.
    if savestem is not None:
        np.savetxt(savestem + '_real.txt', names_ref, fmt='%s')
        np.savetxt(savestem + '_sims.txt', names_withindist, fmt='%s')
        
    return names_ref, names_withindist

def display_image_sets(dir1, names1, dir2, names2, savefile):
    """Display selected images from two directories.
    
    Images are plotted in two sets separated by vertical space.
    For each image, left-most image is a max projection (in Z)
    and remaining are individual Z slices.

    Args:
        dir1: str
            Path to directory containing first image set
        names1: iterable
            Filenames from dir1 to display
        dir2: str
            Path to directory containing second image set
        names2: iterable
            Filenames from dir2 to display
        savefile: str
            File to save figure to
    """
    def load_ims(dir, names):
        """Load ims from dir found in names."""
        names_set = set(names)
        l = []
        fnames = []
        for f in os.listdir(dir):
            if (f[0] == '.') or (f[-3:] != 'pkl'):
                continue
            if f in names_set:
                im = fm.load_pickle(os.path.join(dir, f))
                l.append(im)
                fnames.append(f)
        return np.array(l)
    
    def display(ims, fig, vstart, nrows):
        """Plot images in two batches and save."""
        print(ims.shape)
        for im in ims:
            mask = np.where(im > 0, True, False)
            """
            #max_ = np.percentile(im, 99.9)
            print(round(np.max(im) / max_, 2))
            min_ = np.min(im[mask])
            im = (im - min_) / (max_ - min_) * 1000
            im = np.where(mask, im, 0)
            """
            pix = im[mask]
            im = im / np.median(pix)
            vmin = 0
            vmax = 4
            ax = fig.add_axes((0,vstart,0.15,0.15))
            ax.imshow(im.max(axis=0), vmin=vmin, vmax=vmax, cmap='cividis')
            ax.axis('off')
            ax = fig.add_axes((0.18,vstart,0.15,0.15))
            ax.imshow(im[8], vmin=vmin, vmax=vmax, cmap='cividis')
            ax.axis('off')
            ax = fig.add_axes((0.34,vstart,0.15,0.15))
            ax.imshow(im[14], vmin=vmin, vmax=vmax, cmap='cividis')
            ax.axis('off')
            ax = fig.add_axes((0.50,vstart,0.15,0.15))
            ax.imshow(im[20], vmin=vmin, vmax=vmax, cmap='cividis')
            ax.axis('off')
            ax = fig.add_axes((0.66,vstart,0.15,0.15))
            ax.imshow(im[26], vmin=vmin, vmax=vmax, cmap='cividis')
            ax.axis('off')

            vstart = vstart - (1 / nrows * 0.77)

    ims_1 = load_ims(dir1, names1)    
    ims_2 = load_ims(dir2, names2)
    nrows = (len(ims_1) + len(ims_2))
    fig = plt.figure(constrained_layout=False, facecolor='1', figsize=(8.48,12 * nrows / 6))
    display(ims_1, fig, 0.89, nrows)
    
    display(ims_2, fig, 0.89 - (1 / nrows * (len(ims_1) + 0.25)), nrows)
    fig.savefig(savefile, dpi=300)


def display_real_sim_pattern_cutoff(real_embedding_pkl, sims_embedding_pkl, dir_reals, dir_sims, pattern, cutoff_lower, cutoff_upper, savefile, nsample=[3,3]):            
    """Match a set of real images to a supplied pattern, identify simulated images
    that are embedded within specified distance range, display both together.
    
    Args:
        real_embedding_pkl: str
            Path to embedding pkl object for real nucs
        sims_embedding_pkl: str
            Path to embedding pkl object for simulated nucs
        dir_reals: str
            Directory containing real images
        dir_sims: str
            Directory containing simulated images
        pattern: str
            Pattern to match to names of real sim files
        cutoff_lower: float
            Lower limit for embedding distance
        cutoff_upper: float
            Upper limit for embedding distance
        savefile: None or str
            For for saving final image (must be image file)
        nsample: iterable of length 2
            Number of real and simulated images (respectively) to display
        
    """
    em_real, names_real = fm.load_pickle(real_embedding_pkl)
    em_sim, names_sim = fm.load_pickle(sims_embedding_pkl)
    names_real, names_sim = match_names_and_distcutoff(real_embedding_pkl, sims_embedding_pkl, cutoff_lower, cutoff_upper, pattern=pattern)
    # Sample from available images.
    names_real = np.random.choice(names_real, size=np.min([nsample[0], len(names_real)]), replace=False)
    names_sim = np.random.choice(names_sim, size=np.min([nsample[1], len(names_sim)]), replace=False)
    display_image_sets(dir_reals, names_real, dir_sims, names_sim, savefile)

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
    
