import flymovie as fm
import subprocess
import os
import numpy as np
from time import time, process_time

t_start = time()

outfolder = fm.sim_rpb1_batch(
    outfolder = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/match_cterm',
    kernelfile='/Users/michaelstadler/Bioinformatics/Projects/rpb1/PSFs/psf_20220210_21x25x25pixels_100x50x50voxel.pkl',
    maskfile='/Users/michaelstadler/Bioinformatics/Projects/rpb1/data/real_masks/mask_files/nc14_1.pkl',
    nsims=2,
    nreps=2,
    nprocesses=4,
    sim_func=fm.sim_rpb1,
    concentration=80, 
    hlb_diam_rng=[8,14], 
    hlb_nmols_rng=[1,1], 
    n_clusters_rng=[1,1], 
    cluster_diam_mean_rng=[1,1], 
    cluster_diam_var_rng=[0,0.001], 
    cluster_nmols_mean_rng=[1,1], 
    cluster_nmols_var_rng=[0.1,0.1],
    gfp_intensity=2.7,
    noise_sigma_rng=[9.8,10.6], 
    dims_init=(85, 85, 85), 
    dims_kernel=(100,50,50), 
    dims_final=(250,85,85),
    dilation_struct=np.ones((1,7,7)),
    mask_nuclei=True,
    only_valid=False 
)

# Sort files into left and right folders.
left = os.path.join(outfolder, 'left')
os.mkdir(left)
right = os.path.join(outfolder, 'right')
os.mkdir(right)

subprocess.call(['mv ' + os.path.join(outfolder, '*rep0*') + ' ' + left], shell=True)
subprocess.call(['mv ' + os.path.join(outfolder, '*rep1*') + ' ' + right], shell=True)

t_end = time()
print (t_end - t_start)