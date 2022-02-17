import flymovie as fm
import subprocess
import os
from time import time, process_time

t_start = time()

outfolder = fm.simnuc.sim_rpb1_batch(
    outfolder = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/testsims_1000',
    kernel=fm.load_pickle('/Users/michaelstadler/Bioinformatics/Projects/rpb1/PSFs/psf_20220210_21x25x25pixels_100x50x50voxel.pkl'),
    nsims=2,
    nreps=2,
    nprocesses=4,
    mask_dims=(100,100,100),
    sim_func=fm.sim_rpb1,
    nuc_rad=40,
    nfree_rng=[5_000, 40_000], 
    hlb_diam_rng=[8,14], 
    hlb_nmols_rng=[200,600], 
    n_clusters_rng=[0,1_000], 
    cluster_diam_mean_rng=[1,2], 
    cluster_diam_var_rng=[0,1], 
    cluster_nmols_mean_rng=[5,50], 
    cluster_nmols_var_rng=[1,10],
    noise_sigma_rng=[20,40], 
    dims_init=(85, 85, 85), 
    dims_kernel=(100,50,50), 
    dims_final=(250,85,85)
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