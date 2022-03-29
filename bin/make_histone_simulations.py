import flymovie as fm
import subprocess
import os
import numpy as np
from time import time, process_time

t_start = time()

outfolder = fm.sim_histones_batch(
    '/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/test-histones',
    kernelfile='/Users/michaelstadler/Bioinformatics/Projects/rpb1/PSFs/psf_20220210_21x25x25pixels_100x50x50voxel.pkl',
    maskfile='/Users/michaelstadler/Bioinformatics/Projects/rpb1/data/real_masks/mask_files/nc13_1.pkl',
    nsims=2,
    nreps=2,
    nprocesses=2,
    nfree_rng=[1_000,1_000],
    genome_size=1.8e8,
    bp_per_nucleosome_rng=[250,2000], 
    fraction_labeled_rng=[0.1,0.8], 
    density_min_rng=[2,4], 
    density_max_rng=[8,10], 
    rad_max_rng=[2.5,4],     
    a1_rng=[0,0],
    p1_rng=[0,0],
    noise_sigma_rng=[3,3],
    dims_init=(85, 85, 85), 
    dims_kernel=(100,50,50), 
    dims_final=(250,85,85),
    mask_nuclei=False, 
    dilation_struct=np.ones((1,7,7))
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