import flymovie as fm
from time import time, process_time

t_start = time()

fm.simnuc.sim_rpb1_batch(
    outfolder = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/testsims_1000',
    kernel=fm.load_pickle('/Users/michaelstadler/Bioinformatics/Projects/rpb1/PSFs/psf_20220210_21x25x25pixels_100x50x50voxel.pkl'),
    nsims=1000,
    nreps=2,
    nprocesses=4,
    mask_dims=(100,100,100),
    sim_func=fm.sim_rpb1,
    nuc_rad=40,
    nfree_rng=[5_000, 40_000], 
    hlb_diam_rng=[8,14], 
    hlb_nmols_rng=[100,800], 
    n_clusters_rng=[0,1_000], 
    cluster_diam_mean_rng=[1,2], 
    cluster_diam_var_rng=[0,1], 
    cluster_nmols_mean_rng=[5,50], 
    cluster_nmols_var_rng=[1,10],
    noise_sigma_rng=[10,30], 
    dims_init=(85, 85, 85), 
    dims_kernel=(100,50,50), 
    dims_final=(250,85,85)
)

t_end = time()
print (t_end - t_start)