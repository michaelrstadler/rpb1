import flymovie as fm
from time import time, process_time

t_start = time()

fm.simnuc.sim_rpb1_rand_batch(
    maskfile = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/real_nuclear_masks_nc13.pkl',
    outfolder = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/test_sims_realnuc_',
    nsims=10,
    nreps=2,
    nprocesses=20,
    nuc_bg_mean_rng=[8_000, 10_000], 
    nonnuc_bg_mean_rng=[800,1200], 
    noise_sigma_rng=[200,400], 
    nblobs_rng=[0,100], 
    blob_intensity_mean_rng=[8_000, 12_000], 
    blob_intensity_std_rng=[1_000, 3_000],
    blob_sigma_base_rng=[0.5,0.5],
    blob_sigma_k_rng=[0.5,0.5], 
    blob_sigma_theta_rng=[0.5,0.5], 
    hlb_intensity_rng=[15_000, 25_000],
    hlb_sigma_rng=[4,6], 
    hlb_p_rng=[1.7,2.3]
)

t_end = time()
print (t_end - t_start)