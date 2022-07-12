import flymovie as fm
import os
import string
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("--outfolder", type=str,  required=True)
    parser.add_argument("--hlb_diam_rng", type=float, nargs=2, required=True)
    parser.add_argument("--hlb_nmols_rng", type=float, nargs=2, required=True)
    parser.add_argument("--nclusters_rng", type=float, nargs=2, required=True)
    parser.add_argument("--cluster_diam_mean_rng", type=float, nargs=2, required=True)
    parser.add_argument("--cluster_diam_var_rng", type=float, nargs=2, required=True)
    parser.add_argument("--cluster_nmols_mean_rng", type=float, nargs=2, required=True)
    parser.add_argument("--cluster_nmols_var_rng", type=float, nargs=2, required=True)
    parser.add_argument( "--noise_sigma_rng", type=float, nargs=2, required=True)
    parser.add_argument( "--concentration", type=float, required=True)
    parser.add_argument( "--kernel_file", type=str, required=True)
    parser.add_argument( "--mask_file", type=str, required=True)
    parser.add_argument( "--param_tojitter", type=str, required=True)
    parser.add_argument( "--num_batches", type=int, required=True)
    parser.add_argument( "--num_bins", type=int, required=True)
    parser.add_argument("--gfp_intensity", type=float, required=False, default=1)
    parser.add_argument("--dims_init", type=int, required=False, nargs=3, default=(85,85,85))
    parser.add_argument("--dims_kernel", type=int, required=False, nargs=3, default=(100,50,50))
    parser.add_argument("--dims_final", type=int, required=False, nargs=3, default=(250,85,85))
    parser.add_argument("--dims_dilation", type=int, required=False, nargs=3, default=(1,7,7))
    parser.add_argument("--mask_nuclei", action='store_true')
       
    args = parser.parse_args()
    return args


args = parse_args()

if args.param_tojitter not in ['hlb_diam', 'hlb_nmols', 'nclusters', 'cluster_diam_mean', 'cluster_nmols_mean', 'cluster_nmols_var', 'noise_sigma']:
    raise ValueError('param to jitter not valid')

# Select batch of params.
for _ in range(args.num_batches):
    hlb_diam_rng = [fm.randomize_ab(args.hlb_diam_rng)] * 2
    hlb_nmols_rng = [fm.randomize_ab(args.hlb_nmols_rng)] * 2
    nclusters_rng = [fm.randomize_ab(args.nclusters_rng)] * 2
    cluster_diam_mean_rng = [fm.randomize_ab(args.cluster_diam_mean_rng)] * 2
    cluster_diam_var_rng = [fm.randomize_ab(args.cluster_diam_var_rng)] * 2
    cluster_nmols_mean_rng = [fm.randomize_ab(args.cluster_nmols_mean_rng)] * 2
    cluster_nmols_var_rng = [fm.randomize_ab(args.cluster_nmols_var_rng)] * 2
    noise_sigma_rng = [fm.randomize_ab(args.noise_sigma_rng)] * 2

    if args.param_tojitter == 'hlb_diam':
        hlb_diam_rng = args.hlb_diam_rng
    if args.param_tojitter == 'hlb_nmols':
        hlb_nmols_rng = args.hlb_nmols_rng
    if args.param_tojitter == 'nclusters':
        nclusters_rng = args.nclusters_rng
    if args.param_tojitter == 'cluster_diam_mean':
        cluster_diam_mean_rng = args.cluster_diam_mean_rng
    if args.param_tojitter == 'cluster_diam_var':
        cluster_diam_var_rng = args.cluster_diam_var_rng
    if args.param_tojitter == 'cluster_nmols_mean':
        cluster_nmols_mean_rng = args.cluster_nmols_mean_rng    
    if args.param_tojitter == 'cluster_nmols_var':
        cluster_nmols_var_rng = args.cluster_nmols_var_rng   
    if args.param_tojitter == 'noise_sigma':
        noise_sigma_rng = args.noise_sigma_rng   

    outfolder = fm.sim_rpb1_batch(
        outfolder=args.outfolder,
        kernelfile=args.kernel_file,
        maskfile=args.mask_file,
        nsims=args.num_batches,
        nreps=1,
        nprocesses=4,
        sim_func=fm.sim_rpb1,
        concentration=args.concentration, 
        hlb_diam_rng=hlb_diam_rng, 
        hlb_nmols_rng=hlb_nmols_rng, 
        n_clusters_rng=nclusters_rng, 
        cluster_diam_mean_rng=cluster_diam_mean_rng, 
        cluster_diam_var_rng=cluster_diam_var_rng, 
        cluster_nmols_mean_rng=cluster_nmols_mean_rng, 
        cluster_nmols_var_rng=cluster_nmols_var_rng,
        gfp_intensity=args.gfp_intensity,
        noise_sigma_rng=noise_sigma_rng, 
        dims_init=args.dims_init, 
        dims_kernel=args.dims_kernel, 
        dims_final=args.dims_final,
        dilation_struct=np.ones((args.dims_dilation)),
        mask_nuclei=args.mask_nuclei,
        unique_folder_id = False,
        write_logfile=False
    )
# Set up 0 ranges for all params but the jittered one.

# Call rpb1 batch 