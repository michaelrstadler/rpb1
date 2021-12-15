import flymovie as fm
import numpy as np
import flymovie
from time import time, process_time
import sys
import os
import random
import string
import re
import subprocess

outfolder = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/data/simulations/sims-test'
num_sims = 100
num_replicates=2
bg_mean_range = [9_000, 11_000]
bg_var_range = [500, 1000]
blob_intensity_mean_range = [10_000, 15_000]
blob_intensity_var_range = [5, 10]
blob_radius_mean_range = [0.5, 1]
blob_radius_var_range = [0.5, 1]
blob_number_range = [20, 200]
z_ij_ratio=2
zdim=20
idim=100
jdim=100
nuc_spacing=1000
nuc_rad=50

# Get unique string to identify files and log.
characters = string.ascii_letters + string.digits
unique_id = ''.join(random.choice(characters) for i in range(10))
outfolder = outfolder + '_' + unique_id

if not os.path.isdir(outfolder):
    os.mkdir(outfolder)

fm.make_simulations_from_sampled_params(outfolder=outfolder, bg_mean_range= bg_mean_range, bg_var_range = bg_var_range, blob_intensity_mean_range=blob_intensity_mean_range, 
    blob_intensity_var_range=blob_intensity_var_range, blob_radius_mean_range=blob_radius_mean_range, blob_radius_var_range=blob_radius_var_range, 
    blob_number_range=blob_number_range, num_sims=num_sims, num_replicates=num_replicates, z_ij_ratio=z_ij_ratio, zdim=zdim, idim=idim, jdim=jdim, 
    nuc_spacing=nuc_spacing, nuc_rad=nuc_rad)

# Log parameters in logfile.
logfile = os.path.join(outfolder, unique_id + '_log.txt')

param_names_cat = 'num_sims, num_replicates, bg_mean_range, bg_var_range, blob_intensity_mean_range, blob_intensity_var_range, blob_radius_mean_range, blob_radius_var_range, blob_number_range, z_ij_ratio, zdim, idim, jdim, nuc_spacing, nuc_rad'
param_names = re.split(',\s*', param_names_cat)
params = [num_sims, num_replicates, bg_mean_range, bg_var_range, blob_intensity_mean_range, blob_intensity_var_range, blob_radius_mean_range, blob_radius_var_range, blob_number_range, z_ij_ratio, zdim, idim, jdim, nuc_spacing, nuc_rad]

with open(logfile, 'w') as log:
    for i in range(0, len(params)):
        log.write(param_names[i] + ': ' + str(params[i]) + '\n')

# Split files into left and right.
left_folder = os.path.join(outfolder, 'left')
right_folder = os.path.join(outfolder, 'right')
os.mkdir(left_folder)
os.mkdir(right_folder)
subprocess.run('mv ' + os.path.join(outfolder, '*_0.pkl') + ' ' + str(left_folder) , shell=True)
subprocess.run('mv ' + os.path.join(outfolder, '*_1.pkl') + ' ' + str(right_folder) , shell=True)

sys.exit()