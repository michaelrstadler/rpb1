import flymovie as fm
import numpy as np
import flymovie
from time import time, process_time
import sys
import os
import random
import string
import re

#outfile = '/global/home/users/mstadler/scratch/sims1-6-57k-2'
outfile_stem = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/data/simulations/test'

# Get unique string to identify files and log.
characters = string.ascii_letters + string.digits
unique_id = ''.join(random.choice(characters) for i in range(10))
outfile = outfile_stem + '_' + unique_id + '.pkl'
logfile = outfile_stem + '_' + unique_id + '_log.txt'

num_sims = 100_000
bg_mean_range=[9_000, 11_000]
bg_var_range=[2700, 3300] 
blob_intensity_mean_range=[15_000, 25_000]
blob_intensity_var_range=[500, 2500]
blob_radius_mean_range=[0.3, 1]
blob_radius_var_range=[0.25, 0.5]
blob_number_range = [0, 300]

z_ij_ratio=2
zdim = 20
idim = 300
jdim = 200
nuc_spacing=100 
nuc_rad=50
process_function = fm.make_scalespace_dog_hist

# For process_function
numbins=325
ss_sigmas=[0,0.5,1,2,4]
ss_histrange=(0,66_000)
dog_sigmas=[(1,2),(1,3),(1,5)] 
dog_histrange=(-10_000, 10_000)

t_start = time()
data_ = fm.make_parameter_hist_data(num_sims, bg_mean_range, bg_var_range, 
    blob_intensity_mean_range, blob_intensity_var_range, 
    blob_radius_mean_range, blob_radius_var_range, blob_number_range, 
    z_ij_ratio, zdim, idim, jdim, nuc_spacing, nuc_rad, process_function, 
    numbins=numbins, ss_sigmas=ss_sigmas, ss_histrange=ss_histrange, 
    dog_sigmas=dog_sigmas, dog_histrange=dog_histrange)
                                       
t_end = time()
print (t_end - t_start)
fm.save_pickle(data_, outfile)

param_names_cat = 'num_sims, bg_mean_range, bg_var_range, blob_intensity_mean_range, blob_intensity_var_range, blob_radius_mean_range, blob_radius_var_range, blob_number_range, z_ij_ratio, zdim, idim, jdim, nuc_spacing, nuc_rad, process_function, numbins, ss_sigmas, ss_histrange, dog_sigmas, dog_histrange'
param_names = re.split(',\s*', param_names_cat)

params = (num_sims, bg_mean_range, bg_var_range, 
    blob_intensity_mean_range, blob_intensity_var_range, 
    blob_radius_mean_range, blob_radius_var_range, blob_number_range, 
    z_ij_ratio, zdim, idim, jdim, nuc_spacing, nuc_rad, process_function, 
    numbins, ss_sigmas, ss_histrange, dog_sigmas, dog_histrange)

with open(logfile, 'w') as log:
    for i in range(0, len(params)):
        log.write(param_names[i] + ': ' + str(params[i]) + '\n')

sys.exit()