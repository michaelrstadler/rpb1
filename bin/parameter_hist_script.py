import flymovie as fm
import dask
import numpy as np
import flymovie
from time import time, process_time

from dask.distributed import Client
if __name__ == "__main__":
    client = Client(n_workers=4)

outfile = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/histparams_test2.pkl'
"""
bg_mean_range=[9000, 9500, 10500, 11000, 11500]
bg_var_range=[600]
blob_intensity_mean_range=[20000]
blob_intensity_var_range=[5000]
blob_radius_mean_range=[3]
blob_radius_var_range=[0.5]
blob_number_range=[250]
"""
bg_mean_range=[9000, 9500, 10500, 11000, 11500]
bg_var_range=[600]
blob_intensity_mean_range=[30000]
blob_intensity_var_range=[10000]
blob_radius_mean_range=[5]
blob_radius_var_range=[0.5]
blob_number_range=[500]

t_start = time()
data_delayed = fm.make_parameter_hist_data(bg_mean_range, bg_var_range, blob_intensity_mean_range, 
    blob_intensity_var_range, blob_radius_mean_range, blob_radius_var_range, 
    blob_number_range)

data_ = dask.compute(data_delayed)
t_end = time()
print (len(data_), t_end - t_start)
fm.save_pickle(data_, outfile)