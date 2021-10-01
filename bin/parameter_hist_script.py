import flymovie as fm
import dask
import numpy as np
import flymovie
from time import time, process_time

outfile = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/results/histparams_test2.pkl'
bg_mean_range=[9000, 9500, 10500, 11000, 11500]
bg_var_range=[600, 800, 1000,1200, 1400]
blob_intensity_mean_range=np.arange(20000, 30000, 1000)
blob_intensity_var_range=np.arange(1000,10_000, 1000)
blob_radius_mean_range=np.arange(1, 5)
blob_radius_var_range=[0.5]
blob_number_range=np.arange(0, 500, 25)

t_start = time()
data_delayed = fm.make_parameter_hist_data(bg_mean_range, bg_var_range, blob_intensity_mean_range, 
    blob_intensity_var_range, blob_radius_mean_range, blob_radius_var_range, 
    blob_number_range)

data_ = dask.compute(data_delayed)
t_end = time()
print (len(data_), t_end - t_start)
fm.save_pickle(data_, outfile)