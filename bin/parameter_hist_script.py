import flymovie as fm
import dask
import numpy as np
import flymovie
from time import process_time

bg_mean_range=[50,100]
bg_var_range=[1]
blob_intensity_mean_range=[200]
blob_intensity_var_range=[10]
blob_radius_mean_range=np.arange(1)
blob_radius_var_range=[1]
blob_number_range=[100]

t_start = process_time()
data_delayed = fm.make_parameter_hist_data(bg_mean_range, bg_var_range, blob_intensity_mean_range, 
    blob_intensity_var_range, blob_radius_mean_range, blob_radius_var_range, 
    blob_number_range)
t_end = process_time()

data_ = dask.compute(data_delayed)
print (len(data_), t_end - t_start)