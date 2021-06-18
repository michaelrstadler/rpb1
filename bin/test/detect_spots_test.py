import unittest
import numpy as np
import pandas as pd
import sys

from flymovie.detect_spots import *
from flymovie.load_save import load_pickle

# Workaround for github file size limits...needed to split 
# test data into multiple files. To do this I load them with
# load_test_data function in test package.
wkdir = os.getcwd()
sys.path.append(wkdir)
from load_test_data import load_test_data

class TestData():
    def __init__(self):
        pass

test_data = load_test_data(wkdir)

#---------------------------------------------------------------------------

class TestDetectSpots(unittest.TestCase):

	def test_fit_ms2(self):
		stack = test_data.stack[1]
		min_distances = (stack.shape[-2], 25, 25)
		fitwindow_rad_xy = 10
		fitwindow_rad_z = 2
		sigma_small = 1
		correct_output = test_data.fit_ms2_output

		# Just make sure it runs in nucleus mode.
		_ = fit_ms2(stack, fitwindow_rad_xy=fitwindow_rad_xy, 
		sigma_small=sigma_small, mode='nucleus', nucmask = 
		np.ones_like(stack))
		
		test_output = fit_ms2(stack, min_distances=min_distances, 
			fitwindow_rad_xy=fitwindow_rad_xy, sigma_small=sigma_small)

		for n in range(0, len(test_output)):
			# Fitting is not deterministic, so some tolerance must be allowed.
			self.assertTrue(np.allclose(test_output[n], correct_output[n], 
				atol=0.5), 'Should be the same')
	
	def test_filter_ms2fits(self):
		input_ = test_data.fit_ms2_output
		correct_output = test_data.filter_ms2fits_output
		peakiness = 4.2
		test_output = filter_ms2fits(input_, peakiness)

		for n in range(0, len(test_output)):
			self.assertTrue(np.array_equal(test_output[n], correct_output[n]), 
				'Should be the same')
	
	def test_connect_ms2_frames_via_nuclei(self):
		input_ = test_data.filter_ms2fits_output
		nucmask = test_data.filter_labelmask_circularity_apply4d_output
		correct_output = test_data.connect_ms2_frames_via_nuclei_output
		test_output = connect_ms2_frames_via_nuclei(input_, nucmask)

		for n in correct_output.keys():
			self.assertTrue(np.array_equal(test_output[n], correct_output[n]), 
				'Should be the same')
		
	def test_connect_ms2_fits_focuscorrect(self):
		input_ = test_data.filter_ms2fits_output
		nucmask = test_data.filter_labelmask_circularity_apply4d_output
		# This function is potentially soon for deprecation, 
		# so just check if it runs for now.
		connect_ms2_fits_focuscorrect(input_, [0], [0], nucmask) 

#---------------------------------------------------------------------------

if __name__ == '__main__':
	unittest.main()