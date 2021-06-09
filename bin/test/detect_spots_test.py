import unittest
import numpy as np
import pandas as pd

from flymovie.detect_spots import *
from flymovie.load_save import load_pickle

class TestData():
    def __init__(self):
        pass

test_data = load_pickle(os.path.join(os.getcwd(), 'test_data', 'test_data.pkl'))

#---------------------------------------------------------------------------

class TestFitMs2(unittest.TestCase):
	
	def test_fit_ms2(self):
		stack = test_data.stack[1]
		min_distances = (stack.shape[-2], 25, 25)
		fitwindow_rad_xy = 10
		fitwindow_rad_z = 2
		sigma_small = 1
		correct_output = test_data.fit_ms2_output
		test_output = fit_ms2(stack, min_distances=min_distances, 
			fitwindow_rad_xy=fitwindow_rad_xy, sigma_small=sigma_small)

		for n in range(0, len(test_output)):
			# Fitting is not deterministic, so some tolerance must be allowed.
			self.assertTrue(np.allclose(test_output[n], correct_output[n], 
				atol=0.5), 'Should be the same')

#---------------------------------------------------------------------------

class TestFilterMS2Fits(unittest.TestCase):
	
	def test_filter_ms2fits(self):
		input_ = test_data.fit_ms2_output
		correct_output = test_data.filter_ms2fits_output
		peakiness = 4.2
		test_output = filter_ms2fits(input_, peakiness)

		for n in range(0, len(test_output)):
			self.assertTrue(np.array_equal(test_output[n], correct_output[n]), 
				'Should be the same')

#---------------------------------------------------------------------------

class TestConnectMS2FramesViaNuclei(unittest.TestCase):
	
	def test_connect_ms2_frames_via_nuclei(self):
		input_ = test_data.filter_ms2fits_output
		nucmask = test_data.filter_labelmask_circularity_apply4d_output
		correct_output = test_data.connect_ms2_frames_via_nuclei_output
		test_output = connect_ms2_frames_via_nuclei(input_, nucmask)

		for n in correct_output.keys():
			self.assertTrue(np.array_equal(test_output[n], correct_output[n]), 
				'Should be the same')

#---------------------------------------------------------------------------

if __name__ == '__main__':
	unittest.main()