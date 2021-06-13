import unittest
import numpy as np
import pandas as pd
import os

from flymovie.analyze import *
from flymovie.load_save import load_pickle

class TestData():
    def __init__(self):
        pass

test_data = load_pickle(os.path.join(os.getcwd(), 'test_data', 'test_data.pkl'))

#---------------------------------------------------------------------------

class TestAnalyzeFunctions(unittest.TestCase):
	
	def test_add_volume_mean(self):
		input_ = test_data.filter_spot_duration_output
		test_output = add_volume_mean(input_, test_data.stack, channel=1, 
			ij_rad=3, z_rad=3)
		# Test if function added a column.
		self.assertTrue((test_output[1].shape[1]) == (input_[1].shape[1] + 1)
		, 'Should have added a column')
	
	def test_add_gaussian_integration(self):
		input_ = test_data.filter_spot_duration_output
		test_output = add_gaussian_integration(input_)
		# Test if function added a column.
		self.assertTrue((test_output[1].shape[1]) == (input_[1].shape[1] + 1)
		, 'Should have added a column')

	def test_align_traces(self):
		input_ = test_data.df
		locs = [(1,2), (2,2), (3,3)]
		# Currently, just test that it runs.
		align_traces(input_, locs, window_size=3)
	
	def test_spotdf_bleach_correct(self):
		# Test if it runs.
		spotdf_bleach_correct(test_data.df, test_data.df_stack[1])

	def test_spotdf_plot_traces(self):
		# Test if it runs.
		spotdf_plot_traces(test_data.df, test_data.df, 0)

	def test_spotdf_plot_traces_bleachcorrect(self):
		# Test if it runs.
		spotdf_plot_traces_bleachcorrect(test_data.df, test_data.df, 0, 
			test_data.df_stack[1])	

	def test_threshold_w_slope(self):
		input_ = np.ones((5,10,10)) * 10
		test_output_1 = threshold_w_slope(input_, 0, 2, 0)
		test_output_2 = threshold_w_slope(input_, 10, 2, -1)
		test_output_3 = threshold_w_slope(input_, 10, 2, 1)

		self.assertTrue(np.array_equal(test_output_1, np.ones((1,5,10,10))), "Should be equal")
		self.assertTrue(np.array_equal(test_output_2[0,0], np.zeros((10,10))), "Should be equal")
		self.assertTrue(np.array_equal(test_output_2[0,4], np.ones((10,10))), "Should be equal")
		self.assertTrue(np.array_equal(test_output_3[0,0], np.ones((10,10))), "Should be equal")
		self.assertTrue(np.array_equal(test_output_3[0,4], np.zeros((10,10))), "Should be equal")

		
if __name__ == '__main__':
	unittest.main()