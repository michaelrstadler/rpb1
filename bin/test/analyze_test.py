import unittest
import numpy as np
import pandas as pd
import os
import sys

from flymovie.analyze import *
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

	def test_spot_data_add_depth(self):
		input_ = {}
		for n in range(0,10):
			input_[n] = np.ones((20, 13))
			input_[n][:, 0] = np.arange(0,20)
		surface_before = 0
		surface_after = 1
		join_frames = [5]
		start_positions = [5, 4]

		output = spot_data_add_depth(input_, surface_before, surface_after, 
			join_frames, start_positions, z_interval=0.5)
		
		# Test if it added a column.
		for n in range(0, 10):
			self.assertEqual(output[n].shape[1], 14, 'Failed to add column')

	def test_spot_data_extract_depthbinned_intensities(self):
		input_ = {}
		for n in range(0,10):
			input_[n] = np.ones((20, 13))
			# Set frames.
			input_[n][:, 0] = np.arange(0,20)
			# Set depths.
			input_[n][:, 12] = np.arange(0, 10, 0.5)
			# Set intensities.
			input_[n][:, 9] = np.arange(0, 10, 0.5)

		output = spot_data_extract_depthbinned_intensities(input_, col_depth=12, col_to_bin=9, 
        	bin_size=0.5, nbins=100)

		self.assertEqual(len(output), 100, 'Length should be 100.')
		self.assertEqual(np.mean(output[0]), 0, 'Should be 0.')
		self.assertEqual(np.mean(output[11]), 5.5, 'Should be 5.5.')
		self.assertEqual(len(output[90]), 0, 'Should be 0.')

	def test_spot_data_depth_correct_stdcandle(self):
		input_ = {}
		for n in range(0,10):
			input_[n] = np.ones((20, 13))
			# Set frames.
			input_[n][:, 0] = np.arange(0,20)
			# Set depths.
			input_[n][:, 12] = np.arange(0, 10, 0.5)
			# Set intensities.
			input_[n][:, 9] = np.arange(0, 10, 0.5)

		paramgrid_a = np.ones((250,200))
		paramgrid_b = np.zeros((250,200))
		paramgrid_c = np.ones((250,200))
		paramgrids = (paramgrid_a, paramgrid_b, paramgrid_c)
		output = spot_data_depth_correct_stdcandle(input_, paramgrids, col_to_correct=9, col_depth=12, target_depth=10)
		for spot_id in output:
			self.assertTrue(np.array_equal(output[spot_id][:,9], np.repeat(2, 20)), 'Should be all 2s.')

	def test_spot_data_depth_correct_fromdata(self):
		input_ = {}
		# Note: If intensities and depths are 'too discrete' (I originally used increments of 0.5),
		# curve_fit throws a warning that it can't estimate covariances. Using more continuous
		# values solved this.
		intensities = np.arange(500, 1500, 0.1)
		depths = np.arange(5,25, 0.05)
		for n in range(0,10):
			input_[n] = np.ones((20, 13))
			# Set frames.
			input_[n][:, 0] = np.arange(0,20)
			# Set depths.
			input_[n][:, 12] = np.random.choice(depths, 20) + np.random.normal(0,1,20)
			# Set intensities.
			input_[n][:, 9] = np.random.choice(intensities, 20) + np.random.normal(0,5,20)

		spot_data_depth_correct_fromdata(input_, col_to_correct=9, 
			col_depth=12, target_depth=10, fit_depth_min=12, 
			fit_depth_max=18)

	def test_spot_data_bleach_correct_framemean(self):
		# Spot data uniformly 1 except time column.
		spot_data = {}
		for i in range(1,5):
			spot_data[i] = np.ones((5,12))
			spot_data[i][:,0] = np.arange(0,5)
		
		stack = np.ones((2, 5, 5, 100, 100))
		# Make stack means 1, 2, 3, 4, 5.
		for i in range(1,5):
			stack[:, i] = stack[:, i] * (i+1)
		#print(np.mean(stack[1,3]))
		output = spot_data_bleach_correct_framemean(spot_data, stack, 1, sigma=0.1)
		
		self.assertAlmostEqual(output[1][0, 9], 1, 2, 'Should be 1.')
		self.assertAlmostEqual(output[2][1, 10], 0.5, 2, 'Should be 0.5.')
		self.assertAlmostEqual(output[3][2, 11], 0.333333333, 2, 'Should be 0.33.')
		self.assertAlmostEqual(output[4][3, 10], 0.25, 2, 'Should be 0.25.')

if __name__ == '__main__':
	unittest.main()