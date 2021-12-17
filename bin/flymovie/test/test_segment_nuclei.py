import unittest
import numpy as np
import pandas as pd
import os
import sys

from flymovie.segment_nuclei import *
from flymovie.load_save import load_pickle

# Workaround for github file size limits...needed to split 
# test data into multiple files. To do this I load them with
# load_test_data function in test package.

wkdir = os.getcwd()
sys.path.append(wkdir)
from test.load_test_data import load_test_data

class TestData():
    def __init__(self):
        pass

test_data = load_test_data(wkdir)

#---------------------------------------------------------------------------
class TestSegmentNuclei3DStackRpb1(unittest.TestCase):

	def test_segment_nuclei_3Dstack_rpb1(self):
		nucchannel = 0
		seed_window = (30,30)
		min_nuc_center_dist = 30
		sigma = 5
		usemax = True
		stack = test_data.stack
		correct_output = test_data.segment_nuclei_3Dstack_rpb1_output

		test_output = segment_nuclei_3Dstack_rpb1(stack[nucchannel, 0], 
			seed_window=seed_window, min_nuc_center_dist=min_nuc_center_dist, 
			sigma=sigma, usemax=usemax)


		self.assertTrue(np.array_equal(test_output, correct_output),
			"Should be same")
		
#---------------------------------------------------------------------------
class TestSegmentNuclei4dStack(unittest.TestCase):

	def test_segment_nuclei_4dstack(self):
		nucchannel = 0
		seed_window = (30,30)
		min_nuc_center_dist = 30
		sigma = 5
		usemax = True
		stack = test_data.stack
		correct_output = test_data.segment_nuclei_4dstack_output

		test_output = segment_nuclei_4dstack(stack[nucchannel], 
			segment_nuclei_3Dstack_rpb1, seed_window=seed_window, 
			min_nuc_center_dist=min_nuc_center_dist, sigma=sigma, usemax=usemax)


		self.assertTrue(np.array_equal(test_output, correct_output),
			"Should be same")

#---------------------------------------------------------------------------
class TestLabelmaskFilterObjsize(unittest.TestCase):

	def test_labelmask_filter_objsize(self):
		size_min = 0
		size_max = 90000
		
		input_ = test_data.segment_nuclei_4dstack_output[0]
		correct_output = test_data.labelmask_filter_objsize_output
		test_output = labelmask_filter_objsize(input_, size_min, size_max)

		self.assertTrue(np.array_equal(test_output, correct_output),
			"Should be same")

#---------------------------------------------------------------------------
class TestLabelmaskFilterObjsizeApply4d(unittest.TestCase):

	def test_labelmask_filter_objsize_apply4d(self):
		size_min = 0
		size_max = 90000
		
		input_ = test_data.segment_nuclei_4dstack_output
		correct_output = test_data.labelmask_filter_objsize_apply4d_output
		test_output = labelmask_filter_objsize_apply4d(input_, size_min, size_max)

		self.assertTrue(np.array_equal(test_output, correct_output),
			"Should be same")

#---------------------------------------------------------------------------
class TestObjectCircularity(unittest.TestCase):

	def test_object_circularity(self):

		self.assertAlmostEqual(object_circularity(
			test_data.labelmask_filter_objsize_output, 1), 
		0.7789622038689256, "Should be same")

#---------------------------------------------------------------------------
class TestFilterLabelmaskCircularity(unittest.TestCase):

	def test_filter_labelmask_circularity(self):
		input_ = test_data.labelmask_filter_objsize_output
		correct_output = test_data.filter_labelmask_circularity_output
		test_output = filter_labelmask_circularity(input_, 5)
		self.assertTrue(np.array_equal(test_output, correct_output),
			"Should be same")

#---------------------------------------------------------------------------
class TestFilterLabelmaskCircularityApply4d(unittest.TestCase):

	def test_filter_labelmask_circularity_apply4d(self):
		input_ = test_data.labelmask_filter_objsize_apply4d_output
		correct_output = test_data.filter_labelmask_circularity_apply4d_output
		test_output = filter_labelmask_circularity_apply4d(input_, slicenum=6, 
			circularity_min=0.7)
		self.assertTrue(np.array_equal(test_output, correct_output),
			"Should be same")

#---------------------------------------------------------------------------
class TestUpdateLabels(unittest.TestCase):

	def test_update_labels(self):
		input1 = test_data.filter_labelmask_circularity_apply4d_output[0]
		input2 = test_data.filter_labelmask_circularity_apply4d_output[1]
		correct_output = test_data.update_labels_output
		test_output = update_labels(input1, input2)
		self.assertTrue(np.array_equal(test_output, correct_output),
			"Should be same")
#---------------------------------------------------------------------------
class TestConnectNuclei(unittest.TestCase):

	def test_connect_nuclei(self):
		input_ = test_data.filter_labelmask_circularity_apply4d_output
		correct_output = test_data.connect_nuclei_output
		test_output = connect_nuclei(input_)
		self.assertTrue(np.array_equal(test_output, correct_output),
			"Should be same")

#---------------------------------------------------------------------------
class TestInterpolateNuclearMask(unittest.TestCase):

	def test_interpolate_nuclear_mask(self):
		input_ = test_data.interpolate_nuclear_mask_input
		correct_output = test_data.interpolate_nuclear_mask_output
		test_output = interpolate_nuclear_mask(input_)
		self.assertTrue(np.array_equal(test_output, correct_output),
			"Should be same")

#---------------------------------------------------------------------------

class TestClamp(unittest.TestCase):

	def test_clamp(self):
		pass


#---------------------------------------------------------------------------

if __name__ == '__main__':
	unittest.main()