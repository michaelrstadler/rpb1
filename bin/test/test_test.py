import unittest
import numpy as np

from flymovie.general_functions import concatenate_5dstacks, clamp, labelmask_filter_objsize, labelmask_filter_objsize_apply4d, imfill, local_max, peak_local_max_nD, get_object_centroid

class TestConcatenate5dstacks(unittest.TestCase):
	
	def test_concatenate_5dstacks(self):
		input_ = [np.zeros([2,4,10,25,25]), np.zeros([2,4,10,25,25])]
		output = (np.zeros([2,8,10,25,25]), [4])
		test_output = concatenate_5dstacks(input_)
		self.assertTrue(np.array_equal(test_output[0],output[0]), 'Concatenation is wrong')
		self.assertTrue(np.array_equal(test_output[1],output[1]), 'Join frame is wrong')

#---------------------------------------------------------------------------

class TestClamp(unittest.TestCase):

	def test_clamp(self):
		self.assertEqual(clamp(10, 5, 15), 10,"Should be 10")
		self.assertEqual(clamp(16, 5, 15), 15,"Should be 15")
		self.assertEqual(clamp(1, 5, 15), 5,"Should be 5")

#---------------------------------------------------------------------------

class TestLabelmaskFilterObjsize(unittest.TestCase):

	def test_labelmask_filter_objsize(self):
		# Make a 10x100x100 labelmask with a 3x3x3 object 1.
		input_ = np.zeros([10, 100, 100])
		input_[4:7, 4:7, 4:7] = 1
		filt_20_30 = np.max(labelmask_filter_objsize(input_, 20, 30))
		filt_29_30 = np.max(labelmask_filter_objsize(input_, 29, 30))
		filt_20_25 = np.max(labelmask_filter_objsize(input_, 20, 25))
		self.assertEqual(filt_20_30, 1, "Should be 1")
		self.assertEqual(filt_29_30, 0, "Should be 0")
		self.assertEqual(filt_20_25, 0, "Should be 0")

#---------------------------------------------------------------------------

class TestLabelmaskFilterObjsizeApply4d(unittest.TestCase):

	def test_labelmask_filter_objsize_apply4d(self):
		# Make a 3x10x100x100 labelmask with a 3x3x3 object 1.
		input_ = np.zeros([3, 10, 100, 100])
		input_[:, 4:7, 4:7, 4:7] = 1
		filt_20_30 = np.max(labelmask_filter_objsize_apply4d(input_, 20, 30))
		filt_29_30 = np.max(labelmask_filter_objsize_apply4d(input_, 29, 30))
		filt_20_25 = np.max(labelmask_filter_objsize_apply4d(input_, 20, 25))
		self.assertEqual(filt_20_30, 1, "Should be 1")
		self.assertEqual(filt_29_30, 0, "Should be 0")
		self.assertEqual(filt_20_25, 0, "Should be 0")

#---------------------------------------------------------------------------

class TestImfill(unittest.TestCase):

	def test_imfill(self):
		# Make a 10x100x100 labelmask with a 3x3x3 object 1 with a 1 pixel
		# hole in it.
		mask = np.zeros([10, 100, 100])
		mask[4:7, 4:7, 4:7] = 1
		mask[5,5,5] = 0
		# Seed is default.
		self.assertEqual(np.count_nonzero(imfill(mask).flatten()), 27, 
			"Default background failed")
		self.assertEqual(np.count_nonzero(imfill(mask, (0,0,0)).flatten()), 
			27, "Seed in background failed")
		# Test behavior if bad seed point (in foreground) chosen.
		self.assertEqual(np.count_nonzero(imfill(mask, (4,4,4)).flatten()), 
			1e5, "Seed in foreground failed")

#---------------------------------------------------------------------------

class TestLocalMax(unittest.TestCase):

	def test_local_max(self):
		# Make a 3x10x100x100 labelmask with a 3x3x3 object 1.
		input_ = np.zeros([3, 10, 10])
		input_[0, 0, 0] = 10
		input_[2, 9, 9] = 20

		output_61818 = local_max(input_, (6, 18, 18))
		output_62020 = local_max(input_, (6 , 20, 20))
		self.assertEqual(np.sum(output_61818), 2, "Should be 2")
		self.assertEqual(np.sum(output_62020), 1, "Should be 1")

#---------------------------------------------------------------------------

class TestPeakLocalMaxnD(unittest.TestCase):

	def test_peak_local_max_nD(self):
		# Make a 3x10x100x100 labelmask with a 3x3x3 object 1.
		input_ = np.zeros([3, 10, 10])
		input_[0:2, 0:2, 0:2] = 10
		input_[2:, 8:, 8:] = 20

		output_61616 = peak_local_max_nD(input_, (6, 16, 16))
		output_62020 = peak_local_max_nD(input_, (6 , 20, 20))
		self.assertEqual(np.sum(output_61616[0]), 3, "Should be 3")
		self.assertEqual(np.sum(output_62020[0]), 1, "Should be 1")

#---------------------------------------------------------------------------

class TestGetObjectCentroid(unittest.TestCase):

	def test_get_object_centroid(self):
		# Make a 3x10x100x100 labelmask with a 3x3x3 object 1.
		input_ = np.zeros([10, 10, 10])
		input_[4:7, 4:7, 4:7] = 1
		self.assertEqual(get_object_centroid(input_, 1), (5, 5, 5), 
			"Bad centroid")

#---------------------------------------------------------------------------

class TestSum(unittest.TestCase):

	def test_sum(self):
		self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")


if __name__ == '__main__':
	unittest.main()