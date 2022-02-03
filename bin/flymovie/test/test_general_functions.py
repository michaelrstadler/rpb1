import unittest
import numpy as np
import pandas as pd

from flymovie.general_functions import *

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

class TestLabelmaskApplyMorphology(unittest.TestCase):

	def test_labelmask_apply_morphology(self):
		from scipy import ndimage as ndi
		# Make a 10x10 labelmask with a 3x3 object 3.
		labelmask = np.zeros([10, 10, 10])
		labelmask[4:7, 4:7, 4:7] = 10
		dilate1 = labelmask_apply_morphology(labelmask, 
			ndi.morphology.binary_dilation, struct=np.ones((3, 3, 3)), 
			expand_size=(3, 3, 3))
		erode1 = labelmask_apply_morphology(labelmask, 
			ndi.morphology.binary_erosion, struct=np.ones((3, 3, 3)), 
			expand_size=(3, 3, 3))
		self.assertEqual(np.sum(dilate1), 1250, 'Dilation failed')
		self.assertEqual(np.sum(erode1), 10, 'Erosion failed')
#---------------------------------------------------------------------------

class TestMeshLike(unittest.TestCase):

	def test_mesh_like(self):
		self.assertEqual(np.sum(mesh_like( np.ones((3, 3, 3)), 3)), 81, "Mesh is wrong")

#---------------------------------------------------------------------------

class TestFindBackgroundPoint(unittest.TestCase):

	def test_find_background_point(self):
		mask = np.zeros([10, 100, 100])
		mask[4:7, 4:7, 4:7] = 1
		point = find_background_point(mask)
		for i in range(0,100):
			self.assertEqual(mask[point], 0,  "Point is not in background")

#---------------------------------------------------------------------------

class TestRelabelLabelmask(unittest.TestCase):

	def test_relabel_labelmask(self):
		mask = np.zeros([10, 100, 100])
		mask[4:7, 4:7, 4:7] = 10
		mask[0:2, 0:2, 0:2] = 81
		relabeled = relabel_labelmask(mask)
		
		self.assertTrue(np.array_equal(np.unique(relabeled), [0,1,2]),  
			"Relabel failed")

#---------------------------------------------------------------------------

class TestSortfreq(unittest.TestCase):

	def test_sortfreq(self):
		list_ = [10, 10, 10, 10, 4, 4, 4, 7, 7]
		
		self.assertTrue(np.array_equal(sortfreq(list_, True), [10, 4, 7]),  
			"Sort descending failed")
		self.assertTrue(np.array_equal(sortfreq(list_, False), [7, 4, 10]),  
			"Sort ascending failed")

#---------------------------------------------------------------------------

class TestDfFilterMinlen(unittest.TestCase):

	def test_df_filter_minlen(self):
		def array_equal_nan(in_a, in_b):
			a = in_a.copy()
			b = in_b.copy()
			a[np.isnan(a)] = 12345
			b[np.isnan(b)] = 12345
			return np.array_equal(a, b)
		orig = pd.DataFrame(np.array([[10,10,10,10,10],[1,1,1,np.nan,np.nan],[np.nan,np.nan,np.nan,2,2]]).T)
		minlen4 = pd.DataFrame(np.array([[10,10,10,10,10]]).T)
		minlen3 = pd.DataFrame(np.array([[10,10,10,10,10],[1,1,1,np.nan,np.nan]]).T)
		minlen2 = pd.DataFrame(np.array([[10,10,10,10,10],[1,1,1,np.nan,np.nan],[np.nan,np.nan,np.nan,2,2]]).T)
		self.assertTrue(array_equal_nan(df_filter_minlen(orig, 2), minlen2),  "Minlen 2 failed")
		self.assertTrue(array_equal_nan(df_filter_minlen(orig, 3), minlen3),  "Minlen 3 failed")
		self.assertTrue(array_equal_nan(df_filter_minlen(orig, 4), minlen4),  "Minlen 4 failed")

#---------------------------------------------------------------------------

class TestDfDeriv(unittest.TestCase):

	def test_df_deriv(self):
		def array_equal_nan(in_a, in_b):
			a = in_a.copy()
			b = in_b.copy()
			a[np.isnan(a)] = 12345
			b[np.isnan(b)] = 12345
			return np.array_equal(a, b)
		orig = pd.DataFrame(np.array([[10,9,8,7,6,5,4,3,2,1],[1,2,3,4,5,6,7,8,9,10]]).T)
		w1s1 = pd.DataFrame(np.array([[np.nan,-1,-1,-1,-1,-1,-1,-1,-1,-1],[np.nan,1,1,1,1,1,1,1,1,1]]).T)
		w2s1 = pd.DataFrame(np.array([[np.nan,np.nan,-1,-1,-1,-1,-1,-1,-1,-1],[np.nan,np.nan,1,1,1,1,1,1,1,1]]).T)
		w2s2 = pd.DataFrame(np.array([[np.nan,np.nan,np.nan,-2,-2,-2,-2,-2,-2,-2],[np.nan,np.nan,np.nan,2,2,2,2,2,2,2]]).T)
		self.assertTrue(array_equal_nan(df_deriv(orig, 1,1), w1s1),  "w1s1 failed")
		self.assertTrue(array_equal_nan(df_deriv(orig, 2,1), w2s1),  "w2s1 failed")
		self.assertTrue(array_equal_nan(df_deriv(orig, 2,2), w2s2),  "w2s2 failed")

#---------------------------------------------------------------------------

class TestExpandMip(unittest.TestCase):

	def test_expand_mip(self):
		orig = np.random.rand(10, 256, 256)
		expand = expand_mip(orig, 5)
		self.assertTrue(np.array_equal(expand[0,1], orig[0]), "Should be equal")
		self.assertTrue(np.array_equal(expand[3,4], orig[3]), "Should be equal")

#---------------------------------------------------------------------------

class TestGradientND(unittest.TestCase):

	def test_gradient_nD(self):
		mesh = mesh_like(np.zeros((3,3)), 2)
		grad0 = np.array([[4,4,4], [8,8,8],[4,4,4]])
		grad1 = np.array([[4,8,4], [4,8,4],[4,8,4]])

		self.assertTrue(np.array_equal(gradient_nD(mesh[0]), grad0), "Should be equal")
		self.assertTrue(np.array_equal(gradient_nD(mesh[1]), grad1), "Should be equal")

#---------------------------------------------------------------------------

class TestDogFilter(unittest.TestCase):

	def test_dog_filter(self):
		orig = np.zeros((5,5))
		orig[2:4,2:4] = 100
		filtered_10_1 = np.array([[ 16,  14,  13,  13,  13],
							[ 14,   7,  -3,  -3,   5],
							[ 12,  -3, -25, -25,  -7],
							[ 12,  -3, -25, -25,  -7],
							[ 13,   5,  -7,  -7,   3]])

		self.assertTrue(np.array_equal(dog_filter(orig, 10, 1), filtered_10_1), "Should be equal")

#---------------------------------------------------------------------------

class TestLogFilter(unittest.TestCase):

	def test_log_filter(self):
		orig = np.zeros((5,5))
		orig[2:4,2:4] = 100
		filtered_3 = np.array([[ 0,  1,  0, -1,  1],
								[ 1,  3,  2,  1, -1],
								[ 1, -2, -3,  1, -1],
								[-1,  2,  1,  0, -2],
								[ 1, -1, -1, -2,  0]])

		self.assertTrue(np.array_equal(log_filter(orig, 3), filtered_3), "Should be equal")

#---------------------------------------------------------------------------

class TestZstackNormalizeMean(unittest.TestCase):

	def test_zstack_normalize_mean(self):
		orig = np.ones((5,5,5))
		orig[3] = 2
		output = zstack_normalize_mean(orig)

		self.assertEqual(np.unique(zstack_normalize_mean(orig)), 1.2, "Should be equal")

#---------------------------------------------------------------------------

class TestExtractBox(unittest.TestCase):

	def test_extract_box(self):
		stack = np.ones((5,11,11))
		ex = extract_box(stack, (2,5,5), (7,13,13), pad=True)
		self.assertEqual(ex[0,0,0], 0, "Should be equal")
		self.assertEqual(ex[6,5,5], 0, "Should be equal")
		self.assertEqual(ex[5,5,5], 1, "Should be equal")
		self.assertEqual(np.unique(ex[0,:,:]), [0], "Should be equal")
		self.assertEqual(np.unique(ex[1:6,1:12,1:12]), [1], "Should be equal")
		ex = extract_box(stack, (2,5,5), (7,13,13), pad=False)
		self.assertTrue(np.array_equal(stack.shape, ex.shape), "Should be equal")
		self.assertEqual(np.unique(ex), [1], "Should be equal")

#---------------------------------------------------------------------------

if __name__ == '__main__':
	unittest.main()