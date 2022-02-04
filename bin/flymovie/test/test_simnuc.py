import unittest
import numpy as np
import scipy.ndimage as ndi
from flymovie.simnuc import *

#---------------------------------------------------------------------------
class TestInit(unittest.TestCase):

    def test__init__(self):
        # Make sure it runs.
        mask = np.zeros((20,50,50))
        Sim(mask)

#---------------------------------------------------------------------------
class TestMakeSphericalMask(unittest.TestCase):

    def test_make_spherical_mask(self):
        mask = Sim.make_spherical_mask(zdim=100, idim=100, jdim=100, 
        nuc_rad=40)
        self.assertEqual(np.count_nonzero(mask), 267731, 
            "Wrong number of mask True pixels.")

#---------------------------------------------------------------------------
class TestExtractNuclearMasks(unittest.TestCase):

    def test_extract_nuclear_masks(self):
        mask1 = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, nuc_rad=8)
        mask2 = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, nuc_rad=10)
        mask = np.hstack([mask1, mask2, mask1, mask1, mask1[:,:10,:]])
        lmask, _ = ndi.label(mask)
        lmask = lmask.astype(int)
        target_size=[10,50,50]
        masks = Sim.extract_nuclear_masks(lmask, target_size=target_size)
        self.assertEqual(len(masks), 3, "Should be 3 masks.")
        for m in masks:
            self.assertTrue(np.array_equal(m.shape, target_size), 
                'Masks should match target size.')

#---------------------------------------------------------------------------
class TestExtractResizeMaskobject(unittest.TestCase):

    def test_extract_resize_maskobject(self):
        mask = np.zeros((10,100,100))
        mask[4:6, 10:20, 10:20] = 7
        extraction1 = Sim.extract_resize_maskobject(mask, (2,10,10), 7)
        self.assertEqual(np.max(extraction1), 1, 'Should be a binary mask.')
        self.assertEqual(np.sum(extraction1), 200, 'Should be 200 pixels')
        extraction2 = Sim.extract_resize_maskobject(mask, (2,20,20), 7)
        self.assertEqual(np.sum(extraction2), 800, 'Should be 800 pixels')

#---------------------------------------------------------------------------
class TestRotateBinaryMask(unittest.TestCase):

    def test_rotate_binary_mask(self):
        mask = np.zeros((3,100,100))
        mask[1:, 30:70, 40:60] = 1
        rot45 = Sim.rotate_binary_mask(mask, 45)
        self.assertEqual(rot45[1,60,20], 0, 'Should be 0.')
        self.assertEqual(rot45[1,35,0], 1, 'Should be 1.')
        self.assertEqual(rot45[1,20,60], 0, 'Should be 0.')
        self.assertEqual(rot45[1,82,80], 1, 'Should be 1.')

#---------------------------------------------------------------------------
class TestAddNoise(unittest.TestCase):

    def test_add_noise(self):
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, nuc_rad=5)
        sim = Sim(mask)
        # Fill nucleus with uniform value.
        sim.im[sim.mask] = 10
        std_before = np.std(sim.im[sim.mask])
        self.assertEqual(std_before, 0, 'Std before should be 0.')
        sim.add_noise(model="poisson")
        std_after = np.std(sim.im[sim.mask])
        self.assertGreater(std_after, std_before, 
            'Std should increase with added noise.')
        std_before = std_after
        sim.add_noise(model="gaussian", sigma=100)
        std_after = np.std(sim.im[sim.mask])
        self.assertGreater(std_after, std_before, 
            'Std should increase with added noise.')

#---------------------------------------------------------------------------
class TestGetErodedCoordinates(unittest.TestCase):

    def test_get_eroded_coordinates(self):
        mask = Sim.make_spherical_mask(zdim=20, idim=100, jdim=100, 
        nuc_rad=50)
        sim = Sim(mask)
        eroded_coords = sim.get_eroded_coordinates(5)
        self.assertGreater(np.count_nonzero(sim.mask), len(eroded_coords[0]), 
            '1s in mask should have shrunk.')

#---------------------------------------------------------------------------
class TestAddObject(unittest.TestCase):

    def test_add_object(self):
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask)
        self.assertEqual(sim.im.sum(), 0, 'Should be a blank image.')
        sim.add_object((10,10,10), 10, 1, 1)
        self.assertEqual(sim.im.sum(), 10, 'Should just be 10.')
        sim.add_object((14,14,14), 10, 4, 2)
        self.assertEqual(sim.im.sum(), 50, 'Should be 50.')
        self.assertEqual(np.count_nonzero(sim.im.flatten()), 9, 'Should be 9.')
        sim.add_object((0,0,0), 10, 100, 3)
        self.assertEqual(sim.im.sum(), 1050, 'Should be 1050.')
        self.assertEqual(np.count_nonzero(sim.im.flatten()), 36, 'Should be 36.')

#---------------------------------------------------------------------------
class TestAddNObjects(unittest.TestCase):

    def test_add_n_objects(self):
        # Test nuc and nonnuc modes.
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask, res_z=200, res_ij=200)
        self.assertEqual(np.sum(sim.im[sim.mask]), 0, 'Should be a blank image.')
        self.assertEqual(np.sum(sim.im[~sim.mask]), 0, 'Should be a blank image.')
        sim.add_n_objects(100, 10, 1, 1, mode='nuc')
        self.assertEqual(np.sum(sim.im[sim.mask]), 1000, 'should be 100')
        self.assertEqual(np.mean(sim.im[~sim.mask]), 0, 'Should be a blank image.')
        sim.add_n_objects(100, 10, 1, 1, mode='nonnuc')
        self.assertEqual(np.sum(sim.im[sim.mask]), 1000, 'should have increased')
        self.assertEqual(np.sum(sim.im[~sim.mask]), 1000, 'should have increased')

        # Test all mode.
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask, res_z=200, res_ij=200)
        self.assertEqual(np.mean(sim.im[sim.mask]), 0, 'Should be a blank image.')
        self.assertEqual(np.mean(sim.im[~sim.mask]), 0, 'Should be a blank image.')
        sim.add_n_objects(100, 10, 1, 1, mode='all')
        self.assertGreater(np.sum(sim.im[sim.mask]), 0, 'should have increased')
        self.assertGreater(np.sum(sim.im[~sim.mask]), 0, 'should have increased')
    

        # Test erosion.
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask, res_z=200, res_ij=200)
        sim.add_n_objects(100, 10, 1, 1, mode='nuc', erosion_size=2)
        mask = ndi.morphology.binary_erosion(sim.mask, np.ones([2,2,2]))
        mask = mask.astype('bool')
        self.assertEqual(np.sum(sim.im[~mask]), 0, 'Should be 0 outside eroded zone.')
        self.assertEqual(np.sum(sim.im[mask]), 1000, 'Should be 0 outside eroded zone.')
"""    
#---------------------------------------------------------------------------
class TestAddNBlobsZSchedule(unittest.TestCase):

    def test_add_nblobs_zschedule(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        probs = np.concatenate([np.repeat(0.075, 10), np.repeat(0.025, 10)])
        sim.add_nblobs_zschedule(100, 1_000, 1, 1, probs)
        mean_bottom = np.mean(sim.im[:10])
        mean_top = np.mean(sim.im[10:])
        self.assertGreater(mean_bottom, mean_top, 'Bottom should be stronger than top.')
        probs = np.arange(20) / np.sum(np.arange(20))
        sim.add_nblobs_zschedule(200, 10_000, 1, 1, probs)
        mean_bottom = np.mean(sim.im[:10])
        mean_top = np.mean(sim.im[10:])
        self.assertGreater(mean_top, mean_bottom, 'Top should be stronger than bottom.')


#---------------------------------------------------------------------------
class TestAddNBlobsZScheduleExponential(unittest.TestCase):

    def test_add_nblobs_zschedule_exponential(self):
        # Test flat.
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        probs = sim.add_nblobs_zschedule_exponential(100, 1_000, 1, 1, 0, 
            return_probs=True)
        self.assertTrue(np.array_equal(probs, np.repeat(0.05, 20)), 'Should be all 0.05')

        # Test slant to upper.
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        probs = sim.add_nblobs_zschedule_exponential(100, 5_000, 1, 1, 5, 
            return_probs=True)
        mean_bottom = np.mean(sim.im[:10])
        mean_top = np.mean(sim.im[10:])
        self.assertGreater(mean_top, mean_bottom, 'Top should be stronger than bottom.')
        
        # Test slant to bottom.
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        probs = sim.add_nblobs_zschedule_exponential(100, 5_000, 1, 1, -5, 
            return_probs=True)
        mean_bottom = np.mean(sim.im[:10])
        mean_top = np.mean(sim.im[10:])
        self.assertGreater(mean_bottom, mean_top, 'Bottom should be stronger than top.')

#---------------------------------------------------------------------------
    class TestRandomizeAB(unittest.TestCase):

        def test_randomize_ab(self):
            r = randomize_ab(0,1)
            self.assertTrue((r >= 0) and (r <= 1), 'Should be between 0 and 1')
            r = randomize_ab(0,10_000)
            self.assertTrue((r >= 0) and (r <= 10_000), 'Should be between 0 and 10_000')
            r = randomize_ab(500,600)
            self.assertTrue((r >= 500) and (r <= 600), 'Should be between 500 and 600')
            self.assertRaises(randomize_ab(10,0))
            self.assertEqual(randomize_ab(5,5), 5, 'Should be 5.')

"""

if __name__ == '__main__':
	unittest.main()