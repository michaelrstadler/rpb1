import unittest
import numpy as np
from flymovie.simnuc import Sim

#---------------------------------------------------------------------------
class TestInit(unittest.TestCase):

    def test__init__(self):
        # Make sure it runs.
        mask = np.zeros((20,50,50))
        Sim(mask)

#---------------------------------------------------------------------------
class TestMakeDummyMask(unittest.TestCase):

    def test_make_dummy_mask(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        self.assertEqual(np.count_nonzero(mask), 114388, "Wrong number of mask True pixels.")

#---------------------------------------------------------------------------
class TestExtractNuclearMasks(unittest.TestCase):

    def test_extract_nuclear_masks(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=20, 
            nuc_rad=10, z_ij_ratio=3)
        mask = mask[:,1:,1:] # The circle math in make_dummy_mask leaves row/column 0 empty.
        mask = mask.astype(int)
        target_size=[10,50,50]
        masks = Sim.extract_nuclear_masks(mask, target_size=target_size)
        self.assertEqual(len(masks), 9, "Should be 9 masks.")
        for m in masks:
            self.assertTrue(np.array_equal(m.shape, target_size), 'Masks should match target size.')

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
class TestAddBackground(unittest.TestCase):

    def test_add_background(self):
        # poisson + gaussian.
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        sim.add_background(model='poisson+gaussian', inverse=False, lam=1_000,sigma=100)
        fg_mean = np.mean(sim.im[sim.mask])
        bg_mean = np.mean(sim.im[~sim.mask])
        self.assertGreater(fg_mean, bg_mean, "Foreground should be greater than background.")
        sim.add_background(model='poisson+gaussian', inverse=True, lam=10_000,sigma=100)
        fg_mean = np.mean(sim.im[sim.mask])
        bg_mean = np.mean(sim.im[~sim.mask])
        self.assertGreater(bg_mean, fg_mean, "Background should be greater than foreground.")

        # uniform.
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        sim.add_background(inverse=False, val = 1_000)
        nuc_val = np.unique(sim.im[sim.mask])[0]
        self.assertEqual(nuc_val, 1_000., 'Nucleus should just be 1,000.')
        sim.add_background(inverse=True, val = 500)
        nuc_val = np.unique(sim.im[sim.mask])[0]
        bg_val = np.unique(sim.im[~sim.mask])[0]
        self.assertEqual(nuc_val, 1_000., 'Nucleus should just be 1,000.')
        self.assertEqual(bg_val, 500., 'Background should just be 500.')

#---------------------------------------------------------------------------
class TestAddNoise(unittest.TestCase):

    def test_add_noise(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        sim.add_background(inverse=False, val=10_000)
        std_before = np.std(sim.im[sim.mask])
        self.assertEqual(std_before, 0, 'Std before should be 0.')
        sim.add_noise(model="poisson+gaussian", sigma=100)
        std_after = np.std(sim.im[sim.mask])
        self.assertGreater(std_after, std_before, 
            'Std should increase with added noise.')


#---------------------------------------------------------------------------
class TestMake3dGaussianInABox(unittest.TestCase):

    def test_make_3d_gaussian_inabox(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        box = sim.make_3d_gaussian_inabox(intensity=100, sigma=10, 
            z_windowlen=20, ij_windowlen=100)
        self.assertGreater(box[10,50,50], box[0,0,0], 
            'Should be brighter at center')
        self.assertEqual(box.shape[0], 20, 'Box z dimension is wrong.')
        self.assertEqual(box.shape[1], 100, 'Box ij dimension is wrong.')

#---------------------------------------------------------------------------
class TestAddGaussianBlob(unittest.TestCase):

    def test_add_gaussian_blob(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        sim.add_gaussian_blob((10,50,50),intensity=2_000,sigma=10)
        self.assertAlmostEqual(sim.im[10,50,50], 2001.0, 3,'Should be equal')
        self.assertAlmostEqual(sim.im[9,53,53], 1545.032005, 3, 'Should be equal')

#---------------------------------------------------------------------------
class TestGetErodedCoordinates(unittest.TestCase):

    def test_get_eroded_coordinates(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        eroded_coords = sim.get_eroded_coordinates(5)
        self.assertGreater(np.count_nonzero(sim.mask), len(eroded_coords[0]), 
            '1s in mask should have shrunk.')

#---------------------------------------------------------------------------
class TestAddNBlobs(unittest.TestCase):

    def test_add_nblobs(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        mean1 = np.mean(sim.im[sim.mask])
        var1 = np.std(sim.im[sim.mask])
        sim.add_nblobs(100, 10_000, 1, 1)
        mean2 = np.mean(sim.im[sim.mask])
        var2 = np.std(sim.im[sim.mask])

        self.assertGreater(mean2, mean1, 'Mean should have increased')
        self.assertGreater(var2, var1, 'Variance should have increased')

#---------------------------------------------------------------------------
class TestAddHLB(unittest.TestCase):

    def test_add_hlb(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        mean1 = np.mean(sim.im[sim.mask])
        var1 = np.std(sim.im[sim.mask])
        sim.add_hlb(10_000, 5)
        mean2 = np.mean(sim.im[sim.mask])
        var2 = np.std(sim.im[sim.mask])

        self.assertGreater(mean2, mean1, 'Mean should have increased')
        self.assertGreater(var2, var1, 'Variance should have increased')

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