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
class TestAddBackground(unittest.TestCase):

    def test_add_background(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        sim.add_background(inverse=False, lam=1_000,sigma=100)
        fg_mean = np.mean(sim.im[sim.mask])
        bg_mean = np.mean(sim.im[~sim.mask])
        self.assertGreater(fg_mean, bg_mean, "Foreground should be greater than background.")
        sim.add_background(inverse=True, lam=10_000,sigma=100)
        fg_mean = np.mean(sim.im[sim.mask])
        bg_mean = np.mean(sim.im[~sim.mask])
        self.assertGreater(bg_mean, fg_mean, "Background should be greater than foreground.")

#---------------------------------------------------------------------------
class TestAddGaussianBlob(unittest.TestCase):

    def test_add_gaussian_blob(self):
        mask = Sim.make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5)
        sim = Sim(mask)
        sim.add_gaussian_blob((10,50,50),intensity=2_000,sigma=10)
        self.assertAlmostEqual(sim.im[10,50,50], 2001.0, 3,'Should be equal')
        self.assertAlmostEqual(sim.im[9,53,53], 1652.8521616372966, 'Should be equal')

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
class TestHLB(unittest.TestCase):

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