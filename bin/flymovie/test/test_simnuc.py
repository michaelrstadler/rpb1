import unittest
import numpy as np
import scipy.ndimage as ndi
import tempfile
from copy import deepcopy
from flymovie import save_pickle
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
        # Repeat 10 times because one failure mode I encountered was the 
        # gaussian mode ignoring sigma, which would give correct order 50%
        # of the time.
        for _ in range(10):
            sim = Sim(mask)
            # Fill nucleus with uniform value.
            sim.im[sim.mask] = 10
            std_init = np.std(sim.im[sim.mask])
            self.assertEqual(std_init, 0, 'Std before should be 0.')
            sim.add_noise('poisson')
            std_poisson = np.std(sim.im)

            sim = Sim(mask)
            sim.im[sim.mask] = 10
            sim.add_noise('gaussian', sigma=20)
            std_gauss20 = np.std(sim.im)

            sim = Sim(mask)
            sim.im[sim.mask] = 10
            sim.add_noise('gaussian', sigma=100)
            std_gauss100 = np.std(sim.im)

            self.assertTrue(std_gauss100 > std_gauss20 > std_poisson, 'Should be in this order')
            

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
class TestConcToNMolecules(unittest.TestCase):

    def test_conc_to_nmolecules(self):
        mask = np.zeros((10,10,10))
        sim = Sim(mask, res_z=100, res_ij=100)
        x = sim.conc_to_nmolecules(60)
        self.assertEqual(x, 0, 'Should be 0.')

        mask[5,5,5] = 1
        sim = Sim(mask, res_z=100, res_ij=100)
        x = sim.conc_to_nmolecules(100)
        self.assertAlmostEqual(x, 0.0602, 4,'Wrong for 1 pixel')

        mask[5:8,5:8,5:8] = 1
        sim = Sim(mask, res_z=100, res_ij=100)
        x = sim.conc_to_nmolecules(100)
        self.assertAlmostEqual(x, 1.6259, 2,'Wrong for 1 pixel')

#---------------------------------------------------------------------------
class TestAddSphere(unittest.TestCase):

    def test_add_sphere(self):
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask)

        self.assertEqual(sim.im.sum(), 0, 'Should be a blank image.')
        sim.add_sphere((10,10,10), 10, 1, 5)
        self.assertAlmostEqual(sim.im.sum(), 10, 3, 'Should just be 10.')
        self.assertGreater(sim.im[10,10,14], 0, 'Should be nonzero.')
        self.assertEqual(sim.im[14,14,14], 0, 'Should be 0.')

        # Check that the corner position [14,14,14] is in fact non-zero
        # for a cube-shaped object.
        sim = Sim(mask)
        self.assertEqual(sim.im.sum(), 0, 'Should be a blank image.')
        sim.add_object((10,10,10), 10, 1, 10)
        self.assertGreater(sim.im[14,14,14], 0, 'Should be nonzero.')

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
        sim.add_n_objects(100, 10, [1], [1], mode='nonnuc')
        self.assertEqual(np.sum(sim.im[sim.mask]), 1000, 'should have increased')
        self.assertEqual(np.sum(sim.im[~sim.mask]), 1000, 'should have increased')

        # Test all mode.
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask, res_z=200, res_ij=200)
        self.assertEqual(np.mean(sim.im[sim.mask]), 0, 'Should be a blank image.')
        self.assertEqual(np.mean(sim.im[~sim.mask]), 0, 'Should be a blank image.')
        sim.add_n_objects(100, 10, [1], [1], mode='all')
        self.assertGreater(np.sum(sim.im[sim.mask]), 0, 'should have increased')
        self.assertGreater(np.sum(sim.im[~sim.mask]), 0, 'should have increased')
    

        # Test erosion.
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask, res_z=200, res_ij=200)
        sim.add_n_objects(100, 10, [1,1], 1, mode='nuc', erosion_size=2, 
            fluors_per_object_probs=[0.5,0.5])
        mask = ndi.morphology.binary_erosion(sim.mask, np.ones([2,2,2]))
        mask = mask.astype('bool')
        self.assertEqual(np.sum(sim.im[~mask]), 0, 'Should be 0 outside eroded zone.')
        self.assertEqual(np.sum(sim.im[mask]), 1000, 'Should be 0 outside eroded zone.')

        # Test probability modes.
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask, res_z=200, res_ij=200)
        sim.add_n_objects(100, 10, fluors_per_object=[1,2], 
            size=[1,2], mode='nuc', erosion_size=2, 
            fluors_per_object_probs=[1,0], size_probs=[1,0])
        mask = ndi.morphology.binary_erosion(sim.mask, np.ones([2,2,2]))
        mask = mask.astype('bool')
        self.assertEqual(np.sum(sim.im[~mask]), 0, 'Should be 0 outside eroded zone.')
        self.assertEqual(np.sum(sim.im[mask]), 1000, 'Should be 0 outside eroded zone.')
    
#---------------------------------------------------------------------------
class TestAddKernel(unittest.TestCase):

    def test_add_kernel(self):
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask, res_z=200, res_ij=200)
        kernel = np.ones((3,3,3))
        sim.add_kernel(kernel, res_z=50, res_ij=50)
        self.assertEqual(sim.kernel_res_ij, 50, 'Should be 50.')
        self.assertEqual(sim.kernel_res_z, 50, 'Should be 50.')
        self.assertAlmostEqual(np.max(sim.kernel), 1, 5, 'Should be 1.')

#---------------------------------------------------------------------------
class TestConvolve(unittest.TestCase):

    def test_convolve(self):
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask, res_z=50, res_ij=50)
        kernel = np.ones((3,3,3))
        sim.add_kernel(kernel, res_z=50, res_ij=50)
        sim.add_object([10,10,10], 10, 1, 1)
        sum_before = np.sum(sim.im)
        sim.convolve()
        sum_after = np.sum(sim.im)
        self.assertAlmostEqual(sum_after, np.sum(kernel) * sum_before, 3, 'Should be equal.')
        self.assertAlmostEqual(np.count_nonzero(sim.im.flatten()), 27, 5, 'Should be 27.')

#---------------------------------------------------------------------------
class TestResize(unittest.TestCase):

    def test_resize(self):
        mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
            nuc_rad=8)
        sim = Sim(mask, res_z=50, res_ij=50)
        sim.add_object((10,10,10), 1000, 1, 2)
        self.assertEqual(np.max(sim.im), 125, 'Should be 125.')
        self.assertEqual(np.mean(sim.im), 0.125, 'Should be 0.125.')
        self.assertTrue(np.array_equal(sim.im.shape, (20,20,20)), 'Wrong size.')
        self.assertEqual(sim.res_z, 50, 'Should be 50')
        
        sim.resize([100,100,100], 0)
        self.assertEqual(np.max(sim.im), 125, 'Should be 125.')
        self.assertEqual(np.mean(sim.im), 0.125, 'Should be 0.125.')
        self.assertTrue(np.array_equal(sim.im.shape, (10,10,10)), 'Wrong size.')
        self.assertEqual(sim.res_z, 100, 'Should be 100')

        sim.resize([25,25,25], 0)
        self.assertEqual(np.max(sim.im), 125, 'Should be 125.')
        self.assertEqual(np.mean(sim.im), 0.125, 'Should be 0.125.')
        self.assertTrue(np.array_equal(sim.im.shape, (40,40,40)), 'Wrong size.')
        self.assertEqual(sim.res_z, 25, 'Should be 25')
   
#---------------------------------------------------------------------------
class TestRandomizeAB(unittest.TestCase):

    def test_randomize_ab(self):
        r = randomize_ab([0,1])
        self.assertTrue((r >= 0) and (r <= 1), 'Should be between 0 and 1')
        r = randomize_ab([0,10_000])
        self.assertTrue((r >= 0) and (r <= 10_000), 'Should be between 0 and 10_000')
        r = randomize_ab([500,600])
        self.assertTrue((r >= 500) and (r <= 600), 'Should be between 500 and 600')
        with self.assertRaises(ValueError):
            randomize_ab([10,0])
        self.assertEqual(randomize_ab([5,5]), 5, 'Should be 5.')

#---------------------------------------------------------------------------

class TestRunPooledProcesses(unittest.TestCase):
    # Just test to see if it runs.
    def test_run_pooled_processes(self):
        kwarglist = [{
            'a': [1,2,3,4]
        }]
        run_pooled_processes(kwarglist, 2, np.sum)

#---------------------------------------------------------------------------

class TestWriteLogfile(unittest.TestCase):
    # Just test to see if it runs.
    def test_write_logfile(self):
        with tempfile.NamedTemporaryFile() as tfile:
            logitems = {'a': 4}
            write_logfile(tfile.name, logitems)

#---------------------------------------------------------------------------

class TestSimRpb1(unittest.TestCase):
    # Just test to see if it runs.
    def test_sim_rpb1(self):
        with tempfile.TemporaryDirectory() as tdir:
            mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
                nuc_rad=8)
            masks = [mask, mask]            
            sim_rpb1(masks=masks, kernel=np.ones((2,2,2)), 
                outfolder=tdir, nreps=2, ntotal_rng=[10_000,10_000], hlb_diam_rng=[1,1], 
                hlb_nmols_rng=[1,1], n_clusters_rng=[1,1], 
                cluster_diam_mean_rng=[1,1], cluster_diam_var_rng=[1,1], 
                cluster_nmols_mean_rng=[1,1], cluster_nmols_var_rng=[1,1], 
                noise_sigma_rng=[1,1], hlb_coords=[(10,10,10),(11,11,11),(5,5,5),(17,17,17)])

#---------------------------------------------------------------------------

class TestSimRpb1Batch(unittest.TestCase):
    # Just test to see if it runs.
    def test_sim_rpb1_batch(self):
        with tempfile.TemporaryDirectory() as tdir:
            maskfile = os.path.join(tdir, 'masks.pkl')
            kernelfile = os.path.join(tdir, 'kernel.pkl')
            save_pickle(np.ones((2,2,2)), kernelfile)
            mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
                nuc_rad=8)
            masks = [mask, mask]
            save_pickle(masks, maskfile)
            
            sim_rpb1_batch(outfolder=tdir, kernelfile=kernelfile, 
                maskfile=maskfile, nsims=2, nreps=2, nprocesses=2,
                ntotal_rng=[20_000,20_000], hlb_diam_rng=[7,15], 
                hlb_nmols_rng=[100,1000], n_clusters_rng=[0,30], 
                cluster_diam_mean_rng=[1,2], cluster_diam_var_rng=[0.5, 2], 
                cluster_nmols_mean_rng=[10,25], cluster_nmols_var_rng=[1,3], 
                noise_sigma_rng=[1, 2])  

#---------------------------------------------------------------------------

class TestSimHistones(unittest.TestCase):
    # Just test to see if it runs.
    def test_sim_histones(self):
        with tempfile.TemporaryDirectory() as tdir:
            mask = Sim.make_spherical_mask(zdim=20, idim=20, jdim=20, 
                nuc_rad=8)

            # Test writing file.
            sim_histones([mask, mask], kernel=np.ones((2,2,2)), outfolder=tdir,
                nfree=100, fraction_labeled=0.5, genome_size=5_000_000, bp_per_nucleosome=250, a1=1, 
                p1=0, noise_sigma=0.1, nreps=2,  
            )

            # Test genome_size.
            sim1 = sim_histones([mask, mask], kernel=np.ones((2,2,2)), outfolder=tdir,
                nfree=100, fraction_labeled=0.5, genome_size=5_000_000, bp_per_nucleosome=250, a1=1, 
                p1=0, noise_sigma=0.1, nreps=2, return_sim=True)
            
            sim2 = sim_histones([mask, mask], kernel=np.ones((2,2,2)), outfolder=tdir,
                nfree=100, fraction_labeled=0.5, genome_size=10_000_000, bp_per_nucleosome=250, a1=1, 
                p1=0, noise_sigma=0.1, nreps=2, return_sim=True)

            self.assertGreater(np.mean(sim2.im), np.mean(sim1.im), 'Mean should go up.')
            
            # Test fraction labeled.
            sim1 = sim_histones([mask, mask], kernel=np.ones((2,2,2)), outfolder=tdir,
                nfree=100, fraction_labeled=0.1, genome_size=5_000_000, bp_per_nucleosome=250, a1=1, 
                p1=0, noise_sigma=0.1, nreps=2, return_sim=True)

            sim2 = sim_histones([mask, mask], kernel=np.ones((2,2,2)), outfolder=tdir,
                nfree=100, fraction_labeled=0.9, genome_size=5_000_000, bp_per_nucleosome=250, a1=1, 
                p1=0, noise_sigma=0.1, nreps=2, return_sim=True)

            self.assertGreater(np.mean(sim2.im), np.mean(sim1.im), 'Mean should go up.')

            # Test size_distribution.
            sim1 = sim_histones([mask, mask], kernel=np.ones((2,2,2)), outfolder=tdir,
                nfree=0, fraction_labeled=0.5, genome_size=5_000_000, bp_per_nucleosome=3_000_000, a1=10, 
                p1=0, noise_sigma=0.1, nreps=2, return_sim=True)

            sim2 = sim_histones([mask, mask], kernel=np.ones((2,2,2)), outfolder=tdir,
                nfree=0, fraction_labeled=0.5, genome_size=5_000_000, bp_per_nucleosome=3_000_000, a1=-10, 
                p1=0, noise_sigma=0.1, nreps=2, return_sim=True)

            self.assertGreater(np.max(sim2.im), np.max(sim1.im), 'Max should go up.')

            # Test p1.
            # TO DO.

            self.assertGreater(np.mean(sim2.im), np.mean(sim1.im), 'Mean should go up.')

#---------------------------------------------------------------------------

class TestSimHistonesBatch(unittest.TestCase):
    # Just test to see if it runs.
    def test_sim_histones_batch(self):
        with tempfile.TemporaryDirectory() as tdir:
            maskfile = os.path.join(tdir, 'masks.pkl')
            kernelfile = os.path.join(tdir, 'kernel.pkl')
            save_pickle(np.ones((2,2,2)), kernelfile)
            mask = Sim.make_spherical_mask(zdim=20, idim=200, jdim=200, 
                nuc_rad=42)
            masks = [mask, mask]
            save_pickle(masks, maskfile)
            
            sim_histones_batch(outfolder=tdir, kernelfile=kernelfile, 
                maskfile=maskfile, nsims=2, nreps=2, nprocesses=2,
                nfree_rng=[100,200], genome_size=1.8e7, bp_per_nucleosome_rng=[1800,2000], 
                fraction_labeled_rng=[0.1,0.2], density_min_rng=[2,4], density_max_rng=[8,10], 
                rad_max_rng=[2.5,4], a1_rng=[0,2], p1_rng=[0,2], noise_sigma_rng=[1,3]
                )  
"""
""" 

if __name__ == '__main__':
	unittest.main()