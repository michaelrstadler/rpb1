#!/usr/bin/env python

"""
A class for simulating nuclei.

The Sim class 

Important programming note: when running in parallel, using numpy.random
seems to not perform random initialization, and threads launched on
multiple cores will produce identical simulations. This is solved by
separately initializing a numpy random state using 
rs = np.random.RandomState() within each function where random processes
are needed. This must be used for parallel operations.

v1.0: simulations done via adding gaussians.
v2.0: simulations start with "real" image, convolve with experimentally-derived kernel, add noise

"""
__version__ = '2.0.0'
__author__ = 'Michael Stadler'

from flymovie.general_functions import mesh_like
from flymovie.load_save import save_pickle, load_pickle
import scipy
import random
import numpy as np
import scipy.ndimage as ndi
import multiprocessing as mp
from datetime import datetime

import os
import random
import string

#-----------------------------------------------------------------------
##### Start of Sim class. #####
#-----------------------------------------------------------------------

class Sim():
    """A class to simulate fluorescent signal in nuclei."""

    def __init__(self, mask, res_z=220, res_ij=85):
        self.mask = mask.astype('bool')
        self.im = np.zeros_like(mask)
        self.res_z = res_z
        self.res_ij = res_ij
        self.z_ij_ratio = self.res_z / self.res_ij
        self.kernel = None

    #-----------------------------------------------------------------------
    @staticmethod
    def make_spherical_mask(zdim=20, idim=100, jdim=100, nuc_rad=50):
        """Make a mask of a spherical nucleus.

        Args:
            zdim: int
                Size of mask in z dimension
            idim: int
                Size of mask in i dimension
            jdim: int
                Size of mask in j dimension
            nuc_rad: int
                Radius of nucleus

        Returns:
            mask: ndarray
                Mask
        """
        mask = np.zeros((zdim, idim, jdim))
        z, i, j = mesh_like(mask, 3)
        z_center = int(mask.shape[0] / 2)
        i_center = int(mask.shape[1] / 2)
        j_center = int(mask.shape[2] / 2)

        mask[(((z - z_center) ** 2) + ((i - i_center) ** 2) + 
            ((j - j_center) ** 2)) < (nuc_rad ** 2)] = 1

        return mask
    
    #-----------------------------------------------------------------------
    @staticmethod
    def extract_nuclear_masks(lmask, target_size=(20,100,100), dtype='float64'):
        """Extract individual nuclei from a labelmask, resize them to match
        a target size, create list of masks.
        
        Args:
            lmask: ndarray, labelmask of segmented nuclei
            target size: tuple of ints, size to which to match nuclei
            dtype: type, type of numbers in returned mask

        Returns:
            masks: list of ndarrays, nuclear masks
        """
        def touches_edge(coords, shape):
            """Determine whether nucleus touches image edge (only in last
            two dimensions)."""
            for dim in range(1, len(shape)):
                if np.min(coords[dim]) == 0:
                    return True
                if np.max(coords[dim]) == (shape[dim] - 1):
                    return True
            return False
                
        masks = []
        for n in range(1, np.max(lmask) + 1):
            # Current implementation does this np.where twice (once in 
            # mask_extract_resize). Suboptimal but this function shouldn't
            # be called much.
            coords = np.where(lmask == n)
            if touches_edge(coords, lmask.shape):
                continue
            # Extract image segment constituting a bounding box for nucleus 
            # and resize.
            cutout = Sim.extract_resize_maskobject(lmask, target_size, n=n)
            cutout = cutout.astype(dtype)
            masks.append(cutout)

        return masks
    
    #-----------------------------------------------------------------------
    @staticmethod
    def extract_resize_maskobject(stack, target_size, n=1, dtype='float64'):
        """Extract an object from a labelmask by cutting out 
        the bounding box containing the object and resizing it to match
        supplied target size.
        
        Args:
            stack: ndarray, 3D image stack
            target_size: tuple of 3 ints, size of final box
            n: int, label of object to extract
            dtype: type, type of numbers in returned mask
        
        Returns:
            resized: ndarray, object in its bounding box matched
                to target size
        """
        coords = np.where(stack == n)
        cutout = stack[np.min(coords[0]):(np.max(coords[0]+1)),
                np.min(coords[1]):(np.max(coords[1]+1)),
                np.min(coords[2]):(np.max(coords[2]+1))
                ]
        cutout_mask = np.where(cutout == n, 1, 0)
        # Resize to fit target size.
        zoom_factors = [
            target_size[0] / cutout_mask.shape[0],
            target_size[1] / cutout_mask.shape[1],
            target_size[2] / cutout_mask.shape[2]
        ]
        resized = ndi.zoom(cutout_mask, zoom_factors, order=0)
        return resized.astype(dtype)

    #-----------------------------------------------------------------------
    @staticmethod
    def rotate_binary_mask(mask, degree):
        """Rotates a binary mask in a way that doesn't chop things off at
        the edges.
        
        Image is padded, rotated, than the object is re-extracted by finding
        its bounding box and resizing to match original.

        Args:
            mask: ndarray, a 3d binary mask
            degree: int, degrees counter-clockwise to rotate

        Return:
            extracted: ndarray, rotated mask
        """
        padded = np.pad(mask, [(20,20),(50,50),(50,50)])
        rotated = ndi.rotate(padded, degree, axes=(1,2), order=0, reshape=False)
        extracted = Sim.extract_resize_maskobject(rotated, mask.shape, n=1)
        return extracted

    #-----------------------------------------------------------------------
    def add_noise(self, model='poisson',  **kwargs):
        """Add noise to image according to a model.

        Supported models are 'poisson' and 'gaussian'.

        Usage: Add poisson noise to true image (before convolution) to
            simulate shot noise from photon emission. Add gaussian noise
            to the final convolved image to simulate detector (microscope) 
            noise.
        
        Args:
            model: string
                Currently 'poisson' and 'guassian' are supported
            kwargs:
                gaussian:
                    sigma: standard deviation for gaussian noise      
        """
        rs = np.random.RandomState()
        if model == 'gaussian':
            if 'sigma' not in kwargs:
                raise ValueError('gaussian mode requires kwarg sigma.')
            sigma = kwargs['sigma']
            gaussian = rs.normal(scale = (sigma * np.ones_like(self.im)))
            self.im = self.im + gaussian
            self.im[self.im < 0] = 0
        
        elif model == 'poisson':
            self.im = rs.poisson(self.im)
        
        else:
            raise ValueError(
                "Only 'poisson' and 'gaussian' model currently supported.")

    #-----------------------------------------------------------------------
    def get_eroded_coordinates(self, erosion_size):
        """Get the coordinates (pixels >0) of a mask after applying binary
        erosion. Effectively makes a coordinate set excluding pixels at the 
        edge of the nucleus.

        Args:
            erosion_size: numeric, size of the structure for dilation in the
            ij-dimension, is automatically scaled to the equivalent in z.
        """
        # Get size of structuring element, minimum of 1 in each dimension.
        struct_z = np.max([1, int(erosion_size / self.z_ij_ratio)])
        struct_ij = np.max([1, int(erosion_size)])
        eroded_mask = scipy.ndimage.morphology.binary_erosion(self.mask, 
            structure=np.ones((struct_z, struct_ij, struct_ij)))
        eroded_mask_coords = np.where(eroded_mask)
        return eroded_mask_coords

    #-----------------------------------------------------------------------
    def add_object(self, coords, intensity, num_fluors, length):
        """Add a fluorescent object at specified coordinates. Objects can be
        of different sizes and consist of multiple fluors. All objects are 
        cubes of equal length on all sides.

        Intensity is distributed evenly such that the intensity of each
        pixel in the object is equal to:
            intensity * num_fluors / num_pixels

        Args:
            coords: iterable of ints, location lowest-index coordinate (
                bottom in z, top-left in xy)
            intensity: int, intensity of fluors making up object
            num_fluors: int, number of fluors in object
            length: int: side length of object (cube)
        """
        num_pixels = length ** 3
        intensity_per_pixel = intensity * num_fluors / num_pixels
        # Make tuple of slice objects to specify object location
        obj_coords = ()
        for d in range(len(coords)):
            obj_coords = obj_coords + tuple([slice(
                coords[d], 
                coords[d] + length)])
        self.im[obj_coords] += intensity_per_pixel

    #-----------------------------------------------------------------------
    def add_n_objects(self, n_objects, intensity, fluors_per_object_vals,
        size_vals, fluors_per_object_probs=None, size_probs=None, 
        erosion_size=None, mode='nuc'):
        """Add a defined number of objects at random positions.

        Intensity of fluors is fixed, but the number of fluors per object
        can be varied according to supplied probabilities. Objects can be
        added to the nucleus, outside the nucleus, or uniformly across the
        image.

        Args:
            n_objects: int, number of objects to add
            intensity: int, intensity of fluors making up object
            fluors_per_object_val: iterable of ints, list of possible values
                for number of fluors per object
            length: int, side length of objects. Objects are cubes equal on
                all sides.
            fluors_per_object_probs [optional]: iterable of floats, 
                probabilities of each value in fluors_per_object_val. If 
                omitted, probabilities are set to uniform.
            erosion_size [optional]: int, size of erosion kernel used to 
                restrict available nuclear coordinates. Has the effect of
                shaving off the outter layer of the nucleus for selecting
                object positions.
            mode: string, one of:
                'nuc': objects placed in nucleus (mask==1)
                'nonnuc': objects placed outside nucleus (mask==0)
                'all': objects placed in whole image
        """
        def draw_vals(vals, probs, n_objects, rs):
            # Create uniform probabilities if probs omitted.
            if probs is None:
                probs = np.ones(len(vals))/ len(vals)

            # Randomly draw fluors_per_object based on probabilities, store in array.
            vals_arr = rs.choice(
                    vals,
                    n_objects,
                    p=probs
                )
            return vals_arr

        rs = np.random.RandomState()
        # Get coordinates based on mode.
        if mode == 'nuc':
            if erosion_size is not None:
                coords = self.get_eroded_coordinates(erosion_size)
            else:
                coords = np.where(self.mask == 1)
        elif mode == 'nonnuc':
            coords = np.where(self.mask == 0)
        elif mode == 'all':
            coords = np.where(np.ones_like(self.mask))
        else:
            raise ValueError("Mode must be 'nuc', 'nonnuc', or 'all'.")
        
        num_pixels = len(coords[0])

        
        
        fluors_per_object_arr = draw_vals(fluors_per_object_vals, fluors_per_object_probs, n_objects, rs)
        size_arr = draw_vals(size_vals, size_probs, n_objects, rs)

        # For each iteration get random coordinates, add object.
        for i in range(n_objects):
            
            px = rs.randint(0, num_pixels - 1)
            fluors_per_object = fluors_per_object_arr[i]
            size = size_arr[i]
            print(size)
            random_coords = (coords[0][px], coords[1][px], 
                coords[2][px])
            self.add_object(random_coords, intensity, fluors_per_object, size)

    #-----------------------------------------------------------------------
    def add_kernel(self, kernel, res_z, res_ij):
        """Add kernel, normalize to probability (sum=1).

        Args:
            kernel: ndarray, convolution kernel, dimensions must be same as
                image
            res_z: float, resolution in Z of kernel (typical: nm)
            res_ij: float, resolution in ij of kernel (typical: nm)
        """
        if kernel.ndim != self.im.ndim:
            raise ValueError('Dimensions of kernel do not match image.')
        self.kernel = kernel / np.sum(kernel)
        self.kernel_res_z = res_z
        self.kernel_res_ij = res_ij
    
    #-----------------------------------------------------------------------
    def convolve(self):
        """"""
        if self.kernel is None:
            raise ValueError('Must add kernel to convolve.')
        
        # Resize (re-sample) kernel to match image resolution.
        kernel = self.kernel.copy()
        if (self.res_ij != self.kernel_res_ij) or (self.res_z != self.kernel_res_z):
            kernel = ndi.zoom(self.kernel, [
                self.kernel_res_z / self.res_z, 
                self.kernel_res_ij  / self.res_ij, 
                self.kernel_res_ij  / self.res_ij
                ])

        self.im = ndi.convolve(self.im, kernel)
    
    def resize(self, dims):
        # Resize image.
        zoom_factors = (
                self.res_z / dims[0],
                self.res_ij / dims[1],
                self.res_ij / dims[2]
            )

        im_rs = ndi.zoom(self.im, zoom_factors)
        self.im = im_rs

#-----------------------------------------------------------------------
##### End of Sim class. #####
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
def randomize_ab(ab):
    """Find random float number between a and b, given b > a.
    
    Args:
        ab: iterable
            Two numbers in an iterable, b > a.
    
    Returns:
        random float between a and b
    """
    a, b = ab
    if (b < a):
        raise ValueError('b must be greater than a')
    if (b == a):
        return a
    rs = np.random.RandomState()
    return (rs.random() * (b - a)) + a

#-----------------------------------------------------------------------
def sim_rpb1(mask, kernel, outfolder, nreps, nfree_rng, hlb_len_rng, 
    hlb_nummols_rng, n_clusters_rng, cluster_diam_mean_rng, 
    cluster_diam_var_rng, cluster_nmols_mean_rng, cluster_nmols_var_rng):
    """Simulate an rpb1 nucleus, write to file.

    Take ranges, do selection and simulation. Could abstract again but let's not.

        fluor intensity: fixed
        ranges:

            number free: uniform sample
            hlb length: uniform sample
            hlb num molecules: uniform sample
            cluster_number
            cluster_size
            cluster_size_variance
            cluster_nummolecules
            cluster_nummolecules variance

            noise

    Args:
        mask: ndarray, input mask for building nucleus
        
    """
    sim = Sim(mask, res_z=50, res_ij=50)
    kernel = np.ones((3,3,3))
    #kernel = kernel / np.sum(kernel)
    sim.add_kernel(kernel, res_z=50, res_ij=50)
    sim.add_object([10,10,10], 10, 1, 1)
    
    sim.im[sim.im < 0] = 0
    sim.im[sim.im > 65_536] = 65_536

    file_id = ''.join(random.choice(string.ascii_letters) for i in range(3))
    filepath = os.path.join(outfolder, file_id + '.pkl')
    save_pickle(sim.im, filepath)

#-----------------------------------------------------------------------
def sim_batch(outfolder, kernel, nsims, nreps, nprocesses, 
    sim_func=sim_rpb1, z_dim=200, ij_dim=200, nuc_rad=90, **kwargs):
    """Perform parallelized simulations of rpb1 nuclei by randomly drawing
    parameters from supplied ranges.

    Args:
        maskfile: path, pickled file containing list of masks as ndarrays
        outfolder: path, folder to which to write simulations
        nsims: int, the number of parameter sets to simulate
        nreps: int, the number of simulations to perform for each param set
        nprocesses: int, the number of processes to launch with 
            multiprocessing Pool
    
    Outputs:
        A unique 8-character identifier is associated with the entire 
        batch. This ID is appended to the folder name. A log file is 
        saved in the folder containing all supplied paramaters. In 
        addition, each parameter batch gets a 3-letter random ID to 
        avoid collisions (unlikely). Parameters for each simulation 
        are saved in the filename, separated by underscores.

    Note: "Replicates" use identical parameters but different masks. For
    each replicate, a mask is drawn at random from the supplied list and 
    also rotated randomly (0-360 uniform).
    """
    mp.set_start_method('fork', force=True) # Important for macOS.

    # Set folder name with unique identifier and create it.
    folder_id = ''.join(random.choice(string.ascii_letters) for i in range(8))
    folder = outfolder + folder_id
    os.mkdir(folder)
    
    mask = Sim.make_spherical_mask(z_dim, ij_dim, ij_dim, nuc_rad)
    # For each sim, built a local set of args that will be sent to sim_rpb1,
    # add each set of parameters to arglist.
    arglist = []

    for _ in range(nsims):
        kwargs_loc = kwargs.copy()
        kwargs_loc['mask'] = mask
        kwargs_loc['kernel'] = kernel
        kwargs_loc['outfolder'] = folder
        kwargs_loc['nreps'] = nreps
        arglist.append(kwargs_loc)
    print('arglist done')

    # Launch simulations in parallel using pool method.
    for i in range(0, len(arglist), 1000):
        end = i + 1000
        arglist_sub = arglist[i:end]   
        pool = mp.Pool(processes=nprocesses)
        results = [pool.apply_async(sim_func, (), x) for x in arglist_sub]
        [p.get() for p in results]
    
    # Write logfile.
    logfilepath = os.path.join(folder, 'logfile_' + folder_id + '.txt')
    with open(logfilepath, 'w') as logfile:
        logfile.write(datetime.now().ctime() + '\n')
        logfile.write('nsims: ' + str(nsims) + '\n')
        logfile.write('nreps: ' + str(nreps) + '\n')
        for key in kwargs:
            logfile.write(key + ': ' + str(kwargs[key]))
            logfile.write('\n')

