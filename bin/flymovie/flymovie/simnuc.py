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

"""
__version__ = '1.0.0'
__author__ = 'Michael Stadler'

from flymovie.general_functions import mesh_like
from flymovie.load_save import save_pickle, load_pickle
import scipy
import random
import numpy as np
import scipy.ndimage as ndi
import multiprocessing as mp

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
    def add_n_objects(self, n_objects, intensity, fluors_per_object, length, 
        erosion_size=None, mode='nuc'):
        """Add a defined number of objects objects at random positions.
        
        """
        rs = np.random.RandomState()
        # Use erosion to generate candidate positions that avoid the edge
        # of the nucleus.
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
        for _ in range(n_objects):
            # Get random coordinates 
            px = rs.randint(0, num_pixels - 1)
            random_coords = (coords[0][px], coords[1][px], 
                coords[2][px])
            self.add_object(random_coords, intensity, fluors_per_object, length)

    #-----------------------------------------------------------------------
    def add_kernel(self, kernel, res_z, res_ij):
        if kernel.ndim != self.im.ndim:
            raise ValueError('Dimensions of kernel do not match image.')
        #kernel = ndi.zoom(kernel, [res_z / self.res_z, res_ij / self.res_ij, res_ij / self.res_ij])
        self.kernel = kernel
        self.kernel_res_z = res_z
        self.kernel_res_ij = res_ij
    
    def convolve(self):
        if self.kernel is None:
            raise ValueError('Must add kernel to convolve.')
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
        # Rescale to match input.
        im_rs = im_rs * (self.im.sum() / im_rs.sum())
        self.im = im_rs

    #-----------------------------------------------------------------------
    def add_nblobs_zschedule(self, numblobs, intensity_mean, intensity_std, 
            sigma_base, z_probs, sigma_k=0.5, sigma_theta=0.5):
        """Add gaussian blobs according with asymmetrical Z location 
        probabilities determined by a supplied distribution.

        Args:
            numblobs: int, number of blobs to add
            intensity_mean: numeric, mean of the distribution from which 
                blob intensities are drawn.
            intensity_std: numeric, std. deviation of the distribution from
                which blob intensities are drawn.
            sigma_base: numeric, minimum value of gaussian width (sigma)
            z_probs: iterable, probabilities of blobs occurring in each z slice
            sigma_k: numeric, shape parameter of gamma distribution added to 
                sigma_base to determine gaussian width
            sigma_theta: numeric, scale parameter of gamma distribution added 
                to sigma_base to determine gaussian width
        """
        if len(z_probs) != self.im.shape[0]:
            raise ValueError('Length of z_probs must equal number of z slices.')
        if abs(np.sum(z_probs) - 1) > 0.01:
            raise ValueError('z_probs must sum to 1.')
        rs = np.random.RandomState()
        num_pixels = len(self.fg_coords[0])
        # Divide coords by Z slice.
        eroded_coords = self.get_eroded_coordinates(sigma_base * 3)
        coord_list = np.array(list(zip(eroded_coords[0], eroded_coords[1], eroded_coords[2])))
        coords_byz = []
        num_zslices = self.im.shape[0]
        for z in range(num_zslices):
            coords_byz.append(coord_list[eroded_coords[0] == z])
        # Draw z slices from probability dist: end up with list of Z-slices.
        z_slices = rs.choice(np.arange(0, num_zslices), p=z_probs, size=numblobs)
        # Add spots in randomly selected Z slices.
        for z in z_slices:
            # Draw intensity and sigma.
            sigma = sigma_base + rs.gamma(sigma_k, sigma_theta)
            intensity = rs.normal(intensity_mean, intensity_std)
            intensity = np.max([0, intensity])
            # Draw random coordinate from the drawn Z-slice.
            rand_idx = rs.randint(0, len(coords_byz[z]) - 1)
            rand_coords = coords_byz[z][rand_idx]
            self.add_gaussian_blob(rand_coords, intensity, sigma)

    #-----------------------------------------------------------------------
    def add_nblobs_zschedule_exponential(self, numblobs, intensity_mean, 
            intensity_std, sigma_base, exp_shape=1, sigma_k=0.5, sigma_theta=0.5, 
            return_probs=False):
        """Add blobs with a Z distribution defined by an exponential 
        function.
        
        Shape of exponential is determined by exp_shape according to:
            y = exp(exp_shape * x) for 0 < x < 1
            probs = y / sum(y)

            0: Flat
            More positive: steeper bias toward higher z slices
            More negative: steeper bias toward lower z slices

        Args:
            numblobs: int, number of blobs to add
            intensity_mean: numeric, mean of the distribution from which 
                blob intensities are drawn.
            intensity_std: numeric, std. deviation of the distribution from
                which blob intensities are drawn.
            sigma_base: numeric, minimum value of gaussian width (sigma)
            exp_shape: numeric, shape parameter of exponential
            sigma_k: numeric, shape parameter of gamma distribution added to 
                sigma_base to determine gaussian width
            sigma_theta: numeric, scale parameter of gamma distribution added 
                to sigma_base to determine gaussian width
            return_probs: bool, return z-slice probabilities
        
        Possible issue: the number of available pixels is different at different
        Z slices. Even a uniform distribution, done the way it's done here, will
        crowd spots at the poles. It's possible there's a different version needed
        that weights Z-slices by pixels and uses the distribution as some kind
        of bias.
        """
        x = np.arange(0, 1, 1/self.im.shape[0])
        y = np.exp(exp_shape * x)
        probs = y / np.sum(y)
        self.add_nblobs_zschedule(numblobs, intensity_mean, intensity_std, sigma_base,
            z_probs=probs, sigma_k=0.5, sigma_theta=0.5)
        if return_probs:
            return probs

    #-----------------------------------------------------------------------

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
def sim_rpb1(mask, filename, nuc_bg_mean=10_000, nonnuc_bg_mean=500, 
    noise_sigma=300, nblobs=40, blob_intensity_mean=10_000, 
    blob_intensity_std=2_000, blob_sigma_base=0.5, blob_sigma_k=0.5, 
    blob_sigma_theta=0.5, hlb_intensity=19_000, hlb_sigma=5, hlb_p=2):
    """Simulate an rpb1 nucleus, write to file.

    Args:
        mask: ndarray, input mask for building nucleus
        filename: string or filehandle, file to write image to
        nuc_bg_mean: number, value for nuclear background (uniform)
        nonnuc_bg_mean: number, value for cytoplasm background (uniform)
        noise_sigma: number, sigma value for gaussian component of 
            add_noise function
        nblobs: int, number of blobs to add
        blob_intensity_mean: number, mean of distribution for blob
            intensities
        blob_intensity_std: number, std of distribution of blob 
            intensities
        blob_sigma_base: float, base value for blob size sigma
        blob_sigma_k: float, k parameter for blob size distribution
        blob_sigma_theta: float, theta parameter for blob size 
            distribution
        hlb_intensity: number, intensity for HLB
        hlb_sigma: number, sigma (width) or HLB
        hlb_p: number, shape of gaussian for HLB (higher = flatter)
    """

    # Make sure nblobs is an int.
    nblobs = int(nblobs)
    rs = np.random.RandomState()
    mask = Sim.rotate_binary_mask(mask, rs.randint(0, 360)).astype('float64')

    # Build nucleus with input values.
    sim = Sim(mask)
    sim.add_background(val=nuc_bg_mean)
    sim.add_background(inverse=True, val=nonnuc_bg_mean)
    sim.smooth_edges(1)
    sim.add_hlb(hlb_intensity, hlb_sigma, hlb_p)
    sim.add_nblobs(nblobs, blob_intensity_mean, blob_intensity_std, 
        sigma_base=blob_sigma_base, sigma_k=blob_sigma_k, 
        sigma_theta=blob_sigma_theta)
    sim.add_noise(sigma=noise_sigma)
    sim.im[sim.im < 0] = 0
    sim.im[sim.im > 65_536] = 65_536
    save_pickle(sim.im, filename)

#-----------------------------------------------------------------------
def sim_rpb1_rand_batch( 
    maskfile, 
    outfolder,
    nsims,
    nreps,
    nprocesses,
    nuc_bg_mean_rng, 
    nonnuc_bg_mean_rng, 
    noise_sigma_rng, 
    nblobs_rng, 
    blob_intensity_mean_rng, 
    blob_intensity_std_rng,
    blob_sigma_base_rng,
    blob_sigma_k_rng, 
    blob_sigma_theta_rng, 
    hlb_intensity_rng,
    hlb_sigma_rng, 
    hlb_p_rng):
    """Perform parallelized simulations of rpb1 nuclei by randomly drawing
    parameters from supplied ranges.

    Args:
        maskfile: path, pickled file containing list of masks as ndarrays
        outfolder: path, folder to which to write simulations
        nsims: int, the number of parameter sets to simulate
        nreps: int, the number of simulations to perform for each param set
        nprocesses: int, the number of processes to launch with 
            multiprocessing Pool
        Iterables of form (lower_limit, upper_limit) are required to supply
            the ranges from which parameters are drawn. For a constant, 
            supply the same number twice:

            nuc_bg_mean_rng
            nonnuc_bg_mean_rng
            noise_sigma_rng 
            nblobs_rng 
            blob_intensity_mean_rng 
            blob_intensity_std_rng
            blob_sigma_base_rng
            blob_sigma_k_rng
            blob_sigma_theta_rng
            hlb_intensity_rng
            hlb_sigma_rng 
            hlb_p_rng
    
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
    
    # Load masks.
    masks = load_pickle(maskfile)

    # Set list of args to use for calling and logging.
    args = (maskfile, outfolder, nsims, nreps, nprocesses, nuc_bg_mean_rng, nonnuc_bg_mean_rng, noise_sigma_rng, 
        nblobs_rng, blob_intensity_mean_rng, blob_intensity_std_rng, blob_sigma_base_rng, blob_sigma_k_rng, 
        blob_sigma_theta_rng, hlb_intensity_rng, hlb_sigma_rng, hlb_p_rng)
    
    # For each sim, built a local set of args that will be sent to sim_rpb1,
    # add each set of parameters to arglist.
    rs = np.random.RandomState()
    arglist = []
    for _ in range(nsims):
        # Build up args for rpb1 using random draws from supplied ranges 
        # (uniform prob).
        args_this_sim = []
        for arg in args[5:]:
            choice = randomize_ab(arg)
            args_this_sim.append(choice)

        # Create a unique 3-letter id for this batch.
        file_id = ''.join(random.choice(string.ascii_letters) for i in range(3))
        # Add the parameters to arglist once for every replicate, with the only
        # changes being 1) the rep number in the filename 2) Selection and 
        # rotation of a random mask.
        for n in range(1, nreps+1):
            # Select mask and rotate.
            mask = masks[rs.randint(0, len(masks))]
            #mask = Sim.rotate_binary_mask(mask, rs.randint(0, 360)).astype('float64')
            # Create filename, build final args, add to list.
            filepath = os.path.join(folder, file_id + '_' + '_'.join([str(round(x,1)) for x in args_this_sim]) + '_rep' + str(n) + '.pkl')
            args_this_rep = [mask, filepath] + args_this_sim
            arglist.append(args_this_rep)
    print('arglist done')
    # Launch simulations in parallel using pool method.
    for i in range(0, len(arglist), 1000):
        end = i + 1000
        arglist_sub = arglist[i:end]   
        pool = mp.Pool(processes=nprocesses)
        results = [pool.apply_async(sim_rpb1, args=(x)) for x in arglist_sub]
        [p.get() for p in results]

    # Write logfile.
    logfilepath = os.path.join(folder, 'logfile_' + folder_id + '.txt')
    varnames = sim_rpb1_rand_batch.__code__.co_varnames
    with open(logfilepath, 'w') as logfile:
        for i in range(len(args)):
            logfile.write(varnames[i] + ': ')
            logfile.write(str(args[i]))
            logfile.write('\n')

