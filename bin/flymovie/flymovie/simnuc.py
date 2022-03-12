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

from flymovie.general_functions import mesh_like, stack_normalize_minmax
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
            gaussian = rs.normal(np.zeros_like(self.im), scale=sigma)
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
    def add_sphere(self, center_coords, intensity, num_fluors, rad):
        """Add a spherical object at specified coordinates.

        Intensity is distributed evenly such that the intensity of each
        pixel in the object is equal to:
            intensity * num_fluors / num_pixels

        Args:
            coords: iterable of ints, location lowest-index coordinate (
                bottom in z, top-left in xy)
            intensity: int, intensity of fluors making up object
            num_fluors: int, number of fluors in object
            rad: float: radius of object
        """
        mesh = mesh_like(self.im, 3)
        z, i, j = mesh

        pix_coords = np.where((((z - center_coords[0]) ** 2) + ((i - center_coords[1]) ** 2) + 
            ((j - center_coords[2]) ** 2)) < (rad ** 2))
        num_pixels = len(pix_coords[0])
        intensity_per_pixel = intensity * num_fluors / num_pixels
        self.im[pix_coords] = self.im[pix_coords] + intensity_per_pixel

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
    def add_n_objects(self, n_objects, intensity, fluors_per_object,
        size, mode='nuc', fluors_per_object_probs=None, size_probs=None, 
        erosion_size=None):
        """Add a defined number of objects at random positions.

        Intensity of fluors is fixed, but object size and the number of 
        fluors per object can be randomized according to supplied possible
        values and probabilities. Objects can be added to the nucleus, 
        outside the nucleus, or uniformly across the image.

        Args:
            n_objects: int, number of objects to add
            intensity: int, intensity of fluors making up object
            fluors_per_object: int or iterable of ints,  
                possible values for number of fluors per object
            size: int or iterable of ints, possible side lengths of 
                objects. Objects are cubes equal of all sides.
            mode: string, one of:
                'nuc': objects placed in nucleus (mask==1)
                'nonnuc': objects placed outside nucleus (mask==0)
                'all': objects placed in whole image
            fluors_per_object_probs [optional]: iterable of floats, 
                probabilities of each value in fluors_per_object_val. If 
                omitted, probabilities are set to uniform.
            size_probs [optional]: iterable of floats, 
                probabilities of each value in size_vals. If omitted, 
                probabilities are set to uniform.
            erosion_size [optional]: int, size of erosion kernel used to 
                restrict available nuclear coordinates. Has the effect of
                shaving off the outter layer of the nucleus for selecting
                object positions.
        """
        def draw_vals(vals, probs, n_objects, rs):
            """Draw values for random variables."""
            # If it is a non-iterable value, simply return.
            try:
                iter(vals)
            except:
                return np.repeat(vals, n_objects)

            # Create uniform probabilities if probs omitted.
            if probs is None:
                probs = np.ones(len(vals))/ len(vals)

            # Randomly draw fluors_per_object based on probabilities, store in array.
            vals_arr = rs.choice(vals, n_objects, p=probs)
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

        # Get values for random variables.
        fluors_per_object_arr = draw_vals(fluors_per_object, fluors_per_object_probs, n_objects, rs)
        size_arr = draw_vals(size, size_probs, n_objects, rs)

        # For each iteration get random coordinates, add object.
        for i in range(n_objects):
            fluors_per_object = fluors_per_object_arr[i]
            size = size_arr[i]
            px = rs.randint(0, num_pixels - 1)
            random_coords = (coords[0][px], coords[1][px], coords[2][px])
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
        """Convolve image with its associated kernel (PSF).

        Args: None
        """
        if self.kernel is None:
            raise ValueError('Must add kernel to convolve.')
        
        # Resize (re-sample) kernel to match image resolution.
        kernel = self.kernel.copy()
        # Resize kernel to match image dimensions.
        if ((self.res_ij != self.kernel_res_ij) 
            or (self.res_z != self.kernel_res_z)):
            kernel = ndi.zoom(self.kernel, [
                self.kernel_res_z / self.res_z, 
                self.kernel_res_ij  / self.res_ij, 
                self.kernel_res_ij  / self.res_ij
                ])
            kernel = kernel / np.sum(kernel)

        conv = ndi.convolve(self.im, kernel)
        self.im = conv

    #-----------------------------------------------------------------------
    def resize(self, dims, order=3):
        """Resize image.

        Args:
            dims: iterable of ints, resolution in each dimension of new 
                image (in nm)
            order: int 0-5, order of the polynomial used for interpolation
        """
        zoom_factors = (
                self.res_z / dims[0],
                self.res_ij / dims[1],
                self.res_ij / dims[2]
            )
        im_rs = ndi.zoom(self.im, zoom_factors, order=order)
        self.im = im_rs
        self.res_z = dims[0]
        self.res_ij = dims[1]

#-----------------------------------------------------------------------
##### End of Sim class. #####
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
def randomize_ab(ab, rs=None):
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
    if rs is None:
        rs = np.random.RandomState()
    return (rs.random() * (b - a)) + a

#-----------------------------------------------------------------------
def make_imperfect_masks(dims=(34,100,100), nucrad_mean=45, 
        nucrad_range=3, center_range=5, n=20):
    """Make spherical nuclear masks that vary in center position and
    radius.

    Nuclear radius and center position are drawn from uniform 
    distributions defined by range parameters.
    
    Args:
        dims: tuple of ints
            Dimensions of masks to make
        nucrad_mean: int
            Average nuclear radius
        nucrad_range: int
            Range (+ and -) to vary nuclear radius
        center_range: int
            Range (+ and -) to vary the nucleus position from the 
            image center in lateral dimensions only. Variation applied
            independently in i and j dimension.
        n: int
            Number of masks to make

    Returns:
        masks: list of ndarrays
            List of masks
    """
    rs = np.random.RandomState()
    # Create mask in expanded mask, differentially slice to vary
    # center position.
    maskdims = (dims[0], dims[1] + (2 * center_range), dims[2] + (2 * 
            center_range))
    masks = []
    for _ in range(n):
        nucrad_adj = rs.randint(-1 * nucrad_range, nucrad_range)
        i_adj = rs.randint(0, 2 * center_range)
        j_adj = rs.randint(0, 2 * center_range)
        mask = Sim.make_spherical_mask(maskdims[0], maskdims[1], 
                maskdims[2], nucrad_mean + nucrad_adj)
        mask = mask[:, i_adj:(i_adj + dims[1]), j_adj:(j_adj + dims[2])]
        masks.append(mask)

    return masks

#-----------------------------------------------------------------------
def sim_rpb1(masks, kernel, outfolder, nreps, nfree_rng, hlb_diam_rng, 
    hlb_nmols_rng, n_clusters_rng, cluster_diam_mean_rng, 
    cluster_diam_var_rng, cluster_nmols_mean_rng, cluster_nmols_var_rng,
    noise_sigma_rng, hlb_coords, dims_init=(85, 85, 85), 
    dims_kernel=(250,85,85), dims_final=(250,85,85), return_sim=False):
    """Simulate an rpb1 nucleus from parameters drawn from ranges, 
        write to file.

    Parameters are supplied as ranges defining the upper and lower bounds
    of the value. Parameter values are drawn from a uniform distribution
    defined by these bounds. 

    The parameters governing clusters are a mean and standard deviation
    for the cluster diameter and number of molecules. These means and 
    standard deviations are first drawn from uniform distribution
    defined by supplied range, and they are then used to generate lists
    of values and associated probabilities for a range +- 5 * sigma 
    to feed to the function add_n_objects which performs random draws 
    to generate clusters.

    Args:
        masks: list of ndarrays; list of input masks for building nucleus.
            Length must equal number of reps
        kernel: ndarray; kernel for convolution
        outfolder: string; folder in which to write outputs
        nreps: int; number of replicate simulations to make with each 
            parameter set.
        xxx_rng: iterable of 2 ints; upper/lower bounds for parameter xxx
        hlb_coords: iterable of tuples; coordinates for hlb locations, will 
            be drawn in order
        dims_init: tuple; dimensions (in nm) of voxels in initial image
        dims_kernel: tuple; dimensions (in nm) of kernel
        dims_final: tuple; dimensions (in nm) of final image
        return_sim: bool; if true, performs one simulation and returns,
            does not write to file.

    Output:
        Simulated images are saved as pickled ndarrays. Filenames contain 
        a random 3-letter string followed by parameters separated by _ :

            0: nfree
            1: hlb_diam
            2: hlb_nmols
            3: n_clusters
            4: cluster_diam_mean, 
            5: cluster_diam_var
            6: cluster_nmols_mean
            7: cluster_nmols_var
            8: noise_sigma
    """
    def make_vals_probs(mean, sigma):
        """Get values (>= 1) and probabilities for random variables."""
        # Get values in the range of +- 5 sigma.
        vals = np.arange(np.max([1, mean - (5 * sigma)]), mean + (5 * sigma))
        probs = scipy.stats.norm(loc=mean, scale=sigma).pdf(vals)
        probs = probs / np.sum(probs) # Normalize to sum=1.
        vals = [int(x) for x in vals]
        return vals, probs

    # Generate random file prefix.
    file_id = ''.join(random.choice(string.ascii_letters) for i in range(3))

    gfp_intensity = 100
    rs = np.random.RandomState()

    ### Randomly draw parameters from supplied ranges. ###
    nfree = round(randomize_ab(nfree_rng, rs))
    hlb_diam = float(randomize_ab(hlb_diam_rng, rs))
    hlb_nmols = round(randomize_ab(hlb_nmols_rng, rs))
    n_clusters = round(randomize_ab(n_clusters_rng, rs))
    cluster_diam_mean = round(randomize_ab(cluster_diam_mean_rng, rs))
    cluster_diam_var = randomize_ab(cluster_diam_var_rng, rs)
    cluster_nmols_mean = round(randomize_ab(cluster_nmols_mean_rng, rs))
    cluster_nmols_var = randomize_ab(cluster_nmols_var_rng, rs)
    noise_sigma = float(randomize_ab(noise_sigma_rng, rs))

    cluster_diam_vals, cluster_diam_probs = make_vals_probs(cluster_diam_mean, cluster_diam_var)
    cluster_nmols_vals, cluster_nmols_probs = make_vals_probs(cluster_nmols_mean, cluster_nmols_var)

    ### Simulate an Rpb1 nucleus with selected parameters. ###
    for nrep in range(nreps):
        mask = masks[nrep]
        sim = Sim(mask, res_z=dims_init[0], res_ij=dims_init[1])
        sim.add_kernel(kernel, res_z=dims_kernel[0], res_ij=dims_kernel[1])
        # Add free population.
        sim.add_n_objects(nfree, gfp_intensity, fluors_per_object=1, size=1, mode='nuc')
        # Add HLB.
        sim.add_sphere(hlb_coords[nrep * 2], gfp_intensity, hlb_nmols, hlb_diam / 2)
        sim.add_sphere(hlb_coords[(nrep * 2) + 1], gfp_intensity, hlb_nmols, hlb_diam / 2)

        # Add clusters.
        sim.add_n_objects(n_clusters, gfp_intensity, fluors_per_object=cluster_nmols_vals, 
            size=cluster_diam_vals, fluors_per_object_probs=cluster_nmols_probs, 
            size_probs=cluster_diam_probs)
        
        # Add noise and convolve.
        sim.add_noise('poisson')
        sim.convolve()
        sim.resize(dims_final, order=1)
        sim.im = stack_normalize_minmax(sim.im) * 1000
        sim.add_noise('gaussian', sigma=noise_sigma)
        # Bound values.
        sim.im[sim.im < 0] = 0
        sim.im[sim.im > 65_536] = 65_536

        if return_sim:
            return sim

        paramstring = '_'.join([str(round(x, 2)) for x in [nfree, 
                hlb_diam, hlb_nmols, n_clusters, cluster_diam_mean, 
                cluster_diam_var, cluster_nmols_mean, cluster_nmols_var, 
                noise_sigma]])
        filepath = os.path.join(outfolder, file_id + '_' + paramstring 
            + '_rep' + str(nrep) + '.pkl')
        save_pickle(sim.im, filepath)

#-----------------------------------------------------------------------
def run_pooled_processes(arglist, nprocesses, func, batch_size=1000):
    """Launch a parallel process using multiprocessing Pool function.
    
    I have had some odd problems when supplying large numbers of processes
    to Pool, and I have found empirically that batching helps speed things
    up. This seems to defeat the purpose of Pool, but it works and achieves
    parallelization since batch sizes can still be quite large (1000 seems
    to work just fine; 10,000 runs very slowly).

    Args:
        arglist: iterable; list of arguments to map to function
        nprocesses: int; number of parallel processes to launch (typically,
            number of available cores)
        func: function to call in parallel (target of arglist)
        batch_size: int; size of batches sent to Pool. I think this can be
            left at 1000.
    
    """
    mp.set_start_method('fork', force=True) # Important for macOS.
    with mp.Pool(nprocesses) as pool:
        for i in range(0, len(arglist), batch_size):
            end = i + batch_size
            arglist_sub = arglist[i:end]   
            results = [pool.apply_async(func, (), x) for x in arglist_sub]
            [p.get() for p in results]

#-----------------------------------------------------------------------
def write_logfile(filepath, logitems):
    """Write a logfile with parameters.
    
    Args:
        filepath: string; path for logfile
        logitems: dict; dict containing items to log
    """
    with open(filepath, 'w') as logfile:
        logfile.write(datetime.now().ctime() + '\n')
        for key in logitems:
            logfile.write(key + ': ' + str(logitems[key]))
            logfile.write('\n')

#-----------------------------------------------------------------------
def sim_rpb1_batch(outfolder, kernel, nsims, nreps, nprocesses, mask_dims,
    sim_func=sim_rpb1, nuc_rad=90, nmasks=50, **kwargs):
    """Perform parallelized simulations of Rpb1 nuclei in batch.

    Note: I tried to make a batch function that would be general
    purpose but I couldn't make it work because the erosion for 
    getting HLB coordinates is quite slow and needs to not be repeated.

    Args:
        outfolder: path, folder to which to write simulation outputs
        kernel: ndarray, convolution kernel for images
        nsims: int, number of simulations to perform
        nreps: int, number of replicate simulations to make for each
            parameter set
        nprocesses: int, the number of processes to launch with 
            multiprocessing Pool
        mask_dims: iterable of ints, dimensions in pixels of mask
        sim_func: function, function that recieved kwargs, performs
            simulations, and writes to file
        nuc_rad: number, radius in pixels of spherical nucleus in mask
        nmasks: int, the number of different masks to make (different
            nucleus sizes and shapes)
        kwargs: args supplied to sim_func
    
    Outputs:
        A unique 8-character identifier is associated with the entire 
        batch. This ID is appended to the folder name. A log file is 
        saved in the folder containing all supplied paramaters. In 
        addition, each parameter batch gets a 3-letter random ID to 
        avoid collisions (unlikely). Parameters for each simulation 
        are saved in the filename, separated by underscores.

    Note: "Replicates" use identical parameters but different masks. 
    """
    # Set folder name with unique identifier and create it.
    folder_id = ''.join(random.choice(string.ascii_letters) for i in range(8))
    folder = outfolder + '_' + folder_id
    os.mkdir(folder)
    
    # Generate masks.
    masks = make_imperfect_masks(dims=mask_dims, nucrad_mean=nuc_rad, 
            nucrad_range=round(nuc_rad * 0.05), center_range=5, n=nmasks)

    # Get a list of candidate HLB coordinates by performing erosion on nuclear mask.
    # This ensures that the HLB won't be placed at the nuclear periphery and end up
    # outside the nucleus.
    hlb_coords_possible_list = []
    num_hlb_coords_list = []
    for mask in masks:
        hlb_coords_possible = Sim(mask).get_eroded_coordinates(10)
        num_hlb_coords = len(hlb_coords_possible[0])
        hlb_coords_possible = list(zip(hlb_coords_possible[0], hlb_coords_possible[1], hlb_coords_possible[2]))
        hlb_coords_possible_list.append(hlb_coords_possible)
        num_hlb_coords_list.append(num_hlb_coords)

    # Set constant args.
    f_kwargs = kwargs.copy()
    f_kwargs['kernel'] = kernel
    f_kwargs['outfolder'] = folder
    f_kwargs['nreps'] = nreps
    
    # For each replicate, perform selection of random masks and matching
    # HLB coordinates, add all kwargs to arglist for parallel calling.
    arglist = []
    rs = np.random.RandomState()
    # Select random masks, from masks select random HLB coordinates.
    for _ in range(nsims):
        f_kwargs_loc = f_kwargs.copy()
        mask_idxs = rs.randint(0, len(masks), nreps)
        hlb_coords = []
        for idx in mask_idxs:
            num_coords = num_hlb_coords_list[idx]
            hlb_coords_possible = hlb_coords_possible_list[idx]
            coords = [hlb_coords_possible[x] for x in rs.choice(num_coords, 2)]
            hlb_coords = hlb_coords + coords
        
        f_kwargs_loc['hlb_coords'] = hlb_coords
        f_kwargs_loc['masks'] = [masks[x] for x in mask_idxs]
        arglist.append(f_kwargs_loc)
    print('arglist done')

    run_pooled_processes(arglist, nprocesses, sim_func)
    
    # Write logfile.
    logitems = kwargs.copy()
    logitems['nsims'] = nsims
    logitems['nreps'] = nreps
    logfilepath = os.path.join(folder, 'logfile_' + folder_id + '.txt')
    write_logfile(logfilepath, logitems)
    
    return folder

