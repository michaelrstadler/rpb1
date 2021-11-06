import numpy as np
import matplotlib.pyplot as plt
import scipy
from flymovie.general_functions import mesh_like
from flymovie.load_save import save_pickle
import scipy.ndimage as ndi
import dask
import warnings
import gc
import os
import multiprocessing
from time import sleep

############################################################################
def simulate_blobs(nucmask, bg_mean=10000, bg_var=10, blob_intensity_mean=20000, 
    blob_intensity_var=1000, blob_radius_mean=2, blob_radius_var=0.5, blob_number=50, 
    z_ij_ratio=2):

    """Simulate the distribution of fluorescence signal within nuclei in a 
    supplied mask.
    
    Args:
        nucmask: ndarray
            3D mask (binary or label) defining positions of nuclei
        bg_mean: numeric
            The mean value of the nuclear background signal
        bg_var: numeric
            Variance of the nuclear background signal
        blob_intensity_mean: numeric
            Mean fluorescence intensity of gaussian "blobs"
        blob_intensity_var: numeric
            Variance of the gaussian blob intensity
        blob_radius_mean: numeric
            Mean radius, in pixels, of gaussian blobs
        blob_radius_var: numeric
            Variance of blob radius
        blob_number: int
            Number of gaussian blobs per nucleus
        z_ij_ratio: numeric
            Ratio of the voxel size in Z to the size in ij

    Returns:
        simstack: ndarray
            Stack of same shape as nucmask containing simulated data.
    """
    def make_odd(n):
        """Ensure box dimension is an odd integer to make math work."""
        n = int(n)
        if (n % 2) == 0:
            return n + 1
        else:
            return n

    def make_3d_gaussian_inabox(intensity, sigma_z, sigma_ij, z_winlen, ij_winlen):
        """Make a 3D gaussian signal within a box of defined size.
        
        I used trial and error to find vectorized ways to multiply 1D numpy
        vectors (generated from 1D gaussian functions) together to produce
        a proper 3D gaussian."""
        d1 = scipy.signal.gaussian(ij_winlen, sigma_ij)
        d2 = np.outer(d1, d1)
        z_1dvector = scipy.signal.gaussian(z_winlen, sigma_z)
        d3 = d2 * np.expand_dims(z_1dvector, axis=(1,2))
        return intensity * d3

    def add_box_to_stack(stack, box, coords):
        """Add pixel values from a supplied "box" to a stack at a position 
        centered at supplied coordinates."""
        # Initialize arrays to store the start and end coordinates for the 
        # stack and the box in each dimension.
        box_starts = []
        box_ends = []
        stack_starts = []
        stack_ends = []
        # For each dimension, find the start and stop positions for the box and 
        # the stack to be centered at coords; handle the cases where the supplied 
        # coordinates and box size will result in trying to assign positions 
        # outside the stack.
        for dim in range(0, 3):
            # Initialize start and stop locations for case where box is entirely
            # within stack dimensions.
            start = coords[dim] - int(box.shape[dim] / 2)
            end = coords[dim] + int(box.shape[dim] / 2) + 1
            stack_start = start
            stack_end = end
            box_start = 0
            box_end = box.shape[dim]
            # Adjust for cases where box falls out of bounds of stack.
            if start < 0:
                stack_start = 0
                box_start = -start
            if end > stack.shape[dim]:
                stack_end = stack.shape[dim]
                box_end = box.shape[dim] - (end - stack.shape[dim])
            # Append corrected starts and stops for this dimension.
            stack_starts.append(stack_start)
            stack_ends.append(stack_end)
            box_starts.append(box_start)
            box_ends.append(box_end)
        # Ensure that the shapes of the subsection of the stack to add to
        # and the box to add are the same. If so, add box values to stack.
        substack_shape = stack[stack_starts[0]:stack_ends[0], stack_starts[1]:stack_ends[1], stack_starts[2]:stack_ends[2]].shape
        box_to_add = box[box_starts[0]:box_ends[0], box_starts[1]:box_ends[1], box_starts[2]:box_ends[2]]
        if substack_shape == box_to_add.shape:
            stack[stack_starts[0]:stack_ends[0], stack_starts[1]:stack_ends[1], stack_starts[2]:stack_ends[2]] += box_to_add
        else:
            warnings.warn('Dimensions of box to add and stack to replace do not match.')
    
    ij_windowlen = make_odd(blob_radius_mean * 10.5)
    z_windowlen = make_odd(ij_windowlen / z_ij_ratio)
    # Initialize stack with just nuclear backgrounds.
    bg_stack = np.random.normal(bg_mean, bg_var, size=nucmask.shape)
    simstack = np.where(nucmask, bg_stack, 0)
    # Go through each nucleus, add specified blobs.
    for nuc_id in np.unique(nucmask)[1:]:
        # Get the set of coordinates for this nucleus.
        nuc_coords = np.where(nucmask == nuc_id)
        nuc_numpixels = len(nuc_coords[0])
        for n in range(0, blob_number):
            # Get blob radii in ij (lateral) and z (axial) dimensions.
            r_ij = np.random.normal(blob_radius_mean, blob_radius_var)
            r_z = r_ij / z_ij_ratio
            # Get randomly-generated intensity.
            intensity = np.random.normal(blob_intensity_mean, blob_intensity_var)
            # Select a random coordinat in the nucleus.
            rand_pixel_num = np.random.randint(0, nuc_numpixels - 1)
            z, i, j = nuc_coords[0][rand_pixel_num], nuc_coords[1][rand_pixel_num], nuc_coords[2][rand_pixel_num]
            #gaussian_function = gaussian3d(z, i, j, intensity, r_z, r_ij, r_ij)
            # Make a 3D "box" with values from specified 3D gaussian function, then
            # add that to the stack in a way that is relatively fast.
            box = make_3d_gaussian_inabox(intensity, r_z, r_ij, z_windowlen, ij_windowlen)
            add_box_to_stack(simstack, box, (z, i, j))
    return simstack

############################################################################
def make_dummy_mask(zdim=20, idim=800, jdim=800, nuc_spacing=200, nuc_rad=50):
    """Make a label mask of spherical dummy nuclei.
    
    Nuclei are equally spaced spheres.

    Args:
        zdim: int
            Size of mask in z dimension
        idim: int
            Size of mask in i dimension
        jdim: int
            Size of mask in j dimension
        nuc_spacing: int
            Number of pixels separating nuclear centers in ij dimension
        nuc_rad: int
            Radius of nuclei
    """
    mask = np.zeros((zdim, idim, jdim))
    z, i, j = mesh_like(mask, 3)
    z_midpoint = int(mask.shape[0] / 2)
    nuc_id = 1
    for i_center in range(nuc_rad, mask.shape[1], nuc_spacing):
        for j_center in range(nuc_rad, mask.shape[2], nuc_spacing):
            # Basic equation of circle.
            mask[(((z - z_midpoint) ** 2) + ((i - i_center) ** 2) + 
                ((j - j_center) ** 2)) < (nuc_rad ** 2)] = nuc_id
            nuc_id += 1
    return mask

############################################################################
def make_scalespace_representation(stack, sigmas=[0,0.5,1,1.5,2,4,6]):
    """Make a scalespace representation of an input stack using gaussian 
    kernel with a range of sigmas.
    
    Args:
        stack: ndarray
            Image stack
        sigmas: iterable
            Range of sigmas to use for gaussian filtering in scale-space
            representation

    Returns:
        scalespace: ndarray
            Zeroeth dimension corresponds to the supplied sigma levels,
            shape of each 0-dimension entry is equal to input stack.
    """
    dims = tuple([len(sigmas)]) + stack.shape
    scalespace = np.zeros(dims)
    for i in range(0, len(sigmas)):
        sigma = sigmas[i]
        scalespace[i] = ndi.gaussian_filter(stack, sigma)
    return scalespace

############################################################################
def make_scalespace_hist(scalespace, mask=None, numbins=100, histrange=(0,25)):
    """Make a 2D histogram of a scalespace representation of an image stack.
    
    Args:
        scalespace: ndarray
            Scalespace representation of image stack (scale in dimension 0)
        mask: ndarray
            (optional) Mask for values to be included in histogram calculation
        numbins: int
            Number of bins to use for histogram
        histrange: (int, int)
            Range of values to be included in histogram
    
    Returns:
        hist_data: ndarray
            Histogram data for each sigma level of input data. Dimensions
            are number of sigmas x numbins
        
    """
    def get_pixel_vals(stack, mask):
        """Create flattened array of values in mask foreground of image 
        stack."""
        if mask is not None:
            return stack[np.where(mask)]
        else:
            return stack.flatten()
    # Initialize histogram data with 0s.
    hist_data = np.zeros(tuple([scalespace.shape[0], numbins]))
    # Make histogram from each sigma level, add to hist_data.
    for s in range(0, scalespace.shape[0]):
        vals = get_pixel_vals(scalespace[s], mask)
        hist_ = np.histogram(vals, bins=numbins, range=histrange)[0]
        hist_data[s] = hist_
    
    return hist_data

############################################################################
def make_parameter_hist_data(bg_mean_range, bg_var_range, blob_intensity_mean_range, 
    blob_intensity_var_range, blob_radius_mean_range, blob_radius_var_range, 
    blob_number_range, z_ij_ratio=2, sigmas=[0,0.5,1,1.5,2,4,6], numbins=100, 
    histrange=(0,25), zdim=40, idim=800, jdim=800, nuc_spacing=200, nuc_rad=50):
    """Simulate blobs with a range of parameters, record scale-space histograms.
    
    Args:
        bg_mean_range: iterable
            Range of values for the nuclear baground mean
        bg_var_range= iterable
            Range of values for the nuclear background variance
        blob_intensity_mean_range: iterable
            Range of values for the blob intensity mean
        blob_intensity_var_range: iterable
            Range of values for the blob intensity variance
        blob_radius_mean_range: iterable
            Range of values for the blob radius mean
        blob_radius_var_range: iterable
            Range of values for the blob radius variance
        blob_number_range: iterable
            Range of values for the number of blobs per nucleuc
        z_ij_ratio: numeric
            Ratio of the size of voxels in z to ji dimensions
        sigmas: iterable
            Sigma values to use in constructing scale-space representations
            of simulated stacks.
        numbins: int
            Number of bins for histogram
        histrange: tuple of ints
            End points for histogram values
        zdim: int
            Size of simulated stack in pixels in z.
        idim: int
            Size of simulated stack in pixels in i
        jdim: int
            Size of simulated stack in pixels in j
        nuc_spacing: int
            The spacing, in pixels, between nuclei in i and j.
        nuc_rad: int
            Radius, in pixels, of simulated nuclei.

    Returns:
        data_: delayed list
            Dask delayed object, must be computed using dask.compute(data_).
            Each list item is the outcome of a simulation. Items are tuples.
            First (0) item is a list of simulation parameters:
                0: bg_mean 
                1: bg_var 
                2: blob_intensity_mean 
                3: blob_intensity_var 
                4: blob_radius_mean 
                5: blob_radius_var
                6: blob_number
            Second item (1) is the 2D histogram of the scale-space representation  
    """
    def sim_and_hist(mask, bg_mean, bg_var, blob_intensity_mean, 
        blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
        z_ij_ratio, sigmas, histrange, numbins, data_):
        """Call simulation, generate histogram, add to shared list."""
        simstack = simulate_blobs(mask, bg_mean, bg_var, blob_intensity_mean, 
            blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
            z_ij_ratio)
        hist_ = make_scalespace_2dhist(simstack, sigmas, mask, numbins, histrange)
        params = [bg_mean, bg_var, blob_intensity_mean, blob_intensity_var, blob_radius_mean, 
            blob_radius_var, blob_number]
        data_.append((params, hist_))

    mask = make_dummy_mask(zdim, idim, jdim, nuc_spacing, nuc_rad)
    manager = multiprocessing.Manager()
    processes = []
    with multiprocessing.Manager() as manager:
        data_ = manager.list()
        l = manager.list()
        for bg_mean in bg_mean_range:
            for bg_var in bg_var_range:
                for blob_intensity_mean in blob_intensity_mean_range:
                    for blob_intensity_var in blob_intensity_var_range:
                        for blob_radius_mean in blob_radius_mean_range:
                            for blob_radius_var in blob_radius_var_range:
                                for blob_number in blob_number_range:
                                    p = multiprocessing.Process(target=sim_and_hist, args=(mask, bg_mean, bg_var, blob_intensity_mean, 
                                        blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
                                        z_ij_ratio, sigmas, histrange, numbins, data_))
                                    p.start()
                                    processes.append(p)
                                    
        for process in processes:
            process.join()
        data_ = list(data_)
        sleep(2) # Prevents errors at the end of computation

    return data_
    

############################################################################
def simulate_param_range(outfolder, bg_mean_range, bg_var_range, blob_intensity_mean_range, 
    blob_intensity_var_range, blob_radius_mean_range, blob_radius_var_range, 
    blob_number_range, z_ij_ratio=2, zdim=20, idim=200, jdim=200, 
    nuc_spacing=100, nuc_rad=50):
    """Simulate blobs with a range of parameters, return simulations.
    
    Args:
        bg_mean_range: iterable
            Range of values for the nuclear baground mean
        bg_var_range= iterable
            Range of values for the nuclear background variance
        blob_intensity_mean_range: iterable
            Range of values for the blob intensity mean
        blob_intensity_var_range: iterable
            Range of values for the blob intensity variance
        blob_radius_mean_range: iterable
            Range of values for the blob radius mean
        blob_radius_var_range: iterable
            Range of values for the blob radius variance
        blob_number_range: iterable
            Range of values for the number of blobs per nucleuc
        z_ij_ratio: numeric
            Ratio of the size of voxels in z to ji dimensions
        zdim: int
            Size of simulated stack in pixels in z.
        idim: int
            Size of simulated stack in pixels in i
        jdim: int
            Size of simulated stack in pixels in j
        nuc_spacing: int
            The spacing, in pixels, between nuclei in i and j.
        nuc_rad: int
            Radius, in pixels, of simulated nuclei.

    Returns:
        data_: delayed list
            Dask delayed object, must be computed using dask.compute(data_).
            Each list item is the outcome of a simulation. Items are tuples.
            First (0) item is a list of simulation parameters:
                0: bg_mean 
                1: bg_var 
                2: blob_intensity_mean 
                3: blob_intensity_var 
                4: blob_radius_mean 
                5: blob_radius_var
                6: blob_number
            Second item (1) is the simulated image stack.
    """
    def sim_and_save(mask, bg_mean, bg_var, blob_intensity_mean, 
        blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
        z_ij_ratio):
        simstack = simulate_blobs(mask, bg_mean, bg_var, blob_intensity_mean, 
            blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
            z_ij_ratio)
        params = [bg_mean, bg_var, blob_intensity_mean, blob_intensity_var, blob_radius_mean, 
            blob_radius_var, blob_number]
        param_string = '_'.join([str(x) for x in params])
        filename = os.path.join(outfolder, param_string + '.pkl')
        save_pickle(simstack, filename)

    mask = make_dummy_mask(zdim, idim, jdim, nuc_spacing, nuc_rad)
    args = []
    for bg_mean in bg_mean_range:
        for bg_var in bg_var_range:
            for blob_intensity_mean in blob_intensity_mean_range:
                for blob_intensity_var in blob_intensity_var_range:
                    for blob_radius_mean in blob_radius_mean_range:
                        for blob_radius_var in blob_radius_var_range:
                            for blob_number in blob_number_range:
                                args.append([mask, bg_mean, bg_var, blob_intensity_mean, 
                                    blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
                                    z_ij_ratio])
                                
    batch_size = 2000
    for i in range(0, len(args), batch_size):
        processes = []
        start = i
        end = min(start + batch_size, len(args))
        for j in range(start, end):
            p = multiprocessing.Process(target=sim_and_save, args=args[j])
            p.start() 
                               
############################################################################
def make_scalespace_2dhist(stack, sigmas, mask, numbins=100, histrange=(0,66000)):
    """Directly calculate a scalespace histogram on an image stack.
        
    More memory efficient than first storing scalespace representation.
        
    Args:
        stack: ndarray
            Image stack
        mask: ndarray
            (optional) Mask for values to be included in histogram calculation
        numbins: int
            Number of bins to use for histogram
        histrange: (int, int)
            Range of values to be included in histogram
    
    Returns:
        hist_data: ndarray
            Histogram data for each sigma level of input data. Dimensions
            are number of sigmas x numbins
    """
    def get_pixel_vals(stack, mask):
        """Create flattened array of values in mask foreground of image 
        stack."""
        if mask is not None:
            return stack[np.where(mask)]
        else:
            return stack.flatten()
    # Initialize empty hist_ container.        
    hist_ = np.ndarray((0, numbins))
    # Calculate 1D hist for each sigma level, concatenate.
    for sigma in sigmas:
        stack_filtered = ndi.gaussian_filter(stack, sigma)
        vals = get_pixel_vals(stack_filtered, mask)
        hist_thissigma = np.histogram(vals, bins=numbins, range=histrange)[0]
        hist_ = np.vstack([hist_, hist_thissigma])
    return hist_