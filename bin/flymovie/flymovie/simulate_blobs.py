import numpy as np
import matplotlib.pyplot as plt
import scipy
from flymovie.general_functions import mesh_like, dog_filter
from flymovie.load_save import save_pickle, load_pickle, listdir_nohidden
import scipy.ndimage as ndi
import dask
import warnings
import gc
import os
import multiprocessing
import random
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
    # Initialize stack with just randomized nuclear backgrounds.
    rs = np.random.RandomState()
    bg_stack = rs.normal(bg_mean, bg_var, size=nucmask.shape)
    bg_stack = np.where(bg_stack >= 0, bg_stack, 0)
    simstack = np.where(nucmask, bg_stack, 0)
    # Initialize a randomly seeded random state to make thread safe.
    # Go through each nucleus, add specified blobs.
    for nuc_id in np.unique(nucmask)[1:]:
        # Get the set of coordinates for this nucleus.
        nuc_coords = np.where(nucmask == nuc_id)
        nuc_numpixels = len(nuc_coords[0])
        for n in range(0, blob_number):
            # Get blob radii in ij (lateral) and z (axial) dimensions.
            r_ij = rs.normal(blob_radius_mean, blob_radius_var)
            r_z = r_ij / z_ij_ratio
            # Get randomly-generated intensity.
            intensity = rs.normal(blob_intensity_mean, blob_intensity_var)
            intensity = np.max([0, intensity])
            # Select a random coordinat in the nucleus.
            rand_pixel_num = rs.randint(0, nuc_numpixels - 1)
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
def make_DoG_histograms(stack, mask=None, sigmas=[(1,2),(1,3),(1,5)], 
    numbins=325, histrange=(-10_000, 10_000)):
    """Make a 2D scale-space histogram of an image stack using difference
    of Gaussian (DoG) filter for scale axis.

    Args:
        stack: ndarray
            Image stack
        mask: ndarray
            (optional) Mask for values to be included in histogram calculation
        sigmas: iterable of tuples (numeric, numeric)
            Values to use for the smaller sigma value in DoG filter
        numbins: int
            Number of bins to use for histogram
        histrange: (int, int)
            Range of values to be included in histogram
    
    Returns:
        hist_data: ndarray
            Histogram data for each pair of sigma values. Dimensions
            are # sigmas x numbins
    """
    def get_pixel_vals(stack, mask):
        """Create flattened array of values in mask foreground of image 
        stack."""
        if mask is not None:
            return stack[np.where(mask)]
        else:
            return stack.flatten()

    hist_data = np.zeros(tuple([len(sigmas), numbins]))  
    n = 0 # Keep track of where to add new histogram.     
    for smallsigma, bigsigma in sigmas:
        dog = dog_filter(stack, bigsigma, smallsigma)
        vals = get_pixel_vals(dog, mask)
        hist_ = np.histogram(vals, bins=numbins, range=histrange)[0]
        hist_data[n] = hist_
        n += 1
    return hist_data

############################################################################
def make_scalespace_dog_hist(stack, mask, numbins=325, ss_sigmas=[0,0.5,1,2,4], 
    ss_histrange=(0,66_000), dog_sigmas=[(1,2),(1,3),(1,5)], 
    dog_histrange=(-10_000, 10_000)):
    """Wrapper for making scalespace and DoG-space histograms.
    
    Args:
        stack: ndarray
            Image stack
        mask: ndarray
            (optional) Mask for values to be included in histogram 
            calculation
        numbins: int
            Number of bins to use for histograms
        ss_sigmas: iterable of numeric
            Values to use for the sigmas in scalespace representation
        ss_histrange: tuple of ints
            Lower and upper range for values to be included in scalespace
            histogram
        dog_sigmas: iterable of tuples (numeric, numeric)
            Values to use for the smaller sigma value in DoG filter
        dog_histrange: (int, int)
            Lower and upper range for values to be included in DoG
            histogram

    Returns:
        ndarray
        Vertical stack of 2D histogram outputs for the two functions
    """
    ss_hist = make_scalespace_2dhist(stack, ss_sigmas, mask, numbins, 
        ss_histrange)
    dog_hist = make_DoG_histograms(stack, mask, dog_sigmas, numbins, 
        dog_histrange)
    return np.vstack((ss_hist, dog_hist))

############################################################################
def randomize_ab(ab):
    """Find random float number between a and b, given b > a.
    
    Args:
        ab: iterable
            Two numbers in an iterable, b > a.
    
    Returns:
        random float between a and b
    """
    a, b = ab
    if (b <= a):
        raise ValueError('b must be greater than a')
    return (np.random.random() * (b - a)) + a

############################################################################
def make_simulations_representations_from_sampled_params(num_sims, bg_mean_range, bg_var_range, 
    blob_intensity_mean_range, blob_intensity_var_range, 
    blob_radius_mean_range, blob_radius_var_range, blob_number_range, 
    z_ij_ratio=2, zdim=20, idim=300, jdim=200, nuc_spacing=100, 
    nuc_rad=50, process_function=make_scalespace_dog_hist, **kwargs):
    """Simulate nuclei with a parameters selected from ranges, create 
    representations of simulated images with supplied function.
    
    Args:
        num_sims: int
            Number of simulations to performs
        bg_mean_range: tuple of numeric
            Lower and upper range values for background mean
        bg_var_range= tuple of numeric
            Lower and upper range values for background variance
        blob_intensity_mean_range: tuple of numeric
            Lower and upper range values for intensity mean
        blob_intensity_var_range: tuple of numeric
            Lower and upper range values for blob intensity variance
        blob_radius_mean_range: tuple of numeric
            Lower and upper range values for blob radius mean
        blob_radius_var_range: tuple of numeric
            Lower and upper range values for blob radius variance
        blob_number_range: tuple of ints
            Lower and upper range values for number of blobs per nucleus
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
        process_function: function
            Function that processes simulated images. Takes stack and mask
            as first two arguments, returns ndarray.
        kwargs:
            Arguments to process_function

    Returns:
        data_: list of tuples
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
        z_ij_ratio, data_, process_function, **kwargs):
        """Call simulation, generate histogram, add to shared list."""
        simstack = simulate_blobs(mask, bg_mean, bg_var, blob_intensity_mean, 
            blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
            z_ij_ratio)
        # Normalize simulated stack using mean and a scale factor.
        simstack = simstack / np.median(simstack[np.where(mask)]) * 10_000
        hist_ = process_function(simstack, mask, **kwargs)
        params = [bg_mean, bg_var, blob_intensity_mean, blob_intensity_var, blob_radius_mean, 
            blob_radius_var, blob_number]
        data_.append((params, hist_))

    multiprocessing.set_start_method('fork') # Needed to work python 3.5+ on macOS
    mask = make_dummy_mask(zdim, idim, jdim, nuc_spacing, nuc_rad)
    # Make a manager to allow passing of list around to multiprocessing threads.
    manager = multiprocessing.Manager()
    processes = []
    args = []
    with multiprocessing.Manager() as manager:
        # Create sharable list to store outputs.
        data_ = manager.list()
        # Generate parameters by sampling for the sims to run, store in args.
        for sim_num in range(num_sims):
            bg_mean = randomize_ab(bg_mean_range)
            bg_var = randomize_ab(bg_var_range)
            blob_intensity_mean = randomize_ab(blob_intensity_mean_range)
            blob_intensity_var = randomize_ab(blob_intensity_var_range)
            blob_radius_mean = randomize_ab(blob_radius_mean_range)
            blob_radius_var = randomize_ab(blob_radius_var_range)
            blob_number = np.random.randint(blob_number_range[0], blob_number_range[1])
            args.append([mask, bg_mean, bg_var, blob_intensity_mean, 
                blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
                z_ij_ratio, data_, process_function])

        # Execute simulations in parallel in batches of 200. The proper way to do this
        # is to use multiprocessing pool but I had some trouble getting it to work. This
        # solution is sub-optimal but works and probably doesn't have huge costs in speed.
        batch_size = 1000
        for i in range(0, len(args), batch_size):
            processes = []
            start = i
            end = min(start + batch_size, len(args))
            for j in range(start, end):
                p = multiprocessing.Process(target=sim_and_hist, args=args[j], kwargs=kwargs)
                p.start()
                processes.append(p)                           
            for process in processes:
                process.join()
        # Convert manager shareable list to regular list (necessary before exiting 'with' block)
        data_ = list(data_)
        sleep(20) # Prevents errors at the end of computation

    return data_
    

############################################################################
def make_simulations_from_sampled_params(outfolder, bg_mean_range, bg_var_range, blob_intensity_mean_range, 
    blob_intensity_var_range, blob_radius_mean_range, blob_radius_var_range, 
    blob_number_range, num_sims, num_replicates, z_ij_ratio=2, zdim=20, idim=200, jdim=200, 
    nuc_spacing=100, nuc_rad=50):
    """Simulate blobs with a range of parameters, return simulations.
    
    Args:
        outfolder: folder
            Folder in which to write simulation files.
        bg_mean_range: tuple of numeric
            Lower and upper range values for background mean
        bg_var_range= tuple of numeric
            Lower and upper range values for background variance
        blob_intensity_mean_range: tuple of numeric
            Lower and upper range values for intensity mean
        blob_intensity_var_range: tuple of numeric
            Lower and upper range values for blob intensity variance
        blob_radius_mean_range: tuple of numeric
            Lower and upper range values for blob radius mean
        blob_radius_var_range: tuple of numeric
            Lower and upper range values for blob radius variance
        blob_number_range: tuple of ints
            Lower and upper range values for number of blobs per nucleus
        num_sims: int
            Number of parameter combinations to simulate
        num_replicates: int
            Number of simulations to perform for each parameter set
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
        Pickled ndarrays representing simulated image stacks. Parameters 
        (with replicate number) are separated by _ in filenames.
    """
    def sim_and_save(mask, bg_mean, bg_var, blob_intensity_mean, 
        blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
        z_ij_ratio, rep_num):
        """Simulate stack, save to disk.
        (This is the packet that is parallelized)
        """
        simstack = simulate_blobs(mask, bg_mean, bg_var, blob_intensity_mean, 
            blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
            z_ij_ratio)
        params = [bg_mean, bg_var, blob_intensity_mean, blob_intensity_var, blob_radius_mean, 
            blob_radius_var, blob_number]
        param_string = '_'.join([str(round(x, 2)) for x in params])
        filename = os.path.join(outfolder, param_string + '_' + str(rep_num) + '.pkl')
        save_pickle(simstack, filename)

    mask = make_dummy_mask(zdim, idim, jdim, nuc_spacing, nuc_rad)
    args = []
    # Populate args with parameter sets drawn randomly from supplied ranges.
    for sim_num in range(num_sims):
        bg_mean = randomize_ab(bg_mean_range)
        bg_var = randomize_ab(bg_var_range)
        blob_intensity_mean = randomize_ab(blob_intensity_mean_range)
        blob_intensity_var = randomize_ab(blob_intensity_var_range)
        blob_radius_mean = randomize_ab(blob_radius_mean_range)
        blob_radius_var = randomize_ab(blob_radius_var_range)
        blob_number = np.random.randint(blob_number_range[0], blob_number_range[1])
        for rep_num in range(num_replicates):
            args.append([mask, bg_mean, bg_var, blob_intensity_mean, 
                blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
                z_ij_ratio, rep_num])

    multiprocessing.set_start_method('fork', force=True) # Important for macOS.
    random.shuffle(args) # Unecessary, but helps alleviate potential concerns with random number generation.
    batch_size = 1000
    for i in range(0, len(args), batch_size):
        processes = []
        start = i
        end = min(start + batch_size, len(args))
        for j in range(start, end):
            p = multiprocessing.Process(target=sim_and_save, args=args[j])
            p.start() 
                               
############################################################################
def make_scalespace_2dhist(stack, sigmas, mask, numbins=100, 
    histrange=(0,66000), flatten=False):
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
        flatten: bool
            If true, return flattened array
    
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
    if flatten:
        return hist_.flatten()
    return hist_

def make_scalespace_2dhist_flattened(stack, sigmas, mask, numbins=100, 
    histrange=(0,66000)):
    """Wrapper for make_scalespace_2dhist with flattened toggled on."""
    return make_scalespace_2dhist(stack, sigmas, mask, numbins, histrange, 
    flatten=True)

############################################################################
def sims_to_data(folder, mask, width, t_function, **kwargs):
    """Take a folder of pickled simulation data, apply a transform
    function, return transformed data and parameters.
    
    Args:
        folder: path
            Folder containing pickled simulation output
        mask: ndarray
            Mask specifying regions of simulated images to use
        width: int
            Width of output data array 
        t_function: function
            Function to transform simulation data. Output is
            1D numpy array of length equal to width (above)
        **kwargs:
            kwargs for transfer function
        
    Returns:
        Tuple data, params
        data: ndarray
            Output data from transfer function, each row is data
            for a different simulation
        params: ndarray
            Parameters used for simulation, rows correspond to rows
            in data
    """
    def functionX(file, folder, mask, list, t_function, **kwargs):
        f_base = os.path.splitext(file)[0]
        params = [float(x) for x in f_base.split('_')]
        # Get data from simulated stack.
        stack = load_pickle(os.path.join(folder, file))
        list.append((params, t_function(stack=stack, mask=mask, **kwargs)))

    multiprocessing.set_start_method('fork') # On MacOS, pythons after 3.8 use 'spawn' by default, causes errors
    files = listdir_nohidden(folder)
    manager = multiprocessing.Manager()
    processes = []
    args = []
    with multiprocessing.Manager() as manager:
        data = manager.list()
        batch_size = 200
        for i in range(0, len(files), batch_size):
            processes = []
            start = i
            end = min(start + batch_size, len(files))
            for j in range(start, end):
                file = files[j]
                p = multiprocessing.Process(target=functionX, args=[file, folder, mask, data, t_function], kwargs=kwargs)
                p.start()
                processes.append(p)                           
            for process in processes:
                process.join()

        data = list(data)
        sleep(20) # Prevents errors at the end of computation
    
    return data
