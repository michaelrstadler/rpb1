#!/usr/bin/env python

"""
Insert description here.

"""
__version__ = '1.2.0'
__author__ = 'Michael Stadler'


import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re
from skimage import filters, io
from ipywidgets import interact, IntSlider, Dropdown, IntRangeSlider, fixed
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from functools import partial
import skimage as ski
from skimage.filters.thresholding import threshold_li, threshold_otsu
from skimage.segmentation import flood_fill, watershed
from scipy.stats import mode
from skimage.measure import label, regionprops
from scipy.spatial import distance
import pickle
import czifile
import pandas as pd
# Bug in skimage: skimage doesn't bring modules with it in some environments.
# Importing directly from submodules (above) gets around this.

# Import my packages.
import sys
sys.path.append('/Users/MStadler/Bioinformatics/Projects/Zelda/Quarantine_analysis/bin')
from fitting import fitgaussian3d, gaussian3d

############################################################################
# Classes
############################################################################
class movie():
    # Class attributes    
    # Initializer
    def make_spot_table(self, colnum):
        """"""
        nframes = self.nucmask.shape[0]
        data = {}
        for spot in self.spot_data:
            arr = self.spot_data[spot]
            vals = np.empty(nframes)
            vals.fill(np.nan)
            for i in range(0, len(arr)):
                t = int(arr[i,0])
                val = arr[i,colnum]
                vals[t] = val
            data[spot] = vals
        return pd.DataFrame(data)
    
    def __init__(self, stack, nucmask, fits, spot_data):
        self.stack = stack
        self.nucmask = nucmask
        self.fits = fits
        self.spot_data = spot_data
        self.intvol = self.make_spot_table(9)
        self.intfit = self.make_spot_table(10)
        self.prot = self.make_spot_table(11)

############################################################################
# General image processing functions
############################################################################

def labelmask_filter_objsize(labelmask, size_min, size_max):
    """Filter objects in a labelmask by size.

    Args:
        labelmask: ndarray
            Integer labelmask
        size_min: int
            Minimum size, in pixels, of objects to retain
        size_max: int
            Maximum size, in pixels, of objects to retain

    Returns:
        labelmask_filtered: ndarray
            Input labelmask with all pixels not corresponding to objects
            within size range set to 0.
    """
    # Count pixels in each object.
    (labels, counts) = np.unique(labelmask, return_counts=True)
    # Select and keep only objects within size range.
    labels_selected = labels[(counts >= size_min) & (counts <= size_max)]
    labelmask_filtered = np.where(np.isin(labelmask, labels_selected), labelmask, 0)
    return labelmask_filtered

############################################################################
def imfill(mask, seed_pt='default', min_fraction_filled=0.1):
    '''Fill holes within objects in a binary mask.
    
    Equivalent to matlab's imfill function. seed_pt needs to be a point in 
    the "background" area. All 0 or False pixels directly contiguous with 
    the seed are defined as background, all other pixels are declared foreground.
    Thus any "holes" (0 pixels that are not contiguous with background) are 
    filled in. Conceptually, this is like using the "fill" function in classic
    paint programs to fill the background, and taking all non-background as 
    foreground.
    
    Args:
        mask: ndarray
            Binary mask of n dimensions.
        seed_pt: tuple
            Pixel in mask to use for seeding "background". This is the equi-
            valent of the point you click when filling in paint. Default
            behavior is to select a random 0-valued point under the constraint
            that the resulting fill covers a fraction of the total pixels
            given by min_fraction_filled.
        min_fraction_filled: numeric
            Fraction of total pixels in the mask that must be filled for a 
            randomly chosen seed point (from 'default' setting) to be 
            accepted. 
    Returns:
        mask_filled: ndarray
            Binary mask filled    
    '''
    if (seed_pt == 'default'):
        # Try 20 random seeds.
        for i in range(0,20):
            # Get a random 0-valued pixel.
            seed_pt = find_background_point(mask)
            # Try flooding background from this point.
            mask_flooded = flood_fill(mask, seed_pt,1)
            fraction_filled = np.count_nonzero(mask != mask_flooded) / mask.size
            if (fraction_filled >= min_fraction_filled):
                mask_filled = np.where(mask == mask_flooded, 1, 0)
                return mask_filled
        raise Error('Cannot find a seed point that satisfies minimum fraction filled')
    else:
        mask_flooded = flood_fill(mask, seed_pt,1)
        mask_filled = np.where(mask == mask_flooded, 1, 0)
        return mask_filled

############################################################################
def local_max(img, size=(70,100,100)):
    """Find local maxima pixels in an image stack within a given window.
    
    Defines local maxima as pixels whose value is equal to the maximum value
    within a defined window centered on that point. Implementation is to
    first run a maximum filter, then define pixels in the original image 
    whose value is equal to the max-filter result at the same positions as
    local maximum pixels. Returns a binary mask of such pixels.
    
    Args:
        img: ndarray
            Image stack
        size: tuple of ints
            Size of the window for finding local maxima. The sizes are the
            dimensions of the filter used to search for maxima. So a size
            of (100, 100) will use a square with side lengths of 100 pixels.
            Generally, you want the size dimensions to match the dimensions
            of the objects you're searching for.
    
    Returns:
        local_max: ndarray
            A binary mask with dimensions equal to img of pixels whose value 
            is equal to the local maximum value. 
    """
    # Apply a maximum filter.
    max_f = ndi.maximum_filter(img, size=size)
    # Find pixels that are local maxima.
    local_max = np.where(max_f == img, 1, 0)
    return(local_max)

############################################################################
def peak_local_max_nD(img, size=(70,100,100), min_dist=0):
    """Find local maxima in an N-dimensional image.
    
    Generalizes scikit's peak_local_max function to three (or more) 
    dimensions. Finds local maxima pixels within supplied window, determines
    centroids for connected pixels, and returns a mask of these centroid
    positions and a list of them.
    
    Suggested usage: finding seed points for watershed segmentation from 
    distance transform. It is necessary because there are often multiple
    pixels with the same distance value, leaving little clusters of connected
    pixels. For circular objects (nuclei), distance transforms will form
    nice connected local max clusters. For less circular nuclei/objects,
    sometimes multiple unconnected clusters occur within an object. This
    is the reason for adding the minimum distance function.
    
    Args:
        img: ndarray
            N-dimensional image stack
        size: tuple of ints
            Size of the window for finding local maxima. The sizes are the
            dimensions of the filter used to search for maxima. So a size
            of (100, 100) will use a square with side lengths of 100 pixels.
            Generally, you want the size dimensions to match the dimensions
            of the objects you're searching for.
        min_dist: numeric
            Minimum (euclidean) distance in pixels allowed between peaks. If
            two peaks are within the minimum distance, the numerically lower
            peak (arbitrary) wins. 
    
    Returns:
        tuple: (local_peak_mask, local_peaks)
        local_peak_mask: ndarray
            A labelmask with dimensions equal to img of single labeled 
            pixels representing local maxima.
        local_peaks: list of tuples
            Coordinates of pixels masked in local_peak_mask  
    """
    def has_neighbor(peak, peak_list, min_dist):
        """Find whether a peak already exists within minimum distance of this peak"""
        for testpeak in peak_list:
            if (distance.euclidean(peak, testpeak) < min_dist):
                return True
        return False
    # Find pixels that represent local maxima. Produces clusters of connected
    # pixels at the centers of objects.
    maxes = local_max(img, size)
    # Connect these pixels in a labelmask.
    conn_comp, info = ndi.label(maxes)
    # Get the centroids of each local max object, update mask and list.
    local_peak_mask = np.zeros_like(img)
    local_peaks = []
    peak_num=1
    for id_ in np.unique(conn_comp)[1:]:
        centroid = get_object_centroid(conn_comp, id_)
        # If there is no already-added seed within the minimum distance,
        # add this seed to the mask and list.
        if (not has_neighbor(centroid, local_peaks, min_dist)):
            local_peak_mask[centroid] = peak_num
            local_peaks.append(centroid)
            peak_num = peak_num + 1
    return local_peak_mask, local_peaks

############################################################################
def get_object_centroid(labelmask, id):
    """Find the centroid of an object in a labelmask.
    
    Args:
        labelmask: ndarray
            Labelmask of arbitrary dimensions
        id: int
            Label of object to find centroid for
            
    Returns:
        centroid: tuple of ints
            Coordinates of the object's centroid
    """
    # Get coordinates 
    coords = np.where(labelmask == id)
    # Find mean of each coordinate, remove negatives, make int.
    return tuple([int(np.mean(x)) for x in coords])

############################################################################
def get_object_centroid(labelmask, id):
    """Find the centroid of an object in a labelmask.
    
    Args:
        labelmask: ndarray
            Labelmask of arbitrary dimensions
        id: int
            Label of object to find centroid for
            
    Returns:
        centroid: tuple of ints
            Coordinates of the object's centroid
    """
    # Get coordinates 
    coords = np.where(labelmask == id)
    # Find mean of each coordinate, remove negatives, make int.
    return tuple([int(np.mean(x)) for x in coords])

############################################################################
def labelmask_apply_morphology(labelmask, mfunc, struct=np.ones((2,2,2)), 
                               expand_size=(1,1,1), **kwargs):
    """Apply morphological functions to a labelmask.
    
    Args:
        labelmask: ndarray
            N-dimensional integer labelmask
        mfunc: python function
            Function from scikit.ndimage.morphology module
        struct: ndarray
            Structuring element for morpholocial operation
        expand_size: tuple of ints
            Size in N dimensions for maximum filter to produce label lookup
            mask (see details below)
        **kwargs: keyword arguments
            Keyword arguments supplied to morphological function
            
    Returns:
        new_labelmask: ndarray
            Updated labelmask matching shape of labelmask
    
    This is an imperfect solution to applying morphological operations to 
    labelmasks, so one needs to be a bit careful. The basic strategy is to
    binarize the mask, perform operations on it, and then from that mask 
    look up labels in the previous mask. This is better than re-labeling 
    because it allows the operations to produce objects that touch without 
    merging them. The issue is looking up the labels, which seems to be non-
    trivial. 
    
    The solution here is to generate a "lookup mask" by applying a maximum 
    filter (size determined by expand_size) to the labelmask, which expands 
    each object into its local area. As long as resulting morphological 
    operations keep the object within this area, they'll get the proper label. 
    As long as objects in the original image are spaced farther than the 
    supplied sizes in the three dimensions, this will work perfectly well. If 
    this isn't true, the object with the numerically larger label (arbitrary) 
    will expand at the expense of its neighbor. Of note, this maximum filter is 
    mathematially identical to morpholocial dilation when their is non conflict 
    between objects.
    
    For labelmasks with well-spaced objects, the function works as expected. 
    For closely spaced objects, one needs to select an expand_size that will
    generally be less than or equal to the object separation. For most
    applications, conflicts at edges won't be of great consequence but should
    be kept in mind.
    
    Suggested settings:
    
        -For operations that reduce the size of objects, leave expand_size 
        at (1,1,1), as resulting objects will be entirely contained within
        original objects.
        
        -For operations that increase the size of objects, set expand_size 
        to be equal to or slightly greater than expected increases in object
        size, and no more than the typical separation between objects.
        
    Examples:
    
    Dilation: 
        labelmask_apply_morphology(mylabelmask, ndi.morphology.binary_dilation,
        struct=np.ones((1,7,7)), expand_size=(1,8,8))
    
    Erosion:
       labelmask_apply_morphology(mylabelmask, ndi.morphology.binary_erosion,
        struct=np.ones((1,7,7))) 
        
    
    """
    # Expand the objects in the original mask to provide a "lookup" mask for
    # matching new objects to labels.
    lookupmask = ndi.maximum_filter(labelmask, expand_size)
    
    # Perform morphological operation on binarized labelmask.
    new_binmask = mfunc(labelmask, struct, **kwargs)
    
    # Match labels in new mask to those of lookup mask.
    new_labelmask = np.where(new_binmask, lookupmask, 0)
    return new_labelmask

############################################################################
def mesh_like(arr, n):
    """Make mesh grid for last n dimensions of an array
    
    Makes a meshgrid with the same shape as the last n dimensions of input
    array-like object.
    
    Args:
        arr: array-like
            Array-like object that has a shape parameter
        n: int
            Number of dimensions, from the right, for which to make meshgrid.
    
    Returns:
        meshes: list of ndarrays
            Each element of list corresponds to ordered dimension of input,
            ndarrays are corresponding meshgrids of same shape as arr.
    """
    if (n > arr.ndim):
        raise ValueError('n is larger than the dimension of the array')
    # Make vectors of linear ranges for each dimension.
    vectors = []
    for i in reversed(range(1, n+1)):
        a = np.arange(0, arr.shape[-i])
        vectors.append(list(a))
    # Make meshgrids from vectors.
    meshes = np.meshgrid(*vectors, sparse=False, indexing='ij')
    return meshes

############################################################################
def find_background_point(mask):
    """Find a background (value=0) pixel in a mask.
    
    Searches for a background pixel in a mask, defined as a pixel with 
    value 0. Background pixel is chosen at random from among all 0 pixels.

    Args:
        mask: ndarray
            Mask of abritrary dimensions, background must be 0.

    Returns: 
        coord: tuple of ints
            Coordinates of a single background pixel.
    """
    zerocoords = np.where(mask == 0)
    i = np.random.randint(0,len(zerocoords[0]))
    coord = zerocoords[0][i]
    for n in range(1, len(zerocoords)):
        coord = np.append(coord, zerocoords[n][i])
    return tuple(coord)  

############################################################################
def relabel_labelmask(labelmask, preserve_order=True):
    """Relabel labelmask to set background to 0 and object IDs to be linearly 
    ascending from 1. 
    
    Args:
        labelmask: ndarray
            N-dimensional labelmask.
        preserve_order: bool
            If true, order of labels in original mask is maintained (except
            background). Otherwise, labels will be in descending order of
            object size.
    
    Returns:
        labelmask: ndarray
            Labelmask of same shape as input, with the largest object's (
            background) ID set to 0 and all other objects labeled 1..n
    
    """
    mask = np.copy(labelmask)
    # Get all object labels and their counts.
    labels, counts = np.unique(mask, return_counts=True)
    # Get the indexes of sorted counts, descending.
    ordered_indexes = np.argsort(counts)[::-1]
    # Set largest object as background (ID=0).
    background_label = labels[ordered_indexes[0]]
    mask[mask == background_label] = 0
    # Renumber the rest of the objects 1..n.
    obj_num=1
    if (preserve_order):
        oldlabels = labels
    else:
        oldlabels = labels[ordered_indexes]
    for old_label in oldlabels:
        if (old_label != background_label):
            mask[labelmask == old_label] = obj_num
            obj_num = obj_num + 1
    return mask

############################################################################
def sortfreq(x, descending=True):
    """Sort the items of a list by the frequency with which they occur
    
    Args:
        x: list-like
            List-like object to sort
        descending: bool
            If true, most frequent list item appears first.
    
    Returns:
        items_sorted: array
            List of items from original array sorted by frequency  
    """
    # Get unique items from list and their frequencies of occurence (counts).
    items, counts = np.unique(x, return_counts=True)
    # Get ordered indexes by sorting counts.
    if (descending):
        ordered_indexes = np.argsort(counts)[::-1]
    else:
        ordered_indexes = np.argsort(counts)
    # Sort item list by ordered indexes.
    return items[ordered_indexes]

############################################################################
# Function implementing filters
############################################################################

def gradient_nD(stack):
    """Find the gradient of an n-dimensional image.
    
    Approximates an nD (typically: 3D) gradient by applying a gradient filter
    separately on each axis and taking the root of the sum of their squares.

    Args:
        stack: ndarray
            Image stack in [z, x, y] or [x, y]
            
    Returns:
        gradient: ndarray
            Gradient transform of image in same shape as stack
    """
    # Convert for 64-bit to avoid large number problems in squares.
    stack = np.copy(stack)
    stack = stack.astype(np.float64)
    sumsq = ndi.filters.sobel(stack, axis=0) ** 2
    for d in range(1, stack.ndim):
         sumsq = sumsq + (ndi.filters.sobel(stack, axis=d) ** 2)
    gradient = np.sqrt(sumsq)
    return gradient

############################################################################
def dog_filter(stack, sigma_big, sigma_small):
    """Difference of Gaussians filter
    
    Args:
        stack: ndarray
            n-dimensional image stack
        sigma_big: int
            Larger sigma for gaussian filter
        sigma_small: int
            Smaller sigma for gaussian filter
    
    Returns:
        dog: ndarray
            DoG filtered stack of same shape as input stack.
    """
    stack_cp = stack.astype(np.int16)
    return ndi.filters.gaussian_filter(stack_cp, sigma=sigma_big) - ndi.filters.gaussian_filter(stack_cp, sigma=sigma_small)

############################################################################
def log_filter(stack, sigma):
    """Laplacian of Gaussian filter
    
    Args:
        stack: ndarray
            n-dimensional image stack
        sigma: int
            Sigma for gaussian filter
    
    Returns:
        log: ndarray
            LoG filtered stack of same shape as input stack.
    """
    stack_cp = stack.astype(np.int16)
    gauss = ndi.filters.gaussian_filter(stack_cp, sigma=sigma)
    log = ndi.filters.laplace(gauss)
    return log

############################################################################
# Functions for loading reading and writing data
############################################################################

# Main function for loading TIFF stacks
def _read_tiff_stack(tif_folder, tif_files, **kwargs):
    """Read a folder of 2D or 3D TIFF files into a numpy ndarray.
    
    Args:
        tif_folder: string
            Directory containing multiple TIFF files.
        tif_files: list
            List of files in the folder to load. Must be sorted in order
            desired.
        span: tuple of ints
            Optional key-word argument specifying first and last file to 
            load, both inclusive. Example: span=(0, 5) loads the first 6 
            images, numbers 0 through 5.
    
    Returns:
        stack: ndarray
            n-dimensional image stack with the new dimension (file number) 
            in the 0 position(file_num, z, x, y) for 3D stacks, (filenum, x,
            y) for 2D stacks
            
    Raises:
        ValueError: 
            If dimensions of TIFF file don't match those of the first
            file
    """
    if 'span' in kwargs:
        first, last = (kwargs['span'])
        if (first <= last) and (last < len(tif_files)):
            tif_files = tif_files[first:(last + 1)]
        else:
            raise ValueError('Span exceeds the dimensions of the stack')

    # Create stack with dimensions from first file.
    img = io.imread(join(tif_folder, tif_files[0]))
    dims = img.shape
    num_files = len(tif_files)
    stack = np.ndarray(((num_files,) + dims), dtype=img.dtype)
    stack[0] = img
    img_num = 1
    
    # Add the rest of the files to the stack.
    for tif_file in tif_files[1:]:
        # Add image data to ndarray
        img = io.imread(join(tif_folder, tif_file))
        # Check dimensions
        if not stack[0].shape == img.shape:
            raise ValueError(f'Dimensions do not match previous files: {tif_file}')
        stack[img_num] = img
        img_num = img_num + 1
        
    return stack

# Wrapper function for loading all TIFF files in a folder
def read_tiff_folder(tif_folder):
    """Read all TIFF files in a folder into an ndarray.
    
        Args:
            tif_folder: string
                Directory containing multiple TIFF files. Must be sortable
                asciibetically.
            span: tuple of ints
                Optional key-word argument specifying first and last file to 
                load, both inclusive. Example: span=(0, 5) loads the first 6 
                images, numbers 0 through 5.
        
        Returns:
            stack: ndarray
                n-dimensional image stack with the new dimension (file number) 
                in the 0 position(file_num, z, x, y) for 3D stacks, (filenum, 
                x, y) for 2D stacks
                
        Raises:
            ValueError: 
                If dimensions of TIFF file don't match those of the first
                file
    """
    
    # Compile files that are files and have .tif extension (case-insensitive).
    tif_files = [f for f in listdir(tif_folder) if (isfile(join(tif_folder, f)) 
        and (os.path.splitext(f)[1][0:4].upper() == '.TIF'))]
    # Sort the files: asciibetical sorting produces files ascending in time 
    # (sorting is *in place*)
    tif_files.sort()
    return _read_tiff_stack(tif_folder, tif_files)

# Wrapper function for loading TIFF files in a lattice-style folder, with
# CamA and CamB channels.
def read_tiff_lattice(tif_folder, **kwargs):
    """Read all TIFF files in a lattice output folder into an ndarray.
    
        Args:
            tif_folder: string
                Directory containing multiple TIFF files with 'CamA' and 'CamB' 
                Must be equal numbers of CamA and CamB and files must be 
                sortable asciibetically.
            span: tuple of ints
                Optional key-word argument specifying first and last file to 
                load, both inclusive. Example: span=(0, 5) loads the first 6 
                images, numbers 0 through 5.
        
        Returns:
            stack: ndarray
                n-dimensional image stack with the new dimension (channel) 
                in the 0 position, e.g. (channel, t, z, x, y) for 3D stacks. 
                
        Raises:
            ValueError: 
                If dimensions of TIFF file don't match those of the first file
            ValueError: 
                If there are non-identical numbers of CamA and CamB files
    """
    
    # Compile files that are files and have .tif extension (case-insensitive).
    tif_files = [f for f in listdir(tif_folder) if (isfile(join(tif_folder, f)) 
        and (os.path.splitext(f)[1][0:4].upper() == '.TIF'))]
    # Sort files into two lists based on containing 'CamA' and 'CamB' in filename.
    regex_camA = re.compile('CamA')
    regex_camB = re.compile('CamB')
    camA_files = [*filter(regex_camA.search, tif_files)] # This syntax unpacks filter into a list.
    camB_files = [*filter(regex_camB.search, tif_files)]

    # Sort the files: asciibetical sorting produces files ascending in time 
    # (sorting is *in place*)
    camA_files.sort()
    camB_files.sort()
    # Read both sets of files, combine if they are of same dimensions.
    camA_stack = _read_tiff_stack(tif_folder, camA_files, **kwargs)
    camB_stack = _read_tiff_stack(tif_folder, camB_files, **kwargs)
    if camA_stack.shape == camB_stack.shape:
        return np.stack((camA_stack, camB_stack), axis=0)
    else:
        raise ValueError('Unequal number of CamA and CamB files.')

############################################################################
def read_czi(filename, trim=False, swapaxes=True):
    """Read a czi file into an ndarray
    
    Args:
        filename: string
            Path to czi file
        trim: bool
            If true, remove last frame
        swapaxes: bool
            If true, switches first two axes to produce a stack order ctzxy
            
    Returns:
        stack: ndarray
            Image stack in dimensions [t,c,z,x,y] (no swap) or 
            [c,t,z,x,y] (swapped)
    """
    stack = czifile.imread(filename)
    stack = np.squeeze(stack)
    # Trim off last frame 
    if trim:
        stack = stack[0:stack.shape[0]-1]
    if (swapaxes):
        stack = np.swapaxes(stack,0,1)
    return stack

############################################################################
def save_pickle(obj, filename):
    """Pickel (serialize) an object into a file

    Args:
        filename: string
            Full path to save to
        obj: object
            Python object to serialize
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

############################################################################
def load_pickle(filename):
    """Load a pickled (serialized) object

    Args:
        filename: string
            Full path containing pickled object
    
    Returns:
        obj: object
            Object(s) contained in pickled file
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj

############################################################################
# Functions for interactive image viewing/analysis
############################################################################

def viewer(stacks, order='default', figsize=6):
    """Interactive Jupyter notebook viewer for n-dimensional image stacks.
    
    Args:
        stack: list of ndarrays
            List of n-dimensional image stacks; last two dimensions must 
            be x-y to display. Image shapes must be identical.
        order: string
            String specifying order of image dimensions. Examples: 'ctzxy' 
            or 'tzxy'. Last two dimensions must be 'xy'.
            
    Returns: none
        
    Raises:
        ValueError: 
            If final two dimensions are not xy
        ValueError:
            If the number of dimensions in the order string does not match
            the dimensions of the stack
    """
    # Update the displayed image with inputs from widgets.
    def _update_view(order, **kwargs):
        numplots = len(stacks)
        indexes = []
        colmap = kwargs['colmap']
        min_ = kwargs['contrast'][0]
        max_ = kwargs['contrast'][1]
        
        # Unpack order variable into array.
        order_arr = [char for char in order[:-2]]
        # Populate indexes list with widgets.
        for n in order_arr:
            indexes.append(kwargs[n])
        
        # Set up frame for plots.
        fig, ax = plt.subplots(1, numplots, figsize=(figsize * numplots, figsize * numplots))
        # If only one plot, pack ax into list
        if (type(ax) is not np.ndarray):
            ax = [ax]
        for n in range(0, numplots):
            stack_local = stacks[n]
            # Slice stack, leaving last two dimensions for image.
            # Note: the (...,) in the following is not required, but I think 
            # it is clarifying.
            ax[n].imshow(stack_local[tuple(indexes) + (...,)], cmap=colmap, vmin=min_, 
            vmax=max_);    
    
    # Make a new slider object for dimension selection and return it
    def _make_slider(n):
        widget = IntSlider(min=0, max=(stack.shape[n] - 1), step=1, 
            continuous_update=False,)
        return(widget)
    
    # Make a dropdown widget for selecting the colormap.
    def _make_cmap_dropdown():
        dropdown = Dropdown(
            options={'viridis', 'plasma', 'magma', 'inferno','cividis',
                'Greens', 'Reds', 'gray', 'gray_r', 'prism'},
            value='viridis',
            description='Color',
        )
        return dropdown
    
    # Make a range slider for adjusting image contrast.
    def _make_constrast_slider():
        min_ = stack.min()
        max_ = stack.max()
        contrast_slider = IntRangeSlider(
            value=[min_, max_],
            min=min_,
            max=max_,
            step=1,
            description='Contrast',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )
        return contrast_slider
    
    def main(order):
        if not (order[-2:] == 'xy'):
            raise ValueError("Dimension order must end in 'xy'.")
        if not (len(order) == len(stack.shape)):
            raise ValueError("Order string doesn't match dimensions of stack")
        # Split order string (minus trailing 'xy') into list.
        order_arr = [char for char in order[:-2]]
        
        interact_call = {} 
        #interact_call['order'] = order # passing arrays through interact no good. 
        interact_call['colmap'] = _make_cmap_dropdown()
        interact_call['contrast'] = _make_constrast_slider()
               
        # Build call to interact by appending widgets for all leading dimensions.   
        for n in range(0, len(order_arr)):
            func = _make_slider(n)
            interact_call[order_arr[n]] = func

        # Make color and contrast widgets.
        interact(_update_view, order=fixed(order), **interact_call)
    # Use first stack as reference for sizes, etc.
    if (type(stacks) is list):
        stack = stacks[0]
    else:
        stack = stacks
        stacks = [stack]
    if (order == 'default'):
            order = 'ctzxy'[5-stacks[0].ndim:]
    main(order)

############################################################################
def qax(n, ncol=4):
    """Quick axes: generate 1d list of axes objects of specified number
    
    Args:
        n: int
            Number of plots desired
            
    Returns:
        ax1d: list
            1D list of axes objects in order top left to bottom right (columns
            then rows)
    """
    nrow = int(np.ceil(n / ncol))
    if (n < ncol):
        ncol = n
    fig, ax = plt.subplots(nrow, ncol, figsize=(16, 4*nrow))
    ax1d = []
    pos1d = 0
    if (nrow > 1):
        for r in range(0, nrow):
            for c in range(0, ncol):
                ax1d.append(ax[r][c])
                pos1d = pos1d + 1
    else:
        for c in range(0, ncol):
            ax1d.append(ax[c])
            pos1d = pos1d + 1
    
    return ax1d

############################################################################
def plot_ps(func, span=range(0,8)):
    """Plot a parameter series in a specified range
    
    User supplies a plotting function that takes a single integer input as
    a parameter. plot_ps builds axes to display all parameter values and
    serially calls plot function on them.
    
    Example:
       def temp(x):
            dog = dog_filter(red, x, 3)
            plt.imshow(dog)

        plot_ps(temp, range(8,13)) 
    
    Args:
        func: function
            Must take a single integer value as a parameter and call a plot
            function on the active axes object.
        span: range
            Range object containing values of parameter to plot. 
    """
    nplots = len(span)
    ax = qax(int(len(span)))
    for pln in range(0, len(span)):
        plt.sca(ax[pln])
        func(span[pln])

############################################################################
def box_spots(stack, spot_data, max_mult=1.3, halfwidth_xy=15, 
              halfwidth_z=8, linewidth=3, shadows=True):
    """Draw boxes around detected MS2 spots.
    
    Usage suggestions: Useful with a Z-projection to examine effectiveness
    of spot segmentation. Can also use with very small halfwidths to create
    artificial "dots" representing called spots to overlay on other data,
    e.g. a nuclear mask or a blank matrix of 0s (to examine spot movement
    alone).
    
    Args: 
        stack: ndarray of uint16
            Multi-dimensional image stack of dimensions [t,z,x,y]
        channel: int
            Channel containing MS2 spots
        spot_data: dict of ndarrays
            Data containing tracking of spots detected. Dict entries are unique 
            spot IDs (numeric 1...), rows of ndarray are detections of the spot 
            in a single frame. Time must be column 0, [z,x,y] in columns 2:4.
        max_multi: numeric
            Multiplier of maximum value to use for drawing box.
        halfwidth_xy: int
            Halfwidth in pixels of the boxes in xy direction (sides will be 
            2*halfwidth)
        halfwidth_z: int
            Halfwidth of the boxes in z direction(vertical sides will be 
            2*halfwidth)
        linewidth: int
            Width of lines used to draw boxes
        shadows: bool
            Draw "shadows" (dark boxes) in non-boxed z-slices.
        
    Return:
        boxstack: ndarray
            Selected channel of input image stack with boxes drawn around 
            spots. Dimensions [t,z,x,y]
    """
    if (stack.dtype != 'uint16'):
        raise ValueError("Stack must be uint16.")
    boxstack = np.copy(stack)
    hival = max_mult * boxstack.max()
    if (hival > 65535):
        hival = 65535
    
    def drawbox(boxstack, point, halfwidth_xy, halfwidth_z, linewidth, hival, shadows):
        t, z, i, j = point
        z_min = max(0, z - halfwidth_z)
        z_max = min(boxstack.shape[1], z + halfwidth_z + 1)
        i_min = max(0, i - halfwidth_xy)
        i_max = min(boxstack.shape[2], i + halfwidth_xy + 1)
        j_min = max(0, j - halfwidth_xy)
        j_max = min(boxstack.shape[3], j + halfwidth_xy + 1)
        if shadows:
            # Draw shadow boxes in all Z-frames.
            boxstack[t, :, i_min:i_max, j_min:(j_min + linewidth)] = 0
            boxstack[t, :, i_min:i_max, (j_max-linewidth):j_max] = 0
            boxstack[t, :, i_min:(i_min+linewidth), j_min:j_max] = 0
            boxstack[t, :, (i_max-linewidth):i_max, j_min:j_max] = 0
        # Draw left line.
        boxstack[t, z_min:z_max, i_min:i_max, j_min:(j_min + linewidth)] = hival     
        # Draw right line. 
        boxstack[t, z_min:z_max, i_min:i_max, (j_max-linewidth):j_max] = hival
        # Draw top line. 
        boxstack[t, z_min:z_max, i_min:(i_min+linewidth), j_min:j_max] = hival
        # Draw bottom line.
        boxstack[t, z_min:z_max, (i_max-linewidth):i_max, j_min:j_max] = hival
    
    # Main.
    for spot in spot_data:
        arr = spot_data[spot]
        for row in arr:
            row = row.astype(int)
            point = (row[[0,2,3,4]])
            drawbox(boxstack, point, halfwidth_xy, halfwidth_z, linewidth, hival, shadows)
    return boxstack   

############################################################################
def quickview_ms2(stack, spot_data, channel=0):
    """View image stack with boxes drawn around detected spots
    
    Args:
        stack: ndarray
            5d image stack [c,t,z,x,y]
        spot_data: dict of ndarrays
            Data containing tracking of spots detected. Dict entries are unique 
            spot IDs (numeric 1...), rows of ndarray are detections of the spot 
            in a single frame. Time must be column 0, [z,x,y] in columns 2:4.
        channel: int
            Channel (dimension 0) to be viewed
    """
    substack = stack[channel]
    boxes = box_spots(substack, spot_data, halfwidth_xy=6, linewidth=2)
    viewer(boxes.max(axis=1), 'txy', 15)

############################################################################
def spot_movies(stack, spot_data, channel=0, len_ij=15, len_z=7, fill=np.nan, view=True):
    """Make image stack for viewing MS2 spot raw data
    
    Given spot coordinates and original movie, builds an image stack with 
    spot ID on the first dimension and raw data centered on the spot in the
    remaining three. In cases where the window exceeds the boundaries of the
    image, the remaining portions of the frame will be filled by the value
    specified by fill.
    
    Args:
        stack: ndarray
            5-D image stack [c,t,z,x,y]
        spot_data: dict of ndarrays
            Data containing tracking of spots detected. Dict entries are unique 
            spot IDs (numeric 1...), rows of ndarray are detections of the spot 
            in a single frame. Required columns: 5: gaussian fit height, 6: 
            gaussian fit z-width, 7: gaussian fit x-width, 8: gaussian fit 
            y-width.
        channel: int
            Channel from which to use data
        len_ij: int
            Length (in pixels) of the box to collect around each spot in the 
            lateral (ij) dimension.
        len_z: int
            Length (in pixels) of the box to collect around each spot in the 
            axial (z) dimension.
        fill: numeric
            Value to fill in regions that exceed the boundaries of the image
        view: bool
            If true, calls viewer function on the resulting stack using a max 
            projection on Z dimension (order spot_ID-t-x-y)
            
    Returns:
        movies: ndarray
            5D image stack of input data centered around each spot. In order 
            [spot_id, t,z,x,y]
    """
    def adjust_coords(c, rad, frame_max, stack_max):
        """Adjusts coordinates to account for spots located at the edge of the 
        stack such that the window will cross the boundary of the image. 
        For a given axis, returns both the minimum and max coordinate
        for slicing the stack and the min and max coordinates for the position
        in the frame to which to assign this slice."""

        # Initialize coordinate and frame boundaries.
        cmin = int(c - rad)
        cmax = int(c + rad)
        frame_min = 0

        # Correct for cases where bounds of the stack are exceeded.
        if (cmin < 0):
            frame_min = -1 * cmin
            cmin = 0
        if (cmax > stack_max):
            frame_max = frame_max - (cmax - stack_max)
            cmax = stack_max
        return cmin, cmax, frame_min, frame_max

    # Check that window dimensions are appropriate.
    if ((len_ij % 2 == 0) or (len_z % 2 == 0)):
        raise ValuError('len_ij and len_z must be odd')
    # Define 'radii' for window (distance to go from home pixel in each direction).
    z_rad = int((len_z - 1) / 2)
    ij_rad = int((len_ij - 1) / 2)
    # Initialize movies as a zeros array.
    movies = np.zeros((len(spot_data)+1, stack.shape[1], len_z, len_ij, len_ij))
    # Add data for each spot.
    for spot in spot_data:
        arr = spot_data[spot]
        for row in arr:
            # Get t,z,x,y coordinates for spot.
            t = int(row[0])
            z,i,j = row[2:5]
            # Initialize frame.
            frame = np.empty((len_z, len_ij, len_ij))
            frame.fill(fill)
            # Get the min and max values for each coordinate with reference to the stack and the
            # coordinates in the frame that they will replace.
            zmin, zmax, frame_zmin, frame_zmax = adjust_coords(z, z_rad, len_z - 1, stack.shape[-3]-1)
            imin, imax, frame_imin, frame_imax = adjust_coords(i, ij_rad, len_ij - 1, stack.shape[-2]-1)
            jmin, jmax, frame_jmin, frame_jmax = adjust_coords(j, ij_rad, len_ij - 1, stack.shape[-1]-1)
            # Update frame from stack and assign to movies.
            frame[frame_zmin:(frame_zmax+1), frame_imin:(frame_imax+1), frame_jmin:(frame_jmax+1)] = stack[channel, t, zmin:(zmax+1), imin:(imax+1), jmin:(jmax+1)]
            movies[spot, t] = frame
    # If viewer call specified, call with mean (sum) Z-projection.
    if(view):
        #viewer(movies.mean(axis=2), 'itxy')
        viewer(np.nanmean(movies, axis=2), 'itxy')
    return movies

############################################################################    
############################################################################
# Functions for segmenting images
############################################################################
############################################################################

def segment_embryo(stack, channel=0, sigma=5, walkback = 50):
    """Segment the embryo from extra-embryo space in lattice data.
    
    Details: Crudely segments the embryo from extra-embryonic space in 5-
    dimensional stacks. The mean projection across all time slices is first
    taken, followed by performing a gaussian smoothing, then thresholding,
    then using morphological filtering to fill holes and then "walking
    back" from right-to-left, based on the observation that segmentation 
    tends to extend too far, and lattice images always have the sample on
    the left.
    
    Args:
        stack: ndarray
            Image stack in order [c, t, z, x, y]
        channel: int
            Channel to use for segmentation (channel definted as first
            dimension of the stack)
        sigma: int
            Sigma factor for gaussian smoothing
        walkback: int
            Length in pixels to "walk back" from right
            
    Returns:
        stack_masked: ndarray
            Input stack with masked (extra-embryo) positions set to 0
    """
    # Create a 3D mask from the mean projection of a 4D stack.
    def _make_mask(stack, channel, sigma, walkback):
        # Make a mean projection (on time axis) for desired channel. 
        im = stack[channel].mean(axis=0)
        # Smooth with gaussian kernel.
        im_smooth = ndi.filters.gaussian_filter(im, sigma=sigma)
        # Find threshold with minimum method.
        t = ski.filters.threshold_minimum(im_smooth)
        # Make binary mask with threshold.
        mask = np.where(im_smooth > t, im, 0)
        mask = mask.astype('bool')
        # Fill holes with morphological processing.
        mask = ndi.morphology.binary_fill_holes(mask, structure=np.ones((1,2,2)))
        # Build structure for "walking back" from right via morphological processing.
        struc = np.ones((1,1, walkback))
        midpoint = int(walkback / 2)
        struc[0, 0, 0:midpoint] = 0
        # Walk back mask from right.
        mask = ndi.morphology.binary_erosion(mask, structure=struc)
        return mask
    
    def main(stack, channel, sigma, walkback):
        mask = _make_mask(stack, channel, sigma, walkback)
        stack_masked = np.where(mask, stack, 0) # Broadcasting mask onto stack
        return(stack_masked)
    
    return main(stack, channel, sigma, walkback)

############################################################################
def labelmask_filter_objsize(labelmask, size_min, size_max):
    """Filter objects in a labelmask for size
    
    Args:
        labelmask: ndarray
            n-dimensional integer labelmask
        size_min: int
            Minimum size in total pixels, of the smallest object
        size_max: int
            Maximum size, in total pixels, of the largest object
    
    Return:
        labelmask_filtered: ndarray
            Labelmask of same shape as input mask, containing only objects
            between minimum and maximum sizes.
    """
    # Count pixels in each object.
    (labels, counts) = np.unique(labelmask, return_counts=True)
    # Select objects in desired size range, update filtered mask.
    labels_selected = labels[(counts >= size_min) & (counts <= size_max)]
    labelmask_filtered = np.where(np.isin(labelmask, labels_selected), 
        labelmask, 0)
    return labelmask_filtered

############################################################################
def object_circularity(labelmask, label):
    """Calculate circularity for and object in a labelmask
    
    Implements imageJ circularity measure: 4pi(Area)/(Perimeter^2).
    
    Args:
        labelmask: ndarray
            n-dimensional integer labelmask
        label: int
            ID of object for which to calculate circularity
            
    Return:
        circularity: float
            output of circularity calculation 
    """
    # Find z slice with most pixels from object.
    z, i, j = np.where(labelmask == label)
    zmax = mode(z)[0][0]
    # Select 2D image representing object's max Z-slice.
    im = np.where(labelmask[zmax] == label, 1, 0)
    # Calculate circularity from object perimeter and area.
    regions = regionprops(im)
    perimeter = regions[0].perimeter
    area = regions[0].area
    circularity = 4 * np.pi * area / (perimeter ** 2) 
    return circularity
    
############################################################################
def filter_labelmask(labelmask, func, above=0, below=1e6):
    """Filter objects from a labelmask based on object properties
    
    Applies a user-supplied function that returns a numeric value for an
    object in a labelmask, filters mask to contain only objects between
    minimum and maximum values.
    
    Args:
        labelmask: ndarray
            n-dimensional integer labelmask
        func: function
            Function that accepts a labelmask as its first argument, object
            ID as second argument, and returns a numeric value.
        above: numeric
            Lower limit for object's value returned from the function.
        below: numeric
            Upper limit for object's value returned from the function.
            
    Return:
        labelmask_filtered: ndarray
            Labelmask of same shape as input mask, containing only objects
            between minimum and maximum values from supplied function.
    """
    labels = []
    for x in np.unique(labelmask):
        prop = func(labelmask, x)
        if (prop >= above and prop <= below):
            labels.append(x)
    labelmask_filtered = np.where(np.isin(labelmask, labels), labelmask, 0)
    return labelmask_filtered

############################################################################
def zstack_normalize_mean(instack):
    """Normalize each Z-slice in a Z-stack to by dividing by its mean

    Args:
        instack: ndarray
            Image stack in order [z, x, y]

    Returns:
        stack: ndarray
            Image stack of same shape as instack.
    """
    stack = np.copy(instack)    
    stackmean = stack.mean()
    for x in range(0,stack.shape[0]):
        immean = stack[x].mean()
        stack[x] = stack[x] / immean * stackmean
    return(stack)

############################################################################
def stack_bgsub(stack, bgchannel=0, fgchannel=1):
    """Use one channel of image stack to background subtract a second channel.

    Built for 2-color lattice MS2 stacks. Observation is that low-frequency
    features in MS2 channel (typically red) are almost all shared background
    structures, particularly the embryo boundary. Subtraction is a very 
    effective method of removing this boundary and other non-specific signals.
    The mean projection in time of the background channel is used for
    subtraction.
    
    Args:
        stack: ndarray
            5D image stack of dimensions [c,t,z,x,y].
        bgchannel: int
            Channel to use for background (to be subtracted)
        fgchannel: int
            Channel to use for foreground (to be subtracted from)
    
    Returns:
        bgsub: ndarray
            Background-subtracted stack in same shape as input stack
    """
    # Generate background from mean projection in time.
    bg = stack[bgchannel].mean(axis=0)
    # Find scale factor to equalize mean intensities.
    scale = stack[fgchannel].mean() / bg.mean()
    # Subtract background (broadcast to whole array, in both channels)
    bgsub = stack - (scale * bg)
    # Set minimum value to 0 (remove negative values).
    bgsub = bgsub + abs(bgsub.min())
    return bgsub

############################################################################
def segment_nuclei3D_5(instack, sigma1=3, sigma_dog_small=5, sigma_dog_big=40, seed_window=(70,100,100),
                       erosion_length=5, dilation_length=10, sensitivity=0.5, size_min=1e4, 
                       size_max=5e5, circularity_min=0.5, display=False):
    """Segment nuclei from a 3D imaging stack
   
    Args:
        instack: ndarray
            3D image stack of dimensions [z, x, y].
        sigma1: int
            Sigma for initial Gaussian filter, for making initial mask
        sigma_dog_small: int
            Smaller sigma for DoG filter used as input to gradient for watershed
        sigma_dog_big: int
            Larger sigma for DoG filter used as input to gradient for watershed
        seed_window: tuple of three ints
            Size in [z, x, y] for window for determining local maxes in distance
            transform. Generally want size to be ~ size of nuclei.
        erosion_length: int
            Size in x and y of structuring element for erosion of initial mask.
        dilation_length: int
            Size in x and y of structuring element for dilating objects after
            final segmentation.
        size_min: int
            Minimum size, in pixels, of objects to retain
        size_max: int
            Maximum size, in pixels, of objects to retain
        circularity_min: float
            Minimum circularity measure of objects to retain
    
    Returns:
        labelmask: ndarray
            Mask of same shape as input stack with nuclei segmented and labeled
    
    """


    def smart_dilate(stack, labelmask, sensitivity, dilation_length):
        """
        Dilate nuclei, then apply a threshold to the newly-added pixels and
        only retains pixels that cross it. Change mask in place.
        """
        # Get mean pixel values of foreground and background and define threshold.
        bg_mean = np.mean(stack[labelmask == 0])
        fg_mean = np.mean(stack[labelmask > 0])
        t = bg_mean + ((fg_mean - bg_mean) * sensitivity)
        # Dilate labelmask, return as new mask.
        labelmask_dilated = labelmask_apply_morphology(labelmask, 
                mfunc=ndi.morphology.binary_dilation, 
                struct=np.ones((1, dilation_length, dilation_length)), 
                expand_size=(1, dilation_length + 1, dilation_length + 1))
        # Remove any pixels from dilated mask that are below threshhold.
        labelmask_dilated[stack < t] = 0
        # Add pixels matching nuc in dilated mask to old mask, pixels in old mask that are n
        # and 0 in dilated mask are kept at n. So dilation doesn't remove any nuclear pixels.
        for n in np.unique(labelmask)[1:]:
            if (n != 0):
                labelmask[labelmask_dilated == n] = n

    # Normalize each Z-slice to mean intensity to account for uneven illumination.
    stack = zstack_normalize_mean(instack)
    # Apply gaussian filter.
    stack_smooth = ndi.filters.gaussian_filter(stack, sigma=sigma1)
    # Threshold, make binary mask, fill.
    t = threshold_otsu(stack_smooth)
    mask = np.where(stack_smooth >= t, 1, 0)
    mask = imfill(mask, find_background_point(mask))
    # Use morphological erosion to remove spurious connections between objects.
    mask = ndi.morphology.binary_erosion(mask, structure=np.ones((1, erosion_length, erosion_length)))
    # Perform distance transform of mask.
    dist = ndi.distance_transform_edt(mask)
    # Find local maxima for watershed seeds.
    seeds, _ = peak_local_max_nD(dist, size=seed_window)
    # Add a background seed.
    seeds[find_background_point(mask)] = seeds.max() + 1
    # Re-smooth, do gradient transform to get substrate for watershedding.
    dog = dog_filter(stack, sigma_dog_small, sigma_dog_big)
    grad = gradient_nD(dog)
    # Remove nan from grad, replace with non-nan max values.
    grad[np.isnan(grad)] = grad[~np.isnan(grad)].max()
    # Segment by watershed algorithm.
    ws = watershed(grad, seeds.astype(int))
    # Filter nuclei for size and circularity.
    labelmask = labelmask_filter_objsize(ws, size_min, size_max)
    labelmask = filter_labelmask(labelmask, object_circularity, circularity_min, 1000)
    # Dilate labeled structures.
    smart_dilate(stack_smooth, labelmask, sensitivity, dilation_length)

    if (display):
        middle_slice = int(stack.shape[0] / 2)
        fig, ax = plt.subplots(3,2, figsize=(10,10))
        # Display mask.
        ax[0][0].imshow(mask.max(axis=0))
        ax[0][0].set_title('Initial Mask')
        # Display watershed seeds.
        seeds_vis = ndi.morphology.binary_dilation(seeds, structure=np.ones((1,8,8)))
        ax[0][1].imshow(stack_smooth.max(axis=0), alpha=0.5)
        ax[0][1].imshow(seeds_vis.max(axis=0), alpha=0.5)
        ax[0][1].set_title('Watershed seeds')
        # Display gradient.
        ax[1][0].imshow(grad[middle_slice])
        ax[1][0].set_title('Gradient')
        # Display watershed output.
        ax[1][1].imshow(ws.max(axis=0))
        ax[1][1].set_title('Watershed')
        # Display final mask.
        ax[2][0].imshow(labelmask.max(axis=0))
        ax[2][0].set_title('Final Segmentation')
        
    return labelmask

############################################################################
def segment_nuclei3D_monolayer(stack, sigma1=3, sigma_dog_big=15, 
        sigma_dog_small=5, seed_window=(30,30), min_seed_dist=25, 
        dilation_length=5, size_min=0, size_max=np.inf, display=False):
    """Segment nuclei from confocal nuclear monolayers
    
    Segment nuclei from nuclear monolayers, such as standard MS2 confocal
    stacks. Monolayers don't generally require 3D segmentation, so this
    function uses the max projection in Z to define the domain of each 
    nucleus in XY. 
    
    Args:
        stack: ndarray
            3D image stack of dimensions [z, x, y].
        sigma1: int
            Sigma for Gaussian smoothing used to make gradient input to watershed
        sigma_dog_small: int
            Smaller sigma for DoG filter used to create initial mask
        sigma_dog_big: int
            Larger sigma for DoG filter used to create initial mask
        seed_window: tuple of three ints
            Size in [z, x, y] for window for determining local maxes in distance
            transform. Generally want size to be ~ size of nuclei.
        min_seed_dist: numeric
            The minimum euclidean distance (in pixels) allowed between watershed
            seeds. Typically set as ~the diameter of the nuclei.
        size_min: int
            Minimum size, in pixels, of objects to retain
        size_max: int
            Maximum size, in pixels, of objects to retain
        dilation_length: int
            Size in x and y of structuring element for dilating objects after
            final segmentation.
        
    Returns:
        labelmask: ndarray
            2D labelmask of nuclei.
    """
    # Make max projection on Z.
    maxp = stack.max(axis=0)
    # Filter with DoG to make nuclei into blobs.
    dog = dog_filter(maxp, sigma_dog_small, sigma_dog_big)
    # Get threshold, use thresh to make initial mask and fill holes.
    t = threshold_otsu(dog)
    mask = np.where(dog > t, 1, 0)
    mask = imfill(mask)
    # Perform distance transform, find local maxima for watershed seeds.
    dist = ndi.distance_transform_edt(mask)
    seeds, _ = peak_local_max_nD(dist, size=seed_window, min_dist=min_seed_dist)
    # Smooth image and take gradient, use as input for watershed.
    im_smooth = ndi.filters.gaussian_filter(maxp, sigma=sigma1)
    grad = gradient_nD(im_smooth)
    ws = watershed(grad, seeds.astype(int))
    # Filter object size, relabel to set background to 0.
    labelmask = labelmask_filter_objsize(ws, size_min, size_max)
    labelmask = relabel_labelmask(labelmask)
    # Dilate segmented nuclei.
    labelmask = labelmask_apply_morphology(labelmask, 
                    mfunc=ndi.morphology.binary_dilation, 
                    struct=np.ones((dilation_length, dilation_length)), 
                    expand_size=(dilation_length + 1, dilation_length + 1))

    if (display):
        fig, ax = plt.subplots(3,2, figsize=(10,10))
        # Display mask.
        ax[0][0].imshow(mask)
        ax[0][0].set_title('Initial Mask')
        # Display watershed seeds.
        seeds_vis = ndi.morphology.binary_dilation(seeds, structure=np.ones((8,8)))
        ax[0][1].imshow(im_smooth, alpha=0.5)
        ax[0][1].imshow(seeds_vis, alpha=0.5)
        ax[0][1].set_title('Watershed seeds')
        # Display gradient.
        ax[1][0].imshow(grad)
        ax[1][0].set_title('Gradient')
        # Display watershed output.
        ws = relabel_labelmask(ws)
        ax[1][1].imshow(ws.astype('bool'))
        ax[1][1].set_title('Watershed')
        # Display final mask.
        ax[2][0].imshow(labelmask.astype('bool'))
        ax[2][0].set_title('Final Segmentation')
    
    # Make 2D labelmask into 3D mask by repeating.
    labelmask = np.repeat([labelmask], stack.shape[0], axis=0)
    return labelmask

############################################################################
def segment_nuclei3D_monolayer_rpb1(stack, sigma1=3, sigma_dog_big=15, 
        sigma_dog_small=5, seed_window=(30,30), min_seed_dist=25, 
        dilation_length=5, dilation_length_foci=10, size_min=0, 
        circularity_min=0, size_max=np.inf, display=False):
    """Segment nuclei from confocal nuclear monolayers based on Rpb1 signal
    
    Segment nuclei using Rpb1 signal. The segment_nuclei3D_monolayer function runs
    into problems because of the strong Rpb1 foci (presumed histone locus bodies)
    that create problems in the gradient used as input for watershed segmentation.
    This function has an additional masking step that segments these foci and masks
    them out of the gradient image, and also adds an object circularity filter.
    
    Args:
        stack: ndarray
            3D image stack of dimensions [z, x, y].
        sigma1: int
            Sigma for Gaussian smoothing used to make gradient input to watershed
        sigma_dog_small: int
            Smaller sigma for DoG filter used to create initial mask
        sigma_dog_big: int
            Larger sigma for DoG filter used to create initial mask
        seed_window: tuple of three ints
            Size in [z, x, y] for window for determining local maxes in distance
            transform. Generally want size to be ~ size of nuclei.
        min_seed_dist: numeric
            The minimum euclidean distance (in pixels) allowed between watershed
            seeds. Typically set as ~the diameter of the nuclei.
        size_min: int
            Minimum size, in pixels, of objects to retain
        size_max: int
            Maximum size, in pixels, of objects to retain
        dilation_length: int
            Size in x and y of structuring element for dilating objects after
            final segmentation.
        dilation_length_foci: int
            Size in x and y of structuring element for dilating nuclear foci (HLB)
            mask.
        circularity_min: float 0 to 1
            Minimum circularity for objects to be retained in final mask
        
    Returns:
        labelmask: ndarray
            2D labelmask of nuclei.
    """
    # Make max projection on Z.
    maxp = stack.max(axis=0)
    # Filter with DoG to make nuclei into blobs.
    dog = dog_filter(maxp, sigma_dog_small, sigma_dog_big)
    # Get threshold, use thresh to make initial mask and fill holes.
    t = threshold_otsu(dog)
    mask = np.where(dog > t, 1, 0)
    mask = imfill(mask)
    # Perform distance transform, find local maxima for watershed seeds.
    dist = ndi.distance_transform_edt(mask)
    seeds, _ = peak_local_max_nD(dist, size=seed_window, min_dist=min_seed_dist)
    # Smooth image and take gradient, use as input for watershed.
    im_smooth = ndi.filters.gaussian_filter(maxp, sigma=sigma1)
    grad = gradient_nD(im_smooth)
    # Make second mask of pol2 foci (presumed HLBs) by re-thresholding within nuclei.
    t_foci = threshold_otsu(im_smooth[mask.astype('bool')])
    mask_foci = np.where(im_smooth > t_foci, True, False)
    mask_foci = ndi.morphology.binary_dilation(mask_foci, structure=np.ones((dilation_length_foci, dilation_length_foci)))
    # Mask out pol2 foci in gradient.
    grad = np.where(mask_foci, 0, grad)
    # Perform watershed segmentation.
    ws = watershed(grad, seeds.astype(int))
    # Filter object size and circularity, relabel to set background to 0.
    labelmask = labelmask_filter_objsize(ws, size_min, size_max)
    # Note: object_circularity works on 3D labelmasks, requiring adding (expand_dims) and removing (squeeze) a dimension.
    labelmask = np.squeeze(filter_labelmask(np.expand_dims(labelmask, axis=0), object_circularity, circularity_min, 1000))
    labelmask = relabel_labelmask(labelmask)
    # Dilate segmented nuclei.
    labelmask = labelmask_apply_morphology(labelmask, 
                    mfunc=ndi.morphology.binary_dilation, 
                    struct=np.ones((dilation_length, dilation_length)), 
                    expand_size=(dilation_length + 1, dilation_length + 1))

    if (display):
        fig, ax = plt.subplots(3,2, figsize=(10,10))
        # Display mask.
        ax[0][0].imshow(mask)
        ax[0][0].set_title('Initial Mask')
        # Display watershed seeds.
        seeds_vis = ndi.morphology.binary_dilation(seeds, structure=np.ones((8,8)))
        ax[0][1].imshow(im_smooth, alpha=0.5)
        ax[0][1].imshow(seeds_vis, alpha=0.5)
        ax[0][1].set_title('Watershed seeds')
        # Display gradient.
        ax[1][0].imshow(grad)
        ax[1][0].set_title('Gradient')
        # Display watershed output.
        ws = relabel_labelmask(ws)
        ax[1][1].imshow(ws.astype('bool'))
        ax[1][1].set_title('Watershed')
        # Display final mask.
        ax[2][0].imshow(labelmask.astype('bool'))
        ax[2][0].set_title('Final Segmentation')
    
    # Make 2D labelmask into 3D mask by repeating.
    labelmask = np.repeat([labelmask], stack.shape[0], axis=0)
    return labelmask

############################################################################
def segment_nuclei_4dstack(stack, seg_func, **kwargs):
    """Perform segmentation on a time-series of 3D stacks
    
    Accepts a function that performs nuclear segmentation on a 3D image stack, 
    calls this function on each frame of a 4D image stack, and returns a 4D 
    nuclear mask. Individual time frames are independently segmented and
    labels are not connected between them.
    
    Args:
        stack: ndarray
            4D image stack [t,z,x,y]
        seg_func: function
            A function that performs segmentation on a 3D image stack
        **kwargs: key-word arguments
            Arguments to supply to segmentation function
    
    Returns:
        labelmask: ndarray
            Mask in the shape of input stack with nuclei independently
            segmented at each time point.
    """
    # Segment first frame, initialize 4D labelmask.
    frame0 = seg_func(stack[0], **kwargs)
    labelmask = np.array([frame0])
    # Segment subsequent frames, stack results together.
    for n in range(1, stack.shape[0]):
        print(n)
        frame = seg_func(stack[n], **kwargs)
        labelmask = np.vstack((labelmask, [frame]))
    
    return labelmask

############################################################################
def update_labels(mask1, mask2):
    """Match labels of segmented structures between two labelmasks.
    
    Uses a simple principle of reciprocal best hits: for each labeled object
    in mask 2, find the object in mask1 with the most overlapping pixels. 
    Then do the reverse: find the maximallly overlapping object in mask 1 for
    the objects in mask 2. For objects that are each other's best hit (most
    overlapping pixels), the labels in mask2 are replaced with those of mask1.
    Labels that do not have reciprocal best hits are dropped from the mask.
    
    Args:
        mask1: ndarray
            Labelmask in order [z, x, y]. Labels from this mask will be 
            propagated to mask2.
        mask2: ndarray
            Labelmask of same shape as mask1. Labels in this mask will be
            replaced by corresponding labels from mask1.
        
    Returns:
        updated_mask: ndarray
            Labelmask of identical shape to mask1 and mask2, updated to
            propagate labels from mask1 to mask2.
    
    Raises:
        ValueError:
            If the shapes of the two masks are not the same.
    """
    # Find the object in mask2 that has maximum overlap with an object in max1,
    # (as a fraction of the objects pixels in mask1)
    def get_max_overlap(mask1, mask2, label1):
        # Count overlapping pixels.
        labels, counts = np.unique(mask2[mask1 == label1], return_counts=True)
        # Sort labels by counts (ascending).
        labels_sorted = labels[np.argsort(counts)]
        counts_sorted = counts[np.argsort(counts)]
        # Select new label with maximum overlap.
        max_overlap = labels_sorted[-1]
        return max_overlap
    
    def main(mask1, mask2):
        if not (mask1.shape == mask2.shape):
            raise ValueError("Masks do not have the same shape.")
        # Initialize blank mask.
        updated_mask = np.zeros(mask2.shape)
        # Go one-by-one through the labels in mask2
        for label in np.unique(mask2)[1:]:
            # Find label in mask1 with maximum overlap with nuc from mask2.
            mask1_besthit = get_max_overlap(mask2, mask1, label)
            # Find reverse: best hit for the mask1 label in mask2.
            mask2_besthit = get_max_overlap(mask1, mask2, mask1_besthit)
            # If the labels are reciprocal best hits, update label in 
            # new mask to have the shape of the object in mask 2 with 
            # the label propagated from mask1.
            if ((mask2_besthit == label) and (mask1_besthit != 0)):
                updated_mask[mask2 == label] = mask1_besthit

        return updated_mask
    return main(mask1, mask2)

############################################################################
def connect_nuclei(maskstack, max_frames_skipped=2, 
                             update_func=update_labels):
    """Match labels in a labelmask to masks from previous frames
    
    Takes a stack of nuclear masks, for each frame, walk backward in time 
    looking for segmented nuclei in previous frames that correspond to 
    nuclei in the new frame. Default is to use update_labels function to 
    do comparison of two masks, but this can be changed. Nuclei in a frame
    that cannot be matched to previous nucleus are kept and initialized 
    with new labels (change from previous behavior).
    
    Note: Written with adding 3d masks to 4d stacks, but works for 3d-
    2d.
    
    Args:
        maskstack:
            4D labelmask [t,z,x,y] of segmented nuclei, labels not connected
            between time frames (mask at each time generated independently).
        max_frames_skipped: int
            Maximum number of frames that can be "skipped" to find a nucleus
            to connect to. Compensates for errors in segmentation that cause
            nuclei to sometimes drop from frames. e.g., for a value of 1, 
            function will search for a connected nucleus in the last frame 
            and, if unsuccessful, in the second-to-last frame.
        update_func: function
            Function that takes two label masks, updates labels in the second
            based on matches in the first. Default is to use update_labels.
    
    Returns:
        connected_mask: ndarray
            4D labelmask of same shape as maskstack with object labels connected
            between frames. 
    """
    # Initialize blank mask.
    connected_mask = np.zeros_like(maskstack)
    # Add first frame.
    connected_mask[0] = maskstack[0]
    # Walk through subsequent frames, connecting them to previous frames.
    for n in range(1, maskstack.shape[0]):
        print(n)
        newmask = maskstack[n]
        # Step sequentially backwards through frames, bound by max skip.
        for i in range(n-1, n-2-max_frames_skipped, -1):
            # Make sure the frame exists.
            if (i >= 0):
                # Make updated mask with connections to earlier frame.
                mask_updated_thisframe = update_func(connected_mask[i], newmask)
                # Only update objects that are connected in this frame but were NOT
                # connected to a more recent frame. Do this by restricting updates
                # to positions in connected_mask that are currently 0. Positions that
                # are either non-zero in the current connected_mask or 0 in both are
                # kept the same. Only pixels that are non-zero in new update and 0
                # in current mask are changed (to the labels from the current frame).
                connected_mask[n] = np.where((connected_mask[n] == 0) & (mask_updated_thisframe != 0),
                    mask_updated_thisframe, connected_mask[n])
                #connected_mask[n] = mask_updated_thisframe
        # Initialize un-matched nuclei with new labels.
        unmatched = newmask[(newmask != 0) & (connected_mask[n] == 0)]
        unmatched_labels = np.unique(unmatched)
        for orphan in unmatched_labels:
            new_label = connected_mask.max() + 1
            connected_mask[n][newmask == orphan] = new_label
            
    return connected_mask

############################################################################
def interpolate_nuclear_mask(mask, max_missing_frames=2):
    """Fill-in frames of nuclear mask to compensate for dropout.
    
    So far all nuclear segmentation routines I've tried are susceptible to
    occasional dropout — cases where a nucleus is absent from one or more 
    frames. This function fills in those gaps. Interpolated nuclei have the 
    same shape as the nucleus in the preceding frame with the centroid of
    that shape lying at the mean position of the centroids of the nucleus
    in the preceding and succeeding frame. So if a nucleus is missing from 
    frame 4, the centroid of the nucleus is taken in frame 3 and 5 and the 
    mean of those is used as the centroid of the new nucleus. All the pixels
    belonging to the nucleus in frame 3 are then shifted to match this new
    centroid and this is the final interpolated nuclear object.
    
    Args:
        mask: ndarray
            4D labelmask of segmented nuclei
        max_missing_frames: int
            Maximum allowable number of consecutive frames missing a nucleus for
            interpolation to be executed.
            
    Returns:
        newmask: ndarray
            Labelmask in the same shape as input with interpolated nuclei.
    
    """
    def find_surrounding_frames(frame, frames_with):
        """Find nearest frames containing nucleus before and after frame missing nucleus"""
        # If frame is missing from the beginning, return initial frame twice.
        if (frame < frames_with[0]):
            return frames_with[0], frames_with[0], frames_with[0] - frame
        # If frame is missing from the end, return last frame twice.
        elif (frame > frames_with[-1]):
            return frames_with[-1], frames_with[-1], frame - frames_with[-1]
        # If frame is missing in the middle, return surrounding frames.
        else:
            for i in range(1, len(frames_with)):
                frame_before = frames_with[i-1]
                frame_after = frames_with[i]
                if (frame > frame_before and frame < frame_after):
                    return frame_before, frame_after, frame_after - frame_before - 1

    def interpolate(mask, newmask, label, frame, frames_with, coords, frame_before, frame_after):
        """Use position of nucleus in preceding and succeeding frames to interpolate 
        nucleus in missing frame"""
        # Find centroid of nucleus in before and after frames, take the mean centroid for
        # the position of interpolated nucleus.
        centroid_before = get_object_centroid(mask[frame_before], label)
        centroid_after = get_object_centroid(mask[frame_after], label)
        mean_centroid = [sum(y) / len(y) for y in zip(*(centroid_before, centroid_after))]
        mean_centroid = [int(i) for i in mean_centroid]
        # Get difference of mean centroid and before centroid.
        centroid_diff = ([mean_centroid[0] - centroid_before[0]])
        for i in range(1,len(mean_centroid)):
            centroid_diff = centroid_diff + ([mean_centroid[i] - centroid_before[i]])
        # Make a boolean for all the coordinates in the before frame.
        obj_bool = coords[0] == frame_before
        # Assign the frame number to the first position for all interpolated coordinates.
        interp_coords = tuple([np.repeat(frame, np.count_nonzero(obj_bool))])
        # For remaining dimensions, use centroid difference to "move" pixels of nucleus
        # from before frame to the interpolated position.
        for i in range(0, len(centroid_diff)):
            interp_coords = interp_coords + tuple([coords[i+1][obj_bool] + centroid_diff[i]])
        # Update newmask in place.
        newmask[interp_coords] = label

    # Main.
    newmask = np.copy(mask)
    labels = np.unique(mask)[1:]
    t_frames = np.arange(0,mask.shape[0])
    for label in labels:
        # Find all point coordinates with the label.
        coords = np.where(np.isin(mask, label))
        # Find frames containing and lacking label.
        frames_with = np.unique(coords[0])
        frames_without = t_frames[~np.isin(t_frames, frames_with)]
        # Interpolate nucleus for each frame lacking it.
        for frame in frames_without:
            frame_before, frame_after, num_consecutive_skipped = find_surrounding_frames(frame, frames_with)
            if (num_consecutive_skipped <= max_missing_frames):
                interpolate(mask, newmask, label, frame, frames_with, coords, frame_before, frame_after)
    return newmask

############################################################################
def fit_ms2(stack, peak_window_size=(70,50,50), sigma_small=0.5, 
                   sigma_big=4, bg_radius=4, fitwindow_rad_xy=5, 
                   fitwindow_rad_z=9):  
    """Perform 3D gaussian fitting on local maxima in a 4D image stack
    
    Alrigthm: bandbass filter -> background subtraction -> find local maxima
    -> fit gaussian to windows around maxima
    
    Args:
        stack: ndarray
            4D image stack [t,z,x,y] containing MS2 spots
        peak_window_size: tuple of three ints
            Size in [z,x,y] of window used to find local maxima. Typically
            set to the approximage dimensions of nuclei.
        sigma_small: numeric
            Lower sigma for difference-of-gaussians bandpass filter
        sigma_small: numeric
            Upper sigma for difference-of-gaussians bandpass filter
        bg_radius: int
            Radius for minimum filter used for background subtraction
        fitwindow_rad_xy: int
            Radius in pixels in the xy-dimension of the window around local
            maxima peaks within which to do gaussian fitting.
        fitwindow_rad_z: int
            Radius in pixels in the z-dimension of the window around local
            maxima peaks within which to do gaussian fitting.
    
    Returns:
        fit_data: list of ndarrays
            Each entry in the list is a time point (frame). Each row in
            array is a fit (a single local maxima), columns are: 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: width_z, 4: fit_height, 5: width_x, 6: width_y). Coordinates 
            are adjusted so that if fit center lies outside the image, 
            center is moved to the edge.
    """
    def get_fitwindow(data, peak, xy_rad=5, z_rad=9):
        """Retrieve section of image stack corresponding to given
        window around a point and the coordinate adjustments necessary
        to convert window coordinates to coordinates in the original image"""
        
        # Set the start points for windows and "adjust" them if they get 
        # to negative numbers.
        zmin = peak[0] - z_rad
        xmin = peak[1] - xy_rad
        ymin = peak[2] - xy_rad
        # Initialize adjustments to values that are correct if no edge problems 
        # are encountered.
        z_adj = -z_rad
        x_adj = -xy_rad
        y_adj = -xy_rad
        # Update mins and adjustments if windows start at negative coordinates.
        if (zmin < 0):
            zmin = 0
            z_adj = -peak[0]
        if (xmin < 0):
            xmin = 0
            x_adj = -peak[1]
        if (ymin < 0):
            ymin = 0
            y_adj = -peak[2]

        # Get end points, constained by max coordinate in data.
        zmax = min(data.shape[0] - 1, peak[0] + z_rad)
        xmax = min(data.shape[1] - 1, peak[1] + xy_rad)
        ymax = min(data.shape[2] - 1, peak[2] + xy_rad)

        return (data[zmin:(zmax+1), xmin:(xmax+1), ymin:(ymax+1)], z_adj, x_adj, y_adj)
    
    def relabel(peak_ids, oldparams, mask):
        """Renumber labelmask and corresponding fit parameters
        Set background as 0, objects in order 1...end.
        """
        spot_data = {}
        peak_num = 1
        for peak in peak_ids:
            #coords = np.where(mask == peak)
            paramsnew = oldparams[peak-1,:] # object 1 will be fitparams row 0
            # Rearrange params from fit function so coordinates lead.
            spot_data[peak_num] = paramsnew[[1,2,3,0,4,5,6]]
            peak_num = peak_num + 1
        return spot_data

    def clamp(n, minn, maxn):
        """Bound a number between two constants"""
        return max(min(maxn, n), minn)
    
    def fit_frame(substack, peak_window_size, sigma_small, 
                   sigma_big, bg_radius, fitwindow_rad_xy, 
                   fitwindow_rad_z):
        """Perform 3D gaussian fitting on a 3D image stack."""

        # Filter and background subtract image.
        dog = dog_filter(substack, sigma_small, sigma_big)
        bg = ndi.filters.minimum_filter(dog, bg_radius)
        dog_bs = dog - bg

        # Make a labelmask corresponding to local maxima peaks.
        mask, peaks = peak_local_max_nD(dog_bs, peak_window_size)

        # Fit 3D gaussian in window surrounding each local maximum.
        fitparams = np.ndarray((0,7))
        for peak in peaks:
            fitwindow, z_adj, x_adj, y_adj = get_fitwindow(substack, peak, fitwindow_rad_xy, 
                fitwindow_rad_z)
            opt = fitgaussian3d(fitwindow)
            if opt.success:
                peak_fitparams = opt.x
                # Move center coordinates to match center of gaussian fit, ensure they're within image. 
                # If they're outside the image, coordinate is assigned as the edge of the image.
                peak_fitparams[0] = int(round(clamp((peak[0] + peak_fitparams[0] + z_adj), 0, substack.shape[-3]-1)))
                peak_fitparams[1] = int(round(clamp((peak[1] + peak_fitparams[1] + x_adj), 0, substack.shape[-2]-1)))
                peak_fitparams[2] = int(round(clamp((peak[2] + peak_fitparams[2] + y_adj), 0, substack.shape[-1]-1)))
                fitparams = np.vstack((fitparams, peak_fitparams))
            # If fit fails, add dummy entry for spot.
            else:
                fitparams = np.vstack((fitparams, np.array([0,0,0,0,np.inf,np.inf,np.inf])))
        return fitparams
    
    #### Main ####
    # Do fitting on first frame.
    fit_data_frame0 = fit_frame(stack[0], peak_window_size, sigma_small, 
                   sigma_big, bg_radius, fitwindow_rad_xy, 
                   fitwindow_rad_z)
    # Make fit_data a list of ndarrays.
    fit_data = [fit_data_frame0]
    
    # Fit the rest of the frames, add their data to fit_data.
    for i in range(1, stack.shape[0]):
        print(i)
        fit_data_thisframe = fit_frame(stack[i], peak_window_size, sigma_small, 
                   sigma_big, bg_radius, fitwindow_rad_xy, 
                   fitwindow_rad_z)
        fit_data.append(fit_data_thisframe)
        
    return fit_data

############################################################################
def filter_ms2fits(stack, fit_data, h_stringency=0, xy_max_width=15):
    """Filter MS2 spot fit data based on fit parameters
    
    Select spots that have a minimum fit height (intensity) and a maximum 
    lateral (xy) width.
    
    Args:
        stack: ndarray
            4D imaging stack [t,z,x,y] that the fitting was performed on
        fit_data: list of ndarrays
            Each entry in list is a distinct frame (in time), rows in array
            are individual spot fits and columns are 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: width_z, 4: fit_height, 5: width_x, 6: width_y.
        h_stringency: numeric
            Determines minimum fit height cutoff, expressed as the number
            of standard deviations above the mean for the entire stack.
        xy_max_width: numeric
            Maximum width (in pixels) of the mean of the x and y widths of
            the fit
    
    Returns:
        fit_data: list of ndarrays
            Input data, retaining only rows that pass filter.  
    """
    
    fit_data = fit_data.copy()
    for t in range(0, len(fit_data)):
        frame_data = fit_data[t]
        # Define threshold for height.
        mean_ = np.mean(stack[t])
        std = np.std(stack[t])
        h_thresh = mean_ + (std * h_stringency)
        # Filter array based on min height and max width.
        frame_data_filtered = frame_data[(frame_data[:,3] >= h_thresh) &
                    (np.mean(frame_data[:,5:7], axis=1) < xy_max_width),:]
        fit_data[t] = frame_data_filtered
    return fit_data

############################################################################
def connect_ms2_frames(spot_data, nucmask, max_frame_gap=1, max_jump=10, 
    scale_xy=1, scale_z=1):
    """Connect detected MS2 spots through multiple time frames.
    
    Spots detected in new frame are connected to spots in previous frames
    if they are within specified distance (max_jump). Spots can "disappear" 
    for a number of frames defined by max_frame_gap. Spots that cannot be 
    connected to spots from prior frames are initialized as new spots.
    
    Args:
        spot_data: list of ndarrays
            Each entry in list is a distinct frame (in time), rows in array
            are individual detected spots and columns are 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: width_z, 4: fit_height, 5: width_x, 6: width_y.
        nucmask: ndarray
            4D labelmask of dimensions [t,z,x,y] of segmented nuclei. 0 is 
            background (not a nucleus) and nuclei have integer labels.
        max_frame_gap: int
            Maximum number of frames from which spot can be absent and still
            connected across the gap. Example: for a value of 1, a spot
            detected in frame 6 and absent from frame 7 can be connected to
            a spot in frame 8, but a spot in frame 5 cannot be connected to
            frame 8 if it is absent in frames 6 and 7.
        max_jump: numeric
            Maximum 3D displacement between frames for two spots to be connected
        scale_xy: numeric
            Distance scale for xy direction (typically: nm per pixel)
        scale_z: numeric
            Distance scale for z direction (typically: nm per pixel)
        
    Returns:
        connected_data: dict of ndarrays
            Each key is a unique spot tracked across 1 or more frames. Each row
            of array is the spot's data for a single frame, with columns 0: frame
            number (t), 1: nucleus ID, 2: center Z-coordinate, 3: center X-coord-
            inate, 4: center Y-coordinate, 5: fit height, 6: fit z_width, 7: fit
            x_width, 8: fit y_width
    """
    def initialize_new_spot(new_spot_data, connected_data):
        """Initialize new spot with next numeric ID and entry in connected_data."""
        if (connected_data.keys()):
            new_id = max(connected_data.keys()) + 1
        else:
            new_id = 1
        connected_data[new_id] = np.expand_dims(new_spot_data, 0)

    def sq_euc_distance(coords1, coords2, scale_z=1, scale_xy=1):
        """Find the squared euclidean distance between two points."""
        z2 = ((coords2[0] - coords1[0]) * scale_z) ** 2
        x2 = ((coords2[1] - coords1[1]) * scale_xy) ** 2
        y2 = ((coords2[2] - coords1[2]) * scale_xy) ** 2
        sed = z2 + x2 + y2
        return sed
    
    def coord_list_t(connected_data, t):
        """Make a list of [z,x,y] coordinate tuples for all spots in a given
        frame"""
        coord_list = []
        for spot_id in connected_data:
            this_spot_data = connected_data[spot_id]
            row = this_spot_data[this_spot_data[:,0] == t]
            if (len(row) > 0):
                row = list(row[0])
                spot_coords = [spot_id] + row[2:5]
                coord_list.append(spot_coords)
        return coord_list
            
    
    def find_nearest_spot(this_coord, coord_list, scale_z, scale_xy):
        """For a given point, find the closest spot in a coordinate list
        and the distance between the points."""
        closest_sed = np.inf
        closest_spot = 0
        for test_data in coord_list:
            test_spot_id = test_data[0]
            test_coords = (test_data[1:4])
            sed = sq_euc_distance(test_coords, this_coord, scale_z, scale_xy)
            if (sed < closest_sed):
                closest_sed = sed
                closest_spot = test_spot_id
                closest_spot_coords = test_coords
        return closest_spot, np.sqrt(closest_sed), closest_spot_coords

    def update_spot(this_spot_data, connected_data, scale_z, scale_xy, max_frame_gap, 
                    t):
        """Walk back one frame at a time within limit set by maximum gap, search 
        for a nearest spot that is within the maximum allowable jump, handle 
        duplicates, add connected points to connected_data."""
        this_spot_coords = (this_spot_data[2:5])
        # Walk back one frame at a time.
        for t_lag in range(1, max_frame_gap + 2):
            if ((t - t_lag) >= 0):
                # Get nearest spot in the current frame.
                spot_coords_tlag = coord_list_t(connected_data, t - t_lag)
                # If there are no previously detected spots, break from for loop and initialize new spot entry.
                if (len(spot_coords_tlag) == 0):
                    break
                nearest_spot_id, dist, nearest_spot_coords = find_nearest_spot(this_spot_coords, spot_coords_tlag, scale_z, scale_xy)
                # Check is spot is within max distance.
                if (dist <= max_jump):
                    this_spot_nucID = this_spot_data[1]
                    nearest_spot_nucID = connected_data[nearest_spot_id][-1,1]
                    # Check if there's already a spot added for this time.
                    existing = connected_data[nearest_spot_id][connected_data[nearest_spot_id][:,0] == t]
                    # If there's no existing spot, add this spot to the end of the data for connected spot.
                    if (len(existing) == 0):
                        connected_data[nearest_spot_id] = np.append(connected_data[nearest_spot_id], [this_spot_data], axis=0)
                        return
                    # If there is an existing spot, if the current spot is closer to the previous-frame spot
                    # than the existing entry, replace it. Otherwise, continue looking in previous frames (if
                    # applicable) and eventually create new spot after for loop. I'm not sure this is the best
                    # behavior--may consider dumping out of for loop and creating new spot rather than looking
                    # to previous frames in this situation.
                    else:
                        existing_dist = np.sqrt(sq_euc_distance(nearest_spot_coords, existing[0,2:5], scale_z, scale_xy))
                        # If the the current spot is closer than the existing spot, replace 
                        # existing and initialize it as a new spot.
                        if (dist < existing_dist):
                            row_index = np.where(connected_data[nearest_spot_id][:,0] == t)[0][0]
                            superseded_spot_data = connected_data[nearest_spot_id][row_index]
                            # Superseded spot from this frame gets bumped to be a new spot.
                            initialize_new_spot(superseded_spot_data, connected_data)
                            # Replace data for superseded spot with this spot's data.
                            connected_data[nearest_spot_id][row_index] = this_spot_data
                            return

        # If no suitable spot was found in previous frames, make a new spot.
        initialize_new_spot(this_spot_data, connected_data)
    
    def add_time_nuc(this_spot_data, t, nucmask):
        """ Append the frame number (t) and nucleus ID to front of spot_data"""
        spot_coords = tuple(np.append([t], this_spot_data[0:3]).astype(int))
        nuc_id = nucmask[spot_coords]
        # Add time and nuclear ID columns to spot data and call update to search 
        # for connected spots in previous frames.
        this_spot_data = np.append([t, nuc_id], this_spot_data)
        return this_spot_data
    
    def initialize_connected_data(frame_data):
        """Initialize connected_data structure as a dict where each entry
        is a unique spot and the array rows are data for single frames"""
        connected_data = {}
        for i in range(0, len(frame_data)):
            this_spot_data = add_time_nuc(frame_data[i], 0, nucmask)
            connected_data[i+1] = np.array([this_spot_data])
        return connected_data
        
    # Main
    input_data = spot_data.copy()
    connected_data = initialize_connected_data(input_data[0])

    # Go through each frame, attempt to connect each detected spot to previous spots.
    for t in range(1, len(input_data)):
        print(t)
        frame_data = input_data[t]
        for this_spot_data in frame_data:
            this_spot_data = add_time_nuc(this_spot_data, t, nucmask)
            update_spot(this_spot_data, connected_data, scale_z, scale_xy, max_frame_gap, t)
    
    return connected_data  

############################################################################
def filter_spot_duration(connected_data, min_len):
    """Filter connected_data structure for spots lasting a minimum number
    of frames

    Args:
        connected_data: dict of ndarrays
            Output data structure of connect_ms2_frames
        min_len: int
            Minimum number of frames in which a spot must be detected

    Returns:
        filtered_data: dict of ndarrays
            connected_data retaining only spots of defined duration

    """
    filtered_data = {}
    spot_num = 1
    for spot in connected_data:
        if (connected_data[spot].shape[0] >= min_len):
            filtered_data[spot_num] = connected_data[spot]
            spot_num = spot_num + 1
    return filtered_data

############################################################################
############################################################################
# Functions for analyzing segmented images
############################################################################
############################################################################

def add_volume_mean(spot_data, stack, channel, ij_rad, z_rad, ij_scale=1, z_scale=1):
    """Find mean volume within ellipsoid centered on spots, add to spot_info

    Args:
        spot_data: dict of ndarrays
            Data containing tracking of spots detected in previous frames.
            Dict entries are unique spot IDs (numeric 1...), rows of ndarray
            are detections of the spot in a single frame. Column order is
            0: frame no. (time), 1: nucleus ID, 2: z-coordinate, 3: x-
            coordinate, 4: y-coordinate, 5: gaussian fit height, 6: gaussian
            fit z-width, 7: gaussian fit x-width, 8: gaussian fit y-width.
        stack: ndarray
            Image stack of dimensions [c,t,z,x,y]
        channel: int
            Channel containing MS2 spots
        ij_rad: numeric
            Radius in real units of ellipsoid in the ij (xy) dimension.
        z_rad: numeric
            Radius in real units of ellipsoid in the z dimension.
        ij_scale: numeric
            Scale factor for ij_rad (typically nm/pixel)
        z_scale: numeric
            Scale factor for z_rad (typically nm/pixel)
    
    Returns:
        spot_data: dict of ndarrays
            Input dictionary with mean ellipsoid pixel values appended as an 
            additional column (9) to all entries.
    """
    def ellipsoid_mean(coords, stack, meshgrid, ij_rad=7, z_rad=2):
        """Define ellipsoid around point, return mean of pixel values in ellipsoid."""
        # Equation: (x-x0)^2 + (y-y0)^2 + a(z-z0)^2 = r^2
        r = ij_rad # r is just more intuitive for me to think about...
        a = (r ** 2) / (z_rad ** 2)
        z0, i0, j0 = coords
        valsgrid = np.sqrt((a * ((meshgrid[0] - z0) ** 2)) + ((meshgrid[1] - i0) ** 2) + ((meshgrid[2] - j0) ** 2))
        pixels = stack[valsgrid <= r]
        return pixels.mean()
    
    spot_data = spot_data.copy()
    # Make meshgrid for stack.
    meshgrid = mesh_like(stack, 3)
    # Scale radii to pixels.
    ij_rad_pix = ij_rad / ij_scale
    z_rad_pix = z_rad / z_scale
    num_processed = 0
    # Update data for each spot at each time point combination by adding column
    # with the sum of the pixel values within defined ellipses.
    for spot_id in spot_data:
        num_processed = num_processed + 1
        if (num_processed % 10 == 0):
            print('Processed ' + str(num_processed))
        spot_array = spot_data[spot_id]
        # Initialize new array with extra column.
        new_array = np.ndarray((spot_array.shape[0], spot_array.shape[1] + 1))
        for rownum in range(0, spot_array.shape[0]):
            row = spot_array[rownum]
            t = int(row[0])
            coords = tuple(row[2:5].astype(int))
            substack = stack[channel, t]
            pix_mean = ellipsoid_mean(coords, substack, meshgrid, ij_rad_pix, z_rad_pix)
            new_array[rownum] = np.append(row, [pix_mean])
        spot_data[spot_id] = new_array
    return spot_data

############################################################################
def add_gaussian_integration(spot_data, wlength_xy=15, wlength_z=5):
    """Add a column to spot_data that integrates intensity from gaussian fit
    
    For each spot in spot_data, uses gaussian fit parameters (height and 
    widths in z,y,x) to integrate the gaussian function within a window of 
    supplied dimensions ([z,x,y] = [wlength_z, wlength_xy, wlength_xy]). 
    "Integration" is discrete — gaussian function is converted to pixel values, 
    and the mean pixel intensity is then added as an additional column to each 
    entry in spot_data. Mean is used over sum simply to keep numbers low and 
    aid interpretability.
    
    Args:
        spot_data: dict of ndarrays
            Data containing tracking of spots detected. Dict entries are unique 
            spot IDs (numeric 1...), rows of ndarray are detections of the spot 
            in a single frame. Required columns: 5: gaussian fit height, 6: 
            gaussian fit z-width, 7: gaussian fit x-width, 8: gaussian fit 
            y-width.
        wlength_xy: int
            Length of the sides of the window used for integration in the
            lateral dimension. Must be an odd number. To harmonize with
            volume integration, should be 2*ij_rad + 1.
        wlength_z: int
            Length of the sides of the window used for integration in the
            axial dimension. Must be an odd number. To harmonize with 
            volume integration, should be 2*z_rad + 1.
            
    Returns:
        spot_data: dict of ndarrays
            Structure identical to input with an additional column appended to
            all entries containing result of integration.
    """
    def integrate_gaussian(p, wlength_xy, wlength_z):
        """Determine mean pixel intensity within a window given parameters
        of a 3D gaussian function."""
        if ((wlength_xy % 2 == 0) or (wlength_z % 2 == 0)):
            raise ValueError('wlength_xy and wlength_z must be odd.')
        # Get fit parameters, find coords for center pixel within window.    
        h, width_z, width_x, width_y = p[5:9]
        center_xy = int(wlength_xy / 2)
        center_z = int(wlength_z / 2)
        # Get indices for the window.
        z,x,y = np.indices((wlength_z, wlength_xy, wlength_xy))
        # Generate function to receive indexes and return values of gaussian 
        # function with given parameters
        f = gaussian3d(center_z, center_xy, center_xy, h, width_z, width_x, width_y)
        # Generate window with intensity values from 3d gaussian function.
        vals = f(z,x,y)
        # Return mean pixel intensity of window.
        return vals.mean()
    
    # Work on a copy of input data.
    spot_data = spot_data.copy()
    for spot_id in spot_data:
        spot_array = spot_data[spot_id]
        # Initialize new array with extra column.
        new_array = np.ndarray((spot_array.shape[0], spot_array.shape[1] + 1))
        for rownum in range(0, spot_array.shape[0]):
            row = spot_array[rownum]
            pix_mean = integrate_gaussian(row, wlength_xy, wlength_z)
            new_array[rownum] = np.append(row, [pix_mean])
        spot_data[spot_id] = new_array
    return spot_data

############################################################################
def stitch_ms2(mv1, mv2):
    """Stitch two MS2 movies together across a focus break
    
    Connect two confocal MS2 movies together that are separated by a 
    re-focusing break. First connects nuclear masks, then connects spots
    based on nuclear ID (spots in the same nucleus are connected).
    
    Args:
        mv1: movie object
            First object of movie class with nucmask and spot_data features
        mv2: movie object
            Second object of movie class. Nucleus and spot IDs will be updated
            to match those of mv1
    
    Returns:
        mv_stitched: movie object
            Movie object with nucmask and spot_data from mv1 and mv2 stitched
            together
    """
    
    def update_nucleus_id(data, mask):
        """Update the nuclear ID field of a spot data array based on spot 
        coordinates and updated nuclear mask."""
        for i in range(0, data.shape[0]):
            coords = tuple([int(data[i,0]), int(data[i,2]), int(data[i,3]), 
                int(data[i,4])])
            nuc = mask[coords]
            data[i,1] = nuc
            
    def link_nuc_spots(data):
        """Link nuclei to spots. Make dictionary with nucleus ID as key and 
        spot ID as value."""
        links = {}
        for spot in data:
            # Find modal nuclear ID for the spot.
            nucs = sortfreq(data[spot][:,1])
            if (nucs[0] != 0):
                nuc_id = int(nucs[0])
                if (nuc_id not in links):
                    links[nuc_id] = spot
                # If nuc already has a spot, select spot with longest trajectory.
                else:
                    curr_spot_id = links[nuc_id]
                    len_curr_spot = len(data[curr_spot_id][:,1])
                    len_this_spot = len(data[spot][:,1])
                    if (len_this_spot > len_curr_spot):
                        links[nuc_id] = int(spot)
        return links
    
    def stitch_stacks(stack1, stack2):
        """Put two [ctzxy] stacks together (arbitrary number of channels)"""
        stack_stitched = np.vstack((stack1[0], stack2[0]))
        for i in range(1, stack1.shape[0]):
            channel_stitched = np.vstack((stack1[i], stack2[i]))
            stack_stitched = np.stack((stack_stitched, channel_stitched))
        return stack_stitched

    # Main.
    
    ### Connect nuclei ###
    
    # Merge last two frames of mv1 and first two frames of mv2 to account for 
    # potential dropout.
    end1 = mv1.nucmask[-2:].max(axis=0)
    start2 = mv2.nucmask[0:2].max(axis=0)
    # Update labels from mv2 to match those of mv1
    updated_mask2 = update_labels(end1, start2)
    
    # Generate an update table for old (mv2) to new (mv1) nuc IDs
    convert = {}
    for old in np.unique(start2)[1:]:
        new = np.unique(updated_mask2[start2 == old])[0]
        if (new != 0):
            convert[old] = int(new)

    # Update nucmask of mv2 with new labels, stitch mv1 and mv2 masks together.
    labels = list(np.unique(mv1.nucmask))
    newmask = np.zeros_like(mv2.nucmask)
    for old in np.unique(mv2.nucmask)[1:]:
        if (old in convert):
            newmask[mv2.nucmask == old] = convert[old]
        else:
            new_id = max(labels) + 1
            newmask[mv2.nucmask == old] = new_id
            labels.append(new_id)
    nucmask_stitched = np.vstack((mv1.nucmask, newmask))
    
    ### Connect spots ###
    
    # Make dict of nucleus-spot connections in mv1.
    mv1_nuc_spot_links = link_nuc_spots(mv1.spot_data)
    # Get t (frame) number to adjust by.
    t_adjust = mv1.nucmask.shape[0]
    # Initialize stitched data with mv1 data.
    spot_data_stitched = mv1.spot_data.copy()
    # Keep track of spot_ids used.
    spot_ids = list(mv1.spot_data.keys())
    
    # For each spot in mv2, either connect to a spot in mv1 or initialize as new spot.
    for spot in mv2.spot_data:
        data = np.copy(mv2.spot_data[spot])
        # Update time.
        data[:,0] = data[:,0] + t_adjust
        # Update nucleus ID using new mask (in place).
        update_nucleus_id(data, nucmask_stitched)
        # Get modal nucleus.
        nucs = sortfreq(data[:,1])
        nuc_id = int(nucs[0])
        # If linked to an mv1 spot, merge data for spot.
        if nuc_id in mv1_nuc_spot_links:
            linked_spot_id = mv1_nuc_spot_links[nuc_id]
            spot_data_stitched[linked_spot_id] = np.vstack((mv1.spot_data[linked_spot_id], data))
        # If unlinked, make a new spot.
        else:
            new_spot_id = max(spot_ids) + 1
            spot_data_stitched[new_spot_id] = data
            spot_ids.append(new_spot_id)
    
    fits_stitched = mv1.fits + mv2.fits
    stack_stitched = stitch_stacks(mv1.stack, mv2.stack)
    mv_stitched = movie(stack_stitched, nucmask_stitched, fits_stitched, spot_data_stitched)
    return mv_stitched

############################################################################
def align_traces(df, locs, window_size=11, include_nan=True):
    """Align MS2 traces around selected frames
    
    For use in triggered averaging. Takes a dataframe of trace data and a 
    list of spot IDs and frame numbers, makes a panda dataframe of the trace
    data centered on the indicated frames.
    
    Args:
        df: pandas dataframe
            Trace data for MS2 spots. Spot IDs are columns and rows are time 
            points
        locs: list of 2-item lists
            Each list entry is a spot-frame pair to align. Each entry is
            a 2-item list, with the spot ID in the 0 position and the frame
            number in the 1.
        window_size: int
            Size of the windows to be aligned (in number of frames). Must be 
            an odd number.
    
    Returns:
        data: pandas dataframe
            Aligned trace data, rows are individual alignment points and columns
            are time points.
    """
    # Check that window size is appropriate.
    max_frame = df.shape[0] - 1
    if (window_size % 2 == 0):
        raise ValueError("window_size must be odd.")
    if ((window_size - 1) >= max_frame):
        raise ValueError("window_size larger than number of frames.")
    
    # Get number of frames to "extend" on either side of reference time point
    # and make a list to store each aligned segment.
    extend_len = int((window_size - 1) / 2)
    data = []
    
    for item in locs:
        spot = item[0]
        loc = item[1]
        # If the window extends past 0 on the left, pad beginning with nans.
        if ((loc - extend_len) < 0):
            nan_len = abs(loc - extend_len)
            nans = pd.Series(np.repeat(np.nan, nan_len))
            after_nans = df[spot][0:(loc + extend_len + 1)]
            vals = nans.append(after_nans)
            vals.index = range(0, window_size)
            vals.name = spot
            data.append(vals)
        # If the window extends past the last frame on the right, pad with trailing
        # nans.
        elif((loc + extend_len) > max_frame):
            nan_len = (loc + extend_len) - max_frame
            nans = pd.Series(np.repeat(np.nan, nan_len))
            before_nans = df[spot][(loc - extend_len):(max_frame + 1)]
            vals = before_nans.append(nans)
            vals.index = range(0, window_size)
            vals.name = spot
            data.append(vals)
        # Otherwise slice spot data and append to list.
        else:
            vals = df[spot][(loc - extend_len):(loc + extend_len + 1)]
            vals.index = range(0, window_size)
            data.append(vals)
    
    return pd.DataFrame(data)


