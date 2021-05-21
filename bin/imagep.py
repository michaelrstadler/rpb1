#!/usr/bin/env python

"""
Library of functions for image processing of both confocal and lattice light 
sheet movies of fly embryos.
"""
__version__ = '1.2.0'
__author__ = 'Michael Stadler'


import numpy as np
import matplotlib as mpl
import os
from os import listdir
from os.path import isfile, join
import re
import xml.etree.ElementTree as ET
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
import sys
import pandas as pd
from copy import deepcopy
# Bug in skimage: skimage doesn't bring modules with it in some environments.
# Importing directly from submodules (above) gets around this.

# Import my packages.
sys.path.append(os.getcwd())
from fitting import fitgaussian3d, gaussian3d


"""
To fix:

"""

############################################################################
# Classes
############################################################################
class movie():
    """
    A class to store data (processed and unprocessed) for confocal movies
    from fly embryos.

    Style note: movie should be capitalized, but I did not know this style
    convention when I made it and I am leaving it for compatability reasons

    Attributes:
        stack: ndarray
            N-dimensional image stack of the original data.
        nucmask: ndarray
            Labelmask of segmented nuclei
        fits: list of ndarrays
            3D gaussian fits generated in MS2 spot detection as output of 
            function fit_ms2. Each entry in the list is a time point (frame). 
            Each row in array is a fit (a single local maxima), columns are: 
            0: center z-coordinate, 1: center x-coordinate, 2: center 
            y-coordinate, 3: fit_height, 4: width_z, 5: width_x, 6: width_y). 
            Coordinates are adjusted so that if fit center lies outside the 
            image, center is moved to the edge.
        spot_data: dict of ndarrays
            Each key is a unique spot tracked across 1 or more frames. Each row
            of array is the spot's data for a single frame, with columns 0: frame
            number (t), 1: nucleus ID, 2: center Z-coordinate, 3: center X-coord-
            inate, 4: center Y-coordinate, 5: fit height, 6: fit z_width, 7: fit
            x_width, 8: fit y_width, 9: integrated volume for MS2, 10: integrated
            gaussian fit of MS2 spots, 11: integrated volume for protein signal.
        intvol: pandas df
            Intensity values for spots (over time) derived from the mean signal
            within ellipsoid volumes around detected spot centers.
        intfit: pandas df
            Intensity values for spots (over time) derived from integrating the 
            fitted 3D gaussian parameters.
        prot: pandas df
            Equivalent to intvol except integrations performed in the protein
            (nuclear) channel

    Methods:
        make_spot_table:
            Converts a column in spot_data to a pandas df with axes spot_id and 
            frame (time)
    """
    # Class attributes  
    # Initializer

    @staticmethod
    def make_spot_table(spot_data, nucmask, colnum):
        """Make a spot_id x time_frame pandas df from a given column
        of spot_data."""
        nframes = nucmask.shape[0]
        data = {}
        for spot in spot_data:
            arr = spot_data[spot]
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
        self.intvol = movie.make_spot_table(self.spot_data, self.nucmask, 9)
        self.intfit = movie.make_spot_table(self.spot_data, self.nucmask, 10)
        self.prot = movie.make_spot_table(self.spot_data, self.nucmask, 11)

############################################################################
# General image processing functions
############################################################################

def concatenate_5dstacks(stacks):
    """Concatenate multiple [ctzxy] stacks
    
    Args:
        stacks: list of 5d ndarrays
            List of 5d (2-channel) ndarray image stacks to concatenate
    
    Returns:
        stack: 5d ndarray
            Concatenation of input stacks
        frames: list of ints
            List of the frame numbers at which joins occur, with each entry
            representing the 0-indexed location of the first frame of a new
            stack.
    """
    def stack_channel(stacks, channel):
        """Stack multiple 4d ndarrays"""
        cstack = stacks[0][channel].copy()
        frames = []
        for i in range(1, len(stacks)):
            frames.append(len(cstack))
            cstack = np.vstack([cstack, stacks[i][channel]])
        return cstack, frames
    c0_stack, frames = stack_channel(stacks, 0)
    c1_stack, _ = stack_channel(stacks, 1)
    return np.stack((c0_stack, c1_stack)), frames

############################################################################
def clamp(n, minn, maxn):
    """Bound a number between two constants

    Args:
        n: numeric
            Number to clamp
        minn: numeric
            Minimum value
        maxn: numeric
            Maximum value

    Returns:
        labelmask_filtered: ndarray
            Input labelmask with all pixels not corresponding to objects
            within size range set to 0.
    """
    return max(min(maxn, n), minn)

############################################################################
def labelmask_filter_objsize(labelmask, size_min, size_max):
    """Filter objects in a labelmask by size.

    Args:
        labelmask: ndarray
            Integer labelmask (background must be 0 or errors occur)
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

def labelmask_filter_objsize_apply4d(labelmask, size_min, size_max):
    """Filter objects in a 4D labelmask by size.

    Args:
        labelmask: ndarray
            4D Integer labelmask (background must be 0 or errors occur)
        size_min: int
            Minimum size, in pixels, of objects to retain
        size_max: int
            Maximum size, in pixels, of objects to retain

    Returns:
        labelmask_filtered: ndarray
            Input labelmask with all pixels not corresponding to objects
            within size range set to 0.
    """
    labelmask_filtered = np.zeros_like(labelmask)
    for n in range(0, labelmask.shape[0]):
        print(n, end=' ')
        labelmask_filtered[n] = labelmask_filter_objsize(labelmask[n], size_min, size_max)    
    return labelmask_filtered

############################################################################
def imfill(mask, seed_pt='default'):
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
            valent of the point you click when filling in paint. 
    Returns:
        mask_filled: ndarray
            Binary mask filled    
    '''
    if (seed_pt == 'default'):
        # Get a random 0-valued background pixel.
        seed_pt = find_background_point(mask)
        # Flood background from this point.
        mask_flooded = flood_fill(mask, seed_pt, 1)
        mask_filled = np.where((mask == 0) & (mask_flooded == 1), 0, 1)
        return mask_filled
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

    NOTE: because the size is the length of the side in the filter, objects
    that are n/2 pixels apart will both be declared as maxes (for size of
    101, max window extends 50 pixels in each direction, peaks spaced 51 
    pixels apart will each be maxes.)
    
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
    
    Background is defined as the largest contiguous block of 0 pixels. A 
    random pixel coordinate from this background set is returned.

    Args:
        mask: ndarray
            Mask of abritrary dimensions, background must be 0.

    Returns: 
        coord: tuple of ints
            Coordinates of a single background pixel.
    """
    # Label objects of contiguous background (0) pixels.
    bglabelmask, _ = ndi.label(np.where(mask == 0, 1, 0))
    # Count pixels in each background object.
    labels, counts = np.unique(bglabelmask, return_counts=True)
    # Get the indexes of sorted counts, descending.
    ordered_indexes = np.argsort(counts)[::-1]
    # Set largest contiguous 0 block as true background.)
    bg_label = labels[ordered_indexes[0]]
    if (bg_label == 0): # In this mask, 0 is the foreground (confusingly).
        bg_label = labels[ordered_indexes[1]]
    # Select random coordinate from background to be seed.
    zerocoords = np.where(bglabelmask == bg_label)
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
def df_filter_minlen(df, minlen, renumber=False):
    """Filter pandas dataframe columns for minimum number of non-nan entries.

    Args:
        df: pandas dataframe

        minlen: int
            Minimum number of non-nan entries for a column to be retained
        renumber: bool
            If true, renumber columns sequentially from 1

    Returns:
        new_df: pandas dataframe
            Contains columns of input dataframe with sufficient entries
    """
    new_df =  df.loc[:,df.apply(lambda x: np.count_nonzero(~np.isnan(x)), axis=0) > minlen]
    if (renumber):
        new_df.columns = np.arange(1, len(new_df.columns) + 1)
    return new_df

############################################################################
def df_deriv(df, windowsize, stepsize):
    """Take the discrete derivative of each column in a pandas df.

    The (centered) mean is first taken using windows of size windowsize, and 
    derivates are computed as the  difference between means at offsets of 
    stepsize.

    Args:
        df: pandas dataframe
            Pandas df to take the derivative of
        windowsize: int
            Size of window for taking mean for use in derivative calculation
        stepsize: int
            Offset size used in derivative

    Returns:
        df_deriv: Pandas dataframe
            Derivative of input df
    """
    df_deriv = df.rolling(windowsize, center=True).mean().diff(stepsize)
    return df_deriv

############################################################################
def expand_mip(mask, n):
        """Expand a 3d maximum intensity projection by repeating in Z (-3)
        dimension n times.
        
        Main use: using a MIP for speed to do something like create a nuclear
        mask, but then you want to expand it in the Z dimension to match the
        original, non-MIP stack.

        Args:
            mask: ndarray
                n-dimensional 

        """
        expanded_mask = np.zeros([mask.shape[0], n, mask.shape[1], mask.shape[2]])
        for t in range(0, mask.shape[0]):
            expanded_mask[t] = np.repeat([mask[t]], n, axis=0)
        return expanded_mask

############################################################################
# Functions implementing filters
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
            If true, remove last frame if it contains blank slices
        swapaxes: bool
            If true, switches first two axes to produce a stack order ctzxy
            
    Returns:
        stack: ndarray
            Image stack in dimensions [t,c,z,x,y] (no swap) or 
            [c,t,z,x,y] (swapped)
    """
    def frame_incomplete(stack3d):
        """Determine if frame is incomplete."""
        for slice in stack3d:
            # If only value in slice is 0, frame is incomplete.
            if ((np.min(slice) == 0) & (np.max(slice) == 0)):
                return True
        return False

    stack = czifile.imread(filename)
    stack = np.squeeze(stack)
    # Trim off last frame if incomplete.
    if trim:
        if frame_incomplete(stack[-1,0]):
            stack = stack[:-1]
    if (swapaxes):
        stack = np.swapaxes(stack,0,1)
    return stack

############################################################################
def read_czi_multiple(czi_files, folder):
    """Read a list of 5d czi files, combine into single stack, record
    frame junctions and positions of first Z slice.
    
    Args:
        czis: list-like (iterable)
            List of filenames containing 5d .czi movies of dimension 
            [c,t,z,x,y]. Shapes must be identical except for t dimension.
        folder: path
            Path to folder containing czi files
            
    Returns:
        stack: 5d ndarray
            Concatenation of input stacks
        frames: list of ints
            List of the frame numbers at which joins occur, with each entry
            representing the 0-indexed location of the first frame of a new
            stack.
        starting_positions: list of floats
            List of the position, in meters, of the first slice in the Z
            stack of each file, taken from czi file metadata.
    """
    def get_starting_position(czi_file_):
        metadata = czifile.CziFile(czi_file_).metadata()
        root = ET.fromstring(metadata)
        first_dist = root.findall('.//ZStackSetup')[0][8][0][0].text
        #last_dist = root.findall('.//ZStackSetup')[0][9][0][0].text
        return first_dist

    stacks = []
    starting_positions = []
    for czi_file_ in czi_files:
        czi_file_path = os.path.join(folder, czi_file_)
        stacks.append(read_czi(czi_file_path, trim=True))
        starting_positions.append(get_starting_position(czi_file_path))
        
    stack, frames = concatenate_5dstacks(stacks)
    return stack, frames, starting_positions

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

def viewer(stacks, figsize=12, order='default', zmax=False, init_minval=None, init_maxval=None, color="Greens", coordfile=None):
    """Interactive Jupyter notebook viewer for n-dimensional image stacks.
    
    Args:
        stack: ndarray or list of ndarrays
            List of n-dimensional image stacks; last two dimensions must 
            be x-y to display. Image shapes must be identical.
        figsize: int
            Size of the figure to plot
        order: string
            String specifying order of image dimensions. Examples: 'ctzxy' 
            or 'tzxy'. Last two dimensions must be 'xy'.
        zmax: bool
            If True, displays a maximum projection on the Z axis.
        init_minval: int
            Initial value for minimum on contrast slider
        init_maxval: int
            Initial value for maximum on contrast slider
        color: string
            [optional] set the initial color map.
        coordfile: string
            [Optional] File to which to write coordinates of mouse clicks.
            Only works in %matplotlib notebook mode.
            
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
        if (colmap == 'Gators'):
            colmap = combine_colormaps(plt.cm.Blues_r, plt.cm.YlOrBr)
        min_ = kwargs['contrast'][0]
        max_ = kwargs['contrast'][1]
        
        # Unpack order variable into array.
        order_arr = [char for char in order[:-2]]
        # Populate indexes list with widgets.
        for n in order_arr:
            indexes.append(kwargs[n])
        
        # Set up frame for plots.
        fig, ax = plt.subplots(1, numplots, figsize=(figsize * numplots, figsize * numplots))
        if (coordfile != None):
            cid = fig.canvas.mpl_connect('button_press_event', lambda event: click_coord(event, indexes[-1], indexes[-2]))
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
    
    # Write clicked coordinates to file.
    def click_coord(event, z, t):
            f = open(coordfile, 'a')
            f.write(str(t) + '\t' + str(z) + '\t' + str(event.ydata) + '\t' + str(event.xdata) + '\n')
            f.close()

    # Make a new slider object for dimension selection and return it
    def _make_slider(n):
        widget = IntSlider(min=0, max=(stack.shape[n] - 1), step=1, 
            continuous_update=False,)
        return(widget)
    
    # Combine two colormaps.
    def combine_colormaps(cm1, cm2):
        """Combine two mpl colormaps."""
        colors1 = cm1(np.linspace(0., 1, 128))
        colors2 = cm2(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        return(mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors))

    # Make a dropdown widget for selecting the colormap.
    def _make_cmap_dropdown():
        dropdown = Dropdown(
            options={'Greens', 'Reds', 'viridis', 'plasma', 'magma', 'inferno','cividis', 
            'gray', 'gray_r', 'prism', 'Gators'},
            value=color,
            description='Color',
        )
        return dropdown
    
    # Make a range slider for adjusting image contrast.
    def _make_constrast_slider():
        min_ = stack.min()
        max_ = stack.max()
        init_min = min_
        init_max = max_
        if (init_minval != None):
            init_min = init_minval
        if (init_maxval != None):
            init_max = init_maxval
        contrast_slider = IntRangeSlider(
            value=[init_min, init_max],
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
    if zmax:
        axis_max = stacks[0].ndim - 3
        for i in range(0, len(stacks)):
            stacks[i] = stacks[i].max(axis=axis_max)
        stack = stacks[0]

    if (order == 'default'):
        if zmax:
            order = 'ctxy'[4-stacks[0].ndim:]
        else:
            order = 'ctzxy'[5-stacks[0].ndim:]

    # Open coordfile for appending.
    if (coordfile is not None):
        f = open(coordfile, 'w')
        f.write('#\n')
        f.close()

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
    if (type(spot_data) == dict):
        for spot in spot_data:
            arr = spot_data[spot]
            for row in arr:
                row = row.astype(int)
                point = (row[[0,2,3,4]])
                drawbox(boxstack, point, halfwidth_xy, halfwidth_z, linewidth, hival, shadows)
    elif (type(spot_data) == list):
        for t in range(0, len(spot_data)):
            for row in spot_data[t]:
                point = tuple([t]) + tuple(row[0:3].astype(int))
                drawbox(boxstack, point, halfwidth_xy, halfwidth_z, linewidth, hival, shadows)
    return boxstack   

############################################################################
def quickview_ms2(stack, spot_data, channel=0, figsize=12, MAX=True, 
    halfwidth_xy=8, halfwidth_z=8, spotmode=False, spot_id='all', 
    color='cividis', init_minval=0, init_maxval=1000000, shadows=True):
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
        figsize: int
            Size of figure to display via viewer
        MAX: bool
            Display max projection in Z.
        halfwidth_xy: int
            Halfwidth in pixels of the boxes in xy direction (sides will be 
            2*halfwidth)
        halfwidth_z: int
            Halfwidth in pixels of the boxes in z direction (sides will be 
            2*halfwidth)
        spotmode: bool
            (optional) display a spot instead of a box
        spot_id: int 
            (optional) ID of spot to box. Default boxes all detected spots.
        color: string or color object
            Starting colormap for viewer
        init_minval: int
            Initial value for minimum on contrast slider
        init_maxval: int
            Initial value for maximum on contrast slider
        shadows: bool
            If true, show dark boxes in out of focus Z slices.

    """
    if (spot_id == 'all'):
        data = spot_data.copy()
    else:
        data = {spot_id:spot_data[spot_id]}
    
    if spotmode:
        halfwidth_xy = 3

    substack = stack[channel]
    boxes = box_spots(substack, data, halfwidth_xy=halfwidth_xy, halfwidth_z=halfwidth_z, linewidth=2, shadows=shadows)
    if MAX:
        viewer(boxes.max(axis=1), figsize, 'txy', color=color, init_minval=init_minval, init_maxval=init_maxval)
    else:
        viewer(boxes, figsize, 'tzxy', color=color, init_minval=init_minval, init_maxval=init_maxval)

############################################################################
def spot_movies(stack, spot_data, channel=0, len_ij=15, len_z=7, fill=np.nan, view=False):
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
    movies = np.zeros((max(spot_data)+1, stack.shape[1], len_z, len_ij, len_ij))
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
    Circularity calculation is 2D, using the single Z slice in which the
    lebeled object has the most pixels.
    
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
    if (perimeter == 0):
        perimeter = 0.5
    circularity = 4 * np.pi * area / (perimeter ** 2) 
    return circularity
 
def filter_labelmask_circularity(labelmask, slicenum, circularity_min=0.5):
    """Filter a 3D labelmask for object circularity.

    Implements imageJ circularity measure: 4pi(Area)/(Perimeter^2).
    Circularity calculation is 2D, using the single Z slice in which the
    lebeled object has the most pixels.

    Args:
        labelmask: ndarray
            3d integer labelmask
        slicenum: int
            Z slice to use for circularity calculations
        circularity_min: float
            ID of object for which to calculate circularity
            
    Return:
        circularity: float
            Minimum circularity for objects to pass filter
    """
    # Calculate circularity from object perimeter and area.
    good_objs = []
    im = labelmask[slicenum]
    regions = regionprops(im)
    for n in range(0, len(regions)):
        perimeter = regions[n].perimeter
        area = regions[n].area
        if (perimeter == 0):
            perimeter = 0.5
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if (circularity >= circularity_min):
            good_objs.append(regions[n].label)
    labelmask_filtered = np.where(np.isin(labelmask, good_objs), labelmask, 0)
        
    return labelmask_filtered

def filter_labelmask_circularity_apply4d(labelmask, slicenum, circularity_min=0.5):
    """Filter a 3D labelmask for object circularity.

    Wrapper for filter_labelmask_circularity

    Args:
        labelmask: ndarray
            4d integer labelmask
        slicenum: int
            Z slice to use for circularity calculations
        circularity_min: float
            ID of object for which to calculate circularity
            
    Return:
        circularity: float
            Minimum circularity for objects to pass filter
    """
    labelmask_filtered = np.zeros_like(labelmask)
    for n in range(0, labelmask.shape[0]):
        print(n, end=' ')
        labelmask_filtered[n] = filter_labelmask_circularity(labelmask[n], slicenum=slicenum, circularity_min=circularity_min)
        
    return labelmask_filtered

############################################################################
def filter_3dlabelmask_circularity(labelmask, slice):
    """

        label: int
            ID of object for which to calculate circularity
            
    Return:
        circularity: float
            output of circularity calculation 
    """
    # Calculate circularity from object perimeter and area.
    im = labelmask[slice]
    regions = regionprops(im)
    perimeter = regions[0].perimeter
    area = regions[0].area
    if (perimeter == 0):
        perimeter = 0.5
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

def filter_labelmask_apply4d(labelmask, func, above=0, below=1e6):
    labelmask_filtered = np.zeros_like(labelmask)
    for n in range(0, labelmask.shape[0]):
        print(n, end=' ')
        labelmask_filtered[n] = filter_labelmask(labelmask[n], func, above=above, below=below)
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
def segment_nuclei_3Dstack_rpb1(stack, seed_window=(15,50,50), 
    min_seed_dist=25, sigma=5, usemax=False, display=False, 
    return_intermediates=False):
    """Segment nuclei from Rpb1 fluorescence in confocal data.
    
    Algorithm is smooth -> threshold -> distance transform to find seeds ->
    take gradient on binary mask -> watershed on gradient. Does not do
    any filtering on resulting segmented objects.
   
    Args:
        stack: ndarray
            3D image stack of dimensions [z, x, y].
        seed_window: tuple of three ints
            Size in [z, x, y] for window for determining local maxes in 
            distance transform. A point is retained as a seed if there
            exists some window of this size in the image for which the point
            is the max value. Generally want size to be a little less than 2x 
            the distance between nuclear centers. Centers closer than this 
            will not produce two seeds.
        min_seed_dist: numeric
            The minimum euclidean distance (in pixels) allowed between watershed
            seeds. Typically set as ~the diameter of the nuclei.   
        sigma: numeric
            Sigma for use in initial gaussian smoothing
        usemax: bool
            Use maximum intensity projection (in Z) for segmenting
        return_intermediates: bool
            Return (mask, grad, seeds, ws) for troubleshooting
    
    Returns:
        labelmask: ndarray
            Mask of same shape as input stack with nuclei segmented and labeled
    
    """
    # Smooth stack using a Gaussian filter.
    if usemax:
        stack_smooth = ndi.gaussian_filter(stack.max(axis=0), sigma)
    else:
        stack_smooth = ndi.gaussian_filter(stack, sigma)
    #print('smooth')
    # Define a threshold for nuclear signal.
    thresh = threshold_otsu(stack_smooth)
    #print('thresh')
    # Make a binary mask using threshold.
    mask = np.where(stack_smooth > thresh, 1, 0)
    #print('mask')
    # Take the gradient of the mask to produce outlines for use in watershed algorithm.
    grad = gradient_nD(mask)
    # Perform distance transform and run local max finder to determine watershed seeds.
    dist = ndi.distance_transform_edt(mask)
    seeds, _ = peak_local_max_nD(dist, size=seed_window, min_dist=min_seed_dist)
    # Perform watershed segmentation.
    ws = watershed(grad, seeds.astype(int))
    # Filter object size and circularity, relabel to set background to 0.
    if usemax:
        ws = np.repeat(np.expand_dims(ws, axis=0), stack.shape[0], axis=0)
    labelmask = ws
    #labelmask = labelmask_filter_objsize(ws, size_min, size_max)
    #labelmask = filter_labelmask(labelmask, object_circularity, circularity_min, 1000)

    if (display):
        fig, ax = plt.subplots(3,2, figsize=(10,10))
        # Display mask.
        ax[0][0].imshow(mask.max(axis=0))
        ax[0][0].set_title('Initial Mask')
        # Display watershed seeds.
        seeds_vis = ndi.morphology.binary_dilation(seeds.max(axis=0), structure=np.ones((8,8)))
        ax[0][1].imshow(stack_smooth.max(axis=0), alpha=0.5)
        ax[0][1].imshow(seeds_vis, alpha=0.5)
        ax[0][1].set_title('Watershed seeds')
        # Display gradient.
        ax[1][0].imshow(grad.max(axis=0))
        ax[1][0].set_title('Gradient')
        # Display watershed output.
        ws = relabel_labelmask(ws)
        ax[1][1].imshow(ws.astype('bool').max(axis=0))
        ax[1][1].set_title('Watershed')
        # Display final mask.
        ax[2][0].imshow(labelmask.astype('bool').max(axis=0))
        ax[2][0].set_title('Final Segmentation')

    if return_intermediates:
        return (mask, grad, seeds, ws)
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
        print(n, end=' ')
        frame = seg_func(stack[n], **kwargs)
        labelmask = np.vstack((labelmask, [frame]))
    print('')
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
        labels, counts = np.unique(mask2[mask1 == label1], return_counts=True) # The slow step in this function is actually the mask2[mask1 == label1].
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
def connect_nuclei(maskstack_in, max_frames_skipped=2, 
                             update_func=update_labels, usemax=False):
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
        usemax: bool
            If true, uses the maximum intensity projection of the input mask
            to do connecting. This has huge speed benefits. The mask returned
            is still 4d, with the 2D results repeated in the Z dimension.
    
    Returns:
        connected_mask: ndarray
            4D labelmask of same shape as maskstack with object labels connected
            between frames. 
    """

    # Take maximum intensity projection in Z, if indicated.
    if usemax:
        maskstack = maskstack_in.max(axis=-3)
    else:
        maskstack = maskstack_in.copy()

    # Initialize blank mask.    
    connected_mask = np.zeros_like(maskstack)
    # Add first frame.
    connected_mask[0] = maskstack[0]
    # Walk through subsequent frames, connecting them to previous frames.
    for n in range(1, maskstack.shape[0]):
        print(n, end=' ')
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
    # If needed, expand mask in the Z dimension to match dimensions of stack.
    if usemax:
        connected_mask = expand_mip(connected_mask, maskstack_in.shape[-3]) 
    return connected_mask

############################################################################
def interpolate_nuclear_mask(mask_in, max_missing_frames=2, usemax=True):
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
        usemax: bool
            If true, perform interpolation using maximum projection in Z.
            
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
        # Assign the frame number to the first position (time) for all interpolated coordinates.
        interp_coords = tuple([np.repeat(frame, np.count_nonzero(obj_bool))])
        # For remaining dimensions, use centroid difference to "move" pixels of nucleus
        # from before frame to the interpolated position.
        for i in range(0, len(centroid_diff)):
            interp_coords = interp_coords + tuple([coords[i+1][obj_bool] + centroid_diff[i]])
        # Fix out-of bounds coordinates.
        for i in range(1,4):
            maxdim = newmask.shape[-i] - 1
            interp_coords[-i][interp_coords[-i] < 0] = 0
            interp_coords[-i][interp_coords[-i] > maxdim] = maxdim
        # Update newmask in place.
        newmask[interp_coords] = label

    # Main.
    mask = mask_in
    if usemax:
        mask = mask_in.max(axis=-3)
    t_frames = np.arange(0,mask.shape[0])
    newmask = np.copy(mask)
    labels = np.unique(mask)[1:]

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
    # Only accept interpolations in regions of the original mask that were 0.
    newmask = np.where((newmask > 0) & (mask == 0), newmask, mask)
    if usemax:
        newmask = expand_mip(newmask, mask_in.shape[-3])
    return newmask

############################################################################
def fit_ms2(stack, min_distances=(70,50,50), sigma_small=1, 
                   sigma_big=4, bg_radius=4, fitwindow_rad_xy=10, 
                   fitwindow_rad_z=2):  
    """Perform 3D gaussian fitting on local maxima in a 4D image stack
    
    Alrigthm: bandbass filter -> background subtraction -> find local maxima
    -> fit gaussian to windows around maxima
    
    Args:
        stack: ndarray
            4D image stack [t,z,x,y] containing MS2 spots
        min_distances: tuple of three ints
            Minimum distance (in pixels) allowed between spots for them to be
            counted as distinct spots. Minimum distance supplied for each 
            dimension.
        sigma_small: numeric
            Lower sigma for difference-of-gaussians bandpass filter
        sigma_small: numeric
            Upper sigma for difference-of-gaussians bandpass filter. Critical 
            for this to be at least one, otherwise hot pixels create problems
        bg_radius: int
            Radius for minimum filter used for background subtraction
        fitwindow_rad_xy: int
            Radius in pixels in the xy-dimension of the window around local
            maxima peaks within which `to do gaussian fitting.
        fitwindow_rad_z: int
            Radius in pixels in the z-dimension of the window around local
            maxima peaks within which to do gaussian fitting.
    
    Returns:
        fit_data: list of ndarrays
            Each entry in the list is a time point (frame). Each row in
            array is a fit (a single local maxima), columns are: 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: fit_height, 4: width_z, 5: width_x, 6: width_y). Coordinates 
            are adjusted so that if fit center lies outside the image, 
            center is moved to the edge.
    """

    # Task: change size to minimum distance
    def get_fitwindow(data, peak, xy_rad, z_rad):
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
    
    def fit_frame(substack, min_distances, sigma_small, 
                   sigma_big, bg_radius, fitwindow_rad_xy, 
                   fitwindow_rad_z):
        """Perform 3D gaussian fitting on a 3D image stack."""

        # Filter and background subtract image.
        dog = dog_filter(substack, sigma_small, sigma_big)
        bg = ndi.filters.minimum_filter(dog, bg_radius)
        dog_bs = dog - bg

        # Make a labelmask corresponding to local maxima peaks.
        peak_window_size = (min_distances[0] * 2 + 1, min_distances[1] * 2 + 1, min_distances[2] * 2 + 1)
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
                fitparams = np.vstack((fitparams, np.array([z_adj,x_adj,y_adj,0,np.inf,np.inf,np.inf])))
        return fitparams
    
    #### Main ####
    # Do fitting on first frame.
    fit_data_frame0 = fit_frame(stack[0], min_distances, sigma_small, 
                   sigma_big, bg_radius, fitwindow_rad_xy, 
                   fitwindow_rad_z)
    # Make fit_data a list of ndarrays.
    fit_data = [fit_data_frame0]
    
    # Fit the rest of the frames, add their data to fit_data.
    for i in range(1, stack.shape[0]):
        print(i, end=' ')
        fit_data_thisframe = fit_frame(stack[i], min_distances, sigma_small, 
                   sigma_big, bg_radius, fitwindow_rad_xy, 
                   fitwindow_rad_z)
        fit_data.append(fit_data_thisframe)
        
    return fit_data

############################################################################
def filter_ms2fits(fit_data, peakiness=4.5, stack=None, channel=1):
    """Filter MS2 spot fit data based on fit parameters
    
    Select spots based on "peakiness", measured as the ratio of the height
    to width of the fit. Currently uses the mean of the x- and y-widths.
    
    Args:
        fit_data: list of ndarrays
            Each entry in list is a distinct frame (in time), rows in array
            are individual spot fits and columns are 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: fit_height, 4: width_z, 5: width_x, 6: width_y.
        peakiness: numeric
            Spots whose log (natural) ratio of height to width exceeds this
            are kept.
        stack: ndarray
            [Optional] 5D stack from which fits are derived. If supplied,
            fits will be filtered for those whose peak surpasses the median
            value for the frame in this stack.
        channel: int
            Channel in the stack on which fits were performed.
            
    Returns:
        fit_data: list of ndarrays
            Input data, retaining only rows that pass filter.  
    """
    
    fit_data = fit_data.copy()
    for t in range(0, len(fit_data)):
        frame_data = fit_data[t]
        if stack is not None:
            frame_med = np.median(stack[channel, t])
        else:
            frame_med = -np.inf
        xy_width_means = np.mean(frame_data[:,5:7], axis=1)
        peak_heights = frame_data[:,3]
        spot_peakiness = np.log(peak_heights / xy_width_means)
        frame_data_filtered = frame_data[(peak_heights > frame_med) & (spot_peakiness > peakiness),:]
        fit_data[t] = frame_data_filtered
    return fit_data

############################################################################
def connect_ms2_frames(spot_data, nucmask, max_frame_gap=1, max_jump=10, 
    scale_xy=1, scale_z=1):
    """Connect detected MS2 spots in a single movie through multiple time 
    frames based on the distance between spots.
    
    Spots detected in new frame are connected to spots in previous frames
    if they are within specified distance (max_jump). Spots can "disappear" 
    for a number of frames defined by max_frame_gap. Spots that cannot be 
    connected to spots from prior frames are initialized as new spots.
    
    Args:
        spot_data: list of ndarrays
            Each entry in list is a distinct frame (in time), rows in array
            are individual detected spots and columns are 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: fit_height, 4: width_z, 5: width_x, 6: width_y.
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


    def get_peakiness(spot_data):
        """Determine peakiness for a spot from height and mean of x- and y- widths"""
        return spot_data[3] / np.mean((spot_data[5], spot_data[6]))

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
    
    def inbounds(coords, nucmask):
        """Ensure coordinates are within the image for looking up nucleus ID
        (Z-correction for re-focusing can result in out of bounds points, and XY-correction
        for nuclear movement can do the same)"""
        coords = list(coords)
        coords[1] = clamp(coords[1], 0, nucmask.shape[-3] - 1)
        coords[2] = clamp(coords[2], 0, nucmask.shape[-2] - 1)
        coords[3] = clamp(coords[3], 0, nucmask.shape[-1] - 1)
        return tuple(coords)


    def add_time_nuc(this_spot_data, t, nucmask):
        """ Append the frame number (t) and nucleus ID to front of spot_data"""
        # Combine frame number and zxy to for coordinate tuple, accounting for out-of-bounds z-coordinates due to re-focus adjustments.
        spot_coords = tuple(np.concatenate((
            [t], 
            this_spot_data[0:3]
            )).astype(int))
        
        nuc_id = nucmask[inbounds(spot_coords, nucmask)]
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
        print(t, end=' ')
        frame_data = input_data[t]
        for this_spot_data in frame_data:
            this_spot_data = add_time_nuc(this_spot_data, t, nucmask)
            update_spot(this_spot_data, connected_data, scale_z, scale_xy, max_frame_gap, t)
    
    return connected_data  

############################################################################
def connect_ms2_fits_focuscorrect(fits_orig, z_frames, z_corrs, nucmask, 
    max_frame_gap=1, max_jump=18, scale_xy=1, scale_z=1):
    """Connect MS2 spots (from fits) across time frames and from multiple 
    movies with different Z-focus points
    
    Args:
        fits_orig:list of ndarrays
            Output of fit_ms2 function. Each entry in the list is a time point 
            (frame). Each row in array is a fit (a single local maxima), 
            columns are: 0: center z-coordinate, 1: center x-coordinate, 
            2: center y-coordinate, 3: fit_height, 4: width_z, 5: width_x, 
            6: width_y).
        z_frames: list-like
            The frame numbers of the focus changes, with the number
            corresponding to the first frame (0-based) of the new focal point.
        z_corrs: list-like
            The Z-corrections matching the frames in z_frames. This is the 
            number that must be added (can be negative) to the new (frames 
            after the re-focus) frames to match the old. So if the focus is
            moved down by 2 Z-slices, the correction will be +2.
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
        spots_connected: dict of ndarrays
            Each key is a unique spot tracked across 1 or more frames. Each row
            of array is the spot's data for a single frame, with columns 0: frame
            number (t), 1: nucleus ID, 2: center Z-coordinate, 3: center X-coord-
            inate, 4: center Y-coordinate, 5: fit height, 6: fit z_width, 7: fit
            x_width, 8: fit y_width. Coordinates are relative to the original, 
            uncorrected data (which will be correct for a concatenated image 
            stack).
    
    """
    ## FIRST PART: Create a single fits object (list of ndarrays) with the Z-coordinates
    ## adjusted to correct for the refocusing.
    
    # Make a copy of fits to be altered.
    fits_adj = deepcopy(fits_orig) 
    # Make a vector of length equal to number of frames storing the z correction for each
    # frame (relative to first, uncorrected segment).
    frame_corrs = np.repeat(0, z_frames[0])  
    corr = 0
    
    # Do Z-correction for each focus break.
    for i in range(0, len(z_frames)):
        start_frame = z_frames[i]
        corr = corr + z_corrs[i]
        if (i+1 < len(z_frames)):
            end_frame = z_frames[i+1] - 1
        else:
            end_frame = len(fits_orig) - 1
        # Adjust z-coordinates for each frame in the window.
        for f in range(start_frame, end_frame + 1):
            fits_adj[f][:,0] = fits_adj[f][:,0] + corr
        # Add adjustment to frame_corrs vector.
        frame_corrs = np.concatenate((frame_corrs, np.repeat(corr, end_frame - start_frame + 1)))
    
    ## SECOND PART: Run spot connection on "corrected" fits object.
    spots_connected = connect_ms2_frames(fits_adj, nucmask, max_frame_gap, max_jump, scale_xy, scale_z)
    
    ## THIRD PART: Undo the adjustments in Z to make the spot coordinates match the concatenated
    ## stack.
    for spot in spots_connected:
        arr = spots_connected[spot]
        for i in range(0, len(arr)):
            frame_num = int(arr[i,0])
            arr[i,2] = arr[i,2] - frame_corrs[frame_num]
    return spots_connected

############################################################################
def connect_ms2_frames_via_nuclei(frame_data, nucmask, nucmask_dilation=5):
    """Connect MS2 spots detected in separate time frames based on nucleus
    ID.
    
    Args:
        frame_data: list of ndarrays
            Output from fitting. Each entry in the list is a time point 
            (frame). Each row in array is a fit (a single local maxima), 
            columns are: 0: center z-coordinate, 1: center x-coordinate, 
            2: center y-coordinate, 3: fit_height, 4: width_z, 5: width_x,
            6: width_y). 
        nucmask: ndarray
            4d labelmask of segmented nuclei
        nucmask_dilation: int
            Distance to dilate (in x and y directions) nuclear labelmask for
            looking up the nuclear ID of detected spots.
    
    Returns:
        spot_data: dict of ndarrays
            Each key is a unique spot tracked across 1 or more frames. Each row
            of array is the spot's data for a single frame, with columns 0: frame
            number (t), 1: nucleus ID, 2: center Z-coordinate, 3: center X-coord-
            inate, 4: center Y-coordinate, 5: fit height, 6: fit z_width, 7: fit
            x_width, 8: fit y_width
    """
    def find_nuc_id(coords, t, nucmask_lookup):
        """Find object in nuclear mask at particular coordinate."""
        # Make 4d coordinates as a list.
        coords_nucmask = np.concatenate([[t], coords])
        # Convert coordinates to a tuple of ints.
        coords_nucmask = tuple([int(n) for n in coords_nucmask])
        nuc_id = nucmask_lookup[coords_nucmask]
        return nuc_id
    
    def add_row(row, spot_data):
        """Add a new row (spot data for a single frame) to spot data 
        object."""
        # Pull nuc_id and t off of row.
        t, nuc_id = row[0:2].astype(int)
        # If this nucleus is already in spot data, add to existing data.
        if (nuc_id in spot_data):
            # Check to see if there is already a spot for this nucleus and time.
            if (spot_data[nuc_id][-1,0] == t):
                # If spot_data for this nucleus has a previous entry, choose
                # closest spot. Otherwise just leave current.
                if (spot_data[nuc_id].shape[0] > 1):
                    # Find the euclidean distance between the spot position in the
                    # most recent frame and both the candidate spots in the new
                    # frame.
                    coords_previous_frame = spot_data[nuc_id][-2,2:5]
                    coords_existing = spot_data[nuc_id][-1,2:5]
                    coords_new = row[2:5]
                    # full function: scipy.spatial.distance.euclidean
                    dist_existing = distance.euclidean(coords_previous_frame, coords_existing)
                    dist_new = distance.euclidean(coords_previous_frame, coords_new)
                    # Select closest spot to previous to go into spot data.
                    if (dist_new < dist_existing):
                        spot_data[nuc_id][-1] = row
            # If no spot already, just add new row to end.
            else:
                spot_data[nuc_id] = np.vstack([spot_data[nuc_id], row])
        # If nucleus isn't in spot data, initialize new entry.   
        else:
            spot_data[nuc_id] = np.expand_dims(row, axis=0)
            
    def renumber_spots(spot_data):
        """Renumber the spots in spot_data sequentially from 1."""
        spot_data_renumbered = {}
        spot_number_new = 1
        for spot_number_old in spot_data:
            spot_data_renumbered[spot_number_new] = spot_data[spot_number_old]
            spot_number_new += 1
        return spot_data_renumbered
    
    # Make a new lookup version of nuclear mask that dilates (expands in all 
    # directions) the nucleus objects.
    
    nucmask_lookup = labelmask_apply_morphology(nucmask, ndi.morphology.binary_dilation, 
        struct=np.ones((1,1,nucmask_dilation,nucmask_dilation)), 
        expand_size=(1,1,nucmask_dilation,nucmask_dilation)) 
   
    #nucmask_lookup = nucmask
    
    # Go through all frames of input data, lookup nuclei for each detected spot,
    # Add data to spot_data.
    spot_data = {}
    nframes = len(frame_data)
    for t in range(0, nframes):
        print (t, end=' ')
        for row in frame_data[t]:
            coords = row[0:3]
            nuc_id = find_nuc_id(coords, t, nucmask_lookup)
            if (nuc_id != 0):
                new_row = np.concatenate([[t, nuc_id], row])
                add_row(new_row, spot_data)
                
    return renumber_spots(spot_data)

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

    NOTE: For math reasons I don't fully understand, using radii equal to 
    n (e.g. z_rad = 2) will give you a radius equal to n - 1. I recommend
    using non-integers slightly higher than the desired (like z_rad = 2.1)
    radius to avoid this problem.  

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
            Radius in real units of ellipsoid in the z dimension. For reasons
            I don't totally understand, a radius of 1 will actually give a 
            radius of 0, while 1.1 will give the desired behavior.
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
        z0, i0, j0 = [int(x) for x in coords]

        # Ellipsoid will be bounded by a box of dimensions defined by the radii 
        # (2 * r + 1 on each side). Performing the matrix operations only on this
        # box, rather than the whole stack, makes it way way faster.

        # Define a box around the coordinates in both the meshgrid and the stack.
        z_start = int(np.max([0, z0 - z_rad]))
        z_end = int(np.min([stack.shape[-3] - 1, z0 + z_rad]))
        i_start = int(np.max([0, i0 - ij_rad]))
        i_end = int(np.min([stack.shape[-2] - 1, i0 + ij_rad]))
        j_start = int(np.max([0, j0 - ij_rad]))
        j_end = int(np.min([stack.shape[-1] - 1, j0 + ij_rad]))

        substack = stack[z_start:(z_end+1), i_start:(i_end+1), j_start:(j_end+1)]

        submeshgrid = meshgrid.copy()
        for i in range(0, len(submeshgrid)):
            submeshgrid[i] = submeshgrid[i][z_start:(z_end+1), i_start:(i_end+1), j_start:(j_end+1)]
        
        # Use meshgrid to select pixels within the ellipsoid.
        valsgrid = np.sqrt((a * ((submeshgrid[0] - z0) ** 2)) + ((submeshgrid[1] - i0) ** 2) + ((submeshgrid[2] - j0) ** 2))
        pixels = substack[valsgrid <= r]
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
            print(num_processed, end=' ')
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

############################################################################
def spotdf_bleach_correct(df, stack4d, sigma=10):
    """Perform bleach correction on a 4d image stack using the (smoothed) 
    frame average, apply correction to columns of a pandas df.

    Args:
        df: pandas dataframe
            Spot values, rows are time frames and columns are spots
        stack4d: ndarray
            4D image stack to use for bleach correction
        sigma: number-like
            Sigma value for gaussian filtering of frame means

    Returns:
        df_corr: pandas dataframe
            Input dataframe with bleaching correction applied 


    """
    frame_means = np.mean(stack4d, axis=(1,2,3))
    frame_means_smooth = ndi.gaussian_filter(frame_means, sigma=sigma)
    means_norm = frame_means_smooth / frame_means_smooth[0]
    df_corr = df.apply(lambda x: x / means_norm, axis=0)
    return df_corr

############################################################################
def spotdf_plot_traces(df1, df2, minlen, sigma=0.8, norm=True):
    """Plot individual traces from MS2 spot pandas dfs with a minimum 
    trajectory length filter.
    
    Args:
        df1: pandas df
            First dataset to plot
        df2: pandas df
            Second dataset to plot
        minlen: int
            Minimum length (number of non-nan values) of trajectories to plot
        sigma: numeric
            Sigma for gaussian smoothing of traces
        norm: bool
            If true, traces are normalized between the 5th and 95th percentile
            of all non-nan values in the dataset
    
    Returns: none 
    """
    
    def norm_trace(df, x, lower, upper):
        """Normalize traces given upper and lower bounds"""
        return (df.iloc[:,x] - lower) / (upper - lower)
    def df_filter_trajlen(df, df_to_count, minlen):
        """Filter pandas df columns for minimum number of non-nan entries."""
        return  df.loc[:,df_to_count.apply(lambda x: np.count_nonzero(~np.isnan(x)), axis=0) > minlen]
    
    # Filter input dfs for trajectory length.
    df1_processed = df_filter_trajlen(df1, df1, minlen)
    df2_processed = df_filter_trajlen(df2, df1, minlen)
    
    # Get limits for normalization.
    if norm:
        df1_lower, df1_upper = np.nanpercentile(df1_processed.to_numpy().flatten(), [5, 95])
        df2_lower, df2_upper = np.nanpercentile(df2_processed.to_numpy().flatten(), [5, 95])
    
    num_to_plot=df1_processed.shape[1]
    
    # Define function for making each plot.
    def plot_function(x):
        if norm:
            #plt.plot(ndi.gaussian_filter1d(norm_trace(df1_processed, x, df1_lower, df1_upper), sigma))
            #plt.plot(ndi.gaussian_filter1d(norm_trace(df2_processed, x, df2_lower, df2_upper), sigma))
            plt.plot(norm_trace(df1_processed, x, df1_lower, df1_upper))
            plt.plot(norm_trace(df2_processed, x, df2_lower, df2_upper))
        else:
            plt.plot(ndi.gaussian_filter1d(df1_processed.iloc[:,x], sigma))
            plt.plot(ndi.gaussian_filter1d(df2_processed.iloc[:,x], sigma))
        plt.title(df1_processed.columns[x])
    
    # Multiplot using plot_ps.
    plot_ps(plot_function, range(0,num_to_plot))

############################################################################
def spotdf_plot_traces_bleachcorrect(df1, df2, minlen, stack4d, sigma=0.8, 
        norm=True):
    """Plot individual traces from MS2 spot pandas dfs with a minimum 
    trajectory length filter and bleaching correction with the 
    spotdf_bleach_correct function.
    
    Args:
        df1: pandas df
            First dataset to plot
        df2: pandas df
            Second dataset to plot
        minlen: int
            Minimum length (number of non-nan values) of trajectories to plot
        stack4d: 4d ndarray
            Stack used for bleaching correction
        sigma: numeric
            Sigma for gaussian smoothing of traces
        norm: bool
            If true, traces are normalized between the 5th and 95th percentile
            of all non-nan values in the dataset
    
    Returns: none 
    """
    spotdf_plot_traces(spotdf_bleach_correct(df1, stack4d), spotdf_bleach_correct(df2, stack4d), minlen, sigma, norm)

############################################################################
def correct_spot_data_depth(spot_data, slope=-338, slice_thickness=0.66, 
    cols=[9,10,11], return_dfs=False):
    """Correct a spot_data object for z depth (scattering)
    
    Scattering effects are assumed to be linear (empirically true at typical
    depths of embryonic nuclei). A function must be independently derived for
    the decay of fluorescence intensity as a function of embryo depth. 
    
    Args:
        spot_data: dict of ndarrays
            Each key is a unique spot tracked across 1 or more frames. Each row
            of array is the spot's data for a single frame, with columns 0: frame
            number (t), 1: nucleus ID, 2: center Z-coordinate, 3: center X-coord-
            inate, 4: center Y-coordinate, 5: fit height, 6: fit z_width, 7: fit
            x_width, 8: fit y_width, 9: integrated volume for MS2, 10: integrated
            gaussian fit of MS2 spots, 11: integrated volume for protein signal,
            additional columns if added.
        slope: numeric
            The slope of the intensity function, in units of intensity (a.u.)
            per micron.
        slice_thickness: numeric
            Thickness, in microns, of Z slices
        cols: array-like of ints
            Numbers of the columns to correct (typically the columns containing 
            fluorescence intensity values)
        return_dfs: bool
            If true, creates pandas dataframes for each of the indicated columns
            using make_spot_table function.
    
    Returns:
        corr_spot_data: dict of ndarrays
            If return_dfs is false, corrected version of input spot_data
        dfs: list of pandas dfs
            If return_dfs is true, spot tables corresponding to the corrected
            columns of the input spot_data table
    """
    
    corr_spot_data = {}
    max_frame = 0
    # Correct the data for each spot in spot_data.
    for spot_id in spot_data:
        arr = spot_data[spot_id].copy()
        # Correction vector determines intensity to "add" to each row based
        # on its Z slice
        corr_vector = arr[:,2] * slice_thickness * slope
        # Apply correction to each indicated column.
        for col in cols:
            arr[:,col] = arr[:,col] - corr_vector
        corr_spot_data[spot_id] = arr
        # Track maximum time frame in dataset.
        max_frame_thisspot = np.max(arr[:,0])
        if (max_frame_thisspot > max_frame):
            max_frame = max_frame_thisspot
    
    # Create and return pandas dfs, if indicated.
    if return_dfs:
        dfs = []
        for col in cols:
            df = movie.make_spot_table(corr_spot_data, np.zeros(int(max_frame)+1), col)
            dfs.append(df)
        return dfs
    
    # If dfs not requested, return corrected spot data.
    else:
        return corr_spot_data


