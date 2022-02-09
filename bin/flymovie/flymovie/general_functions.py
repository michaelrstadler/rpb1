#!/usr/bin/env python

"""
General functions for analyzing image data.

"""
__version__ = '1.1.0'
__author__ = 'Michael Stadler'

import numpy as np
import os
from scipy import ndimage as ndi
from skimage.segmentation import flood_fill
from skimage.measure import regionprops
from scipy.spatial import distance
from dataclasses import dataclass
import pandas as pd



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
        clamped number
    """
    return max(min(maxn, n), minn)

############################################################################
def labelmask_filter_objsize(labelmask, size_min, size_max):
    """Filter objects in a labelmask by size.

    Args:
        labelmask: ndarray
            N-dimensional integer labelmask (background must be 0 or errors 
            occur)
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
            valent of the point you click when filling in paint. If left
            as 'default', uses find_background_point to select point.
    Returns:
        mask_filled: ndarray
            Binary mask filled    
    '''
    if (seed_pt == 'default'):
        # Get a random 0-valued background pixel.
        seed_pt = find_background_point(mask)
    # Flood background from seed point.
    mask_flooded = flood_fill(mask, seed_pt, 1)
    # Set whatever changed (from 0 to 1) from the filling to background;
    # all else is foreground.
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
    props = regionprops(conn_comp)
    for id_ in range(0, len(props)):
        centroid = props[id_].centroid
        # Convert floats to ints.
        centroid = tuple([int(x) for x in centroid])
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
            So if arr is a (2,10,16,256,256) array, and n is 3, it will make 
            a mesh of size (16,256,256).
    
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
    """Sort the items of a list by the frequency with which they occur,
    return sorted list of unique items.
    
    Args:
        x: list-like
            List-like object to sort
        descending: bool
            If true, most frequent list item appears first.
    
    Returns:
        items_sorted: array
            List of unique items from original array sorted by frequency  
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
    new_df =  df.loc[:,df.apply(lambda x: np.count_nonzero(~np.isnan(x)), axis=0) >= minlen]
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
        """Expand a mask made from a maximum intensity projection by 
        repeating in Z (-3) dimension n times.
        
        Main use: A maximum intensity projection (in Z) is used for doing 
        some work, and you want to recover the Z dimension to match the 
        shape of the original (no MIP) stack.

        Args:
            mask: ndarray
                3-dimensional binary mask, time (frame) is the first 
                dimension
            n: int
                Number of Z slices to create (will repeat 2D mask n times
                in the Z dimension)

        Returns:
            expanded_mask: ndarray
                4d binary mask of dimensions equal to input except with
                n inserted as the second dimension.
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
def stack_normalize_minmax(instack):
    """Normalize ndarray to have values between 0 and 1.

    Args:
        instack: ndarray
            Image stack in order [z, x, y]

    Returns:
        stack: ndarray
            Image stack of same shape as instack.
    """
    stack = np.copy(instack)    
    return (stack - np.min(stack)) / (np.max(stack) - np.min(stack))

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
    for z in range(0,stack.shape[0]):
        immean = stack[z].mean()
        stack[z] = stack[z] / immean * stackmean
    return(stack)

############################################################################
def make_tables_from_arivis(trackfile, nucfile, spotfile):
    """Take output csv table data from arivis segmentation of MS2 movies
    and wrangel it into a reasonable pandas dataframe object.
    
    Inputs are the filenames for the csv exports of the tracks, nucleus
    objects and spots from arivis.
    
    Output is a list of data containers. Each item in list is a spot.
    Each spot contains 'nuc_all' and 'spot_all' which contain all the exported
    parameters for the nucleus and MS2 spot, and 'nuc' and 'spot' which 
    contain a reduced set of parameters with better names."""

    @dataclass
    class DataCont:
        nuc_all: pd.DataFrame
        spot_all: pd.DataFrame
        nuc: pd.DataFrame
        spot: pd.DataFrame

    tracks = pd.read_csv(trackfile)
    nucs = pd.read_csv(nucfile)
    spots = pd.read_csv(spotfile)

    data = []
    track_num = 0
    for index, row in tracks.iterrows():
        data.append({})
        for child_id in row['Children Ids'].split(','):
            nuc_data = nucs[nucs['Id'] == int(child_id)]
            if nuc_data.shape[0] != 0:
                nuc_id = int(nuc_data['Id'])
                if 'nuc' not in data[track_num]:
                    data[track_num]['nuc'] = nuc_data
                else:
                    data[track_num]['nuc'] = pd.concat([data[track_num]['nuc'], nuc_data]) 
                
                spot_data = spots[spots['Parent Ids'] == nuc_id]
                if spot_data.shape[0] != 0:
                    if 'spot' not in data[track_num]:
                        data[track_num]['spot'] = spot_data
                    else:
                        data[track_num]['spot'] = pd.concat([data[track_num]['spot'], spot_data]) 
        track_num += 1
        
    selected = ['First, Time Point', 'X (px), Center of Bounding Box',	'Y (px), Center of Bounding Box',	'Z (px), Center of Bounding Box',
        'Min, Intensities #1',	'Max, Intensities #1',	'Mean, Intensities #1',	'Sum, Intensities #1',	'SD, Intensities #1',
        'Min, Intensities #2',	'Max, Intensities #2',	'Mean, Intensities #2',	'Sum, Intensities #2',	'SD, Intensities #2']
    selected_simple = ['t', 'x', 'y', 'z', 'min_intensity_1',	'max_intensity_1',	'mean_intensity_1',	'sum_intensity_1',	'sd_intensity_1',
        'min_intensity_2',	'max_intensity_2',	'mean_intensity_2',	'sum_intensity_2',	'sd_intensity_2']
    data_final = []
    
    for i in range(0, len(data)):
        if 'spot' in data[i]:
            nuc = data[i]['nuc']
            spot = data[i]['spot']
            nuc_selected = nuc[selected]
            spot_selected = spot[selected]
            nuc_selected.columns = selected_simple
            spot_selected.columns = selected_simple
            data_final.append(DataCont(nuc, spot, nuc_selected, spot_selected))
            
    return data_final

############################################################################
def extract_box(stack, coords, box_dims, pad=True):
    """Extract a box (subwindow) from an image stack.

    Args:
        stack: ndarray, image stack (n-dimensional)
        coords: iterable of ints, coordinates for box center
        box_dims: iterable of odd ints, dimensions of box to extract
        pad: bool, pad with zeros if box falls outsize stack borders.
            If false, boxes that fall outside edge will return boxes
            that are smaller than box_dims

    Returns:
        box: ndarray, sub-window of image stack
    """
    def get_dim_slice(stack, coords, len_, dim):
        """Get slices for image stack and box, adjusted for edge
        crossing if necessary."""
        halflen = int(len_ / 2)
        stack_start = coords[dim] - halflen
        stack_end = coords[dim] + halflen + 1
        box_start = 0
        box_end = box.shape[dim]
        max_pos = stack.shape[dim]

        if stack_start < 0:
            box_start = -1 * stack_start
            stack_start = 0
        
        if stack_end > max_pos:
            box_end = box_end - (stack_end - max_pos)
            stack_end = max_pos 

        stack_slice = slice(stack_start, stack_end)
        box_slice = slice(box_start, box_end)
        return stack_slice, box_slice

    # Check that all box dimensions are odd.
    for d in box_dims:
        if (d % 2 == 0):
            raise ValueError('Box dimensions must be odd.')
    if len(box_dims) != stack.ndim:
        raise ValueError('Box and stack dimensions do not match.')

    # Initialize an empty box and slice tuples for each.
    box = np.zeros(box_dims)
    stack_slice_all = ()
    box_slice_all = ()
    # Build slice objects for each dimension.
    for d in range(stack.ndim):
        stack_slice_dim, box_slice_dim = get_dim_slice(stack, coords, box_dims[d], d)
        stack_slice_all = stack_slice_all + tuple([stack_slice_dim])
        box_slice_all = box_slice_all + tuple([box_slice_dim])

    if pad:
        box[box_slice_all] = stack[stack_slice_all]
        return box
    
    else:
        return stack[stack_slice_all]

############################################################################
def make_3d_gaussian_inabox(intensity, sigma, 
        z_windowlen, ij_windowlen, z_ij_ratio=2.94, p=1):
    """Make a 3D gaussian signal within a box of defined size.
    
    Implementation is of a generalized or super gaussian:
    1 / (sigma * sqrt(2 pi)) * exp(-1 * (d^2 / (2 * sigma^2))^p)

    When p = 1, it's a normal gaussian. For higher powers, the function
    becomes more 'flat-topped'.

    The distances are scaled to account for anisotropic z vs. ij 
    dimensions.

    Args:
        intensity: numeric, intensity of gaussian (height in 1d)
        sigma: numeric, sigma of gaussian
        z_windowlen: int, length in z dimension
        ij_windowlen: int, length in ij dimension
        z_ij_ratio: float, ratio of voxel size in z to ij dimension
        p: float, shape parameter for generalize (super-) gaussian
    """
    mesh = mesh_like(np.ones((z_windowlen, ij_windowlen, ij_windowlen)), n=3)
    # Adjust z coordinates to account for non-isotropy.
    mesh[0] = mesh[0] * z_ij_ratio
    midpoint_z = int(z_windowlen / 2 * z_ij_ratio)
    midpoint_ij = int(ij_windowlen / 2)
    # Calculate squares distance of each point.
    d2 = ((mesh[0] - midpoint_z) ** 2) + ((mesh[1] - midpoint_ij) ** 2) + ((mesh[2] - midpoint_ij) ** 2)
    # Calculate gaussian as PDF.
    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-1. * ((d2) / (2 * (sigma ** 2))) ** p))
    # Scale gaussian to have max of 1, multiply by intensity.
    gauss = gauss * (1 / np.max(gauss)) * intensity
    return gauss 