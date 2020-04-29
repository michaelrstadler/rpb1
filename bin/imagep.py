#!/usr/bin/env python

"""
Insert description here.

"""
__version__ = '1.0.0'
__author__ = 'Michael Stadler'


import numpy as np
import os
import 
from os import listdir
from os.path import isfile, join
import re
import skimage
from skimage import filters, io
from ipywidgets import interact, IntSlider, Dropdown, IntRangeSlider, fixed
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from functools import partial

############################################################################
# General image processing functions
############################################################################

# matlab imfill
def imfill(mask, seed_pt='default'):
    '''Fill holes in a binary mask.
    
    Equivalent to matlab's imfill function, conceptually identical to fill
    functions in paint programs. seed_pt needs to be a point in the "back-
    ground" area to fill. All 0 or False pixels directly contiguous with the 
    seed are defined as background, all other pixels are declared foreground.
    Thus any "holes" (0 pixels that are not contiguous with background) are 
    filled in.
    
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
    # By default, start in upper left corner.
    if (seed_pt == 'default'):
        seed_pt = tuple(np.zeros(mask.ndim).astype(int))
    # Fill all background pixels by changing them to 1. Changes are made to
    # original mask, so 1s are carried over in mask_flooded.
    mask_flooded = skimage.segmentation.flood_fill(mask, seed_pt,1)
    # Identify background pixels by those that are changed from original mask.
    # Unchanged pixels (0s and 1s) in original mask are now "filled" foreground.
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
def peak_local_max_nD(img, size=(70,100,100)):
    """Find local maxima in an N-dimensional image.
    
    Generalizes scikit's peak_local_max function to three (or more) 
    dimensions. Finds local maxima pixels within supplied window, determines
    centroids for connected pixels, and returns a mask of these centroid
    positions and a list of them.
    
    Suggested usage: finding seed points for watershed segmentation.
    
    Args:
        img: ndarray
            N-dimensional image stack
        size: tuple of ints
            Size of the window for finding local maxima. The sizes are the
            dimensions of the filter used to search for maxima. So a size
            of (100, 100) will use a square with side lengths of 100 pixels.
            Generally, you want the size dimensions to match the dimensions
            of the objects you're searching for.
    
    Returns:
        local_peak_mask: ndarray
            A binary mask with dimensions equal to img of single pixels
            representing local maxima.
        local_peaks: list of tuples
            Coordinates of pixels masked in local_peak_mask  
    """
    # Find pixels that represent local maxima. Produces clusters of connected
    # pixels at the centers of objects.
    maxes = local_max(img, size)
    # Connect these pixels in a labelmask.
    conn_comp, info = ndi.label(maxes)
    # Get the centroids of each local max object, update mask and list.
    local_peak_mask = np.zeros_like(img)
    local_peaks = []
    for x in np.unique(conn_comp)[1:]:
        centroid = get_object_centroid(conn_comp, x)
        local_peak_mask[centroid] = 1
        local_peaks.append(centroid)
        
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
# Functions for loading TIFF stacks
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
# Functions for interactive image viewing/analysis
############################################################################

def viewer(stacks, order='tzxy'):
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
        fig, ax = plt.subplots(1, numplots, figsize=(6 * numplots, 6 * numplots))
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
    main(order)

############################################################################
# Functions for segmenting images
############################################################################

def mask_plane(stack, top_z0, bottom_z0, top_zmax, side='>', maskval=0):
    """Draw a plane through a 3D image and mask all positions on one side of 
       it.

    Args:
        stack: ndarray
            Image stack in order [..., x, y]
        
        ## Note: any three points will work as three points define a plane,
        ## the position descriptions here are just convenient suggestions.
        top_z0: tuple of three ints
            Point in the plane at the top of the image in slice z=0.
        bottom_z0: tuple of three ints
            Point in the plane at the bottom of the image in slice z=0.
        top_zmax: tuple of three ints
            Point in the plane at the top of the image in last z slice.
        
        side: string
            '>' or '<' determine side of plane on which to mask indices (for
            embryo border masking, > masks right, < masks left)
        maskval: int
            Value with which to replace masked values.
            
    Returns:
        stack_masked: ndarray
            Image stack with same dimensions as input stack, masked.
    
    Raises:
        ValueError: side isn't '<' or '>'.

    """
    # Recursive function that applies 3D mask to entire n-dimensional stack
    def _apply_mask(substack, mesh, d, side, maskval):
        # If 3-d stack, apply mask
        # Note: changes occur in place; don't have to return up the recursion chain
        if (len(substack.shape) == 3):            
            if side == '>':
                substack[mesh > d] = maskval
            elif side == '<':
                substack[mesh < d] = maskval
        # If not 3-d stack, call self on each substack of left-most dimension  
        else:
            for n in range(0, substack.shape[0]):
                _apply_mask(substack[n,...], mesh, d, side, maskval)
    
    if side not in {'<', '>'}:
        raise ValueError("side must be < or >") 
    
    # Using z, i, j notation throughout 
    max_i = stack.shape[-2] - 1
    max_z = stack.shape[-3] - 1
    
    ## Use vector solution to find equation of plane given three points.
    # Define 3 points in the plane.
    p1 = np.array(top_z0)
    p2 = np.array(bottom_z0)
    p3 = np.array(top_zmax)

    # Define two vectors that lie in the plane.
    v1 = p3 - p1
    v2 = p2 - p1

    # Take their cross product to produce a vector normal to the plane, this 
    # vector provides coefficients for equation in form ax + by + cz = d
    cp = np.cross(v1, v2)
    a, b, c = cp

    # To solve for d (az + bi + cj = d), take dot product of normal vector and 
    # any point in plane.
    d = np.dot(cp, p3)
    
    ## Make 3D mesh grid where value is the position in the given direction
    # Make linear vectors used to construct meshgrids.
    z = np.arange(0, stack.shape[-3])
    i = np.arange(0, stack.shape[-2])
    j = np.arange(0, stack.shape[-1])
    # Make meshgrids from vectors.
    zmesh, imesh, jmesh = np.meshgrid(z, i, j, sparse=False, indexing='ij')
    # Make mesh_sum array from line equation, value at each position is right 
    # side of line equation
    mesh_sum = a*zmesh + b*imesh + c*jmesh
    stack_masked = np.copy(stack)
    _apply_mask(stack_masked, mesh_sum, d, side, maskval)
    return(stack_masked)

############################################################################
def segment_embryo(stack, channel=0, sigma=5, walkback = 50):
    """Segment the embryo from extra-embryo space in lattice data.
    
    Details: Crudely segments the embryo from extra-embryonic space in 5-
    dimensional stacks. Performs a gaussian smoothing, then thresholds,
    then uses morphological filtering to fill holes and then to "walk
    back" from right-to-left, based on the observation that segementation 
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
        t = filters.threshold_minimum(im_smooth)
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
def segment_nuclei_1(ref_stack, sigma=4, percentile=95, size_max=1e5, 
                     size_min=5000, erode_by=5):
    """Segment nuclei from a single 3D lattice stack.
    
    Details: Segments nuclei in lattice light sheet image substack. Uses
    gaussian smoothing and thresholding with a simple percentile to
    generate initial nuclear mask, then erodes this mask slightly, con-
    nects components, filters resulting objects for size, and returns
    a 3D labelmask of filtered structures.
    
    Optional: Input can be pre-segmented from background by segment_embryo
    function. This can help to standardize use of percentile-based 
    thresholding.
    
    Args:
        ref_stack: 3D ndarray
            Image stack in order [z, x, y]. This is a representative 
            substack (single channel and timepoint) of the full stack
            on which to perform segmentation.
        sigma: int
            Sigma value to use for gaussian smoothing
        percentile: int
            Percentile value to use for thresholding. Only non-zero pixels
            are used in calculating percentiles.
        size_max: int
            Upper size cutoff for connected structures (nuclei)
        size_min: int
            Lower size cutoff for connected structures
        erode_by: int
            Size of the structuring element (in x-y only) used to erode
            preliminary thresholded mask.
            
    Returns:
        labelmask: ndarray
            Same shape as input stack, filtered segmented structures are 
            masked by unique integer labels.
    """
    # Smooth input image.
    ref_smooth = ndi.filters.gaussian_filter(ref_stack, sigma=sigma)
    # Assign threshold value based on percentile of non-zero pixels, mask on threshold.
    t = np.percentile(ref_smooth[ref_smooth > 0], percentile);
    mask = np.where(ref_smooth > t, True, False)
    # Erode binary mask.
    mask = ndi.morphology.binary_erosion(mask, structure=np.ones((1, erode_by, erode_by)))
    # Label connected components to generate label mask.
    conn_comp, info = ndi.label(mask)
    # Filter labelmask based on maximum and minimum structure size.
    (labels, counts) = np.unique(conn_comp, return_counts=True)
    labels_selected = labels[(counts >= size_min) & (counts <= size_max)]
    labelmask = np.where(np.isin(conn_comp, labels_selected), conn_comp, 0)
    return labelmask

############################################################################
def segment_nuclei3D(stack, sigma=4, percentile=95, size_max=2e5, 
                     size_min=5000, erode_by=5):
    """Segment nuclei from a single 3D lattice stack.
    
    Details: Segments nuclei in lattice light sheet image substack. Uses
    gaussian smoothing and thresholding with a simple percentile to
    generate initial nuclear mask, then erodes this mask slightly, con-
    nects components, filters resulting objects for size, and returns
    a 3D labelmask of filtered structures.
    
    Optional: Input can be pre-segmented from background by segment_embryo
    function. This can help to standardize use of percentile-based 
    thresholding.
    
    Args:
        stack: 3D ndarray
            Image stack in order [z, x, y]. This is a representative 
            substack (single channel and timepoint) of the full stack
            on which to perform segmentation.
        sigma: int
            Sigma value to use for gaussian smoothing
        percentile: int
            Percentile value to use for thresholding. Only non-zero pixels
            are used in calculating percentiles.
        size_max: int
            Upper size cutoff for connected structures (nuclei)
        size_min: int
            Lower size cutoff for connected structures
        erode_by: int
            Size of the structuring element (in x-y only) used to erode
            preliminary thresholded mask.
            
    Returns:
        labelmask: ndarray
            Same shape as input stack, filtered segmented structures are 
            masked by unique integer labels.
    """
    # Smooth input image.
    stack_smooth = ndi.filters.gaussian_filter(stack, sigma=sigma)
    # Assign threshold value based on percentile of non-zero pixels, mask on threshold.
    t = np.percentile(stack_smooth[stack_smooth > 0], percentile);
    mask = np.where(stack_smooth > t, True, False)
    # Erode binary mask.
    mask = ndi.morphology.binary_erosion(mask, structure=np.ones((1, erode_by, erode_by)))
    # Label connected components to generate label mask.
    conn_comp, info = ndi.label(mask)
    # Filter labelmask based on maximum and minimum structure size.
    (labels, counts) = np.unique(conn_comp, return_counts=True)
    labels_selected = labels[(counts >= size_min) & (counts <= size_max)]
    labelmask = np.where(np.isin(conn_comp, labels_selected), conn_comp, 0)
    return labelmask

############################################################################
def update_labels(mask1, mask2):
    """Match labels of segmented structures to those of a previous frame.
    
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
        # Select new label with maximum overlap.
        max_overlap = labels_sorted[-1]
        return max_overlap
    
    def main(mask1, mask2):
        if not (mask1.shape == mask2.shape):
            raise ValueError("Masks do not have the same shape.")
        # Initialize blank mask.
        updated_mask = np.zeros(mask2.shape)
        for label1 in np.unique(mask1):
            # Find label in mask2 with maximum overlap with nuc from mask1.
            label2 = get_max_overlap(mask1, mask2, label1)
            # Check that labels are "reciprocal best hits" by determining the 
            # label in mask1 with maximum overlap with label in mask2 found above.
            label2_besthit = get_max_overlap(mask2, mask1, label2)
            if ((label2_besthit == label1) and (label1 != 0)):
                updated_mask[mask2 == label2] = label1
        return updated_mask

    return main(mask1, mask2)

############################################################################
def segment_nuclei4D(stack, seg_func, update_func, **kwargs):
    """Segment nuclei in a 4D image stack (expect lattice data).
    
    A wrapper for two supplied functions: one function that performs
    segmentation of a 3D image stack and a second function that connects
    segmentation outputs for consecutive frames by identifying shared objects
    and harmonizing their labels. Iteratively calls these functions on all
    3D stacks and returns a 4D labelmask of segmented objects contiguous in 
    time.
    
    Args:
        stack: ndarray
            4D image stack of dimensions [t, z, x, y].
        seg_func: function
            Function that performs segmentation on 3D image stacks. Must take 
            as arguments a 3D image stack and optional keyword arguments.
        update_func: function
            Function that compares two 3D labelmasks, assigns object IDs from 
            mask1 to mask2, and updates labels in mask2 to match mask1.
        **kwargs: optional key-word arguments
            Keyword arguments to supply to segmentation function.
    
    Returns:
        labelmask: ndarray
            4D labelmask of dimensions [t, z, x, y] with segmented objects.
    
    Example usage:
        labelmask = segment_nuclei4D(im_stack, segment_nuclei3D, update_labels,
            sigma=5, percentile=90)
    """
    # Create partial form of segmentation function with supplied kwargs.
    seg_func_p = partial(seg_func, **kwargs)
    # Segment first frame, add 4th axis in 0 position.
    labelmask = seg_func_p(stack[0], **kwargs)
    labelmask = np.expand_dims(labelmask, axis=0) 
    
    # Segment subsequent frames, update labels, build 4D labelmask.
    for t in range(1, stack.shape[0]):
        mask = seg_func_p(stack[t], **kwargs)
        mask_updated = update_func(labelmask[t-1], mask)
        mask_updated = np.expand_dims(mask_updated, axis=0)
        labelmask = np.concatenate((labelmask, mask_updated), axis=0)
    
    return labelmask

def lattice_segment_nuclei_1(stack, channel=1, **kwargs):
    """Wrapper for nuclear segmentation routine for lattice data.

    Uses 3D stack segmentation function segment_nuclei3D and label propagator
    update_labels
    
    Args:
        stack: ndarray
            5D image stack of dimensions [c, t, z, x, y].
        channel: int
            Channel (0th dimension) to use for segmentation.
        kwargs: key-word arguments (optional)
            Arguments for 3D segmentation function
        
    Returns:
        labelmask: ndarray
            4D labelmask of dimensions [t, z, x, y]
    
    
    """
    return segment_nuclei4D(stack[channel], segment_nuclei3D, update_labels, **kwargs)   