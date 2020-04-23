#!/usr/bin/env python

"""
Insert description here.

"""
__version__ = '1.0.0'
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

def viewer(stack, order='tzxy'):
    """Interactive Jupyter notebook viewer for n-dimensional image stacks.
    
    Args:
        stack: ndarray
            n-dimensional image, last two dimensions must be x-y to display
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
        indexes = []
        colmap = kwargs['colmap']
        min_ = kwargs['contrast'][0]
        max_ = kwargs['contrast'][1]
        
        # Unpack order variable into array.
        order_arr = [char for char in order[:-2]]
        #order_arr = [char for char in kwargs['order'][:-2]] 
        # Populate indexes list with widgets.
        for n in order_arr:
            indexes.append(kwargs[n])
            
        fig = plt.figure(figsize=(6.5, 6.5))
        # Slice stack, leaving last two dimensions for image.
        # Note: the (...,) in the following is not required, but I think 
        # it is clarifying.
        plt.imshow(stack[tuple(indexes) + (...,)], cmap=colmap, vmin=min_, 
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
