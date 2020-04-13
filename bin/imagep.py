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
from skimage import filters, io
from ipywidgets import interact, IntSlider, Dropdown, IntRangeSlider, fixed
import matplotlib.pyplot as plt

############################################################################

def read_tiff_stack(stack_folder):
    """Read a folder of 2D or 3D TIFF files into a numpy ndarray.
    
    Args:
        stack_folder: string
            Directory containing multiple TIFF files. Must be in asciibetical
            order.
    
    Returns:
        stack: ndarray
            n-dimensional image stack with the new dimension (file number) 
            in the 0 position(file_num, x, y, z) for 3D stacks, (filenum, x,
            y) for 2D stacks
            
    Raises:
        ValueError: if dimensions of TIFF file don't match those of the first
        file
    """
    
    # Compile files that are files and have .tif extension (case-insensitive).
    tif_files = [f for f in listdir(stack_folder) if (isfile(join(stack_folder, f)) 
        and (os.path.splitext(f)[1][0:4].upper() == '.TIF'))]
    # Sort the files: asciibetical sorting produces files ascending in time 
    # (sorting is *in place*)
    tif_files.sort()
    
    # Create stack with dimensions from first file.
    img = io.imread(join(stack_folder, tif_files[0]))
    dims = img.shape
    num_files = len(tif_files)
    stack = np.ndarray(((num_files,) + dims), dtype=img.dtype)
    stack[0] = img
    img_num = 1
    
    # Add the rest of the files to the stack.
    for tif_file in tif_files[1:]:
        # Add image data to ndarray
        img = io.imread(join(stack_folder, tif_file))
        # Check dimensions
        if not stack[0].shape == img.shape:
            raise ValueError(f'Dimensions do not match previous files: {tif_file}')
        stack[img_num] = img
        img_num = img_num + 1
        
    return stack

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
        ValueError: if final two dimensions are not xy
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
        plt.imshow(stack[tuple(indexes) + (...,)], cmap=colmap, vmin=min_, vmax=max_);    
    
    # Make a new slider object for dimension selection and return it
    def _make_slider(n):
        widget = IntSlider(min=0, max=(stack.shape[n] - 1), step=1, continuous_update=False,)
        return(widget)
    
    # Make a dropdown widget for selecting the colormap.
    def _make_cmap_dropdown():
        dropdown = Dropdown(
            options={'viridis', 'plasma', 'magma', 'inferno','cividis','Greens', 
                     'Reds', 'gray', 'gray_r'},
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
