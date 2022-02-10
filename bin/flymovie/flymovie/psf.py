#!/usr/bin/env python

"""
Functions for calculating a point spread function (PSF) from images of 
fluorescent beads.

"""
__version__ = '1.0.0'
__author__ = 'Michael Stadler'

import flymovie as fm
import numpy as np
import scipy.ndimage as ndi
from importlib import reload
from flymovie.general_functions import extract_box, stack_normalize_minmax

#-----------------------------------------------------------------------
def extract_beads(stack, thresh, box_dims):
    """Extract windows of an image centered at beads.
    
    Beads are segmented by user-supplied threshold.

    Args:
        stack: ndarray
            Image stack containing fluorescent beads
        thresh: number
            Threshold intensity for segmenting beads
        box_dims: sequence of ints
            Dimensions of windows to extract. Dimensions must be odd.
    
    Returns:
        boxes: ndarray
            Array of extracted windows (concatenated on axis 0)
    """

    # Make mask with thresholding, use opening to get rid of small objects, label.
    mask = np.where(stack > thresh, 1, 0)
    mask = ndi.morphology.binary_opening(mask, structure=np.ones((3,3,3)))

    # Strategy: ndi.measurements.center_of_mass finds the center of mass of objects
    # in a labelmask. I'm worried that if I use the thresholded mask, subtle
    # differences in thresholding might change the center. It would be better
    # to determine the center of mass within a nice window around each object. 
    # To do this, first expand the objects using morphological
    # dilation and find the center of mass of these larger objects.

    mask = ndi.morphology.binary_dilation(mask, structure=np.ones((4,8,8)))
    lmask,_ = ndi.label(mask)

    # Find center of mass for each spot (within window defined above).
    centers = ndi.measurements.center_of_mass(stack, lmask, np.arange(1, np.max(lmask) + 1))

    # Extract windows (boxes) around each of the centers of mass.
    boxes = np.ndarray(tuple([0]) + box_dims)
    for center in centers:
            center = [round(x) for x in center]
            box = extract_box(stack, center, box_dims, pad=False)
            # If the box extends beyond the image borders, box returned by extract_box
            # will be smaller than box_dims (because padding = false).
            if not np.array_equal(box.shape, box_dims):
                continue
            box = stack_normalize_minmax(box)
            box = np.expand_dims(box, axis=0)
            boxes = np.vstack([boxes, box])
    return boxes

#-----------------------------------------------------------------------
def extract_beads_batch(stacklist, thresh, box_dims):
    """Extract beads from a list of image stacks.
    
    Args:
        stacklist: sequence of ndarrays
            List of stacks of images of beads
        thresh: number
            Threshold intensity for segmenting beads
        box_dims: sequence of ints
            Dimensions of windows to extract. Dimensions must be odd.
    
    Returns:
        boxes: ndarray
            Array of extracted windows (concatenated on axis 0)
    """
    boxes = np.ndarray(tuple([0]) + box_dims)
    for stack in stacklist:
        boxes = np.vstack([boxes, extract_beads(stack, thresh, box_dims)])
    return boxes

#-----------------------------------------------------------------------
def remove_bad_beads(boxes, bad_indexes):
    """Remove frames from boxes (output of extract_beads).

    Args:
        boxes: ndarray
            Array of extracted windows (concatenated on axis 0)
        bad_indexes: sequence of ints
            List of boxes entries to exclude

    Returns:
        boxes with "bad" entries removed
    """
    mask = np.ones(boxes.shape[0])
    mask[bad_indexes] = 0
    mask = mask.astype(bool)
    return(boxes[mask])