#!/usr/bin/env python

"""
Functions for segmenting nuclei from fluorscence images of fly embryos.

"""
__version__ = '1.1.0'
__author__ = 'Michael Stadler'

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage as ski
from skimage.filters.thresholding import threshold_li, threshold_otsu
from skimage.segmentation import flood_fill, watershed
from scipy.stats import mode
from skimage.measure import label, regionprops
from flymovie import gradient_nD, peak_local_max_nD, get_object_centroid, expand_mip, relabel_labelmask

############################################################################    
############################################################################
# Functions for segmenting nuclei
############################################################################
############################################################################


############################################################################
def segment_nuclei_3Dstack_rpb1(stack, min_nuc_center_dist=25, sigma=5, 
    usemax=False, display=False, return_intermediates=False, 
    seed_window=None, thresh=None):
    """Segment nuclei from Rpb1 fluorescence in confocal data.
    
    Algorithm is smooth -> threshold -> gradient -> distance transform to 
    find seeds -> take gradient on binary mask -> watershed on gradient. 
    Does not do any filtering on resulting segmented objects.
   
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
        min_nuc_center_dist: numeric
            The minimum euclidean distance (in pixels) allowed between watershed
            seeds. Typically set as ~the diameter of the nuclei.   
        sigma: numeric
            Sigma for use in initial gaussian smoothing
        usemax: bool
            Use maximum intensity projection (in Z) for segmenting
        return_intermediates: bool
            Return (mask, grad, seeds, ws) for troubleshooting
        seed_window: tuple of ints
            [Optional] 
            Size in [z, x, y] for window for determining local maxes in 
            distance transform. A point is retained as a seed if there
            exists some window of this size in the image for which the point
            is the max value. Generally want size to be a little less than 2x 
            the distance between nuclear centers. Centers closer than this 
            will not produce two seeds. If None, then a seed window is 
            automatically generated from min_nuc_center_dist so that the
            diagonal of the box is equal to twice this distance.
        thresh: float (optional)
            Threshold value to use for creating initial mask
    
    Returns:
        labelmask: ndarray
            Mask of same shape as input stack with nuclei segmented and labeled
    
    """
    # Generate seed window if none supplied.
    if seed_window is None:
        # Window set such that the diagonal is equal to 2 * min_nuc_center_dist.
        seed_window = (stack.shape[0], (min_nuc_center_dist * 2) / np.sqrt(2), (min_nuc_center_dist * 2) / np.sqrt(2))
        # Remove first dimension if max projection used.
        if usemax:
            seed_window = seed_window[1:]

    # Smooth stack using a Gaussian filter.
    if usemax:
        stack_smooth = ndi.gaussian_filter(stack.max(axis=0), sigma)
    else:
        stack_smooth = ndi.gaussian_filter(stack, (1, sigma, sigma))
    # Define a threshold for nuclear signal if not supplied.
    if thresh is None:
        thresh = threshold_otsu(stack_smooth)
    # Make a binary mask using threshold.
    mask = np.where(stack_smooth > thresh, 1, 0)
    # Take the gradient of the mask to produce outlines for use in watershed algorithm.
    grad = gradient_nD(mask)
    # Perform distance transform and run local max finder to determine watershed seeds.
    dist = ndi.distance_transform_edt(mask)
    seeds, _ = peak_local_max_nD(dist, size=seed_window, min_dist=min_nuc_center_dist)
    # Perform watershed segmentation.
    ws = watershed(grad, seeds.astype(int))
    # Filter object size and circularity, relabel to set background to 0.
    if usemax:
        ws = np.repeat(np.expand_dims(ws, axis=0), stack.shape[0], axis=0)
    labelmask = ws

    if (display):
        if usemax:
            mask = np.expand_dims(mask, 0)
            seeds = np.expand_dims(seeds, 0)
            stack_smooth = np.expand_dims(stack_smooth, 0)
            grad = np.expand_dims(grad, 0)
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
def segment_nuclei_3Dstack_victoria(stack, min_nuc_center_dist=25, sigma=5, 
    usemax=False, display=False, return_intermediates=False, 
    seed_window=None):
    """Segment nuclei from Rpb1 fluorescence in confocal data.
    
    Algorithm is smooth -> threshold -> gradient -> distance transform to 
    find seeds -> take gradient on binary mask -> watershed on gradient. 
    Does not do any filtering on resulting segmented objects.
   
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
        min_nuc_center_dist: numeric
            The minimum euclidean distance (in pixels) allowed between watershed
            seeds. Typically set as ~the diameter of the nuclei.   
        sigma: numeric
            Sigma for use in initial gaussian smoothing
        usemax: bool
            Use maximum intensity projection (in Z) for segmenting
        return_intermediates: bool
            Return (mask, grad, seeds, ws) for troubleshooting
        seed_window: tuple of ints
            [Optional] 
            Size in [z, x, y] for window for determining local maxes in 
            distance transform. A point is retained as a seed if there
            exists some window of this size in the image for which the point
            is the max value. Generally want size to be a little less than 2x 
            the distance between nuclear centers. Centers closer than this 
            will not produce two seeds. If None, then a seed window is 
            automatically generated from min_nuc_center_dist so that the
            diagonal of the box is equal to twice this distance.
    
    Returns:
        labelmask: ndarray
            Mask of same shape as input stack with nuclei segmented and labeled
    
    """
    # Generate seed window if none supplied.
    if seed_window is None:
        # Window set such that the diagonal is equal to 2 * min_nuc_center_dist.
        seed_window = (stack.shape[0], (min_nuc_center_dist * 2) / np.sqrt(2), (min_nuc_center_dist * 2) / np.sqrt(2))
        # Remove first dimension if max projection used.
        if usemax:
            seed_window = seed_window[1:]

    # Smooth stack using a Gaussian filter.
    if usemax:
        stack_medfilt = ndi.median_filter(stack.max(axis=0), 15)
        stack_smooth = ndi.gaussian_filter(stack_medfilt, sigma)
    else:
        stack_smooth = ndi.gaussian_filter(stack, sigma)
    # Define a threshold for nuclear signal.
    thresh = threshold_li(stack_smooth)
    print(thresh)
    # Make a binary mask using threshold.
    mask = np.where(stack_smooth > thresh, 1, 0)
    # Take the gradient of the mask to produce outlines for use in watershed algorithm.
    grad = gradient_nD(mask)
    # Perform distance transform and run local max finder to determine watershed seeds.
    dist = ndi.distance_transform_edt(mask)
    seeds, _ = peak_local_max_nD(dist, size=seed_window, min_dist=min_nuc_center_dist)
    # Perform watershed segmentation.
    ws = watershed(grad, seeds.astype(int))
    # Filter object size and circularity, relabel to set background to 0.
    if usemax:
        ws = np.repeat(np.expand_dims(ws, axis=0), stack.shape[0], axis=0)
    labelmask = ws

    if (display):
        if usemax:
            mask = np.expand_dims(mask, 0)
            seeds = np.expand_dims(seeds, 0)
            stack_smooth = np.expand_dims(stack_smooth, 0)
            grad = np.expand_dims(grad, 0)
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

############################################################################ 
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

############################################################################
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

