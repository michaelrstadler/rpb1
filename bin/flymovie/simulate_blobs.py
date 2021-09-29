import numpy as np
import matplotlib.pyplot as plt
from flymovie.general_functions import mesh_like


############################################################################
def simulate_blobs(nucmask, bg_mean, bg_var, blob_intensity_mean, 
    blob_intensity_var, blob_radius_mean, blob_radius_var, blob_number, 
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
            start = coords[dim] - int(box.shape[dim] / 2)
            end = coords[dim] + int(box.shape[dim] / 2) + 1
            if start < 0:
                stack_starts.append(0)
                stack_ends.append(end + start)
                box_starts.append(-start)
                box_ends.append(box.shape[dim] + start)
            elif end > stack.shape[dim]:
                stack_starts.append(start)
                stack_ends.append(stack.shape[dim])
                box_starts.append(0)
                box_ends.append(box.shape[dim] + stack.shape[dim] - end)
            else:
                stack_starts.append(start)
                stack_ends.append(end)
                box_starts.append(0)
                box_ends.append(box.shape[dim])
        # Ensure that the shapes of the subsection of the stack to add to
        # and the box to add are the same. If so, add box values to stack.
        substack_shape = stack[stack_starts[0]:stack_ends[0], stack_starts[1]:stack_ends[1], stack_starts[2]:stack_ends[2]].shape
        box_to_add = box[box_starts[0]:box_ends[0], box_starts[1]:box_ends[1], box_starts[2]:box_ends[2]]
        if substack_shape == box_to_add.shape:
            stack[stack_starts[0]:stack_ends[0], stack_starts[1]:stack_ends[1], stack_starts[2]:stack_ends[2]] += box_to_add
        else:
            raise ValueError('Shape of box and stack section not equal')
    
    ij_windowlen blob_radius_mean * 3.5
    z_windowlen = ij_windowlen / z_ij_ratio
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

    