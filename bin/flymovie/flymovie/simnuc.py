#!/usr/bin/env python

"""
A class for simulating nuclei.

The Sim class 

Important programming note: when running in parallel, using numpy.random
seems to not perform random initialization, and threads launched on
multiple cores will produce identical simulations. This is solved by
separately initializing a numpy random state using 
rs = np.random.RandomState() within each function where random processes
are needed. This must be used for parallel operations.

"""
__version__ = '1.0.0'
__author__ = 'Michael Stadler'

from flymovie.general_functions import mesh_like
import scipy
import random
import numpy as np
import scipy.ndimage as ndi

class Sim():
    """A class to simulate fluorescent signal in nuclei."""

    def __init__(self, mask, z_ij_ratio=4.5):
        self.mask = mask.astype('bool')
        self.im = mask.copy()
        self.fg_coords = np.where(self.mask)
        self.bg_coords = np.where(~self.mask)
        self.z_ij_ratio = z_ij_ratio

    #-----------------------------------------------------------------------
    @staticmethod
    def make_dummy_mask(zdim=20, idim=100, jdim=100, nuc_spacing=200, 
        nuc_rad=50, z_ij_ratio=4.5):
        """Make a label mask of spherical dummy nuclei.
        
        Nuclei are equally spaced spheres. The can be squashed by playing
        with z_ij_ratio, if desired.

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
            z_ij_ratio: numeric
                Ratio of voxel size in z to ij (axial to lateral)
        """
        mask = np.zeros((zdim, idim, jdim))
        z, i, j = mesh_like(mask, 3)
        z_midpoint = int(mask.shape[0] / 2)
        nuc_id = 1
        for i_center in range(nuc_rad, mask.shape[1], nuc_spacing):
            for j_center in range(nuc_rad, mask.shape[2], nuc_spacing):
                # Basic equation of circle.
                mask[((((z - z_midpoint) ** 2) * (z_ij_ratio ** 2)) + ((i - i_center) ** 2) + 
                    ((j - j_center) ** 2)) < (nuc_rad ** 2)] = nuc_id
                nuc_id += 1

        return mask
    
    #-----------------------------------------------------------------------
    @staticmethod
    def extract_nuclear_masks(lmask, target_size=(20,100,100)):
        """Extract individual nuclei from a labelmask, resize them to match
        a target size, create list of masks.
        
        Args:
            lmask: ndarray, labelmask of segmented nuclei
            target size: tuple of ints, size to which to match nuclei

        Returns:
            masks: list of ndarrays, nuclear masks
        """
        def touches_edge(coords, shape):
            """Determine whether nucleus touches image edge (only in last
            two dimensions)."""
            for dim in range(1, len(shape)):
                if np.min(coords[dim]) == 0:
                    return True
                if np.max(coords[dim]) == (shape[dim] - 1):
                    return True
            return False
                
        masks = []
        for n in range(1, np.max(lmask) + 1):
            coords = np.where(lmask == n)
            if touches_edge(coords, lmask.shape):
                continue
            # Extract image segment constituting a bounding box for nucleus.
            cutout = lmask[np.min(coords[0]):np.max(coords[0]),
                np.min(coords[1]):np.max(coords[1]),
                np.min(coords[2]):np.max(coords[2])
                ]
            cutout_mask = np.where(cutout == n, 1, 0)
            # Resize to fit target size.
            zoom_factors = [
                target_size[0] / cutout_mask.shape[0],
                target_size[1] / cutout_mask.shape[1],
                target_size[2] / cutout_mask.shape[2]
            ]
            resized = ndi.zoom(cutout_mask, zoom_factors, order=0)
            masks.append(resized)
        return masks

    #-----------------------------------------------------------------------
    def add_background(self, model='uniform', inverse=False, 
            **kwargs):
        """Replace foreground or background pixels with pixel values from
        a background model.

        poisson+gaussian: Pixels drawn from poisson distribution, then 
            gaussian noise added
        uniform: Pixels are a uniform value (noise added later).

        Args:
            model: string
                Currently only 'poisson+gaussian' is supported
            inverse: bool
                If true, replace foreground, if false, background
            kwargs:
                poisson+gaussian:
                    lam: lambda value for poisson distribution (mean and variance)
                    sigma: standard deviation for gaussian noise   
                uniform:
                    val: numeric, uniform value         
        """
        coords = self.fg_coords
        if inverse:
            coords = self.bg_coords
        
        if model == 'poisson+gaussian':
            if not all(arg in kwargs for arg in ['lam', 'sigma']):
                raise ValueError('poisson+gaussian mode requires kwargs lam, sigma.')
            lam, sigma = kwargs['lam'], kwargs['sigma']
            num_pixels = len(coords[0])
            rs = np.random.RandomState()
            # Start with poisson distributed background pixels.
            pixels = rs.poisson(lam=lam, size=num_pixels)
            # Add gaussian noise.
            noise = rs.normal(scale=sigma, size=num_pixels)
            pixels = pixels + noise
            pixels[pixels < 0] = 0
            self.im[coords] = pixels
        
        elif model == 'uniform':
            if not all(arg in kwargs for arg in ['val']):
                raise ValueError('uniform mode requires kwarg val.')
            val = kwargs['val']
            self.im[coords] = val

        else:
            raise ValueError('Only poisson+gaussian and uniform models currently supported.')
    
    #-----------------------------------------------------------------------
    def add_noise(self, model='poisson+gaussian',  **kwargs):
        """Add noise to image according to a model.

        poisson+gaussian: Pixels drawn from poisson distribution, then 
            gaussian noise added

        Args:
            model: string
                Currently only 'poisson+gaussian' is supported
            kwargs:
                poisson+gaussian:
                    sigma: standard deviation for gaussian noise      
        """
        if model == 'poisson+gaussian':
            if 'sigma' not in kwargs:
                raise ValueError('poisson+gaussian mode requires kwarg sigma.')
            sigma = kwargs['sigma']
            rs = np.random.RandomState()
            poisson = rs.poisson(self.im)
            gaussian = rs.normal(scale = (sigma * np.ones_like(self.im)))
            self.im = poisson + gaussian
            self.im[self.im < 0] = 0
        
        else:
            raise ValueError('Only poisson+gaussian mode supported currently.')

    #-----------------------------------------------------------------------
    @staticmethod
    def make_3d_gaussian_inabox(intensity, sigma_z, sigma_ij, 
            z_windowlen, ij_windowlen):
        """Make a 3D gaussian signal within a box of defined size.
        
        Multiply 1D numpy vectors (generated from 1D gaussian functions) 
        together to produce a proper 3D gaussian.
        
        Args:
            intensity: numeric, intensity of gaussian (height in 1d)
            sigma_z: numeric, sigma of gaussian in Z dimension
            sigma_ij: numeric, sigma of gaussian in ij dimension
            z_windowlen: int, length in z dimension of box will be 2X this
            ij_windowlen: int, length in ij dimension of box will be 2X this
        """
        d1 = scipy.signal.gaussian(ij_windowlen, sigma_ij)
        d2 = np.outer(d1, d1)
        z_1dvector = scipy.signal.gaussian(z_windowlen, sigma_z)
        d3 = d2 * np.expand_dims(z_1dvector, axis=(1,2))
        return intensity * d3

    #-----------------------------------------------------------------------
    #### TO DO
    @staticmethod
    def make_flattop_3d_gaussian_inabox(intensity, sigma_z, sigma_ij, 
            z_windowlen, ij_windowlen):
        """Make a 3D gaussian signal within a box of defined size.
        
        Multiply 1D numpy vectors (generated from 1D gaussian functions) 
        together to produce a proper 3D gaussian.
        
        Args:
            intensity: numeric, intensity of gaussian (height in 1d)
            sigma_z: numeric, sigma of gaussian in Z dimension
            sigma_ij: numeric, sigma of gaussian in ij dimension
            z_windowlen: int, length in z dimension of box will be 2X this
            ij_windowlen: int, length in ij dimension of box will be 2X this
        """
        d1 = scipy.signal.gaussian(ij_windowlen, sigma_ij)
        d2 = np.outer(d1, d1)
        z_1dvector = scipy.signal.gaussian(z_windowlen, sigma_z)
        d3 = d2 * np.expand_dims(z_1dvector, axis=(1,2))
        return intensity * d3

    #-----------------------------------------------------------------------
    @staticmethod
    def add_box_to_stack(stack, box, coords):
        """Add pixel values from a supplied "box" to a stack at a position 
        centered at supplied coordinates.
        
        Args:
            stack: ndarray, image stack
            box: ndarray, the box to add to the image stack
            coords: iterable of ints, coordinates marking center of position
                in stack to which to add box
        """
        # Initialize arrays to store the start and end coordinates for the 
        # stack and the box in each dimension. They are arrays because they
        # will store the start/end position in each of the three dimensions.
        box_starts = [] # Start positions relative to box [z, i, j] 
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

    #-----------------------------------------------------------------------
    def add_gaussian_blob(self, coords, intensity, sigma):
        """Add a gaussian blob to image.
        
        Args:
            coords: tuple of three ints
                Coordinates of the center of gaussian
            intensity: numeric
                "Height" of gaussian
            sigma: numeric
                Standard deviation of gaussian
        """
        def make_odd(n):
            """Ensure box dimension is an odd integer to make math work."""
            n = int(n)
            if (n % 2) == 0:
                return n + 1
            else:
                return n

        ij_windowlen = make_odd(sigma * 10.5)
        z_windowlen = make_odd(ij_windowlen / self.z_ij_ratio)
        sigma_z = sigma / self.z_ij_ratio
        box = self.make_3d_gaussian_inabox(intensity, sigma_z, sigma, 
            z_windowlen, ij_windowlen)
        self.add_box_to_stack(self.im, box, coords)
    
    #-----------------------------------------------------------------------
    def get_eroded_coordinates(self, erosion_size):
        """Get the coordinates (pixels >0) of a mask after applying binary
        erosion. Effectively makes a coordinate set excluding pixels at the 
        edge of the nucleus.

        Args:
            erosion_size: numeric, size of the structure for dilation in the
            ij-dimension, is automatically scaled to the equivalent in z.
        """
        # Get size of structuring element, minimum of 1 in each dimension.
        struct_z = np.max([1, int(erosion_size / self.z_ij_ratio)])
        struct_ij = np.max([1, int(erosion_size)])
        eroded_mask = scipy.ndimage.morphology.binary_erosion(self.mask, 
            structure=np.ones((struct_z, struct_ij, struct_ij)))
        eroded_mask_coords = np.where(eroded_mask)
        return eroded_mask_coords

    #-----------------------------------------------------------------------
    def add_nblobs(self, numblobs, intensity_mean, intensity_std, sigma_base,
            sigma_k=0.5, sigma_theta=0.5):
        """Add gaussian blobs at random positions inside nucleus, with
        intensities and widths drawn from random distributions.
        
        Intensities are drawn from gaussian distribution, widths (sigma of
        gaussian) use a gamma distribution (gamma is non-negative): 
            sigma = sigma_base + gamma(k, theta)

        Args:
            numblobs: int, number of blobs to add
            intensity_mean: numeric, mean of the distribution from which 
                blob intensities are drawn.
            intensity_std: numeric, std. deviation of the distribution from
                which blob intensities are drawn.
            sigma_base: numeric, minimum value of gaussian width (sigma)
            sigma_k: numeric, shape parameter of gamma distribution added to 
                sigma_base to determine gaussian width
            sigma_theta: numeric, scale parameter of gamma distribution added 
                to sigma_base to determine gaussian width
        """
        rs = np.random.RandomState()
        # Use erosion to generate candidate positions that avoid the edge
        # of the nucleus.
        erosion_size = sigma_base * 3
        eroded_coords = self.get_eroded_coordinates(erosion_size)
        num_pixels = len(eroded_coords[0])
        for _ in range(numblobs):
            # Get randomly-generated intensity and sigma for gaussian.
            sigma = sigma_base + rs.gamma(sigma_k, sigma_theta)
            intensity = rs.normal(intensity_mean, intensity_std)
            intensity = np.max([0, intensity])
            # Get random coordinates 
            px = rs.randint(0, num_pixels - 1)
            random_coords = (eroded_coords[0][px], eroded_coords[1][px], 
                eroded_coords[2][px])
            self.add_gaussian_blob(random_coords, intensity, sigma)

    #-----------------------------------------------------------------------
    def add_hlb(self, intensity, sigma):
        """Add histone locus bodies to nucleus. HLBs simulated as gaussian
        blobs.
        
        Possible improvement: the HLBs may not be well-modeled by 3d
        gaussian function. A more sophisticated model might be useful.

        Args:
            intensity: number, intensity of gaussian representing HLB
            sigma: number, width of 3D gaussian representing HLB
        """
        min_dist = sigma
        # Pick location 1, constrained to no to too close to edge.
        erosion_size = sigma * 3
        eroded_mask_coords = self.get_eroded_coordinates(erosion_size)
        rs = np.random.RandomState()
        num_pixels = len(eroded_mask_coords[0])
        px = rs.randint(0, num_pixels - 1)
        coords_hlb1 = eroded_mask_coords[0][px], eroded_mask_coords[1][px], eroded_mask_coords[2][px]

        # Pick location 2, constrained by proximity to 1 and edge.
        good_coords = False
        while not good_coords:
            px = rs.randint(0, num_pixels - 1)
            coords_hlb2 = eroded_mask_coords[0][px], eroded_mask_coords[1][px], eroded_mask_coords[2][px]
            dist_hlb_1_2 = scipy.spatial.distance.euclidean(coords_hlb2, coords_hlb1)
            if dist_hlb_1_2 > min_dist:
                good_coords = True
        self.add_gaussian_blob(coords_hlb1, intensity, sigma)
        self.add_gaussian_blob(coords_hlb2, intensity, sigma)

    #-----------------------------------------------------------------------
    def add_nblobs_zschedule(self, numblobs, intensity_mean, intensity_std, 
            sigma_base, z_probs, sigma_k=0.5, sigma_theta=0.5):
        """Add gaussian blobs according with asymmetrical Z location 
        probabilities determined by a supplied distribution.

        Args:
            numblobs: int, number of blobs to add
            intensity_mean: numeric, mean of the distribution from which 
                blob intensities are drawn.
            intensity_std: numeric, std. deviation of the distribution from
                which blob intensities are drawn.
            sigma_base: numeric, minimum value of gaussian width (sigma)
            z_probs: iterable, probabilities of blobs occurring in each z slice
            sigma_k: numeric, shape parameter of gamma distribution added to 
                sigma_base to determine gaussian width
            sigma_theta: numeric, scale parameter of gamma distribution added 
                to sigma_base to determine gaussian width
        """
        if len(z_probs) != self.im.shape[0]:
            raise ValueError('Length of z_probs must equal number of z slices.')
        if abs(np.sum(z_probs) - 1) > 0.01:
            raise ValueError('z_probs must sum to 1.')
        rs = np.random.RandomState()
        num_pixels = len(self.fg_coords[0])
        # Divide coords by Z slice.
        eroded_coords = self.get_eroded_coordinates(sigma_base * 3)
        coord_list = np.array(list(zip(eroded_coords[0], eroded_coords[1], eroded_coords[2])))
        coords_byz = []
        num_zslices = self.im.shape[0]
        for z in range(num_zslices):
            coords_byz.append(coord_list[eroded_coords[0] == z])
        # Draw z slices from probability dist: end up with list of Z-slices.
        z_slices = rs.choice(np.arange(0, num_zslices), p=z_probs, size=numblobs)
        # Add spots in randomly selected Z slices.
        for z in z_slices:
            # Draw intensity and sigma.
            sigma = sigma_base + rs.gamma(sigma_k, sigma_theta)
            intensity = rs.normal(intensity_mean, intensity_std)
            intensity = np.max([0, intensity])
            # Draw random coordinate from the drawn Z-slice.
            rand_idx = rs.randint(0, len(coords_byz[z]) - 1)
            rand_coords = coords_byz[z][rand_idx]
            self.add_gaussian_blob(rand_coords, intensity, sigma)

    #-----------------------------------------------------------------------
    def add_nblobs_zschedule_exponential(self, numblobs, intensity_mean, 
            intensity_std, sigma_base, exp_shape=1, sigma_k=0.5, sigma_theta=0.5, 
            return_probs=False):
        """Add blobs with a Z distribution defined by an exponential 
        function.
        
        Shape of exponential is determined by exp_shape according to:
            y = exp(exp_shape * x) for 0 < x < 1
            probs = y / sum(y)

            0: Flat
            More positive: steeper bias toward higher z slices
            More negative: steeper bias toward lower z slices

        Args:
            numblobs: int, number of blobs to add
            intensity_mean: numeric, mean of the distribution from which 
                blob intensities are drawn.
            intensity_std: numeric, std. deviation of the distribution from
                which blob intensities are drawn.
            sigma_base: numeric, minimum value of gaussian width (sigma)
            exp_shape: numeric, shape parameter of exponential
            sigma_k: numeric, shape parameter of gamma distribution added to 
                sigma_base to determine gaussian width
            sigma_theta: numeric, scale parameter of gamma distribution added 
                to sigma_base to determine gaussian width
            return_probs: bool, return z-slice probabilities
        
        Possible issue: the number of available pixels is different at different
        Z slices. Even a uniform distribution, done the way it's done here, will
        crowd spots at the poles. It's possible there's a different version needed
        that weights Z-slices by pixels and uses the distribution as some kind
        of bias.
        """
        x = np.arange(0, 1, 1/self.im.shape[0])
        y = np.exp(exp_shape * x)
        probs = y / np.sum(y)
        self.add_nblobs_zschedule(numblobs, intensity_mean, intensity_std, sigma_base,
            z_probs=probs, sigma_k=0.5, sigma_theta=0.5)
        if return_probs:
            return probs