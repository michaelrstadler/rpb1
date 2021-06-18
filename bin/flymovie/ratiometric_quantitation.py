import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from scipy.interpolate import CloughTocher2DInterpolator, RegularGridInterpolator, interp2d, RectBivariateSpline
from scipy.optimize import curve_fit

from flymovie.fitting import fitgaussian3d
from flymovie.general_functions import clamp
from flymovie.detect_spots import connect_ms2_fits_focuscorrect
from flymovie.viewers import spot_movies


############################################################################
# Functions for ratiometric quantitation
############################################################################

############################################################################
def fit_objects_from_mask(stack, mask, fitwindow_rad_xy=10, 
    fitwindow_rad_z=2, sample_size=None):  
    """Find the centroids of all connected objects in a binary mask, perform
    3D gaussian fitting on each object.

    Args:
        stack: ndarray
            4D image stack
        mask: ndarray
            4D binary mask of same dimensions as stack
        fitwindow_rad_xy: int
            'Radius' in pixels of fit window in xy plane
        fitwindow_rad_z: int
            'Radius' in pixels of fit window in z plane

    Returns:
        fit_data: list of ndarrays
            Output of gaussian fitting on objects in mask.
            Each entry in list is a distinct frame (in time), rows in array
            are individual spot fits and columns are 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: fit_height, 4: width_z, 5: width_x, 6: width_y. 

    """
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
    
    def get_centroids(labelmask):
        """Get the centroids of all unique ofjects in mask."""
        centroids = []
        props = regionprops(labelmask)
        for id_ in range(0, len(props)):
            centroid = tuple([int(x) for x in props[id_].centroid])
            centroids.append(centroid)
        return centroids
    
    def fit_frame(substack, submask, fitwindow_rad_xy, fitwindow_rad_z):
        """Perform 3D gaussian fitting on all connected objects in a binary 3D mask."""

        # Label objects in mask and find centroids of resulting objects.
        labelmask = ndi.label(submask)[0]
        peaks = get_centroids(labelmask)
        # Sample peaks if desired.
        if sample_size != None:
            idxs = np.random.choice(np.arange(0, len(peaks)), sample_size, replace=False)
            peaks = [peaks[n] for n in idxs]
        print('# to fit: ' + str(len(peaks)) + '\n# fit: ', end=' ')
        count = 0
        
        # Fit 3D gaussian in window surrounding each centroid.
        fitparams = np.ndarray((0,7))
        for peak in peaks:
            count += 1
            if ((count % 1000) == 0):
                print(count, end=' ')
            # Get subset of data (a 3D box centered on the point) and the adjustments needed
            # to convert from the coordinates of that box to coordinates in the original data.
            fitwindow, z_adj, x_adj, y_adj = get_fitwindow(substack, peak, fitwindow_rad_xy, 
                fitwindow_rad_z)
            # Perform gaussian fitting.
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
    
    # Run fit_frame on all frames in mask/image.
    fit_data = []
    for t in range(0, stack.shape[0]):
        print(t)
        fit_data.append(fit_frame(stack[t], mask[t], fitwindow_rad_xy, fitwindow_rad_z))
    
    return fit_data

############################################################################
def spotstacks_fromfits_byz(stack4d, fits, len_ij=31, len_z=5):
    """Take a set of fitting data, make 4d spot movie structures for each
    Z slice, bundle these into a list where the index is the z slice number.
    
    Args:
        stack4d: ndarray
            4d image stack
        fits: list of ndarrays
            Output of gaussian fitting on objects in mask.
            Each entry in list is a distinct frame (in time), rows in array
            are individual spot fits and columns are 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: fit_height, 4: width_z, 5: width_x, 6: width_y.
        len_ij: int
            Length in ij dimension of the box for making spot movies.
        len_z: int
            Length in z dimension of the box for making spot movies.

    Returns:
        spots_byz: list of ndarrays
            Each entry in list is a spot array for a different z slice,
            first dimension of array is spot id.

    """
    # Create 5D version of stack to feed to spot_movies.
    stack5d = np.expand_dims(stack4d, axis=0)
    spots_byz = []
    # Go through each Z slice.
    for i in range(0, stack4d.shape[-3]-1):
        # Make boolean vector for this z slice.
        # NOTE: only using first frame of fits.
        zslice_bool = fits[0][:,0] == i
        # If there are any fit entries for this slice:
        if (np.count_nonzero(zslice_bool) > 0):
            fits_zslice = [fits[0][zslice_bool]]
            # Use connect_ms2_fits_focuscorrect function to produce spot_data object.
            spot_data = connect_ms2_fits_focuscorrect(fits_zslice, [0], [0], np.zeros_like(stack4d))
            spots = spot_movies(stack5d, spot_data, fill=0, len_ij=len_ij, len_z=len_z, view=False)
            spots = np.squeeze(spots, axis=1)
            spots_byz.append(spots)
        else:
            spots_byz.append(np.zeros((1,len_z, len_ij, len_ij)))
        
    return spots_byz

############################################################################
def calculate_spot_intensities_fg_bg(spot_stack, inner_length_ij=5, 
    inner_length_z=1, outter_length_ij=7, outter_length_z=3, 
    mode='bg_subtract'):
    """Calculate the intensity of segmented spots in a 4d image stack.

    For each spot, define an inner box and a larger outter box.
    For bg_subtract mode, find the mean pixel value in the inner box and then 
    take the mean valueof pixels that are in the outter box but not in inner 
    box ('background'), return the differences as a list. Alternate modes
    return the mean pixel intensity of the inner box of the mean of the 
    3 brightest pixels in the inner box.

    Args:
        spot_stack: ndarray
            4d image stack of 3D boxes centered on detected spots. First
            dimension is spot id.
        inner_length_ij: int
            Length of sides of "inner" (foreground) box in ij dimension
        inner_length_z: int
            Length of sides of "inner" (foreground) box in z dimension
        outter_length_ij: int
            Length of sides of "outter" (background) box in ij dimension
        outter_length_z: int
            Length of sides of "outter" (background) box in z dimension
        mode: string
            'bg_subtract': Perform background subtraction of mean of 
            background from mean of inner box
            'mean_intensity': Simply return mean of inner box
            'top3_max': Return max of top 3 pixels in inner box

    Returns:
        intensities: list
            List of calculated intensities
    """
    if (mode not in ['bg_subtract', 'mean_intensity', 'top3_max']):
        raise ValueError('Not a valid mode.')
    # Get points for ij:
    midpoint = int(spot_stack.shape[-1] / 2)
    inner_rad_ij = int(inner_length_ij / 2)
    outter_rad_ij = int(outter_length_ij / 2)
    inner_start_ij = midpoint - inner_rad_ij
    inner_end_ij = midpoint + inner_rad_ij + 1
    outter_start_ij = midpoint - outter_rad_ij
    outter_end_ij = midpoint + outter_rad_ij + 1
    # Get points for z:
    midpoint_z = int(spot_stack.shape[-3] / 2)
    inner_rad_z = int(inner_length_z / 2)
    outter_rad_z = int(outter_length_z / 2)
    inner_start_z = midpoint_z - inner_rad_z
    inner_end_z = midpoint_z + inner_rad_z + 1
    outter_start_z = midpoint_z - outter_rad_z
    outter_end_z = midpoint_z + outter_rad_z + 1
    
    # Perform calculations on boxes, add intensities to list.
    intensities = []
    for box in spot_stack:
        inner_box = box[inner_start_z:inner_end_z, inner_start_ij:inner_end_ij, inner_start_ij:inner_end_ij]
        if (mode == 'bg_subtract'):
            outter_box = box[outter_start_z:outter_end_z, outter_start_ij:outter_end_ij, outter_start_ij:outter_end_ij] 
            outter_mean = (np.sum(outter_box) - np.sum(inner_box)) / (outter_box.size - inner_box.size)
            inner_mean = np.mean(inner_box)
            bgsub_diff = inner_mean - outter_mean
            intensities.append(bgsub_diff)
        elif (mode == 'mean_intensity'):
            intensities.append(np.mean(inner_box))
        elif(mode == 'top3_max'):
            top3 = np.sort(inner_box.flatten())[-3:]
            intensities.append(np.mean(top3))
    return intensities  

############################################################################
def calculate_spot_intensities_fg_bg_byz(spots_byz, inner_length_ij=5, 
    outter_length_ij=9, inner_length_z=1, outter_length_z=1, 
    return_mean=False, mode='bg_subtract', min_spots_for_mean=4):

    """Call calculate_spot_intensities_fg_bg on stack for each z slice output 
    from spotstacks_fromfits_byz.

    Args:
        spots_byz: list of ndarrays
            Output of spotstacks_fromfits_byz List entries are Z slices, arrays
            are fitting outputs, each row is a spot, columns: 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: fit_height, 4: width_z, 5: width_x, 6: width_y.
        inner_length_ij: int
            Length of sides of "inner" (foreground) box in ij dimension
        inner_length_z: int
            Length of sides of "inner" (foreground) box in z dimension
        outter_length_ij: int
            Length of sides of "outter" (background) box in ij dimension
        outter_length_z: int
            Length of sides of "outter" (background) box in z dimension
        mode: string
            'bg_subtract': Perform background subtraction of mean of 
            background from mean of inner box
            'mean_intensity': Simply return mean of inner box
            'top3_max': Return max of top 3 pixels in inner box
        return_mean: bool
            If true, return the mean intensity value for each z slice
        min_spots_for_mean: int
            The minimum number of spots detected in a z slice for
            a mean to be calculated for that slice. If fewer spots
            are detected, mean is assigned np.nan.

    Returns:
        z_slices: list
            Vector of Z slice numbers
        intensities: list
            Vector of spot values corresponding to z slices in x

    """
    # Initialize x and y vectors for plotting.
    z_slices = []
    intensities = []
    n = len(spots_byz)
    # Perform fg/bg intensity calculation for each z slice, put results into list.
    for z in range(0, n):
        vals = calculate_spot_intensities_fg_bg(spots_byz[z], inner_length_ij=
            inner_length_ij, outter_length_ij=outter_length_ij, inner_length_z=
            inner_length_z, outter_length_z=outter_length_z, mode=mode)
        # Convert list to np.array.
        vals = np.array(vals)
        # Throw out upper and lower 10 percent to avoid outliers.
        low_cut, high_cut = np.percentile(vals, [10,90])
        vals_middle = vals[(vals >= low_cut) & (vals <= high_cut)]
        if (return_mean):
            if (len(vals) > min_spots_for_mean):
                intensities.append(np.mean(vals_middle))
            else:
                intensities.append(np.nan)
            z_slices.append(z)
        else:
            intensities = intensities + list(vals_middle)
            z_slices = z_slices + ([z] * len(vals_middle))
    return z_slices, intensities

############################################################################
def make_depth_vs_intensity_vectors(datasets, surface, start, interval, 
    inner_length_ij=5, outter_length_ij=9, inner_length_z=1, 
    outter_length_z=1, mode='bg_subtract', return_mean=False, 
    min_spots_for_mean=4):
    """Build vectors of spot intensity values vs embryo depth from
    multiple datasets.
    
    Args:
        datasets: iterable
            List (or equivalent) of spots_byz data structures, outputs of 
            spotstacks_fromfits_byz. List entries are Z slices, arrays are 
            fitting outputs
        surface: iterable
            Locations, in µm, of the embryo surface for each dataset
        start: iterable
            Locations in µm of the starting position of each Z stack 
            (obtainable from first_dist in metadata)
        interval: float
            Size, in µm, of Z slices (obtainable from z_interval in metadata)
        inner_length_ij: int
            Length of sides of "inner" (foreground) box in ij dimension
        inner_length_z: int
            Length of sides of "inner" (foreground) box in z dimension
        outter_length_ij: int
            Length of sides of "outter" (background) box in ij dimension
        outter_length_z: int
            Length of sides of "outter" (background) box in z dimension
        mode: string
            'bg_subtract': Perform background subtraction of mean of 
            background from mean of inner box
            'mean_intensity': Simply return mean of inner box
            'top3_max': Return max of top 3 pixels in inner box
        return_mean: bool
            If true, return the mean intensity value for each z slice
        min_spots_for_mean: int
            The minimum number of spots detected in a z slice for
            a mean to be calculated for that slice. If fewer spots
            are detected, mean is assigned np.nan.
    
    Return: 
        depths: list of np.arrays
            Each list item is an array corresponding to an input dataset.
            Array contains embryo depth for each intensity value in intensities
        intensities: list
            Each list item is an array corresponding to an input dataset.
            Array contains spot intensities (or mean spot intensities)
    
    """
    depths = []
    intensities = []
    for i in range(0, len(datasets)):
        # Calculate intensities.
        z_slices, intensities_thisdataset = calculate_spot_intensities_fg_bg_byz(datasets[i], mode=mode, return_mean=return_mean, min_spots_for_mean=min_spots_for_mean, inner_length_ij=inner_length_ij)
        # Add intensities from this dataset to overall list.
        #intensities = intensities + intensities_thisdataset
        intensities.append(np.array(intensities_thisdataset))
        # Convert Z slice to depth.
        depths_thisdataset = (np.array(z_slices) * interval) + start[i] - surface[i]
        # Add depths to overall depths list.
        #depths = depths + list(depths_thisdataset)
        depths.append((depths_thisdataset))
    return depths, intensities

############################################################################
def fit_interpolate_depth_curves(depths, intensities, xgrid_start=0, 
    xgrid_end=20, xgrid_incr=0.1, ygrid_start=0, ygrid_end=30000, 
    ygrid_incr=100, fitfunc='exponential', guess=(0,1,3500), display=False):
    """Fit depth-intensity data with exponential functions, interpolate
    parameters in depth-intensity space for use in depth correction.
    
    Depth-intensity data for each dataset is independently fit with an 
    exponential function (y = a * exp(-b * x) + c), giving a series of 
    curves at different intensities. From these curves, a 2D grid is 
    interpolated for each exponential parameter. I had trouble finding
    a method to do interpolation that resulted in a smooth grid from 
    multiple curves, but I could get a smooth grid from two curves.
    The slightly clunky solution I came up with is to calculate grids
    from pairs of curves and then essentially join them together. This 
    necessitates the curves being "in order", from lowest to highest 
    intensity.
    
    Args:
        depths: list of np.arrays
            Output from make_depth_vs_intensity_vectors. List items are
            input datasets, array values are embryo depths in µm. Curves
            must be in order from lowest to highest intensity.
        intensities: list of np.arrays
            Output from make_depth_vs_intensity_vectors. List items are
            input datasets, array values are spot intensities. Curves
            must be in order from lowest to highest intensity.
        xgrid_start: numeric
            Start position for embryo depth axis of grid used for 
            interpolation.
        xgrid_end: numeric
            End position for embryo depth axis of grid used for 
            interpolation.
        xgrid_incr: numeric
            Increment value for embryo depth axis of grid used for 
            interpolation.
        ygrid_start: numeric
            Start position for spot intensity axis of grid used for 
            interpolation.
        ygrid_end: numeric
            End position for spot intensity axis of grid used for 
            interpolation.
        ygrid_incr: numeric
            Increment value for spot intensity axis of grid used for 
            interpolation.
        fitfunc: string
            Currently only 'exponential' is valid
        guess: tuple of numeric
            Initial parameters used for exponential fit
        display: bool
            If true, plot depth vs. intensity fits for range between 4
            and 15 µm
    Returns: 
        paramgrid_a, paramgrid_b, paramgrid_c: ndarrays
            2D grids (depth, intensity) of exponential parameters a, b, c
    
    """
    def update_paramgrid(paramgrid_old, idx1, idx2, x_all, y_all, vals, grid_x, grid_y):
        """Compute parameter grid from two curves, merge this grid with old grid."""
        # Make vectors of x, y, and parameter values for these two curves.
        x = np.concatenate([x_all[idx1], x_all[idx2]])
        y = np.concatenate([y_all[idx1], y_all[idx2]])
        vals = np.concatenate([vals[idx1], vals[idx2]])
        # Perform interpolation for each parameter of exponential function.
        # Syntax note: CloughTocher function returns a CloughTocher function, this
        # function is called on the meshgrids to produce the interpolated grid.
        paramgrid_new = CloughTocher2DInterpolator((x, y), vals)(grid_x, grid_y)
        # Merge parameter grids by keeping all non-nan values in the new grid, and 
        # replacing all nan positions with the values from the old grid. 
        paramgrids_merged = np.where(~np.isnan(paramgrid_new), paramgrid_new, paramgrid_old)
        return paramgrids_merged

    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Not currently implemented.
    def lin_func(x,m,b):
        return (m * x) + b
    
    if fitfunc not in ['exponential']:
        raise ValueError('fitfunc not valid')
    elif fitfunc == 'exponential':
        fitfunc = exp_func
    
    # Create vectors of x positions, y positions, and values for exponential
    # parameters a, b, and c. These vectors are used for interpolation.
    x_all = []
    y_all = []
    a_all = []
    b_all = []
    c_all = []
    
    # Step through the depths and intensities for each dataset, build 'all'
    # vectors.
    for n in range(0, len(depths)):
        # Get vectors of depths and intensities for this dataset.
        depth_means,intensity_means = np.array(depths[n]), np.array(intensities[n])
        # Fit curve with exponential function.
        (a,b,c),_ = curve_fit(fitfunc, depth_means, intensity_means, p0=guess, maxfev=100000)
        # Deprecated: a bounded version of the fit function.
        #(a,b,c),_ = curve_fit(fitfunc, depth_means, intensity_means, p0=(a_all[-1], b_all[-1], c_all[-1]), maxfev=100000,
        #bounds=((a_all[-1], b_all[-1], c_all[-1]), (np.inf, np.inf, np.inf)))
        
        # Create x and y values from exponential function with fitted parameters.
        x = np.arange(xgrid_start, xgrid_end, 0.1)
        y = (a * np.exp(-b * x)) + c
        # Add x, y, and exponential parameters to existing 'all' vectors.
        x_all.append(x)
        y_all.append(y)
        a_all.append(np.repeat(a, len(x)))
        b_all.append(np.repeat(b, len(x)))
        c_all.append(np.repeat(c, len(x)))
    
    # Create meshgrids to define regions over which to perform interpolation.
    grid_x, grid_y = np.mgrid[xgrid_start:xgrid_end:xgrid_incr, 
        ygrid_start:ygrid_end:ygrid_incr]
    # Initialize parameter grids for each exponential parameter.
    paramgrid_a = np.zeros(grid_x.shape)
    paramgrid_b = np.zeros(grid_x.shape)
    paramgrid_c = np.zeros(grid_x.shape)

    # Perform fitting for all consecutive ascending pairs of curves, continuously
    # update final grid.
    for n in range(1, len(depths)):
        paramgrid_a = update_paramgrid(paramgrid_a, n-1, n, x_all, y_all, a_all, 
            grid_x, grid_y)
        paramgrid_b = update_paramgrid(paramgrid_b, n-1, n, x_all, y_all, b_all, 
            grid_x, grid_y)
        paramgrid_c = update_paramgrid(paramgrid_c, n-1, n, x_all, y_all, c_all, 
            grid_x, grid_y)
    
    # Plot some useful things.
    if display:
        plt.figure(figsize=(10,6))

        # Plot original data and fits.
        plt.subplot(231)
        for n in range(0, len(depths)):
            plt.scatter(depths[n], intensities[n], color='gray')
            plt.scatter(x_all[n], y_all[n], 0.5, color='red')
        plt.title('Fits to input data')

        # Get curves from parameter grids at a series of points, plot in gray.
        plt.subplot(232)
        for pt_x in (50, 150):
            for pt_y in np.arange(20, 200, 20):
                a = paramgrid_a[pt_x, pt_y]
                b = paramgrid_b[pt_x, pt_y]
                c = paramgrid_c[pt_x, pt_y]
                x = np.arange(xgrid_start, xgrid_end, 0.1)
                y = a * np.exp(-b * x) + c
                plt.scatter(x,y, 0.5, color="gray")
                plt.ylim(0,20000)
                plt.grid(color='gray', alpha=0.5)
    
        # Plot original fits to data in red.
        for n in range(0, len(depths)):
            x = x_all[n]
            y = y_all[n]
            plt.scatter(x,y, 0.5, color='red')
            plt.grid(color='black', linestyle='-', linewidth=0.5)
            plt.ylim(0,20000)
        plt.title('Curves from points')
        

        # Plot heatmaps of each parameter grid.
        plt.subplot(234)
        plt.imshow(np.swapaxes(paramgrid_a,0,1), origin='lower')
        plt.title('a')
        plt.subplot(235)
        plt.imshow(np.swapaxes(paramgrid_b,0,1), origin='lower')
        plt.title('b')
        plt.subplot(236)
        plt.imshow(np.swapaxes(paramgrid_c,0,1), origin='lower');
        plt.title('c')
    
    return paramgrid_a, paramgrid_b, paramgrid_c
