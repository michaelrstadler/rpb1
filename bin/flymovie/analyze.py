import numpy as np
from scipy import ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
from flymovie.general_functions import mesh_like
from flymovie.fitting import gaussian3d
from flymovie.viewers import plot_ps
from flymovie.movieclass import movie
import scipy

import copy


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
def spot_data_bleach_correct_framemean(spot_data, stack, channel,  
    cols_to_correct=(9, 10, 11), sigma=7, ref_depth=15):
    """Perform bleach correction using the (smoothed) frame averages from 
    an image stack, apply correction to columns of spot_data object.

    Args:
        spot_data: dict of ndarrays
            Spot_data object
        stack: ndarray
            5D [c,t,z,x,y] image stack to use for bleach correction
        channel: int
            Channel to use for correction
        cols to corrects: iterable of ints
            Columns in spot_data to bleach correct
        sigma: number-like
            Sigma value for gaussian filtering of frame means

    Returns:
        spot_data_corr: dict of ndarrays
            Input spot_data with bleaching correction applied to indicated
            columns
    """
    def get_ref_slices(spot_data):
        
    frame_means = np.mean(stack[channel], axis=(1,2,3))
    frame_means_smooth = ndi.gaussian_filter(frame_means, sigma=sigma)
    # Normalize means to the first frame.
    means_norm = frame_means_smooth / frame_means_smooth[0]
    spot_data_corr = copy.deepcopy(spot_data)
    for spot_id in spot_data:
        for col in cols_to_correct:
            spot_data_corr[spot_id][:, col] = np.apply_along_axis(lambda x: x[col] / means_norm[int(x[0])], 1, spot_data[spot_id])

    return spot_data_corr

#######################################################################
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
            #plt.plot(ndi.gaussian_filter1d(df1_processed.iloc[:,x], sigma))
            #plt.plot(ndi.gaussian_filter1d(df2_processed.iloc[:,x], sigma))
            plt.plot(df1_processed.iloc[:,x])
            plt.plot(df2_processed.iloc[:,x])
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

############################################################################
def threshold_w_slope(stack, ref_thresh, ref_slice, slope, display=False):
    """Threshold a 3D stack using a threshold that varies linearly in Z
    according to a supplied slope, return a binary mask.
    
    To use: Find a reasonable threshold for a single slice in your image
    stack. Run function with various slopes and display=True, which will
    show the number of thresholded objects detected in each Z slice. Play 
    with slope until this relationship is satisfactorally flat.

    Args:
        stack: ndarray
            3D image stack
        ref_thresh: numeric
            Threshold value for the reference slice
        ref_slice: int
            Slice number used for reference
        slope: numeric
            Slope (intensity/slice) of linear adjustment in threshold
        display: bool
            If true, plots number of detected objects vs. z slice
    
    Return:
        mask: ndarray
            4D binary mask resulting from adjusted thresholding
    """
    # Initialize empty mask.
    mask = np.zeros_like(stack)
    n_slices = stack.shape[0]
    # Make a vector of thresholds for different Z slices.
    thresh_start = ref_thresh - (ref_slice * slope)
    thresh_end = ref_thresh + ((n_slices - ref_slice - 1) * slope)
    thresh = np.linspace(thresh_start, thresh_end, n_slices)
    
    # Apply computed thresholds serially to Z slices, make binary
    # mask with a little bit of opening to remove single-pixel
    # objects.
    for z in range(0, n_slices):
        mask[z] = np.where(stack[z] >= thresh[z], 1, 0)
    mask = ndi.morphology.binary_opening(mask, np.ones((1,2,2)))

    # Plot number of detected objects per Z slice
    if display:
        counts = []
        for z in range(0, n_slices):
            _, count = ndi.label(mask[z])
            counts.append(count)
        plt.plot(counts)
        plt.ylim(0,600)
    return np.expand_dims(mask, axis=0)

############################################################################
def spot_data_add_depth(spot_data, surface_before, surface_after,
    join_frames, start_positions, z_interval=0.5):
    """Add a column to spot_data with the absolute embryo depth of every
    detected spot.

    Args:
        spot_data: dict of ndarrays
            Column 0 must be frame number (time), column 2 must be Z slice
        surface_before: float
            Position of embryo surface measured before taking stack, in 
            microns
        surface_after: float
            Position of embryo surface measured after taking stack, in 
            microns
        join_frames: iterable of ints
            Positions of frames where movies are joined
        start_positions: iterable of numeric
            Positions of the start (lowest Z slice) of z stack for each
            constituent stack
        z_interval: float
            Interval between Z slices, in microns
        
    Returns:
        spot_data_wdepth: dict of ndarrays
            Input spot_data object, with column added containing calibrated
            absolute embryo depth
    """
    def get_nframes(spot_data):
        """Get max frame number from spot_data object."""
        last_frame_all = 0
        for spot_id in spot_data:
            last_frame_spot = spot_data[spot_id][-1, idx_t]
            if last_frame_spot > last_frame_all:
                last_frame_all = last_frame_spot
        return last_frame_all + 1

    def make_surface_vector(before, after, nframes):
        """Make vector of inferred embryo surface positions for each frame,
        assuming drift occurs at a constant rate."""
        increment = (after - before) / nframes
        surface_position = np.arange(before, after, increment)
        return surface_position
    
    def make_true_start_vector(surface_positions, join_frames, start_positions):
        """Make vector of the true start position of the Z stack for every frame,
        calculated as the difference between the inferred embryo surface position
        and the z-stack start position stored in metadata."""
        nframes = len(surface_positions)
        # Add last frame to join_frames list.
        join_frames = join_frames + [nframes]
        # First, make a vector (of length = nframes) with the uncorrected
        # z-stack start position for each frame. 
        # Start at 0. Join frames starts with the first junction, so for each 
        # junction make the part of the vector between this junction and the 
        # previous junction.
        curr_segment_start = 0
        start_positions_uncorrected = np.array([])
        for n in range(0, len(join_frames)):
            length = join_frames[n] - curr_segment_start
            start_pos = float(start_positions[n])
            start_positions_uncorrected = np.concatenate([start_positions_uncorrected,
                np.repeat(start_pos, length)])
            curr_segment_start = join_frames[n]

        # Get corrected z-stack start positions by subtracting inferred embryo
        # surface positions.
        start_positions_corrected =  start_positions_uncorrected - surface_positions
        return start_positions_corrected
        
    def add_depth_column(spot_data, start_positions, z_interval):
        """Add new column to spot_data with absolute embryo depth."""
        spot_data_wdepth = {}
        for spot_id in spot_data:
            arr = spot_data[spot_id].copy()
            # Add new blank column of 0s.
            arr = np.append(arr, np.zeros((arr.shape[0],1)), axis=1)
            # Fill in data row by row.
            for row in range(0, arr.shape[0]):
                t,z = arr[row, idx_t], arr[row, idx_z]
                slice_depth = start_positions[int(t)] + (z * z_interval)
                arr[row, -1] = slice_depth
            spot_data_wdepth[spot_id] = arr
        return spot_data_wdepth

    # Set indexes for frame (time) and z slice in spot_data, get nframes.
    idx_t = 0
    idx_z = 2
    nframes = get_nframes(spot_data)
    # Get surface positions, then z-stack start positions, then add depth column.
    surface_positions = make_surface_vector(surface_before, surface_after, nframes)
    start_positions = make_true_start_vector(surface_positions, join_frames, start_positions)
    spot_data_wdepth = add_depth_column(spot_data, start_positions, z_interval)           
    return spot_data_wdepth


############################################################################
def spot_data_extract_depthbinned_intensities(spot_data, col_depth, col_to_bin, 
        bin_size=0.5, nbins=100):
        """Build vector of intensities binned by depth and their means.
        
        Args:
            spot_data: dict of ndarrays
                spot_data object
            col_depth: int
                Column in spot_data containing embryo depth
            col_to_bin: iterable of ints
                Columns in spot_data to correct
            bin_size: numeric
                Size of bins for depth, in microns
            nbins: int
                Number of bins to make
        
        Returns:
            vals: list of lists
                Each value in parent list is a depth bin, each bin contains
                a list of all values from col_to_bin at in that depth range
        """
        vals = [[]] * nbins
        for spot_id in spot_data:
            arr = spot_data[spot_id]
            for row in range(0, arr.shape[0]):
                # Convert depth to array index in 0.5 µm increments.
                depth = int(arr[row, col_depth] / bin_size)
                vals[depth] = vals[depth] + [arr[row, col_to_bin]]
        return vals

############################################################################
def spot_data_depth_correct_stdcandle(spot_data, paramgrids, 
    col_to_correct=9, col_depth=12, target_depth=10, display=True, 
    plot_title=''):
    """Correct spot_data object using parameters derived from standard 
    candles.

    Args:
        spot_data: dict of ndarrays
            Column 0 must be frame number (time), column 2 must be Z slice
        paramgrids: tuple of ndarrays
            Depth by intensity arrays of exponential parameters a, b, c, 
            according to function intensity = a * e^(-b * depth) + c
            Units are 0.1 µm on the depth axis and 100 a.u. on the intensity
            axis. For a spot measures at depth 10 microns of intensity 
            5000, parameters would be drawn from position (100, 50).
        col_to_correct: iterable of ints
            Column in spot_data to correct
        col_depth: int
            Column in spot_data containing embryo depth
        target_depth: numeric
            Reference depth to which all spots will be corrected
        display: bool
            If true, plot before and after boxplots of depth vs. intensity
        plot_title: string
            Label for data to use for plots

    Returns:
        spot_data_corr: dict of ndarrays
            Input spot_data with indicated columns corrected for depth
    """
    def calc_exponential(depth, a, b, c):
        """Return value of exponential function a * e^(-b * depth) + c."""
        return a * np.exp(-b * depth) + c
    
    paramgrid_a, paramgrid_b, paramgrid_c = paramgrids
    spot_data_corr = copy.deepcopy(spot_data)
    for spot_id in spot_data:
        for row in range(0, spot_data[spot_id].shape[0]):
            depth = spot_data[spot_id][row, col_depth]
            intensity = spot_data[spot_id][row, col_to_correct]
            paramgrid_position_intensity = int(intensity / 100)
            paramgrid_position_depth = int(depth / 0.1)
            if paramgrid_position_intensity > paramgrid_a.shape[1]:
                intensity_corr = intensity
            else:
                a = paramgrid_a[paramgrid_position_depth, paramgrid_position_intensity]
                b = paramgrid_b[paramgrid_position_depth, paramgrid_position_intensity]
                c = paramgrid_c[paramgrid_position_depth, paramgrid_position_intensity]
            intensity_corr = calc_exponential(target_depth, a, b, c)
            spot_data_corr[spot_id][row, col_to_correct] = intensity_corr
    
    if display:
        # Plot boxplot of intensity vs. depth for uncorrected and corrected data.
        vals = spot_data_extract_depthbinned_intensities(spot_data, col_depth, col_to_correct)
        plt.figure(figsize=(14, 4))
        plt.subplot(121)
        plt.boxplot(vals[10:40], labels=np.arange(5, 20, 0.5))
        plt.xticks(rotation = 45)
        plt.ylim(0, 15000)
        plt.title(plot_title + ' Before Std. Candle Correction')
        vals_corr = spot_data_extract_depthbinned_intensities(spot_data_corr, col_depth, col_to_correct)
        plt.subplot(122)
        plt.boxplot(vals_corr[10:40])
        plt.ylim(0, 15000)
        plt.title(plot_title + ' After Std. Candle Correction')
    
    return spot_data_corr

############################################################################
def spot_data_depth_correct_fromdata(spot_data, col_to_correct=9, 
    col_depth=12, target_depth=10, fit_depth_min=12, fit_depth_max=20,
    display=True, plot_title=''):
    """Apply sample depth correction to a column in spot_data by equalizing 
    the mean values of the feature across sample depths.

    For the indicated column, mean values are determined for 0.5 µm bins
    of sample depth. These depth vs. intensity curves are fitted with an 
    exponential function, and this function is used to correct all 
    values in the datasest.
    
    Args:
        spot_data: dict of ndarrays
            Column 0 must be frame number (time), column 2 must be Z slice
        col_to_correct: iterable of ints
            Column in spot_data to correct
        col_depth: int
            Column in spot_data containing embryo depth
        target_depth: numeric
            Reference depth to which all spots will be corrected
        fit_depth_min: numeric
            Minimum depth of data to be used for fitting
        fit_depth_max: numeric
            Maximum depth of data to be used for fitting
        display: bool
            If true, plot before and after boxplots of depth vs. intensity
        plot_title: string
            Label for data to use for plots

    Returns:
        spot_data_corr: dict of ndarrays
            Input spot_data with indicated columns corrected for depth
    """
    def exp_func(x, a, b, c):
        """Return value of exponential function a * e^(-b * x) + c."""
        return a * np.exp(-b * x) + c
    
    def get_intercept(x, y, a, b):
        """Solve exponential for intercept given all other parameters and 
        x,y."""
        return y - (a * np.exp(-b * x))

    def get_means_intensity_depth_vectors(vals):
        """Make a vector of mean intensity for each sample depth bin."""
        means = [[]] * 100
        for depth in range(0, len(vals)):
            if (len(vals[depth]) > 0):
                mean_ = np.nanmean(vals[depth])
                means[depth] = mean_
            else:
                means[depth] = np.nan
        return means

    spot_data_corrected = copy.deepcopy(spot_data)
    intensities_bydepth = spot_data_extract_depthbinned_intensities(spot_data, col_depth, col_to_correct)
    means_bydepth = get_means_intensity_depth_vectors(intensities_bydepth)
        
    # Fit depths vs. means with exponential.
    depths = np.arange(fit_depth_min, fit_depth_max, 0.5)
    fit_minbin = int(fit_depth_min / 0.5)
    fit_maxbin = int(fit_depth_max / 0.5)
    means_subset = means_bydepth[fit_minbin:fit_maxbin]
    (a,b,c),_ = scipy.optimize.curve_fit(exp_func, depths, means_subset, maxfev=100000)

    # Correct all values in indicated column based on fitted depth v. intensity
    # function.
    for spot_id in spot_data:
        arr = spot_data[spot_id]
        for row in range(0, arr.shape[0]):
            depth = arr[row, col_depth]
            value = arr[row, col_to_correct]
            intercept = get_intercept(depth, value, a, b)
            corrected_value = exp_func(target_depth, a, b, intercept)
            spot_data_corrected[spot_id][row, col_to_correct] = corrected_value

    if display:
        # Plot a boxplot of uncorrected intensities vs. sample depth.
        plt.figure(figsize=(5, 5))
        plt.subplot(311)
        plt.boxplot(intensities_bydepth[10:40], labels=np.arange(5, 20, 0.5));
        plt.title(plot_title + ' Depth vs. Intensity: Uncorrected')

        # Plot fit vs. data.
        x = np.arange(fit_depth_min, fit_depth_max, 0.02)
        y = a * np.exp(-b * x) + c
        plt.subplot(312)
        plt.scatter(x,y, s=0.5)
        plt.scatter(depths, means_subset)
        plt.title(plot_title + ' Fitting Result')

        # Plot a boxplot of corrected intensity values vs. sample depth.
        vals_corr = spot_data_extract_depthbinned_intensities(spot_data_corrected, col_depth, col_to_correct)
        plt.subplot(313)
        plt.boxplot(vals_corr[10:40]);
        plt.title(plot_title + ' Depth vs. Intensity: Corrected')
        plt.tight_layout()

    return spot_data_corrected

############################################################################
def mv_apply_corrections(mv, spot_data_orig, stack, paramgrids, 
    surface_before, surface_after, join_frames, start_positions, 
    z_interval=0.5, spotchannel=1, protchannel=0, ij_rad=3, z_rad=1.1, 
    ij_scale=1, z_scale=1):

    def plot_bleaching_corr(spot_data_before, spot_data_after, nframes):
        """Plot results of bleach correction."""
        plt.figure(figsize=(14, 8))
        plt.subplot(221)
        ms2_before = spot_data_extract_depthbinned_intensities(spot_data_before, 
            col_depth=0, col_to_bin=9, bin_size=1, nbins=nframes)
        plt.boxplot(ms2_before)
        plt.title('MS2 Before Bleaching Correction')
        plt.ylim(0, 1e4)
        plt.subplot(222)
        ms2_after = spot_data_extract_depthbinned_intensities(spot_data_after, 
            col_depth=0, col_to_bin=9, bin_size=1, bins=nframes)
        plt.boxplot(ms2_after)
        plt.title('MS2 After Bleaching Correction')
        plt.ylim(0, 1e4)
        plt.subplot(223)
        prot_before = spot_data_extract_depthbinned_intensities(spot_data_before, 
            col_depth=0, col_to_bin=11, bin_size=1, bins=nframes)
        plt.boxplot(prot_before)
        plt.title('Prot Before Bleaching Correction')
        plt.ylim(0, 1e4)
        plt.subplot(224)
        prot_after = spot_data_extract_depthbinned_intensities(spot_data_after, 
            col_depth=0, col_to_bin=11, bin_size=1, bins=nframes)
        plt.boxplot(prot_after)
        plt.title('Prot After Bleaching Correction')
        plt.ylim(0, 1e4)

    # Add columns corresponding to integrated MS2 signal (9), integrated
    # Gaussian fit of MS2 (10), and integrated protein signal (11).
    spot_data_plusms2 = add_volume_mean(spot_data_orig, stack, spotchannel, 
        ij_rad, z_rad, ij_scale, z_scale)
    
    wlength_ij = (2 * ij_rad) + 1
    wlength_z = (2 * int(z_rad)) + 1
    spot_data_plusgaussint = add_gaussian_integration(spot_data_plusms2, 
        wlength_ij, wlength_z)

    spot_data_plusprot = add_volume_mean(spot_data_plusgaussint, stack, 
        protchannel, ij_rad, z_rad, ij_scale, z_scale)

    # Add a column (12) containing the sample depth for each spot.
    spot_data_plusdepth = spot_data_add_depth(spot_data_plusprot, 
        surface_before, surface_after, join_frames, start_positions, 
        z_interval)

    # Store names of columns for supplying to plotting functions.
    colnames = {9:'MS2', 11:"Prot"}

    # Possible alternative: use some reference columns to add volume 
    # integrations with smaller radii to more closely match calculations from 
    # 120-mer. Use these intensities to get the shape of the correction curve
    # (parameters a and b), solve for intercept using the preferred intensity, 
    # perform correction. Do bleach correction. Make a version of 
    # spot_data_plus_depth with corrected intensity columns.

    # Perform bleach correction on the three columns containing intensity 
    # measurements. Currently using frame-mean based correction but this 
    # could change.
    spot_data_bleachcorr = spot_data_bleach_correct_framemean(
        spot_data_plusdepth, stack, protchannel, cols_to_correct=(9, 10, 11))

    # Plot results of bleach correction as mean intensity vs. time boxplots.
    nframes = stack.shape[1]
    #plot_bleaching_corr(spot_data_plusdepth, spot_data_bleachcorr, nframes)

    # Perform depth correction using standard candle method.
    spot_data_depthcorr_stdcandle = copy.deepcopy(spot_data_bleachcorr)
    for col_to_correct in (9, 11):
        spot_data_depthcorr_stdcandle = spot_data_depth_correct_stdcandle(
            spot_data_depthcorr_stdcandle, paramgrids, 
            col_to_correct=col_to_correct, col_depth=12, target_depth=12, 
            plot_title=colnames[col_to_correct])

    # Perform depth correction using from_data method.
    spot_data_depthcorr_fromdata = copy.deepcopy(spot_data_bleachcorr)
    for col_to_correct in (9, 11):
        spot_data_depthcorr_fromdata = spot_data_depth_correct_fromdata(
            spot_data_depthcorr_fromdata, col_to_correct=9, col_depth=12, 
            target_depth=12, fit_depth_min=12, fit_depth_max=20, 
            plot_title=colnames[col_to_correct])

    # Generate dataframes, add data structures to movie.
    mv.spot_data_bleachcorr = spot_data_bleachcorr
    mv.spot_data_depthcorr_stdcandle = spot_data_depthcorr_stdcandle
    mv.spot_data_depthcorr_fromdata = spot_data_depthcorr_fromdata
    mv.ms2_stdcandle = movie.make_spot_table(spot_data_depthcorr_stdcandle, 9)
    mv.ms2_fromdata = movie.make_spot_table(spot_data_depthcorr_fromdata, 9)
    mv.prot_stdcandle = movie.make_spot_table(spot_data_depthcorr_stdcandle, 11)
    mv.prot_fromdata = movie.make_spot_table(spot_data_depthcorr_fromdata, 11)
