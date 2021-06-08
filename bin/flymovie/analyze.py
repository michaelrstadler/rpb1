import numpy as np
from scipy import ndimage as ndi
import pandas as pd


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
            plt.plot(ndi.gaussian_filter1d(df1_processed.iloc[:,x], sigma))
            plt.plot(ndi.gaussian_filter1d(df2_processed.iloc[:,x], sigma))
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
