import numpy as np
import os
from scipy import ndimage as ndi
from scipy.spatial import distance
from copy import deepcopy

from .general_functions import dog_filter, peak_local_max_nD, labelmask_apply_morphology
from .fitting import fitgaussian3d
############################################################################
def fit_ms2(stack, min_distances=(70,50,50), sigma_small=1, 
                   sigma_big=4, bg_radius=4, fitwindow_rad_xy=10, 
                   fitwindow_rad_z=2):  
    """Perform 3D gaussian fitting on local maxima in a 4D image stack
    
    Alrigthm: bandbass filter -> background subtraction -> find local maxima
    -> fit gaussian to windows around maxima
    
    Args:
        stack: ndarray
            4D image stack [t,z,x,y] containing MS2 spots
        min_distances: tuple of three ints
            Minimum distance (in pixels) allowed between spots for them to be
            counted as distinct spots. Minimum distance supplied for each 
            dimension.
        sigma_small: numeric
            Lower sigma for difference-of-gaussians bandpass filter
        sigma_small: numeric
            Upper sigma for difference-of-gaussians bandpass filter. Critical 
            for this to be at least one, otherwise hot pixels create problems
        bg_radius: int
            Radius for minimum filter used for background subtraction
        fitwindow_rad_xy: int
            Radius in pixels in the xy-dimension of the window around local
            maxima peaks within which `to do gaussian fitting.
        fitwindow_rad_z: int
            Radius in pixels in the z-dimension of the window around local
            maxima peaks within which to do gaussian fitting.
    
    Returns:
        fit_data: list of ndarrays
            Each entry in the list is a time point (frame). Each row in
            array is a fit (a single local maxima), columns are: 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: fit_height, 4: width_z, 5: width_x, 6: width_y). Coordinates 
            are adjusted so that if fit center lies outside the image, 
            center is moved to the edge.
    """
    # Task: change size to minimum distance
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
    
    def relabel(peak_ids, oldparams, mask):
        """Renumber labelmask and corresponding fit parameters
        Set background as 0, objects in order 1...end.
        """
        spot_data = {}
        peak_num = 1
        for peak in peak_ids:
            #coords = np.where(mask == peak)
            paramsnew = oldparams[peak-1,:] # object 1 will be fitparams row 0
            # Rearrange params from fit function so coordinates lead.
            spot_data[peak_num] = paramsnew[[1,2,3,0,4,5,6]]
            peak_num = peak_num + 1
        return spot_data

    def clamp(n, minn, maxn):
        """Bound a number between two constants"""
        return max(min(maxn, n), minn)
    
    def fit_frame(substack, min_distances, sigma_small, 
                   sigma_big, bg_radius, fitwindow_rad_xy, 
                   fitwindow_rad_z):
        """Perform 3D gaussian fitting on a 3D image stack."""

        # Filter and background subtract image.
        dog = dog_filter(substack, sigma_small, sigma_big)
        bg = ndi.filters.minimum_filter(dog, bg_radius)
        dog_bs = dog - bg

        # Make a labelmask corresponding to local maxima peaks.
        peak_window_size = (min_distances[0] * 2 + 1, min_distances[1] * 2 + 1, min_distances[2] * 2 + 1)
        mask, peaks = peak_local_max_nD(dog_bs, peak_window_size)

        # Fit 3D gaussian in window surrounding each local maximum.
        fitparams = np.ndarray((0,7))
        for peak in peaks:
            fitwindow, z_adj, x_adj, y_adj = get_fitwindow(substack, peak, fitwindow_rad_xy, 
                fitwindow_rad_z)
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
    
    #### Main ####
    # Do fitting on first frame.
    fit_data_frame0 = fit_frame(stack[0], min_distances, sigma_small, 
                   sigma_big, bg_radius, fitwindow_rad_xy, 
                   fitwindow_rad_z)
    # Make fit_data a list of ndarrays.
    fit_data = [fit_data_frame0]
    
    # Fit the rest of the frames, add their data to fit_data.
    for i in range(1, stack.shape[0]):
        print(i, end=' ')
        fit_data_thisframe = fit_frame(stack[i], min_distances, sigma_small, 
                   sigma_big, bg_radius, fitwindow_rad_xy, 
                   fitwindow_rad_z)
        fit_data.append(fit_data_thisframe)
        
    return fit_data

############################################################################
def filter_ms2fits(fit_data, peakiness=4.5, stack=None, channel=1):
    """Filter MS2 spot fit data based on fit parameters
    
    Select spots based on "peakiness", measured as the ratio of the height
    to width of the fit. Currently uses the mean of the x- and y-widths.
    
    Args:
        fit_data: list of ndarrays
            Each entry in list is a distinct frame (in time), rows in array
            are individual spot fits and columns are 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: fit_height, 4: width_z, 5: width_x, 6: width_y.
        peakiness: numeric
            Spots whose log (natural) ratio of height to width exceeds this
            are kept.
        stack: ndarray
            [Optional] 5D stack from which fits are derived. If supplied,
            fits will be filtered for those whose peak surpasses the median
            value for the frame in this stack.
        channel: int
            Channel in the stack on which fits were performed.
            
    Returns:
        fit_data: list of ndarrays
            Input data, retaining only rows that pass filter.  
    """
    
    fit_data = fit_data.copy()
    for t in range(0, len(fit_data)):
        frame_data = fit_data[t]
        if stack is not None:
            frame_med = np.median(stack[channel, t])
        else:
            frame_med = -np.inf
        xy_width_means = np.mean(frame_data[:,5:7], axis=1)
        peak_heights = frame_data[:,3]
        spot_peakiness = np.log(peak_heights / xy_width_means)
        frame_data_filtered = frame_data[(peak_heights > frame_med) & (spot_peakiness > peakiness),:]
        fit_data[t] = frame_data_filtered
    return fit_data

############################################################################
def connect_ms2_frames(spot_data, nucmask, max_frame_gap=1, max_jump=10, 
    scale_xy=1, scale_z=1):
    """Connect detected MS2 spots in a single movie through multiple time 
    frames based on the distance between spots.
    
    Spots detected in new frame are connected to spots in previous frames
    if they are within specified distance (max_jump). Spots can "disappear" 
    for a number of frames defined by max_frame_gap. Spots that cannot be 
    connected to spots from prior frames are initialized as new spots.
    
    Args:
        spot_data: list of ndarrays
            Each entry in list is a distinct frame (in time), rows in array
            are individual detected spots and columns are 0: center 
            z-coordinate, 1: center x-coordinate, 2: center y-coordinate, 
            3: fit_height, 4: width_z, 5: width_x, 6: width_y.
        nucmask: ndarray
            4D labelmask of dimensions [t,z,x,y] of segmented nuclei. 0 is 
            background (not a nucleus) and nuclei have integer labels.
        max_frame_gap: int
            Maximum number of frames from which spot can be absent and still
            connected across the gap. Example: for a value of 1, a spot
            detected in frame 6 and absent from frame 7 can be connected to
            a spot in frame 8, but a spot in frame 5 cannot be connected to
            frame 8 if it is absent in frames 6 and 7.
        max_jump: numeric
            Maximum 3D displacement between frames for two spots to be connected
        scale_xy: numeric
            Distance scale for xy direction (typically: nm per pixel)
        scale_z: numeric
            Distance scale for z direction (typically: nm per pixel)
        
    Returns:
        connected_data: dict of ndarrays
            Each key is a unique spot tracked across 1 or more frames. Each row
            of array is the spot's data for a single frame, with columns 0: frame
            number (t), 1: nucleus ID, 2: center Z-coordinate, 3: center X-coord-
            inate, 4: center Y-coordinate, 5: fit height, 6: fit z_width, 7: fit
            x_width, 8: fit y_width
    """
    def initialize_new_spot(new_spot_data, connected_data):
        """Initialize new spot with next numeric ID and entry in connected_data."""
        if (connected_data.keys()):
            new_id = max(connected_data.keys()) + 1
        else:
            new_id = 1
        connected_data[new_id] = np.expand_dims(new_spot_data, 0)


    def get_peakiness(spot_data):
        """Determine peakiness for a spot from height and mean of x- and y- widths"""
        return spot_data[3] / np.mean((spot_data[5], spot_data[6]))

    def sq_euc_distance(coords1, coords2, scale_z=1, scale_xy=1):
        """Find the squared euclidean distance between two points."""
        z2 = ((coords2[0] - coords1[0]) * scale_z) ** 2
        x2 = ((coords2[1] - coords1[1]) * scale_xy) ** 2
        y2 = ((coords2[2] - coords1[2]) * scale_xy) ** 2
        sed = z2 + x2 + y2
        return sed
    
    def coord_list_t(connected_data, t):
        """Make a list of [z,x,y] coordinate tuples for all spots in a given
        frame"""
        coord_list = []
        for spot_id in connected_data:
            this_spot_data = connected_data[spot_id]
            row = this_spot_data[this_spot_data[:,0] == t]
            if (len(row) > 0):
                row = list(row[0])
                spot_coords = [spot_id] + row[2:5]
                coord_list.append(spot_coords)
        return coord_list
            
    
    def find_nearest_spot(this_coord, coord_list, scale_z, scale_xy):
        """For a given point, find the closest spot in a coordinate list
        and the distance between the points."""
        closest_sed = np.inf
        closest_spot = 0
        for test_data in coord_list:
            test_spot_id = test_data[0]
            test_coords = (test_data[1:4])
            sed = sq_euc_distance(test_coords, this_coord, scale_z, scale_xy)
            if (sed < closest_sed):
                closest_sed = sed
                closest_spot = test_spot_id
                closest_spot_coords = test_coords
        return closest_spot, np.sqrt(closest_sed), closest_spot_coords

    def update_spot(this_spot_data, connected_data, scale_z, scale_xy, max_frame_gap, 
                    t):
        """Walk back one frame at a time within limit set by maximum gap, search 
        for a nearest spot that is within the maximum allowable jump, handle 
        duplicates, add connected points to connected_data."""
        this_spot_coords = (this_spot_data[2:5])
        # Walk back one frame at a time.
        for t_lag in range(1, max_frame_gap + 2):
            if ((t - t_lag) >= 0):
                # Get nearest spot in the current frame.
                spot_coords_tlag = coord_list_t(connected_data, t - t_lag)
                # If there are no previously detected spots, break from for loop and initialize new spot entry.
                if (len(spot_coords_tlag) == 0):
                    break
                nearest_spot_id, dist, nearest_spot_coords = find_nearest_spot(this_spot_coords, spot_coords_tlag, scale_z, scale_xy)
                # Check is spot is within max distance.
                if (dist <= max_jump):
                    this_spot_nucID = this_spot_data[1]
                    nearest_spot_nucID = connected_data[nearest_spot_id][-1,1]
                    # Check if there's already a spot added for this time.
                    existing = connected_data[nearest_spot_id][connected_data[nearest_spot_id][:,0] == t]
                    # If there's no existing spot, add this spot to the end of the data for connected spot.
                    if (len(existing) == 0):
                        connected_data[nearest_spot_id] = np.append(connected_data[nearest_spot_id], [this_spot_data], axis=0)
                        return
                    # If there is an existing spot, if the current spot is closer to the previous-frame spot
                    # than the existing entry, replace it. Otherwise, continue looking in previous frames (if
                    # applicable) and eventually create new spot after for loop. I'm not sure this is the best
                    # behavior--may consider dumping out of for loop and creating new spot rather than looking
                    # to previous frames in this situation.
                    else:
                        existing_dist = np.sqrt(sq_euc_distance(nearest_spot_coords, existing[0,2:5], scale_z, scale_xy))
                        # If the the current spot is closer than the existing spot, replace 
                        # existing and initialize it as a new spot.
                        if (dist < existing_dist):
                            row_index = np.where(connected_data[nearest_spot_id][:,0] == t)[0][0]
                            superseded_spot_data = connected_data[nearest_spot_id][row_index]
                            # Superseded spot from this frame gets bumped to be a new spot.
                            initialize_new_spot(superseded_spot_data, connected_data)
                            # Replace data for superseded spot with this spot's data.
                            connected_data[nearest_spot_id][row_index] = this_spot_data
                            return

        # If no suitable spot was found in previous frames, make a new spot.
        initialize_new_spot(this_spot_data, connected_data)
    
    def inbounds(coords, nucmask):
        """Ensure coordinates are within the image for looking up nucleus ID
        (Z-correction for re-focusing can result in out of bounds points, and XY-correction
        for nuclear movement can do the same)"""
        coords = list(coords)
        coords[1] = clamp(coords[1], 0, nucmask.shape[-3] - 1)
        coords[2] = clamp(coords[2], 0, nucmask.shape[-2] - 1)
        coords[3] = clamp(coords[3], 0, nucmask.shape[-1] - 1)
        return tuple(coords)


    def add_time_nuc(this_spot_data, t, nucmask):
        """ Append the frame number (t) and nucleus ID to front of spot_data"""
        # Combine frame number and zxy to for coordinate tuple, accounting for out-of-bounds z-coordinates due to re-focus adjustments.
        spot_coords = tuple(np.concatenate((
            [t], 
            this_spot_data[0:3]
            )).astype(int))
        
        nuc_id = nucmask[inbounds(spot_coords, nucmask)]
        # Add time and nuclear ID columns to spot data and call update to search 
        # for connected spots in previous frames.
        this_spot_data = np.append([t, nuc_id], this_spot_data)
        return this_spot_data
    
    def initialize_connected_data(frame_data):
        """Initialize connected_data structure as a dict where each entry
        is a unique spot and the array rows are data for single frames"""
        connected_data = {}
        for i in range(0, len(frame_data)):
            this_spot_data = add_time_nuc(frame_data[i], 0, nucmask)
            connected_data[i+1] = np.array([this_spot_data])
        return connected_data
        
    # Main
    input_data = spot_data.copy()
    connected_data = initialize_connected_data(input_data[0])
    
    # Go through each frame, attempt to connect each detected spot to previous spots.
    for t in range(1, len(input_data)):
        print(t, end=' ')
        frame_data = input_data[t]
        for this_spot_data in frame_data:
            this_spot_data = add_time_nuc(this_spot_data, t, nucmask)
            update_spot(this_spot_data, connected_data, scale_z, scale_xy, max_frame_gap, t)
    
    return connected_data  


############################################################################
def connect_ms2_fits_focuscorrect(fits_orig, z_frames, z_corrs, nucmask, 
    max_frame_gap=1, max_jump=18, scale_xy=1, scale_z=1):
    """Connect MS2 spots (from fits) across time frames and from multiple 
    movies with different Z-focus points
    
    Args:
        fits_orig:list of ndarrays
            Output of fit_ms2 function. Each entry in the list is a time point 
            (frame). Each row in array is a fit (a single local maxima), 
            columns are: 0: center z-coordinate, 1: center x-coordinate, 
            2: center y-coordinate, 3: fit_height, 4: width_z, 5: width_x, 
            6: width_y).
        z_frames: list-like
            The frame numbers of the focus changes, with the number
            corresponding to the first frame (0-based) of the new focal point.
        z_corrs: list-like
            The Z-corrections matching the frames in z_frames. This is the 
            number that must be added (can be negative) to the new (frames 
            after the re-focus) frames to match the old. So if the focus is
            moved down by 2 Z-slices, the correction will be +2.
        nucmask: ndarray
            4D labelmask of dimensions [t,z,x,y] of segmented nuclei. 0 is 
            background (not a nucleus) and nuclei have integer labels.
        max_frame_gap: int
            Maximum number of frames from which spot can be absent and still
            connected across the gap. Example: for a value of 1, a spot
            detected in frame 6 and absent from frame 7 can be connected to
            a spot in frame 8, but a spot in frame 5 cannot be connected to
            frame 8 if it is absent in frames 6 and 7.
        max_jump: numeric
            Maximum 3D displacement between frames for two spots to be connected
        scale_xy: numeric
            Distance scale for xy direction (typically: nm per pixel)
        scale_z: numeric
            Distance scale for z direction (typically: nm per pixel) 
        
    Returns:
        spots_connected: dict of ndarrays
            Each key is a unique spot tracked across 1 or more frames. Each row
            of array is the spot's data for a single frame, with columns 0: frame
            number (t), 1: nucleus ID, 2: center Z-coordinate, 3: center X-coord-
            inate, 4: center Y-coordinate, 5: fit height, 6: fit z_width, 7: fit
            x_width, 8: fit y_width. Coordinates are relative to the original, 
            uncorrected data (which will be correct for a concatenated image 
            stack).
    
    """
    ## FIRST PART: Create a single fits object (list of ndarrays) with the Z-coordinates
    ## adjusted to correct for the refocusing.
    
    # Make a copy of fits to be altered.
    fits_adj = deepcopy(fits_orig) 
    # Make a vector of length equal to number of frames storing the z correction for each
    # frame (relative to first, uncorrected segment).
    frame_corrs = np.repeat(0, z_frames[0])  
    corr = 0
    
    # Do Z-correction for each focus break.
    for i in range(0, len(z_frames)):
        start_frame = z_frames[i]
        corr = corr + z_corrs[i]
        if (i+1 < len(z_frames)):
            end_frame = z_frames[i+1] - 1
        else:
            end_frame = len(fits_orig) - 1
        # Adjust z-coordinates for each frame in the window.
        for f in range(start_frame, end_frame + 1):
            fits_adj[f][:,0] = fits_adj[f][:,0] + corr
        # Add adjustment to frame_corrs vector.
        frame_corrs = np.concatenate((frame_corrs, np.repeat(corr, end_frame - start_frame + 1)))
    
    ## SECOND PART: Run spot connection on "corrected" fits object.
    spots_connected = connect_ms2_frames(fits_adj, nucmask, max_frame_gap, max_jump, scale_xy, scale_z)
    
    ## THIRD PART: Undo the adjustments in Z to make the spot coordinates match the concatenated
    ## stack.
    for spot in spots_connected:
        arr = spots_connected[spot]
        for i in range(0, len(arr)):
            frame_num = int(arr[i,0])
            arr[i,2] = arr[i,2] - frame_corrs[frame_num]
    return spots_connected

############################################################################
def connect_ms2_frames_via_nuclei(frame_data, nucmask, nucmask_dilation=5):
    """Connect MS2 spots detected in separate time frames based on nucleus
    ID.
    
    Args:
        frame_data: list of ndarrays
            Output from fitting. Each entry in the list is a time point 
            (frame). Each row in array is a fit (a single local maxima), 
            columns are: 0: center z-coordinate, 1: center x-coordinate, 
            2: center y-coordinate, 3: fit_height, 4: width_z, 5: width_x,
            6: width_y). 
        nucmask: ndarray
            4d labelmask of segmented nuclei
        nucmask_dilation: int
            Distance to dilate (in x and y directions) nuclear labelmask for
            looking up the nuclear ID of detected spots.
    
    Returns:
        spot_data: dict of ndarrays
            Each key is a unique spot tracked across 1 or more frames. Each row
            of array is the spot's data for a single frame, with columns 0: frame
            number (t), 1: nucleus ID, 2: center Z-coordinate, 3: center X-coord-
            inate, 4: center Y-coordinate, 5: fit height, 6: fit z_width, 7: fit
            x_width, 8: fit y_width
    """
    def find_nuc_id(coords, t, nucmask_lookup):
        """Find object in nuclear mask at particular coordinate."""
        # Make 4d coordinates as a list.
        coords_nucmask = np.concatenate([[t], coords])
        # Convert coordinates to a tuple of ints.
        coords_nucmask = tuple([int(n) for n in coords_nucmask])
        nuc_id = nucmask_lookup[coords_nucmask]
        return nuc_id
    
    def add_row(row, spot_data):
        """Add a new row (spot data for a single frame) to spot data 
        object."""
        # Pull nuc_id and t off of row.
        t, nuc_id = row[0:2].astype(int)
        # If this nucleus is already in spot data, add to existing data.
        if (nuc_id in spot_data):
            # Check to see if there is already a spot for this nucleus and time.
            if (spot_data[nuc_id][-1,0] == t):
                # If spot_data for this nucleus has a previous entry, choose
                # closest spot. Otherwise just leave current.
                if (spot_data[nuc_id].shape[0] > 1):
                    # Find the euclidean distance between the spot position in the
                    # most recent frame and both the candidate spots in the new
                    # frame.
                    coords_previous_frame = spot_data[nuc_id][-2,2:5]
                    coords_existing = spot_data[nuc_id][-1,2:5]
                    coords_new = row[2:5]
                    # full function: scipy.spatial.distance.euclidean
                    dist_existing = distance.euclidean(coords_previous_frame, coords_existing)
                    dist_new = distance.euclidean(coords_previous_frame, coords_new)
                    # Select closest spot to previous to go into spot data.
                    if (dist_new < dist_existing):
                        spot_data[nuc_id][-1] = row
            # If no spot already, just add new row to end.
            else:
                spot_data[nuc_id] = np.vstack([spot_data[nuc_id], row])
        # If nucleus isn't in spot data, initialize new entry.   
        else:
            spot_data[nuc_id] = np.expand_dims(row, axis=0)
            
    def renumber_spots(spot_data):
        """Renumber the spots in spot_data sequentially from 1."""
        spot_data_renumbered = {}
        spot_number_new = 1
        for spot_number_old in spot_data:
            spot_data_renumbered[spot_number_new] = spot_data[spot_number_old]
            spot_number_new += 1
        return spot_data_renumbered
    
    # Make a new lookup version of nuclear mask that dilates (expands in all 
    # directions) the nucleus objects.
    
    nucmask_lookup = labelmask_apply_morphology(nucmask, ndi.morphology.binary_dilation, 
        struct=np.ones((1,1,nucmask_dilation,nucmask_dilation)), 
        expand_size=(1,1,nucmask_dilation,nucmask_dilation)) 
   
    #nucmask_lookup = nucmask
    
    # Go through all frames of input data, lookup nuclei for each detected spot,
    # Add data to spot_data.
    spot_data = {}
    nframes = len(frame_data)
    for t in range(0, nframes):
        print (t, end=' ')
        for row in frame_data[t]:
            coords = row[0:3]
            nuc_id = find_nuc_id(coords, t, nucmask_lookup)
            if (nuc_id != 0):
                new_row = np.concatenate([[t, nuc_id], row])
                add_row(new_row, spot_data)
                
    return renumber_spots(spot_data)

############################################################################
def filter_spot_duration(connected_data, min_len):
    """Filter connected_data structure for spots lasting a minimum number
    of frames

    Args:
        connected_data: dict of ndarrays
            Output data structure of connect_ms2_frames
        min_len: int
            Minimum number of frames in which a spot must be detected

    Returns:
        filtered_data: dict of ndarrays
            connected_data retaining only spots of defined duration

    """
    filtered_data = {}
    spot_num = 1
    for spot in connected_data:
        if (connected_data[spot].shape[0] >= min_len):
            filtered_data[spot_num] = connected_data[spot]
            spot_num = spot_num + 1
    return filtered_data