import numpy as np
import matplotlib as mpl
import os
from ipywidgets import interact, IntSlider, Dropdown, IntRangeSlider, fixed
import matplotlib.pyplot as plt


############################################################################
# Functions for interactive image viewing/analysis
############################################################################

def viewer(stacks, figsize=12, order='default', zmax=False, init_minval=None, init_maxval=None, color="Greens", coordfile=None):
    """Interactive Jupyter notebook viewer for n-dimensional image stacks.
    
    Args:
        stack: ndarray or list of ndarrays
            List of n-dimensional image stacks; last two dimensions must 
            be x-y to display. Image shapes must be identical.
        figsize: int
            Size of the figure to plot
        order: string
            String specifying order of image dimensions. Examples: 'ctzxy' 
            or 'tzxy'. Last two dimensions must be 'xy'.
        zmax: bool
            If True, displays a maximum projection on the Z axis.
        init_minval: int
            Initial value for minimum on contrast slider
        init_maxval: int
            Initial value for maximum on contrast slider
        color: string
            [optional] set the initial color map.
        coordfile: string
            [Optional] File to which to write coordinates of mouse clicks.
            Only works in %matplotlib notebook mode.
            
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

        numplots = len(stacks)
        indexes = []
        colmap = kwargs['colmap']
        if (colmap == 'Gators'):
            colmap = combine_colormaps(plt.cm.Blues_r, plt.cm.YlOrBr)
        min_ = kwargs['contrast'][0]
        max_ = kwargs['contrast'][1]
        
        # Unpack order variable into array.
        order_arr = [char for char in order[:-2]]
        # Populate indexes list with widgets.
        for n in order_arr:
            indexes.append(kwargs[n])
        
        # Set up frame for plots.
        fig, ax = plt.subplots(1, numplots, figsize=(figsize * numplots, figsize * numplots))
        if (coordfile != None):
            cid = fig.canvas.mpl_connect('button_press_event', lambda event: click_coord(event, indexes[-1], indexes[-2]))
        # If only one plot, pack ax into list
        if (type(ax) is not np.ndarray):
            ax = [ax]
        for n in range(0, numplots):
            stack_local = stacks[n]
            # Slice stack, leaving last two dimensions for image.
            # Note: the (...,) in the following is not required, but I think 
            # it is clarifying.
            ax[n].imshow(stack_local[tuple(indexes) + (...,)], cmap=colmap, vmin=min_, 
            vmax=max_);    
    
    # Write clicked coordinates to file.
    def click_coord(event, z, t):
            f = open(coordfile, 'a')
            f.write(str(t) + '\t' + str(z) + '\t' + str(event.ydata) + '\t' + str(event.xdata) + '\n')
            f.close()

    # Make a new slider object for dimension selection and return it
    def _make_slider(n):
        widget = IntSlider(min=0, max=(stack.shape[n] - 1), step=1, 
            continuous_update=False,)
        return(widget)
    
    # Combine two colormaps.
    def combine_colormaps(cm1, cm2):
        """Combine two mpl colormaps."""
        colors1 = cm1(np.linspace(0., 1, 128))
        colors2 = cm2(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        return(mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors))

    # Make a dropdown widget for selecting the colormap.
    def _make_cmap_dropdown():
        dropdown = Dropdown(
            options={'Greens', 'Reds', 'viridis', 'plasma', 'magma', 'inferno','cividis', 
            'gray', 'gray_r', 'prism', 'Gators'},
            value=color,
            description='Color',
        )
        return dropdown
    
    # Make a range slider for adjusting image contrast.
    def _make_constrast_slider():
        min_ = stack.min()
        max_ = stack.max()
        init_min = min_
        init_max = max_
        if (init_minval != None):
            init_min = init_minval
        if (init_maxval != None):
            init_max = init_maxval
        contrast_slider = IntRangeSlider(
            value=[init_min, init_max],
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
    
    # Use first stack as reference for sizes, etc.
    if (type(stacks) is list):
        stack = stacks[0]
    else:
        stack = stacks
        stacks = [stack]
    if zmax:
        axis_max = stacks[0].ndim - 3
        for i in range(0, len(stacks)):
            stacks[i] = stacks[i].max(axis=axis_max)
        stack = stacks[0]

    if (order == 'default'):
        if zmax:
            order = 'ctxy'[4-stacks[0].ndim:]
        else:
            order = 'ctzxy'[5-stacks[0].ndim:]

    # Open coordfile for appending.
    if (coordfile is not None):
        f = open(coordfile, 'w')
        f.write('#\n')
        f.close()

    main(order)

############################################################################
def qax(n, ncol=4):
    """Quick axes: generate 1d list of axes objects of specified number
    
    Args:
        n: int
            Number of plots desired
            
    Returns:
        ax1d: list
            1D list of axes objects in order top left to bottom right (columns
            then rows)
    """
    nrow = int(np.ceil(n / ncol))
    if (n < ncol):
        ncol = n
    fig, ax = plt.subplots(nrow, ncol, figsize=(16, 4*nrow))
    ax1d = []
    pos1d = 0
    if (nrow > 1):
        for r in range(0, nrow):
            for c in range(0, ncol):
                ax1d.append(ax[r][c])
                pos1d = pos1d + 1
    else:
        for c in range(0, ncol):
            ax1d.append(ax[c])
            pos1d = pos1d + 1
    
    return ax1d

############################################################################
def plot_ps(func, span=range(0,8)):
    """Plot a parameter series in a specified range
    
    User supplies a plotting function that takes a single integer input as
    a parameter. plot_ps builds axes to display all parameter values and
    serially calls plot function on them.
    
    Example:
       def temp(x):
            dog = dog_filter(red, x, 3)
            plt.imshow(dog)

        plot_ps(temp, range(8,13)) 
    
    Args:
        func: function
            Must take a single integer value as a parameter and call a plot
            function on the active axes object.
        span: range
            Range object containing values of parameter to plot. 
    """
    nplots = len(span)
    ax = qax(int(len(span)))
    for pln in range(0, len(span)):
        plt.sca(ax[pln])
        func(span[pln])

############################################################################
def box_spots(stack, spot_data, max_mult=1.3, halfwidth_xy=15, 
              halfwidth_z=8, linewidth=3, shadows=True):
    """Draw boxes around detected MS2 spots.
    
    Usage suggestions: Useful with a Z-projection to examine effectiveness
    of spot segmentation. Can also use with very small halfwidths to create
    artificial "dots" representing called spots to overlay on other data,
    e.g. a nuclear mask or a blank matrix of 0s (to examine spot movement
    alone).
    
    Args: 
        stack: ndarray of uint16
            Multi-dimensional image stack of dimensions [t,z,x,y]
        spot_data: dict of ndarrays
            Data containing tracking of spots detected. Dict entries are unique 
            spot IDs (numeric 1...), rows of ndarray are detections of the spot 
            in a single frame. Time must be column 0, [z,x,y] in columns 2:4.
        max_multi: numeric
            Multiplier of maximum value to use for drawing box.
        halfwidth_xy: int
            Halfwidth in pixels of the boxes in xy direction (sides will be 
            2*halfwidth)
        halfwidth_z: int
            Halfwidth of the boxes in z direction(vertical sides will be 
            2*halfwidth)
        linewidth: int
            Width of lines used to draw boxes
        shadows: bool
            Draw "shadows" (dark boxes) in non-boxed z-slices.
        
    Return:
        boxstack: ndarray
            Selected channel of input image stack with boxes drawn around 
            spots. Dimensions [t,z,x,y]
    """
    if (stack.dtype != 'uint16'):
        raise ValueError("Stack must be uint16.")
    boxstack = np.copy(stack)
    hival = max_mult * boxstack.max()
    if (hival > 65535):
        hival = 65535
    
    def drawbox(boxstack, point, halfwidth_xy, halfwidth_z, linewidth, hival, shadows):
        t, z, i, j = point
        z_min = max(0, z - halfwidth_z)
        z_max = min(boxstack.shape[1], z + halfwidth_z + 1)
        i_min = max(0, i - halfwidth_xy)
        i_max = min(boxstack.shape[2], i + halfwidth_xy + 1)
        j_min = max(0, j - halfwidth_xy)
        j_max = min(boxstack.shape[3], j + halfwidth_xy + 1)
        if shadows:
            # Draw shadow boxes in all Z-frames.
            boxstack[t, :, i_min:i_max, j_min:(j_min + linewidth)] = 0
            boxstack[t, :, i_min:i_max, (j_max-linewidth):j_max] = 0
            boxstack[t, :, i_min:(i_min+linewidth), j_min:j_max] = 0
            boxstack[t, :, (i_max-linewidth):i_max, j_min:j_max] = 0
        # Draw left line.
        boxstack[t, z_min:z_max, i_min:i_max, j_min:(j_min + linewidth)] = hival     
        # Draw right line. 
        boxstack[t, z_min:z_max, i_min:i_max, (j_max-linewidth):j_max] = hival
        # Draw top line. 
        boxstack[t, z_min:z_max, i_min:(i_min+linewidth), j_min:j_max] = hival
        # Draw bottom line.
        boxstack[t, z_min:z_max, (i_max-linewidth):i_max, j_min:j_max] = hival
    
    # Main.
    if (type(spot_data) == dict):
        for spot in spot_data:
            arr = spot_data[spot]
            for row in arr:
                row = row.astype(int)
                point = (row[[0,2,3,4]])
                drawbox(boxstack, point, halfwidth_xy, halfwidth_z, linewidth, hival, shadows)
    elif (type(spot_data) == list):
        for t in range(0, len(spot_data)):
            for row in spot_data[t]:
                point = tuple([t]) + tuple(row[0:3].astype(int))
                drawbox(boxstack, point, halfwidth_xy, halfwidth_z, linewidth, hival, shadows)
    return boxstack   

############################################################################
def quickview_ms2(stack, spot_data, channel=0, figsize=12, MAX=True, 
    halfwidth_xy=8, halfwidth_z=8, spotmode=False, spot_id='all', 
    color='cividis', init_minval=0, init_maxval=1000000, shadows=True):
    """View image stack with boxes drawn around detected spots
    
    Args:
        stack: ndarray
            5d image stack [c,t,z,x,y]
        spot_data: dict of ndarrays
            Data containing tracking of spots detected. Dict entries are unique 
            spot IDs (numeric 1...), rows of ndarray are detections of the spot 
            in a single frame. Time must be column 0, [z,x,y] in columns 2:4.
        channel: int
            Channel (dimension 0) to be viewed
        figsize: int
            Size of figure to display via viewer
        MAX: bool
            Display max projection in Z.
        halfwidth_xy: int
            Halfwidth in pixels of the boxes in xy direction (sides will be 
            2*halfwidth)
        halfwidth_z: int
            Halfwidth in pixels of the boxes in z direction (sides will be 
            2*halfwidth)
        spotmode: bool
            (optional) display a spot instead of a box
        spot_id: int 
            (optional) ID of spot to box. Default boxes all detected spots.
        color: string or color object
            Starting colormap for viewer
        init_minval: int
            Initial value for minimum on contrast slider
        init_maxval: int
            Initial value for maximum on contrast slider
        shadows: bool
            If true, show dark boxes in out of focus Z slices.

    """
    if (spot_id == 'all'):
        data = spot_data.copy()
    else:
        data = {spot_id:spot_data[spot_id]}
    
    if spotmode:
        halfwidth_xy = 3

    substack = stack[channel]
    boxes = box_spots(substack, data, halfwidth_xy=halfwidth_xy, halfwidth_z=halfwidth_z, linewidth=2, shadows=shadows)
    if MAX:
        viewer(boxes.max(axis=1), figsize, 'txy', color=color, init_minval=init_minval, init_maxval=init_maxval)
    else:
        viewer(boxes, figsize, 'tzxy', color=color, init_minval=init_minval, init_maxval=init_maxval)

############################################################################
def spot_movies(stack, spot_data, channel=0, len_ij=15, len_z=7, fill=np.nan, view=False):
    """Make image stack for viewing MS2 spot raw data
    
    Given spot coordinates and original movie, builds an image stack with 
    spot ID on the first dimension and raw data centered on the spot in the
    remaining three. In cases where the window exceeds the boundaries of the
    image, the remaining portions of the frame will be filled by the value
    specified by fill.
    
    Args:
        stack: ndarray
            5-D image stack [c,t,z,x,y]
        spot_data: dict of ndarrays
            Data containing tracking of spots detected. Dict entries are unique 
            spot IDs (numeric 1...), rows of ndarray are detections of the spot 
            in a single frame. Required columns: 5: gaussian fit height, 6: 
            gaussian fit z-width, 7: gaussian fit x-width, 8: gaussian fit 
            y-width.
        channel: int
            Channel from which to use data
        len_ij: int
            Length (in pixels) of the box to collect around each spot in the 
            lateral (ij) dimension.
        len_z: int
            Length (in pixels) of the box to collect around each spot in the 
            axial (z) dimension.
        fill: numeric
            Value to fill in regions that exceed the boundaries of the image
        view: bool
            If true, calls viewer function on the resulting stack using a max 
            projection on Z dimension (order spot_ID-t-x-y)
            
    Returns:
        movies: ndarray
            5D image stack of input data centered around each spot. In order 
            [spot_id, t,z,x,y]
    """
    def adjust_coords(c, rad, frame_max, stack_max):
        """Adjusts coordinates to account for spots located at the edge of the 
        stack such that the window will cross the boundary of the image. 
        For a given axis, returns both the minimum and max coordinate
        for slicing the stack and the min and max coordinates for the position
        in the frame to which to assign this slice."""

        # Initialize coordinate and frame boundaries.
        cmin = int(c - rad)
        cmax = int(c + rad)
        frame_min = 0

        # Correct for cases where bounds of the stack are exceeded.
        if (cmin < 0):
            frame_min = -1 * cmin
            cmin = 0
        if (cmax > stack_max):
            frame_max = frame_max - (cmax - stack_max)
            cmax = stack_max
        return cmin, cmax, frame_min, frame_max

    # Check that window dimensions are appropriate.
    if ((len_ij % 2 == 0) or (len_z % 2 == 0)):
        raise ValuError('len_ij and len_z must be odd')
    # Define 'radii' for window (distance to go from home pixel in each direction).
    z_rad = int((len_z - 1) / 2)
    ij_rad = int((len_ij - 1) / 2)
    # Initialize movies as a zeros array.
    movies = np.zeros((max(spot_data)+1, stack.shape[1], len_z, len_ij, len_ij))
    # Add data for each spot.
    for spot in spot_data:
        arr = spot_data[spot]
        for row in arr:
            # Get t,z,x,y coordinates for spot.
            t = int(row[0])
            z,i,j = row[2:5]
            # Initialize frame.
            frame = np.empty((len_z, len_ij, len_ij))
            frame.fill(fill)
            # Get the min and max values for each coordinate with reference to the stack and the
            # coordinates in the frame that they will replace.
            zmin, zmax, frame_zmin, frame_zmax = adjust_coords(z, z_rad, len_z - 1, stack.shape[-3]-1)
            imin, imax, frame_imin, frame_imax = adjust_coords(i, ij_rad, len_ij - 1, stack.shape[-2]-1)
            jmin, jmax, frame_jmin, frame_jmax = adjust_coords(j, ij_rad, len_ij - 1, stack.shape[-1]-1)
            # Update frame from stack and assign to movies.
            frame[frame_zmin:(frame_zmax+1), frame_imin:(frame_imax+1), frame_jmin:(frame_jmax+1)] = stack[channel, t, zmin:(zmax+1), imin:(imax+1), jmin:(jmax+1)]
            movies[spot, t] = frame
    # If viewer call specified, call with mean (sum) Z-projection.
    if(view):
        #viewer(movies.mean(axis=2), 'itxy')
        viewer(np.nanmean(movies, axis=2), 'itxy')
    return movies