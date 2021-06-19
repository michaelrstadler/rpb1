import numpy as np
import pandas as pd

class movie():
    """
    A class to store data (processed and unprocessed) for confocal movies
    from fly embryos.

    Style note: movie should be capitalized, but I did not know this style
    convention when I made it and I am leaving it for compatability reasons

    Attributes:
        stack: ndarray
            N-dimensional image stack of the original data.
        nucmask: ndarray
            Labelmask of segmented nuclei
        fits: list of ndarrays
            3D gaussian fits generated in MS2 spot detection as output of 
            function fit_ms2. Each entry in the list is a time point (frame). 
            Each row in array is a fit (a single local maxima), columns are: 
            0: center z-coordinate, 1: center x-coordinate, 2: center 
            y-coordinate, 3: fit_height, 4: width_z, 5: width_x, 6: width_y). 
            Coordinates are adjusted so that if fit center lies outside the 
            image, center is moved to the edge.
        spot_data: dict of ndarrays
            Each key is a unique spot tracked across 1 or more frames. Each row
            of array is the spot's data for a single frame, with columns 0: frame
            number (t), 1: nucleus ID, 2: center Z-coordinate, 3: center X-coord-
            inate, 4: center Y-coordinate, 5: fit height, 6: fit z_width, 7: fit
            x_width, 8: fit y_width, 9: integrated volume for MS2, 10: integrated
            gaussian fit of MS2 spots, 11: integrated volume for protein signal.
        intvol: pandas df
            Intensity values for spots (over time) derived from the mean signal
            within ellipsoid volumes around detected spot centers.
        intfit: pandas df
            Intensity values for spots (over time) derived from integrating the 
            fitted 3D gaussian parameters.
        prot: pandas df
            Equivalent to intvol except integrations performed in the protein
            (nuclear) channel

    Methods:
        make_spot_table:
            Converts a column in spot_data to a pandas df with axes spot_id and 
            frame (time)
    """
    # Class attributes  
    # Initializer

    @staticmethod
    def make_spot_table(spot_data, colnum):
        """Make a spot_id x time_frame pandas df from a given column
        of spot_data."""
        # Initialize dataframe with 1000 frames, trim later.
        nframes_init = 1000
        max_frame = 0
        data = {}
        for spot in spot_data:
            arr = spot_data[spot]
            vals = np.empty(nframes_init)
            vals.fill(np.nan)
            for i in range(0, len(arr)):
                t = int(arr[i,0])
                if t > max_frame:
                    max_frame = t
                val = arr[i,colnum]
                vals[t] = val
            data[spot] = vals
        df = pd.DataFrame(data)
        # Trim off rows beyond max frame.
        df_trimmed = df.iloc[:(max_frame + 1),:]
        return df_trimmed
    
    def __init__(self, stack, nucmask, fits, spot_data):
        self.stack = stack
        self.nucmask = nucmask
        self.fits = fits
        self.spot_data = spot_data
        self.intvol = movie.make_spot_table(self.spot_data, 9)
        self.intfit = movie.make_spot_table(self.spot_data, 10)
        self.prot = movie.make_spot_table(self.spot_data, 11)
