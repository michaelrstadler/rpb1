# Import public packages.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage as ndi 
from importlib import reload
import pickle
import czifile
import imageio
from skimage.filters.thresholding import threshold_otsu

# Import my packages.
import sys
sys.path.append('/Users/michaelstadler/Bioinformatics/Projects/rpb1/bin')
import imagep as imp
reload(imp)

def df_filter_trajlen(df, minlen, renumber=False):
    """Filter pandas df columns for minimum number of non-nan entries."""
    new_df =  df.loc[:,df.apply(lambda x: np.count_nonzero(~np.isnan(x)), axis=0) > minlen]
    if (renumber):
        new_df.columns = np.arange(1, len(new_df.columns) + 1)
    return new_df

def bleach_corr(df, stack4d, sigma=10):
    """Perform bleach correction using the (smoothed) frame average of 
    the rpb1 channel, apply to columns of a pandas df."""
    frame_means = np.mean(stack4d, axis=(1,2,3))
    frame_means_smooth = ndi.gaussian_filter(frame_means, sigma=sigma)
    means_norm = frame_means_smooth / frame_means_smooth[0]
    return df.apply(lambda x: x / means_norm, axis=0)

def findrises_norm_all(df_in, windowsize, lag, minlen, min_, max_, sigma=1, rise=True, norm=True, display=False):
    """Find points in signal where a 'rise' occurs of magnitude between min_ and max_.
    Change is calculated using means within windows of specified size and with offset
    defined by lag. If rise is false, will look for falls. Norm normalizes each spot's
    data 0 to 1."""
    df = df_filter_trajlen(df_in, minlen)
    norm_lower, norm_upper = np.nanpercentile(df.to_numpy().flatten(), [5, 95])
    events = []
    for spot in df:
        data = pd.Series(ndi.gaussian_filter1d(df[spot], sigma))
        if (norm):
            data_norm = (data - norm_lower) / (norm_upper - norm_lower)
        else:
            data_norm = data
        rolling_avg = data_norm.rolling(windowsize, center=True).mean()
        diff = rolling_avg.diff(lag)
        
        if rise:
            indices = np.where((diff > min_) & (diff < max_))
        else:
            indices = np.where((diff < (-1 * min_)) & (diff > (-1 * max_)))
        
        for i in indices[0]:
            events.append([spot, i])
            
    if display:
        nframes = df.shape[1]
        ax = imp.qax(nframes+1)

        #for n in range(0, nframes-1):
        i = 0
        for n in df:
            ax[i].plot(df[n])
            for event in events:
                if (event[0] == n):
                    ax[i].axvline([event[1]], color="black", linestyle="--")
            i += 1
    print(len(events))
    return events

def df_deriv(df, windowsize, stepsize):
    return df.rolling(windowsize, center=True).mean().diff(stepsize)

def plot_traces(df1, df2, stack, minlen, sigma=0.8, norm=True):
    """Plot individual traces with smoothing, bleach correction, and 
    a minimum trajectory length filter."""
    def norm_trace(df, x, lower, upper):
        return (df.iloc[:,x] - lower) / (upper - lower)
    def df_filter_trajlen(df, df_to_count, minlen):
        """Filter pandas df columns for minimum number of non-nan entries."""
        return  df.loc[:,df_to_count.apply(lambda x: np.count_nonzero(~np.isnan(x)), axis=0) > minlen]

    df1_processed = df_filter_trajlen(bleach_corr(df1, stack), df1, minlen)
    df2_processed = df_filter_trajlen(bleach_corr(df2, stack), df1, minlen)

    df1_lower, df1_upper = np.nanpercentile(df1_processed.to_numpy().flatten(), [5, 95])
    df2_lower, df2_upper = np.nanpercentile(df2_processed.to_numpy().flatten(), [5, 95])
    num_to_plot=df1_processed.shape[1]
    def test(x):
        if norm:
            plt.plot(ndi.gaussian_filter1d(norm_trace(df1_processed, x, df1_lower, df1_upper), sigma))
            plt.plot(ndi.gaussian_filter1d(norm_trace(df2_processed, x, df2_lower, df2_upper), sigma))
        else:
            plt.plot(ndi.gaussian_filter1d(df1_processed.iloc[:,x], sigma))
            plt.plot(ndi.gaussian_filter1d(df2_processed.iloc[:,x], sigma))
        plt.title(df1_processed.columns[x])
    imp.plot_ps(test, range(0,num_to_plot))

from imagep import get_object_centroid
def df_find_peaks_thresh(df, thresh, sigma=1, display=False):
    def norm_trace(df, x, lower, upper):
        return (df[x] - lower) / (upper - lower)
    
    events = []
    lower, upper = np.nanpercentile(df.to_numpy().flatten(), [5, 95])
    df_norm = df / upper
    for spot in df_norm:
        data = pd.Series(ndi.gaussian_filter1d(df_norm[spot], sigma))
        
        peakmask = np.where(data >= thresh, 1, 0)
        labelmask, _ = ndi.label(peakmask)
        for peak in np.unique(labelmask)[1:]:
            max_ = np.nanmax(data[labelmask == peak])
            if not np.isnan(max_):
                max_loc = np.where((labelmask == peak) & (data == max_))[0][0]
                #centroid = get_object_centroid(labelmask, peak)
                events.append([spot, max_loc])
    if display:
        nframes = df_norm.shape[1]
        ax = imp.qax(nframes+1)

        #for n in range(0, nframes-1):
        i = 0
        for n in df:
            ax[i].plot(df_norm[n])
            for event in events:
                if (event[0] == n):
                    ax[i].axvline([event[1]], color="black", linestyle="--")
            i += 1
    print(len(events))
    return events


def make_padded_vectors(df1_in, df2_in, minlen=0, pad=100):
    """Take two matched (corresponding data from e.g. transcription and protein) data
    frames, concatenate the data from the dataframe columns with pads of zeros separating
    them, also create a matching control vector with 1s replacing the data vectors."""
    def df_filter_trajlen(df, df_to_count, minlen):
        """Filter pandas df columns for minimum number of non-nan entries."""
        return  df.loc[:,df_to_count.apply(lambda x: np.count_nonzero(~np.isnan(x)), axis=0) > minlen]
    df1 = df_filter_trajlen(df1_in, df1_in, minlen)
    df2 = df_filter_trajlen(df2_in, df1_in, minlen)
    v1 = np.zeros(pad)
    v2 = np.zeros(pad)
    control = np.zeros(pad)
    for n in df1:
        v1 = np.concatenate([v1, df1[n], np.zeros(pad)])
        v2 = np.concatenate([v2, df2[n], np.zeros(pad)])
        control = np.concatenate([control, np.ones(len(df1[n])), np.zeros(pad)])
    return (v1, v2, control)

#mv = eve_4_
def df_cross_corr_intvol_prot(mv, window=5, step=3, minlen=0, pad=30, plot=True, spread=20, bleach_correct=True):
    if bleach_correct:
        df1 = df_deriv(bleach_corr(mv.prot, mv.stack[0]), window, step)
        df2 = df_deriv(bleach_corr(mv.intvol, mv.stack[0]), window, step)
    else:
        df1 = df_deriv(mv.prot, window, step)
        df2 = df_deriv(mv.intvol, window, step)
    v1, v2, c = make_padded_vectors(df1, df2, minlen, pad)
    v1[np.isnan(v1)] = 0
    v2[np.isnan(v2)] = 0
    crosscorr_norm = np.correlate(v1, v2, 'same') / np.correlate(c, c, 'same')
    spread = 20
    crosscorr_norm = crosscorr_norm / np.max(crosscorr_norm)
    midpoint = int(len(crosscorr_norm) / 2)
    data = crosscorr_norm[(midpoint - spread):(midpoint + spread)]
    plt.plot(data, marker=".")
    plt.axvline(spread, alpha=0.3, color="orange")
    return crosscorr_norm












