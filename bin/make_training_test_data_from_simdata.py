#!/usr/bin/env python

"""make_training_test_data_from_simdata.py.
Takes an output from simulation (histogram and params) and prepares training and test data for learning parameters and image differences.

"""
__author__      = "Michael Stadler"
__copyright__   = "Copyright 2021, Planet Earth"

import scipy
import flymovie as fm
import numpy as np
from optparse import OptionParser
from time import time, process_time


def parse_options():
    parser = OptionParser()
    parser.add_option("-s", "--sim_file", dest="sim_file",
                      help="Pickled simulation datasets, comma-separated.", 
                      metavar="SIMS")
    parser.add_option("-o", "--outfilestem", dest="outfilestem",
                      help="File stems for saving pickled output.", 
                      metavar="OUTFILESTEM")
    parser.add_option("-n", "--num_sample", dest="num_sample",
                      help="Number of sims to sample.", 
                      metavar="NUMSAMPLE")
    parser.add_option("-m", "--num_diff", dest="num_diff",
                      help="Number of differences to compute for each sampled sim.", 
                      metavar="NUMDIFF")
    parser.add_option("-l", "--leaveout_fraction", dest="leaveout_fraction",
                      help="Fraction of entries to leave out.", 
                      metavar="LEAVEOUT")
    
    (options, args) = parser.parse_args()
    return options

def prepare_hist_param_data(data, param_pos, data_pos):
    """Process data output from blob simulator for feeding to ML function. 
    
    For histograms: 
        - Normalize each 1d histogram by dividing by sum (-> probabilities)
        - Flatten
        - Take log (after adding a bit to avoid 0)

    For parameters:
        - Combine into single 2d numpy array (rows are entries)
        - Normalize each column (mean 0 sd 1)
    """
    def check2d(a):
        """Add dummy first dimension if array is 1d."""
        if len(a.shape) < 2:
            a = np.expand_dims(a, axis=0)
        return a

    def norm_columns(x):
        """Normalize columns to mean=0 SD=1."""
        x1 = x.copy()
        for i in range(0, x.shape[1]):
            sd = np.std(x[:,i])
            if sd > 0:
                x1[:,i] = (x[:,i] - np.mean(x[:,i])) / sd
            else:
                x1[:,i] = np.zeros(len(x1[:,i]))
        return x1

    # Get some relevant values.
    total_hist_bins = check2d(data[0][data_pos]).size
    num_params = len(data[0][param_pos])
    num_entries = len(data)
    # Initialize data containers.
    histograms_processed = np.zeros((num_entries, total_hist_bins))
    params = np.zeros((num_entries, num_params))

    for i in range(num_entries):
        params[i] = data[i][param_pos]
        # Make a copy and ensure histogram is 2D.
        hist2d = data[i][data_pos].copy()
        hist2d = check2d(hist2d)
        # Normalize by dividing by row sum.
        hist2d = np.apply_along_axis(lambda x: x / np.sum(x), axis=1, arr=hist2d)
        # Flatten.
        hist_flattened = hist2d.flatten()
        # Add small number to account for 0s, take log, assign.
        zero_pad = np.min(hist_flattened[hist_flattened > 0]) / 100
        hist_flatlog = np.log(hist_flattened + zero_pad)
        histograms_processed[i] = hist_flatlog

    # Normalize parameters once they are all collected in array.
    params = norm_columns(params)    
    return histograms_processed, params

def create_training_test_data(data, n_leaveout, param_pos, data_pos, permute):
    """Randomly select entries for training and test data."""
    histograms, params = prepare_hist_param_data(data, param_pos, data_pos)
    # Generate a list of indexes (either randomly permuted or in order).
    if permute:
        p = np.random.permutation(histograms.shape[0])
    else:
        p = np.arange(0, histograms.shape[0])


    if n_leaveout > 0:       
        x_train = histograms[p[:-n_leaveout]]
        y_train = params[p[:-n_leaveout]]
        x_test = histograms[p[-n_leaveout:]]
        y_test = params[p[-n_leaveout:]]
        
        return x_train, y_train, x_test, y_test

    else:
        raise ValueError('Cannot have 0 leaveouts.')

def merge_data(x_train1, x_train2, y_train1, y_train2):
    """Horizontally merge training data based on common y entries."""
    def argsortcols(a, cols):
        cols_ext = []
        for col in cols:
            cols_ext.append(a[:,col])
        return np.lexsort(cols_ext)

    order1 = argsortcols(y_train1, np.arange(0, y_train1.shape[1]))
    order2 = argsortcols(y_train2, np.arange(0, y_train2.shape[1]))
    y_train1_new = y_train1[order1]
    y_train2_new = y_train2[order2]
    if np.max(abs(y_train1_new - y_train2_new)) > 1e-5:
        raise ValueError('Sorted y_train sets are not identical')
    x_train_new = np.hstack((x_train1[order1], x_train2[order2]))
    return x_train_new, y_train1_new

def orient_diffhist(x):
    """Orient difference histogram so that is has a positive sum."""
    if np.sum(x) < 0:
        return -1 * x
    return x

def make_similarity_data(x_in, y_in, num_sample, num_diff):
    """Use random sampling to generate training data using differences
    between entries."""
    total = num_sample * num_diff
    # Initialize containers for difference data.
    x = np.zeros((total, x_in.shape[1]))
    y = np.zeros((total, 1))
    # Sample indexes for the first time to get the entries for comparison.
    indexes = np.arange(0, x_in.shape[0])
    sampled_indexes1 = np.random.choice(indexes, size=num_sample, replace=False)
    for i in range(len(sampled_indexes1)):
        index1 = sampled_indexes1[i]
        # Sample indexes for entries to compare this entry to, excluding self.
        indexes_sansself = np.concatenate((indexes[:index1], indexes[(index1 + 1):]))
        sampled_indexes2 = np.random.choice(indexes_sansself, size=num_diff, replace=False)
        # Generate arrays with data matching second set of sampled indexes.
        x_sub = x_in[sampled_indexes2]
        y_sub = y_in[sampled_indexes2]
        # Get differences between reference entry and sampled comparison entries.
        x_diffs = orient_diffhist(x_sub - x_in[index1])
        y_diffs = np.sum((y_sub - y_in[index1]) ** 2, axis=1) # sum of squared differences
        # Assign difference scores to empty positions in container.
        start = i * num_diff
        end = start + num_diff
        x[start:end] = x_diffs
        y[start:end] = np.expand_dims(y_diffs, axis=1)
    return x, y

if __name__ == '__main__':
        
    t_start = time()
    options = parse_options()

    sim_file = options.sim_file
    num_sample = int(options.num_sample)
    num_diff = int(options.num_diff)
    num_sample_test = int(np.sqrt(num_sample))
    num_diff_test = int(np.sqrt(num_diff))
    leaveout_fraction = float(options.leaveout_fraction)

    data = fm.load_pickle(sim_file)
    num_leaveout = int(leaveout_fraction * len(data))
    x_train, y_train, x_test, y_test = create_training_test_data(data, param_pos=0, data_pos=1, permute=False, n_leaveout=num_leaveout)

    num_sample = int(options.num_sample)
    num_diff = int(options.num_diff)

    x_train_diff, y_train_diff = make_similarity_data(x_train, y_train, num_sample, num_diff)
    #x_test_diff, y_test_diff = make_similarity_data(x_test, y_test, num_sample_test, num_diff_test)

    hist_training_outfile = options.outfilestem + '_hist_training.pkl'
    hist_test_outfile = options.outfilestem + '_hist_test.pkl'
    diff_training_outfile = options.outfilestem + '_diff_training.pkl'
    diff_test_outfile = options.outfilestem + '_diff_test.pkl'

    fm.save_pickle((x_train, y_train), hist_training_outfile)
    fm.save_pickle((x_test, y_test), hist_test_outfile)
    fm.save_pickle((x_train_diff, y_train_diff), diff_training_outfile)
    #fm.save_pickle((x_test_diff, y_test_diff), diff_test_outfile)