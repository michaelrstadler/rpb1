#!/usr/bin/env python

"""train_model.py.


"""
__author__      = "Michael Stadler"
__copyright__   = "Copyright 2021, Planet Earth"

import flymovie as fm
import numpy as np
import tensorflow as tf
import os
from optparse import OptionParser
from importlib.machinery import SourceFileLoader
from time import time, process_time


def parse_options():
    parser = OptionParser()
    parser.add_option("-d", "--data", dest="data",
                      help="Pickled object containing training and test data", 
                      metavar="DATA")
    parser.add_option("-o", "--outfolder", dest="outfolder",
                      help="Folder to save model and output data", 
                      metavar="OUTFOLDER")
    parser.add_option("-m", "--model", dest="model",
                      help="File specififying model architecture", 
                      metavar="MODEL")

    
    (options, args) = parser.parse_args()
    return options    
t_start = time()
options = parse_options()
# Load data.
print('Loading data...')
x, y = fm.load_pickle(options.data)
print('Done.')
# Set up output folder.
outfolder = options.outfolder
if not os.path.isdir(outfolder):
    os.mkdir(outfolder)
    os.mkdir(os.path.join(outfolder, 'model'))

# Load model from file.
tempmodule = SourceFileLoader('',options.model).load_module()
model, history = tempmodule.train_model(x, y)

