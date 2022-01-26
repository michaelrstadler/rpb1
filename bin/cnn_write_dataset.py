#!/usr/bin/env python
"""cnn_write_dataset.py: Write tensorflow dataset from a folder of
3D image files.

To run on savio: 
export PYTHONPATH="$PWD/rpb1/bin/cnn_models" # Put path to cnn_models here...I don't know how to manage this with savio module system yet (facepalm).
"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2022, Berkeley, California, USA"

import cnn_models.siamese_cnn_3d as cn
import cnn_models.siamese_cnn as cn2
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import os
from optparse import OptionParser
from time import time
import pickle



def parse_options():
    parser = OptionParser()
    parser.add_option("-f", "--input_data_folder", dest="input_data_folder",
                      help="Folder containing pickled 3d ndarrays in folders labeled left and right.", 
                      metavar="FOLDER")
    parser.add_option("-o", "--output_data_folder", dest="output_data_folder",
                      help="Folder for output datasets.", 
                      metavar="FOLDER")
    parser.add_option("-r", "--num_repeats", dest="num_repeats", default=1,
                      help="Number of repeats of the dataset to use. Repeats will have shuffled negative images and be rotated.", 
                      metavar="NUMREPEATS")
    parser.add_option("-m", action="store_true", dest="mip",
                      help="Flag: write maximum intensity projection (mip).")
    (options, args) = parser.parse_args()
    return options

def save_dataset(dataset, folder):
    """Save dataset as gzip and save pickled elementspec file in same directory."""
    os.mkdir(folder)
    tf.data.experimental.save(dataset, folder, compression='GZIP')
    elementspec_file = os.path.join(folder, 'elements_spec.pkl')
    with open(elementspec_file, 'wb') as file:
        pickle.dump(dataset.element_spec, file)

if __name__ == "__main__":
    options = parse_options()
    input_data_folder = options.input_data_folder
    num_repeats = int(options.num_repeats)
    mip = options.mip
    output_data_folder = options.output_data_folder
    if os.path.isdir(output_data_folder):
        raise ValueError(output_data_folder + ' already exists.')

    cache_dir = Path(input_data_folder)

    sys.stdout.write('Loading training data...\n')
    sys.stdout.flush()
    t1 = time()
    if (mip):
        train_dataset, val_dataset = cn2.make_triplet_inputs(cache_dir, num_repeats)
    else:
        train_dataset, val_dataset = cn.make_triplet_inputs(cache_dir, num_repeats, mip=mip)
    t2 = time()
    sys.stdout.write('finished.\n')
    sys.stdout.write('time: ' + str(t2 - t1) + '\n')
    sys.stdout.flush()
    
    os.mkdir(output_data_folder)
    train_dataset_folder = os.path.join(output_data_folder, 'training_dataset')
    val_dataset_folder = os.path.join(output_data_folder, 'val_dataset')

    save_dataset(train_dataset, train_dataset_folder)
    save_dataset(val_dataset, val_dataset_folder)