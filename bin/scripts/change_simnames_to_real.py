#!/usr/bin/env python

"""
"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2022, California, USA"

import argparse
import sys
import os
import numpy as np
import shutil

def make_parser():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("-s", "--source_folder", type=str,  required=True,
                        help='Folder containing pickled image files.')
    parser.add_argument("-t", "--target_folder", type=str,  required=True,
                        help='Folder containing images with names to match to.')
    parser.add_argument("-o", "--output_folder", type=str,  required=True,
                        help='Folder to write images to.')
    return parser

def get_files(folder):
    files = []
    for f in os.listdir(folder):
        if f[0] != '.':
            files.append(f)
    return files

def main(argv):
    parser = make_parser()
    args = parser.parse_args(argv)
    source_folder = args.source_folder
    target_folder = args.target_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    target_files = get_files(target_folder)
    source_files = get_files(source_folder)

    for source_file in source_files:
        target_file = np.random.choice(target_files)
        sampleID, stackID, nucID, sliceID = target_file.split('_')
        nucID = str(int(nucID) + 1)
        newfile = '_'.join((sampleID, stackID, nucID, sliceID)) + '.pkl'
        newfilepath = os.path.join(output_folder, newfile)
        shutil.copyfile(os.path.join(source_folder, source_file), newfilepath)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))