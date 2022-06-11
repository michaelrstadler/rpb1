#!/usr/bin/env python

"""decode_sampleID.py: Change stack ID of pickled image files according to supplied
key, write outputs to new file.

"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2022, California, USA"

import argparse
import sys
import shutil
import os
import warnings

def make_parser():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("-i", "--image_folder", type=str,  required=True,
                        help='Folder containing pickeled image files.')
    parser.add_argument("-k", "--keyfile_path", type=str, required=True,
                        help='Path to file with stackID key (tsv)')
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help='Folder to copy renamed images to')
    return parser

def read_keyfile(keyfilepath):
    """Read stackIDs and replacements into dictionary."""
    stack_key = {}
    with open(keyfilepath, 'r') as keyfile:
        for line in keyfile:
            items = line.strip().split('\t')
            if len(items) != 2:
                warnings.warn('There should be two items per line of sample key file.')
            else:
                stack_key[items[0]] = items[1]
    
    if len(stack_key.keys()) == 0:
        raise ValueError('Stack key is empty.')
    
    return stack_key

def decode_files(image_folder, output_path, stack_key):
    """Rename files, copy to new destination"""
    image_files = os.listdir(image_folder)
    nuc_nums = {}
    for image_file in image_files:
        if image_file[0] == '.':
            continue
        items = image_file.split('_')
        if len(items) < 3:
            warnings.warn(image_file + ' does not contain enough _-delimited items.')
            continue
        sampleID, stackID, nucID, remainder = items[0], items[1], items[2], '_'.join(items[3:])
        if stackID in stack_key:
            new_stackID = stack_key[stackID]
            # Get new non-overlapping nucID.
            if new_stackID in nuc_nums:
                new_nucID = nuc_nums[new_stackID]
                nuc_nums[new_stackID] += 1
            else:
                new_nucID = 0
                nuc_nums[new_stackID] = 1
            new_nucID = str(new_nucID)

            # Copy file with new name.
            new_filename = '_'.join((sampleID, new_stackID, new_nucID, remainder))
            shutil.copy(
                os.path.join(image_folder, image_file),
                os.path.join(output_path, new_filename)
            )

def main(argv):
    parser = make_parser()
    args = parser.parse_args(argv)
    stack_key = read_keyfile(args.keyfile_path)
    decode_files(args.image_folder, args.output_path, stack_key)
    
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))