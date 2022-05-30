#!/usr/bin/env python

"""make_file_triplets_fromreal.py: make file triplets for siamese cnn training from real images
or a mixture of real and simulated images.

To run on savio: 
"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2022, California, USA"

import argparse
import sys
import pickle
import os
import gzip
import numpy as np

def make_parser():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("-a", "--anchor_image_folder", type=str,  required=True,
                        help='Folder containing pickled image files.')
    parser.add_argument("-p", "--positive_image_folder", type=str,  default=None,
                        help='(optional) Folder containing pickled image files. Default: anchor image folder')
    parser.add_argument("-n", "--negative_image_folder", type=str, default=None,
                        help='(optional) Folder containing pickled image files. Default: anchor image folder')
    parser.add_argument("-t", "--num_triplets", type=int, required=True,
                        help='Number of triplets to make for each real image')
    parser.add_argument("-o", "--outfile", type=str, required=True,
                        help='File to write triplets to')

    return parser


def get_files(folder):
    files = {}
    for f in os.listdir(folder):
        if f[0] == '.':
            continue
        path = os.path.join(folder, f)
        sampleID = f.split('_')[0]
        stackID = f.split('_')[1]
        nucID = f.split('_')[2]

        if sampleID in files:
            if stackID in files[sampleID]:
                if nucID in files[sampleID]:
                    files[sampleID][stackID][nucID].append(path)
                else:
                    files[sampleID][stackID][nucID] = [path]
            else:
                files[sampleID][stackID] = {nucID: [path]}

        else:
            files[sampleID] = {
                stackID: {
                    nucID: [path]
                }
            }

    return files


def get_negatives(negative_files, sampleID, n, rs):
    count = 0
    selections = []
    for _ in range(n * 10):
        sampleID_choice = rs.choice(list(negative_files.keys()))
        if sampleID_choice != sampleID:
            stackID_choice = rs.choice(list(negative_files[sampleID_choice].keys()))
            nucID_choice = rs.choice(list(negative_files[sampleID_choice][stackID_choice].keys()))
            selection = rs.choice(negative_files[sampleID_choice][stackID_choice][nucID_choice])
            selections.append(selection)
    return selections

def get_positives(positive_files, sampleID, stackID, nucID, n, rs):
    count = 0
    selections = []
    for _ in range(n * 10):
        nucID_choice = rs.choice(list(positive_files[sampleID][stackID].keys()))
        if nucID_choice != nucID:
            selection = rs.choice(positive_files[sampleID][stackID][nucID_choice])
            selections.append(selection)
    return selections


def make_triplets(anchor_files, positive_files, negative_files, outfilepath, num_triplets):
    rs = np.random.RandomState()
    with gzip.open(outfilepath, 'wt') as outfile:
        for sampleID in anchor_files.keys():
            for stackID in anchor_files[sampleID].keys():
                for nucID in anchor_files[sampleID][stackID].keys():
                    anchor_selections = rs.choice(anchor_files[sampleID][stackID][nucID], num_triplets)
                    positive_selections = get_positives(positive_files, sampleID, stackID, nucID, num_triplets, rs)
                    negative_selections = get_negatives(negative_files, sampleID, num_triplets, rs)
                    for i in range(len(anchor_selections)):
                        try:
                            outfile.write(','.join([anchor_selections[i], positive_selections[i], negative_selections[i]]) + '\n')
                        except:
                            print(sampleID, stackID, nucID)




def main(argv):
    parser = make_parser()
    args = parser.parse_args(argv)

    anchor_image_folder = args.anchor_image_folder
    positive_image_folder = args.positive_image_folder
    negative_image_folder = args.negative_image_folder
    if positive_image_folder is None:
        positive_image_folder = anchor_image_folder
    if negative_image_folder is None:
        negative_image_folder = anchor_image_folder

    anchor_files = get_files(anchor_image_folder)
    positive_files = get_files(positive_image_folder)
    negative_files = get_files(negative_image_folder)

    make_triplets(anchor_files, positive_files, negative_files, args.outfile, args.num_triplets)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))