#!/usr/bin/env python

""""""

__author__      = "Michael Stadler"

import cnn_models.siamese_cnn as cn
import pickle
import os
import string
from pathlib import Path
import argparse
import random
import gzip
from time import time

def parse_args():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("-f", "--training_data_folder", type=str,  required=True,
                help="Folder containing training data in folders labeled left and right.")
    parser.add_argument("-l", "--lower_margin", type=float,  required=True,
                help="Lower margin for curriculum learning")
    parser.add_argument("-u", "--upper_margin", type=float,  required=True,
                help="Upper margin for curriculum learning")
    parser.add_argument("-n", "--num_triplets", type=int,  required=True,
                help="Number of triplets to make from each A-P pair")
    
    args = parser.parse_args()
    return args

t1 = time()

args = parse_args()

# Set up input directories.
data_dir = Path(args.training_data_folder)
anchor_images_path = data_dir / "left"
positive_images_path = data_dir / "right"

# Get rid of hidden files (if any).
anchor_files = [x for x in os.listdir(anchor_images_path) if x[0] != '.']
positive_files = [x for x in os.listdir(positive_images_path) if x[0] != '.']

# Sort files, add folder for full path.
anchor_files = sorted([str(anchor_images_path / f) for f in anchor_files])

positive_files = sorted([str(positive_images_path / f) for f in positive_files])

# Generate random 3-letter id.
file_id = ''.join(random.choice(string.ascii_letters) for i in range(3))

# Generate file triplets.
a, p, n = cn.match_file_triplets(anchor_files, positive_files, num_negatives=args.num_triplets, 
lower_margin=args.lower_margin, upper_margin=args.upper_margin)

# Save.
outfilepath = os.path.join(data_dir, 'filetriplets_' + str(args.lower_margin) + 
        '_' + str(args.upper_margin) + '_' + str(args.num_triplets) + '_' + file_id + '.csv.gz')

with gzip.open(outfilepath, 'wt') as outfile:
    for i in range(len(a)):
        outfile.write(','.join([a[i], p[i], n[i]]) + '\n')

t2 = time()
print(t2 - t1)