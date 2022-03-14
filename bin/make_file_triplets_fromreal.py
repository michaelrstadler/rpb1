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
import numpy as np

def make_parser():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("-r", "--real_image_folder", type=str,  required=True,
                        help='Folder containing pickled image files.')
    parser.add_argument("-t", "--num_triplets", type=int, required=True,
                        help='Number of triplets to make for each real image')
    parser.add_argument("-s", "--sim_folder", default=None,
                        help='optional: folder containing matched simulations')
    parser.add_argument("-n", "--negative_folder", default=None,
                        help='optional: folder containing images for negative slot')

    return parser

def get_filepaths(folder, sort=False):
    if sort is False:
        return [os.path.join(folder, x) for x in os.listdir(folder) if x[0] != '.']
    else:
        files = {}
        for f in os.listdir(folder):
            if f[0] == '.':
                continue
            path = os.path.join(folder, f)
            tag = f.split('_')[0]
            if tag in files:
                files[tag].append(path)
            else:
                files[tag] = [path]
        return files

def get_negative_image(real_images, anchor_tag, negative_images, rs):
    if negative_images is not None:
        return rs.choice(negative_images)
    
    tags = rs.choice(list(real_images.keys()), 2, replace=False)
    for tag in tags:
        if tag != anchor_tag:
            return rs.choice(real_images[tag])

def get_positive_image(real_images, real_image, tag, positive_images, rs):
    if positive_images is not None:
        return rs.choice(positive_images[tag])
    else:
        ims = rs.choice(real_images[tag], 2, replace=False)
        for im in ims:
            if im != real_image:
                return im

def make_triplets_withsims(real_image_folder, num_triplets, sim_folder, negative_folder):
    real_images = get_filepaths(real_image_folder, sort=True)
    #sims = get_filepaths(sim_folder, sort=True)
    if negative_folder is not None:
        negative_images = get_filepaths(negative_folder, sort=False)
    else:
        negative_images = None

    if sim_folder is not None:
        positive_images = get_filepaths(sim_folder, sort=True)
    else:
        positive_images = None

    rs = np.random.RandomState()

    triplets = []
    for tag in real_images:
        for real_image in real_images[tag]:
            for n in range(num_triplets):
                positive_image = get_positive_image(real_images, real_image, tag, positive_images, rs)
                negative_image = get_negative_image(real_images, tag, negative_images, rs)
                if rs.choice([0,1]) == 0:
                    triplets.append((real_image, positive_image, negative_image))
                else:
                    triplets.append((positive_image, real_image, negative_image))

    return triplets


def main(argv):
    parser = make_parser()
    args = parser.parse_args(argv)

    real_image_folder = args.real_image_folder
    num_triplets = args.num_triplets
    sim_folder = args.sim_folder
    negative_folder = args.negative_folder

    
    triplets = make_triplets_withsims(real_image_folder, num_triplets, sim_folder, negative_folder)
    for i in triplets:
        print(i)
    

    
    

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))