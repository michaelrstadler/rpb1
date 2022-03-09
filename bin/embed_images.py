#!/usr/bin/env python

"""embed_images.py: Given an embedding CNN model, embed images from a folder.

To run on savio: 
export PYTHONPATH="$PWD/rpb1/bin/cnn_models";module unload python/3.7;module load ml/tensorflow/2.5.0-py37
"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2022, California, USA"

import argparse
import sys
import pickle
import cnn_models.siamese_cnn as cn
from cnn_models.evaluate_models import embed_images

def make_parser():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("-i", "--image_folder", type=str,  required=True,
                        help='Folder containing pickeld image files.')
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help='Path to variables for model')
    parser.add_argument("-s", "--image_shape", nargs='+', type=int, required=True,
                        help='Shape (z, y, x) of input images')
    parser.add_argument("-o", "--outfile", type=str, required=True,
                        help='Path for output file')
    parser.add_argument("-l", "--num_layers", type=int, default=8,
                        help='number of layers in CNN model')

    return parser


def main(argv):
    parser = make_parser()
    args = parser.parse_args(argv)
    target_shape = tuple(args.image_shape)
    
    # Make model.
    base_cnn = cn.make_base_cnn_3d(image_shape=target_shape, nlayers=8)
    embedding = cn.make_embedding(base_cnn)
    embedding.load_weights(args.model_path)

    # Get embeddings.
    embeddings = embed_images(args.image_folder, embedding)

    # Save.
    with open(args.outfile, 'wb') as outfile:
        pickle.dump(embeddings, outfile, protocol=4)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))