#!/usr/bin/env python

"""
Functions for evaluating the quality of siamese CNN models for determining
the similarity of fluorescence microscopy images.
"""
__version__ = '1.0.0'
__author__ = 'Michael Stadler'
__copyright__   = "Copyright 2022, California, USA"

from .siamese_cnn import preprocess_image
import os
import numpy as np

# Idea: compare replicate folders. Run through net, see how often images are closest to their twin.

#---------------------------------------------------------------------------
def embed_images(im_folder, embedding, mip=False):
    """Pass images from a folder through embedding model, return their 
    location and normalized simulation parameters.
    
    Args:
        im_folder: string
            Folder containing pickled ndarray image stacks
        embedding: keras model
            model for image embedding
        mip: bool
            Use maximum intensity projections
        
    Returns:
        im_embeddings: ndarray
            Embedding location of images (each row is an image)
        params: ndarray
            Simulation parameters of images, taken from file name,
            each normalized mean=0 std=1.
    """
    
    def normalize_params(params):
        """Normalize param array as Z-scores."""
        std = np.std(params, axis=0)
        # If the parameter is fixed it will have std=0. Since the difference
        # will be 0 and the desired value is all 0, any non-zero value for 
        # std works to avoid divide by 0 error.
        std[std == 0] = 1e6
        p = params.copy()
        p = (p - p.mean(axis=0)) / std
        return p
        
    ims = []
    files = sorted(os.listdir(im_folder))
    num_params = len(files[-1].split('_')) - 2
    params = np.ndarray((0, num_params))

    # Load images and extract parameters.
    for f in files:
        if f[3] != '_':
            continue
        # Because of the silliness with extracting filename from tensor, 
        # have to add two single quotes flanking filename.
        filename = "_'" + os.path.join(im_folder, f) + "'_"
        im = preprocess_image(filename, mip)
        ims.append(im)
        p= f.split('_')[1:-1]
        p = [float(x) for x in p]
        params = np.vstack([params, p])
    
    params = normalize_params(params)

    # Calculate embedding for each image.
    im_embeddings = np.ndarray((0,256))
    for i in range(len(ims)):
        im = ims[i]
        im = np.expand_dims(im, axis=0)
        e = embedding(im).numpy()
        im_embeddings = np.vstack([im_embeddings, e])
        
    return im_embeddings, params