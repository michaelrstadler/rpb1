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

#---------------------------------------------------------------------------
def rank_embeddingdist_matchedpairs(embeddings1, embeddings2):
    """Determine the ranking of matched pairs of images w.r.t. embedding 
    distance.
    
    Takes two sets of embeddings that represent matched pairs of images 
    (the same row in embeddings1 and embeddings2 correspond to paired images,
    e.g. two different simulations performed with the same parameters). For
    each image, the distance to every other image is calculated, the 
    distances are ranked, and the ranking of its matched pair is recorded. 
    For an ideal model, the ranking will always be 0. 
    
    
    """
    def get_ranks(embeddins1, embeddings2):
        ranks = []
        for n in range(embeddings1.shape[0]):
            emb_ref = embeddings1[n]
            # Stack all the embeddings EXCEPT self from first set onto second
            # set. The row number of the matched pair will be unchanged.
            embeddings_nonself = np.vstack((embeddings2, embeddings1[:n, :], 
                                embeddings1[(1+n):, :]))
            dists = np.sum((embeddings_nonself - emb_ref) ** 2, axis=1)
            # Rank indexes by distance, add the rank of the matched pair.
            idxs_sorted = np.argsort(dists)
            ranks.append(np.where(idxs_sorted == n)[0][0])
            
        return np.array(ranks)
    
    # Get the ranks wrt each image in embeddings1, then embeddings2, return mean 
    # rank for each pair.
    ranks1 = get_ranks(embeddings1, embeddings2)
    ranks2 = get_ranks(embeddings2, embeddings1)
    return (ranks1 + ranks2) / 2