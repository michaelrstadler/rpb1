#!/usr/bin/env python

"""
Functions for evaluating the quality of siamese CNN models for determining
the similarity of fluorescence microscopy images.
"""
__version__ = '1.0.0'
__author__ = 'Michael Stadler'
__copyright__   = "Copyright 2022, California, USA"

from .siamese_cnn import preprocess_image
import flymovie as fm
import cnn_models.siamese_cnn as sm
from fpdf import FPDF
import tempfile
import os
import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt


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
    # Ensure it's a good file, ignore hidden files.
    for f in files:
        if (f[3] != '_') or (f[0] == '.'):
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

#---------------------------------------------------------------------------
def plot_pca(embeddings1, embeddings2):
    """Plot a 2d and 3d PCA of embeddings of matched pair images.
    
    Embeddings represent matched pairs (same row in each represent
    embeddings of similar images). PCA is performed on all embeddings
    together, PCA is applied to each set, and positions of each sample
    are plotted in scatterplots using random color pallet such that matched
    pairs are the same color.

    2D scatterplot features PC1 vs. PC2; 3D scatter added PC3.

    Args:
        embeddings1 and embeddings2: ndarray
            Embedding locations for images, each row is an image
    
    Returns: None
    """
    # Combine and perform PCA dimensionality reduction.
    combined = np.vstack((embeddings1, embeddings2))
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(combined)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Plot separately for each set of embeddings.
    for e in (embeddings1, embeddings2):
        tfm = pca.transform(e)
        ax1.scatter(tfm[:,0], tfm[:,1], c = np.arange(tfm.shape[0]), 
                                    cmap='prism')
        ax2.scatter(tfm[:,0], tfm[:,1], tfm[:,2], 
                    c = np.arange(tfm.shape[0]), cmap='prism')
    
    ax1.set_title('PC1 vs. PC2')
    ax2.set_title('PC1 vs. PC2 vs. PC3')
    return fig

#---------------------------------------------------------------------------
def evaluate_model(folder, weights_file, outfolder, mip=False, nlayers=18, 
            imsize=(34,100,100)):
    """Evaluate performance of an embedding model against multiple image sets.
    
    Test datasets consist of paired images: Each image in the dataset has
    a similar image (e.g., rotation of itself or an independent simulation
    performed with identical parameters) which *should* be the closest image
    in the model. Performance on each dataset is assessed by where similar
    images rank in distance, performed by function 
    rank_embeddingdist_matchedpairs. A PDF is produced where each page is 
    a new test dataset. The dataset name, average pair rank, top1 and top5
    accuracy is printed, and a 2D and 3D PCA is displayed.
    
    Args:
        folder: string
            Path to folder containing image sets. Within folder, each folder
            is a dataset and contains left and right folders with paired
            images
        weights_file: string
            Path to files containing model weights
        outfolder: string
            Path to folder to which to write outfiles
        mip: bool
            Whether model is for maximum intensity projections
        nlayers: int (18 or 34)
            Number of layers in 3D CNN model
        imsize: iterable of ints
            Size of input images to model
    
    Returns:
        avg_ranks: list
            Mean rank for self-pairs in each test dataset
        good_dirs: list
            Names of test file directories

    Writes:
        evaluate_model + name + .pdf: PDF describing model performance
            on different test datasets. Contains paired ranking metrics
            and plots of 2D and 3D PCA.
    """
    # Load model from weights.
    if mip is False:
        base_cnn = sm.make_base_cnn_3d(image_shape=imsize, nlayers=nlayers)
    else:
        base_cnn = sm.make_base_cnn_2d(image_shape=imsize)
    embedding = sm.make_embedding(base_cnn)
    embedding.load_weights(weights_file)

    pdf = FPDF()
    dirs = os.listdir(folder)
    avg_ranks = []
    good_dirs = []
    # For each folder containing test data.
    for dir_ in dirs:
        # Ignore .DS_store (all hidden files)
        if dir_[0] == '.':
            continue
        
        good_dirs.append(dir_)
        # Calculate embeddings, determine ranks for pairs.
        embeddings1, _ = embed_images(os.path.join(folder, dir_, 'left'), embedding, mip)
        embeddings2, _ = embed_images(os.path.join(folder, dir_, 'right'), embedding, mip)
        ranks = rank_embeddingdist_matchedpairs(embeddings1, embeddings2)
        avg_ranks.append(np.mean(ranks))

        # Construct PDF.
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(120, 10, dir_, 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(120, 10, 'Mean rank: ' + str(np.mean(ranks)), 0, 1)
        top1 = np.count_nonzero(ranks == 0) / len(ranks)
        top5 = np.count_nonzero(ranks <= 4) / len(ranks)
        pdf.cell(120, 10, 'Top1 accuracy: ' + str(top1), 0, 1)
        pdf.cell(120, 10, 'Top5 accuracy: ' + str(top5), 0, 1)
        pdf.cell(120, 10, ','.join([str(x) for x in ranks]), 0, 1)
        
        # Create images by saving to temprary files, inserting in pdf.
        with tempfile.NamedTemporaryFile('w', suffix='.png') as f:
            fig = plot_pca(embeddings1, embeddings2)
            plt.savefig(f.name, dpi=150)
            pdf.image(f.name, w=200, h=100)
            plt.close(fig)
    
    outfile = os.path.join(outfolder, 'evaluate_model_' + weights_file.split('/')[-1] + '.pdf')
    pdf.output(outfile, 'F')
    return avg_ranks, good_dirs

#---------------------------------------------------------------------------
def evaluate_models(data_folder, out_folder, weights_folder, nlayers=18):
    """Evaluate multiple CNN models against multiple test datasets.
    
    This is mostly a wrapper for evaluate_model. Identifies each model 
    variable set in the weights_folder and calls evaluate_model. In addition
    to PDFs, save a tsv file with the average rank for each model against
    each dataset.

    ** Currently only for 3D CNNs (no MIP support)

    Args:
        data_folder: string
            Path to folder containing image sets. Within folder, each folder
            is a dataset and contains left and right folders with paired
            images
        out_folder: string
            Path to folder to which to write outfiles
        weights_folder: string
            Folder containing files containing variable for models
        nlayers: int
            Number of layers in CNN model
    
    Returns:
        stats: dict of lists
            Average rank data for each model (model name is key)
        data_dirs: list
            Names of test file directories

    Writes:
        ...evaluation_summary.tsv: average pair ranks for each model against
        each dataset
    """
    stats = {}
    data_dirs = []
    for f in os.listdir(weights_folder):
        if f[-5:] == 'index':
            weights_file = os.path.join(weights_folder, f[:-6])
            try:
                avg_ranks, data_dirs = evaluate_model(data_folder, weights_file, out_folder, nlayers=nlayers)
                stats[weights_file] = avg_ranks
            except:
                print('Failed: ' + weights_file)

    outfilepath = os.path.join(out_folder, weights_folder.split('/')[-1] + ' evaluation_summary.tsv')
    with open(outfilepath, 'w') as outfile:
        # Write colnames.
        outfile.write('\t'.join(data_dirs) + '\n')
        for wf in stats:
            outfile.write(wf.split('/')[-1] + '\t')
            outfile.write('\t'.join([str(x) for x in stats[wf]]) + '\n')

    return stats, data_dirs

#---------------------------------------------------------------------------
def visualize_simfolder(folder, n=32, **kwargs):
    """Load viewer for files in left and right folders in a simulated image
     output folder.
     
    Args:
        folder: str
            Folder containing left and right subfolders with pickled
                image files
        n: int
            Number of images from left and right to load
        kwargs:
            kwargs for viewer function
    """
    def stack_images(subfolder, n):
        files = os.listdir(subfolder)
        sample_file = os.path.join(subfolder, files[-1])
        stack = np.zeros(fm.load_pickle(sample_file).shape)
        stack = np.expand_dims(stack, axis=0)
        im_count = 0
        for f in files:
            if f[0] == '.':
                continue
            if im_count == n:
                break
            im = fm.load_pickle(os.path.join(subfolder, f))
            im = np.expand_dims(im, axis=0)
            stack = np.vstack([stack, im])
            im_count += 1
        return stack[1:]

    left = os.path.join(folder, 'left')
    right = os.path.join(folder, 'right')
    if os.path.exists(left) and os.path.exists(right):
        left_stack = stack_images(left, n)
        right_stack = stack_images(right, n)
    fm.viewer([left_stack, right_stack], **kwargs)

#---------------------------------------------------------------------------
def plot_history(pkl_file, ymax=0.25):
    """Plot history of loss and value loss from keras output.
    
    Args:
        pkl_file: str
            Path to pickled file containing history['history']
        ymax: number
            Maximum value for y axis    
    """
    history = fm.load_pickle(pkl_file)
    val_loss = []
    loss = []
    for i in range(len(history)):
        val_loss = val_loss + history[i]['val_loss']
        loss = loss + history[i]['loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylim((0, ymax))
    plt.legend(['loss', 'val_loss'])

#---------------------------------------------------------------------------
def visualize_batch(ds, figsize=4, **kwargs):
    """Visualize a batch of a triplet dataset.
    
    Args:
        ds: keras dataset
            Triplet image dataset
        figsize: int
            Figsize for viewer
        kwargs:
            kwargs for viewer function
    """
    def process_im(arr):
        im = np.squeeze(arr)
        return (im - np.min(im)) / (np.max(im) - np.min(im)) * 1000
    
    iter = ds.as_numpy_iterator()
    batch = next(iter)
    im1 = process_im(batch[0])
    im2 = process_im(batch[1])
    im3 = process_im(batch[2])

    fm.viewer([im1, im2, im3], figsize, **kwargs)