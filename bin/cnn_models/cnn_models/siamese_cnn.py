#!/usr/bin/env python

"""
A siamese CNN for determining similarity between 3D confocal microscope
images of fluorescent Drosophila nuclei. The model is heavily based on:

https://keras.io/examples/vision/siamese_network/

This model in turn uses resnet architectures as the based for the CNN.
This model can be built to take the maximum intensity projections (MIP)
of the 3D stacks, in which case it operates as a 2D CNN, or to work on 
the full 3D stack as a 3D CNN. The 2D architecture is based on 
resnet50, while the 3D architecture can be either resnet18 or resnet34.

Architectures based specifically on table 1 in 
Deep Residual Learning for Image Recognition
He, Zhang, Ren, Sun
https://arxiv.org/pdf/1512.03385.pdf

Some programming notes/challenges:
    - 

v1.0: 
v1.1: Implemented curriculum learning.

"""
__version__ = '1.1.0'
__author__ = 'Michael Stadler'

import numpy as np
import random
import pickle
import pandas as pd
import scipy.spatial
import scipy.ndimage as ndimage
import tensorflow as tf
import random
import sys
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet

#---------------------------------------------------------------------------
def identity_block_2d(input_tensor, kernel_size, filters, stage, block, 
		channels_axis=-1):
    """A block of layers that has no convolution at the shortcut.

	Architecture:
		- 1x1 convolution + batch norm + relu
		- (k,k) convolution + batch norm + relu
		- 1x1 convolution + batch norm
		- Residual step: add initial input to output of previous layer
		- relu
    
	Args:
        input_tensor: input tensor
        kernel_size: int or tuple of ints
			The kernel size of middle conv layer at main path
        filters: iterable of 3 ints 
			List of 3 integers, the number of filters for each of the 3 
			convolution layers
            **note: The last filter must equal be same as input (for adding 
			in resnet shortcut)
        stage: int
			The current stage label, used for generating layer names
        block: string
			'a','b'..., current block label, used for generating layer names
		channels_axis: int
			Axis containing channel--used for batch normalization. Default 
            is -1 for channels last format. For channels first, should be 1.

	Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

#---------------------------------------------------------------------------
def conv_block_2d(input_tensor, kernel_size, filters, stage, block,
    	strides=(2, 2), channels_axis=-1):
    """A block of layers that has convolution at the shortcut.

	Architecture:
		- 1x1 convolution with (s,s) stride length + batch norm + relu
		- (k,k) convolution + batch norm + relu
		- 1x1 convolution + batch norm
		- Residual step: 
			-(k,k) convolution of initial input wiht (s,s) stride length + batch norm
			- add conv of input to output of previous layer
		- relu
    
	Args:
        input_tensor: input tensor
        kernel_size: int or tuple of ints
			The kernel size of middle conv layer at main path
        filters: iterable of 3 ints 
			List of 3 integers, the number of filters for each of the 3 
			convolution layers
        stage: int
			The current stage label, used for generating layer names
        block: string
			'a','b'..., current block label, used for generating layer names
		strides: int or iterable of ints
			Strides for two non-1x1 conv layers in the block (middle and
			shortcut)
		channels_axis: int
			Axis containing channel. Default is -1 for channels last format.
            For channels first, should be 1.

	Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=channels_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

#---------------------------------------------------------------------------
def make_base_cnn_2d(image_shape=(100,100), name='base_cnn'):
    """Make a CNN network for a single image. Based on Resnet 50

    Note: channel last

    Args:
        image_shape: tuple of ints
            Shape of input images in pixels
        name: string
            Name for model

    Returns:
        Keras model
    """
    img_input = layers.Input(shape=image_shape + (1,)) # Channels last.
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7,7),
                        strides=(2, 2),
                        padding='valid',
                        kernel_initializer='he_normal',
                        name='conv1')(x)
    x = layers.BatchNormalization( name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)


    x = conv_block_2d(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block_2d(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block_2d(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block_2d(x, 3, [128, 128, 512], stage=3, block='a', strides=(2,2))
    x = identity_block_2d(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block_2d(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block_2d(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block_2d(x, 3, [256, 256, 1024], stage=4, block='a', strides=(2,2))
    x = identity_block_2d(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_2d(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_2d(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_2d(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_2d(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block_2d(x, 3, [512, 512, 2048], stage=5, block='a', strides=(1,1))
    x = identity_block_2d(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_2d(x, 3, [512, 512, 2048], stage=5, block='c')

    return Model(img_input, x, name=name)

#---------------------------------------------------------------------------
def identity_block_3d(input_tensor, kernel_size, filters, stage, block, 
		channels_axis=-1):
    """A block of layers that has no convolution at the shortcut.

	Architecture:
		- (k,k) convolution + batch norm + relu
		- (k,k) convolution + batch norm
		- Residual step: add initial input to output of previous layer
		- relu
    
	Args:
        input_tensor: input tensor
        kernel_size: int or tuple of ints
			The kernel size of middle conv layer at main path (default=3)
        filters: iterable of 2 ints 
			List of 2 integers, the number of filters for each of the 2 
			convolution layers
            **note: The last filter must equal be same as input (for adding 
			in resnet shortcut)
        stage: int
			The current stage label, used for generating layer names
        block: string
			'a','b'..., current block label, used for generating layer names
		channels_axis: int
			Axis containing channel--used for batch normalization. Default 
            is -1 for channels last format. For channels first, should be 1.

	Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv3D(filters1, kernel_size, 
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2b')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

#---------------------------------------------------------------------------
def conv_block_3d(input_tensor, kernel_size, filters, stage, block,
    	strides=(2, 2, 2), channels_axis=-1):
    """A block of layers that has convolution at the shortcut.

	Architecture:
		- (k,k) convolution with (s,s) stride length + batch norm + relu
		- (k,k) convolution + batch norm
		- Residual step: 
			-(k,k) convolution of initial input + batch norm
			- add conv of input to output of previous layer
		- relu
    
	Args:
        input_tensor: input tensor
        kernel_size: int or tuple of ints
			The kernel size of middle conv layer at main path
        filters: iterable of 2 ints 
			List of 2 integers, the number of filters for each of the 2 
			convolution layers
        stage: int
			The current stage label, used for generating layer names
        block: string
			'a','b'..., current block label, used for generating layer names
		strides: int or iterable of ints
			Strides for the two reductive conv layers in the block (first and
			shortcut)
		channels_axis: int
			Axis containing channel. Default is -1 for channels last format.
            For channels first, should be 1.

	Returns
        Output tensor for the block. 
    """
    filters1, filters2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv3D(filters1, kernel_size, strides=strides,
                      padding='same', kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv3D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=channels_axis, name=bn_name_base + '2b')(x)

    shortcut = layers.Conv3D(filters2, (1, 1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=channels_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

#---------------------------------------------------------------------------
def make_base_cnn_3d(image_shape=(20, 100,100), name='base_cnn', nlayers=18):
    """Make a CNN network for a single image.

    Args:
        image_shape: tuple of ints
            Shape of input images in pixels
        name: string
            Name for model
        nlayers: int
            Number of layers in the model: 18 or 34

    Returns:
        Keras model
    """
    if nlayers not in [8, 18, 34]:
        raise ValueError('nlayers must be 8, 18 or 34.')

    img_input = layers.Input(shape=image_shape + (1,)) # Channels last.
    x = layers.ZeroPadding3D(padding=(1, 3, 3), name='conv1_pad')(img_input)

    x = layers.Conv3D(64, (3,7,7),
                        strides=(2, 2, 2),
                        padding='valid',
                        kernel_initializer='he_normal',
                        name='conv1')(x)
    
    x = layers.BatchNormalization( name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding3D(padding=(1, 1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)

    if nlayers == 8:
        x = layers.ZeroPadding3D(padding=(1, 3, 3), name='conv2_pad')(x)

        x = layers.Conv3D(64, (3,7,7),
                        strides=(2, 2, 2),
                        padding='valid',
                        kernel_initializer='he_normal',
                        name='conv2')(x)

        x = conv_block_3d(x, (3,3,3), [64, 64], stage=2, block='a', strides=(2,2,2))
        x = identity_block_3d(x, (3,3,3), [64, 64], stage=2, block='b')

        x = conv_block_3d(x, (3,3,3), [128, 128], stage=3, block='a', strides=(2,1,1))
        x = identity_block_3d(x, (3,3,3), [128, 128], stage=3, block='b')

        x = conv_block_3d(x, (3,3,3), [128, 128], stage=4, block='a', strides=(1,1,1))
        x = identity_block_3d(x, (3,3,3), [128, 128], stage=4, block='b')

    if nlayers == 18:
    
        x = conv_block_3d(x, (3,3,3), [64, 64], stage=2, block='a', strides=(2,1,1))
        x = identity_block_3d(x, (3,3,3), [64, 64], stage=2, block='b')

        x = conv_block_3d(x, (3,3,3), [128, 128], stage=3, block='a', strides=(2,2,2))
        x = identity_block_3d(x, (3,3,3), [128, 128], stage=3, block='b')

        x = conv_block_3d(x, (3,3,3), [256, 256], stage=4, block='a', strides=(2,2,2))
        x = identity_block_3d(x, (3,3,3), [256, 256], stage=4, block='b')

        x = conv_block_3d(x, (3,3,3), [512, 512], stage=5, block='a', strides=(1,1,1))
        x = identity_block_3d(x, (3,3,3), [512, 512], stage=5, block='b')

    if nlayers == 34:
    
        x = conv_block_3d(x, (3,3,3), [64, 64], stage=2, block='a', strides=(2,1,1))
        x = identity_block_3d(x, (3,3,3), [64, 64], stage=2, block='b')
        x = identity_block_3d(x, (3,3,3), [64, 64], stage=2, block='c')

        x = conv_block_3d(x, (3,3,3), [128, 128], stage=3, block='a', strides=(2,2,2))
        x = identity_block_3d(x, (3,3,3), [128, 128], stage=3, block='b')
        x = identity_block_3d(x, (3,3,3), [128, 128], stage=3, block='c')
        x = identity_block_3d(x, (3,3,3), [128, 128], stage=3, block='d')

        x = conv_block_3d(x, (3,3,3), [256, 256], stage=4, block='a', strides=(2,2,2))
        x = identity_block_3d(x, (3,3,3), [256, 256], stage=4, block='b')
        x = identity_block_3d(x, (3,3,3), [256, 256], stage=4, block='c')
        x = identity_block_3d(x, (3,3,3), [256, 256], stage=4, block='d')
        x = identity_block_3d(x, (3,3,3), [256, 256], stage=4, block='e')
        x = identity_block_3d(x, (3,3,3), [256, 256], stage=4, block='f')

        x = conv_block_3d(x, (3,3,3), [512, 512], stage=5, block='a', strides=(1,1,1))
        x = identity_block_3d(x, (3,3,3), [512, 512], stage=5, block='b')
        x = identity_block_3d(x, (3,3,3), [512, 512], stage=5, block='c')

    return Model(img_input, x, name=name)
    
#---------------------------------------------------------------------------
def make_embedding(input_model, name='base_cnn'):
    """Make a CNN (from base_cnn) that creates an embedding for a 
      single image.

    Args:
        input_model: keras model (typicall base_cnn)
        name: string
            Name for model

    Returns:
        Keras model
    """
    flatten = layers.Flatten()(input_model.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    return Model(input_model.input, output, name="Embedding")

#---------------------------------------------------------------------------
class DistanceLayer(layers.Layer):
    """
    Compute the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

#---------------------------------------------------------------------------
def make_siamese_network(input_model, name='siamese_network'):
    """Make a siamese network consisting of three copies of input model
    fed to a distance layer.

    Args:
        input_model: keras model, model to use for each subnetwork
        name: string, name of network

    Returns:
        keras model
    """
    input_shape = input_model.input_shape

    anchor_input = layers.Input(name="anchor", shape=input_shape[1:])
    positive_input = layers.Input(name="positive", shape=input_shape[1:])
    negative_input = layers.Input(name="negative", shape=input_shape[1:])

    distances = DistanceLayer()(
        input_model(anchor_input),
        input_model(positive_input),
        input_model(negative_input),
    )

    return Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    

#---------------------------------------------------------------------------
class SiameseModel(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)

    Args:
        Model: keras model, siamese network

    Returns:
        keras model with custom training and testing loops
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    # Override inherited call method.
    def call(self, inputs):
        return self.siamese_network(inputs)
    
    # Override inherited train_step method.
    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # Override inherited test_step method.
    def test_step(self, data):
        loss = self._compute_loss(data)

        # Update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # New method.
    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

#---------------------------------------------------------------------------
def preprocess_image(input, mip=False):
        """
        Load the specified file as an ndarray, preprocess it and
        resize it to the target shape.

        ** Made this a standalone so it can be used by functions for 
        testing and playing with outputs.

        Args:
            filename: string tensor
                Path to pickled ndarray file
            mip: bool
                Whether or not to take/return maximum intensity 
                projection

        Return:
            im: ndarray
                Processed image stack
        """
        # This next part absolutely sucks but I cannot fucking figure out 
        # how else to get the string out.
        a = str(input)
        _, filename, _ = a.split("'")
        
        with open(filename, 'rb') as file:
            im = pickle.load(file)

        if mip:
            im = im.max(axis=0)

        im = im.astype('float32')
        # Dummy axis must be added in position 0.
        im = np.expand_dims(im, axis=-1)
        # Normalize by dividing by mean of unmasked (>0) values.
        min_ = np.min(im[im > 0])
        max_ = np.max(im)
        im = (im - min_) / (max_ - min_)
        return im

#---------------------------------------------------------------------------
def match_file_triplets(anchor_files, positive_files, num_negatives=5, 
    lower_margin=0, upper_margin=100):
    """Make file lists that match anchor-positive pairs with negative images
    within a margin of similarity.

    Image similarity is defined by using the euclidean distance between the 
    (normalized) simulation parameters. The margins are determined by
    randomly sampling image pairs to generate a distribution of parameter
    distances, then using percentiles from that distribution (percentiles
    defined by supplied upper and lower margins) to define the allowable
    distances for negative images.
    
    Images are selected by the following algorithm:
        1. Extract simulation parameters from all file names.
        2. Normalize each parameter (Z-scores)
        3. Determine distribution of distances by randomly sampling
            some images (up to 1000), calculating the distances
            between these sampled images and all other images, 
            and then taking percentiles of the resulting
            distribution.
        4. For each anchor-positive pair, find random negative images 
            that are within specified distance cutoffs (not sure this
            is most efficient but it works -- see get_good_idxs).
        5. Add triplet files for these random matches.

    Args:
        anchor_files: file path
            Iterable of files containing anchor images
        positive_files: file path
            Iterable of files containing positive images
        num_negatives: int
            The number of A-P-N triplets to make for each A-P pair
        upper_margin: number
            Percentile defining the upper limit of image similarity
            for drawing negative images 
        lower_margin: number
            Percentile defining the lower limit of image similarity
            for drawing negative images 
        
        Margin examples:
            lower = 0, upper = 50: negatives limited to most similar half
            lower = 50, upper = 100: negatives limited to least similar half

    Returns:
        a, p, n: lists
            Ordered lists of filepaths for anchor, positive, and negative
            images

    """
    def dist_row_to_all(x, mat):
        """Calculate euclidean distance between a 1d vector and 
        all rows of a matrix."""
        return np.sqrt(np.sum(((mat - x) ** 2), axis=1))

    def sample_distances(params, lower_margin, upper_margin):
        """Sample up to 1000 rows, calculate the distance to all other
        rows, collect the distances, return upper and lower cutoffs
        from percentiles."""
        distances = np.zeros(0)
        idxs = np.random.RandomState().choice(np.arange(params.shape[0]), size=np.min([params.shape[0], 1_000]), replace=False)
        for idx in idxs:
            distance = dist_row_to_all(params[idx, :], params)
            distances = np.concatenate([distances, distance])

        lower_cutoff = np.percentile(distances, lower_margin)
        upper_cutoff = np.percentile(distances, upper_margin)
        return lower_cutoff, upper_cutoff

    def get_params_from_filename(filepath):
        """Extract simulation parameters from filename as floats."""
        filename = filepath.split('/')[-1]
        p = filename.split('_')[1:-1]
        p = [float(x) for x in p]
        return p

    def get_params(files):
        """Extract parameters from a list of files, return as 
        numpy array."""
        num_params = len(get_params_from_filename(files[-1]))
        params = np.ndarray((0, num_params))

        # Load images and extract parameters.
        for f in files:
            p= get_params_from_filename(f)
            params = np.vstack([params, p])
        return params

    def get_good_idxs(ref_params, params, dist_cutoff_lower, 
                dist_cutoff_upper, chunk_size, target_num, rs):
        """Search matrix by chunks to find indexes of rows that are
        within cutoff distances of reference row."""
        shuffled_idxs = rs.choice(np.arange(params.shape[0]), params.shape[0],replace=False)
        good_idxs = np.zeros(0)
        for start in range(0, len(shuffled_idxs), chunk_size):
            chunk_idxs = shuffled_idxs[start:start + chunk_size]
            distances = dist_row_to_all(ref_params, params[chunk_idxs, :])
            good_idxs_chunk = chunk_idxs[np.where((distances != 0) & (distances >= dist_cutoff_lower) 
                    & (distances <= dist_cutoff_upper))[0]]
            good_idxs = np.concatenate([good_idxs, good_idxs_chunk])
            if len(good_idxs) >= target_num:
                break
        return [int(x) for x in good_idxs[:target_num]]

    negative_files = positive_files + anchor_files

    # Get parameters from anchor and negative files.
    anchor_params = get_params(anchor_files)
    negative_params = get_params(negative_files)

    # Determine the mean and std of the entire dataset.
    param_means = np.mean(negative_params, axis=0)
    param_stds = np.std(negative_params, axis=0)
    # In case any parameters are invariant, set std to 1 to avoid divide by zero.
    # Any non-zero value works, since the distances will all be 0, and when 
    # normalized will be (0 - 0) / 1 = 0, which is the desired behavior.
    param_stds[param_stds == 0] = 1

    # Normalize anchor and negative parameters.
    anchor_params = (anchor_params - param_means) / param_stds
    negative_params = (negative_params - param_means) / param_stds

    # Get the upper and lower cutoffs from the sampled distribution of distances.
    dist_cutoff_lower, dist_cutoff_upper = sample_distances(anchor_params, lower_margin, upper_margin)
    # Get chunk size for searching matrix.
    chunk_size = int(100 / (upper_margin - lower_margin) * num_negatives * 2)

    # Initialize lists to contain ordered image files.
    a, p, n = [], [], []

    # Go through each anchor-positive pair, find negative matches.
    rs = np.random.RandomState()
    filecount = 1
    for i in range(len(anchor_files)):
        # Print a helpful counter for monitoring progress.
        if filecount % 10_000 == 0:
            sys.stdout.write(str(filecount) + '\n')
            sys.stdout.flush()
        filecount += 1

        # Get random indexes of rows that are within distance cutoff.
        idxs = get_good_idxs(anchor_params[i, :], negative_params, dist_cutoff_lower, 
                dist_cutoff_upper, chunk_size, num_negatives, rs)

        # Add files of selected triplets.
        for idx in idxs:
            a.append(anchor_files[i])
            p.append(positive_files[i])
            n.append(negative_files[idx])

    # Shuffle lists using a shared order by zipping and unzipping.
    zipped = list(zip(a, p, n))
    random.shuffle(zipped)
    a, p, n = [list(x) for x in zip(*zipped)]
    return a, p, n

#---------------------------------------------------------------------------
def make_triplet_inputs(triplets_file, epoch_size, batch_size=32, rotate=False):
    """Create an input dataset of anchor-positive-negative triplets.

    Args:
        triplets_file: str
            CSV file containing file triplets (anchor, positive, negative)
        epoch_size: int
            Number of triplets to use in each training epoch (randomly sampled
            from triplets in file)
        batch_size: int
            Batch size
        rotate: bool
            If true, apply random transformation to each image: 50% flipped
            along axis 1, random rotation at 90-degree multiples
    
    Returns:
        train_dataset and val_dataset, two tf.data.Datasets ready to be loaded
            by models.
    """
    
    def preprocess_batch(input, mip=False):
        """Wrapper to apply preprocess_image to batch"""
        ims = []
        for ft in input:
            ims.append(preprocess_image(ft))
        return np.array(ims)

    def preprocess_triplets(anchor, positive, negative):
        """Given the filenames corresponding to the three images, load and
        preprocess them."""
        [anchor,] = tf.py_function(preprocess_batch,[anchor,],[tf.float32,])
        [positive,] = tf.py_function(preprocess_batch,[positive,],[tf.float32,])
        [negative,] = tf.py_function(preprocess_batch,[negative,],[tf.float32,])
        return (anchor, positive, negative)

    def rotate_batch(input, mip=False):
        """Apply random rotations/flips to batch of images. Rotations are multiple 
        of 90 degrees, 50% of images are flipped along axis 1."""
        rs = np.random.RandomState()
        ims = []
        for im in input:
            im_rot = ndimage.rotate(im, rs.choice([0,90,180,270]), axes=(1,2), 
                        reshape=False)
            if rs.choice([0,1]) == 1:
                im_rot = np.flip(im_rot, axis=1)
            ims.append(im_rot)
        return np.array(ims)

    def rotate_triplets(anchor, positive, negative):
        """Wrapper to apply random flips/rotations to each image in triplet."""
        [anchor,] = tf.py_function(rotate_batch,[anchor,],[tf.float32,])
        [positive,] = tf.py_function(rotate_batch,[positive,],[tf.float32,])
        [negative,] = tf.py_function(rotate_batch,[negative,],[tf.float32,])
        return (anchor, positive, negative)


    # Create datasets from these sorted files. These are in order and match in pairs.
    file_triplets = pd.read_csv(triplets_file, header=None)
    anchor_files = list(file_triplets.iloc[:,0])
    positive_files = list(file_triplets.iloc[:,1])
    negative_files = list(file_triplets.iloc[:,2])

    dataset_full_size = len(anchor_files)

    if epoch_size > dataset_full_size:
        raise ValueError('Epoch size must be <= full dataset size.')

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_files)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_files)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_files)
    
    # Combine datasets to form triplet images, apply shuffle to the whole
    # dataset. So upon iteration, order of triplets will be random, and
    # within triplets, the negative image will shuffle.
    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=dataset_full_size)
    dataset = dataset.take(epoch_size)

    # Batch (before mapping -- supposed to be faster).
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # Apply preprocessing and rotation via special mappable functions.
    dataset = dataset.map(preprocess_triplets, num_parallel_calls=tf.data.AUTOTUNE)

    if rotate:
        dataset = dataset.map(rotate_triplets, num_parallel_calls=tf.data.AUTOTUNE)

    # Divide into training and evaluation.
    train_size = np.max([round(epoch_size / batch_size  * 0.99), 1])
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset