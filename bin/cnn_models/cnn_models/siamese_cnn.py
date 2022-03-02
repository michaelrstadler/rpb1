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
import os
import random
import pickle
import scipy.spatial
import scipy.ndimage as ndimage
import tensorflow as tf
import random
from pathlib import Path
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
        # Normalize 0-1.
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
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
        3. Randomly sample image pairs, calculate the euclidean distance
            between parameters for each pair, then get distance cutoffs 
            based on percentiles from this distribution.
        4. For each anchor-positive pair, randomly shuffle negative images.
        5. Search in order through shuffled negatives until a negative image
            is found whose normalized (by mean and SD found in 3) distance
            is within the bounds defined by margin parameters, 
        6. Repeat 5 until num_negatives is reached
    
    Programming note: the slowest part of this (normally) is the sampling in 
    get_param_stats, which is fine, but I think fewer pairs can be sampled
    and still get good stats if the timing is a real problem.

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
    def get_params_from_filename(filepath):
        """Extract simulation parameters from filename as floats."""
        filename = filepath.split('/')[-1]
        p = filename.split('_')[1:-1]
        p = [float(x) for x in p]
        return p

    def get_norm_params(filename, means, stds):
        """Z-score normalize parameters."""
        p= get_params_from_filename(filename)
        return (p - means) / stds

    def get_param_stats(files):
        """Get the mean and std for simulation parameters across the dataset,
        and get the mean and std for the euclidean distances between parameters 
        of sampled image pairs."""

        num_params = len(get_params_from_filename(files[-1]))
        params = np.ndarray((0, num_params))

        # Load images and extract parameters.
        for f in files:
            p= get_params_from_filename(f)
            params = np.vstack([params, p])

        # Calculate the mean and std for parameters.
        param_stds = params.std(axis=0)
        param_means = params.mean(axis=0)

        # Sample image pairs, get mean and std for distances.
        distances = []
        for _ in range(2_500):
            rs = np.random.RandomState()
            params1 = get_norm_params(rs.choice(files), param_means, param_stds)
            params2 = get_norm_params(rs.choice(files), param_means, param_stds)
            dist = scipy.spatial.distance.euclidean(params1, params2)
            distances.append(dist)

        return param_means, param_stds, distances
    

    negative_files = positive_files + anchor_files
    param_means, param_stds, dists_sampled = get_param_stats(anchor_files)
    dist_cutoff_upper = np.percentile(dists_sampled, upper_margin)
    dist_cutoff_lower = np.percentile(dists_sampled, lower_margin)

    # In case any parameters are invariant, set std to 1 to avoid divide by zero.
    # Any non-zero value works, since the distances will all be 0, and when 
    # normalized will be (0 - 0) / 1 = 0, which is the desired behavior.
    param_stds[param_stds == 0] = 1
    
    # Initialize lists to contain ordered image files.
    a, p, n = [], [], []

    # Go through each anchor-positive pair, find negative matches.
    rs = np.random.RandomState()
    filecount = 0
    for i in range(len(anchor_files)):
        # Print a helpful counter for monitoring progress.
        if filecount % 5_000 == 0:
            print(filecount)
        filecount += 1

        anchor_params = get_norm_params(anchor_files[i], param_means, param_stds)
        matches_count = 0
        used_indexes = []
        # Try randomly drawn negative images to find images that fall within
        # distance cutoffs, check to make sure negatives aren't repeated and 
        # that the anchor and positive image are excluded.
        for _ in range(int(len(negative_files) * 2)): # Avoiding while loop
            # If enough matches have been found, exit for loop.
            if matches_count == num_negatives:
                break

            # Get random index for negative files.
            idx = rs.randint(len(negative_files))
            
            # Skip rest of loop if this is the positive image.
            if idx == i: 
                continue

            f = negative_files[idx]
            f_params = get_norm_params(f, param_means, param_stds)
            dist = scipy.spatial.distance.euclidean(anchor_params, f_params)
            # If this is the anchor image, skip rest of loop.
            if dist == 0:
                continue
            
            # Add triplet if negative image is within distance cutoffs and 
            # hasn't been used.
            if (dist >= dist_cutoff_lower) and (dist <= dist_cutoff_upper):
                if idx not in used_indexes:
                    a.append(anchor_files[i])
                    p.append(positive_files[i])
                    n.append(f)
                    used_indexes.append(idx)
                    matches_count += 1

    # Shuffle lists using a shared order by zipping and unzipping.
    zipped = list(zip(a, p, n))
    random.shuffle(zipped)
    a, p, n = [list(x) for x in zip(*zipped)]
    return a, p, n

#---------------------------------------------------------------------------
def make_triplet_inputs(folder, lower_margin=0, upper_margin=100, 
    num_negatives=5, n_repeats=1, batch_size=32, rotate=False):
    """Create an input dataset of anchor-positive-negative triplets.

    Args:
        folder: Path object
            Folder containing left and right subfolders with matching images
        upper_margin: number
            Percentile defining the upper limit of image similarity
            for drawing negative images 
        lower_margin: number
            Percentile defining the lower limit of image similarity
            for drawing negative images
        num_negatives: int
            Number of negative images to match with each A-P pair. Defines
            number of possible triplets per A-P pair.
        n_repeats: int
            Number of times dataset is repeated in each epoch, or the mean
            number of triplets for each A-P pair that will be seen in each 
            epoch.
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


    # Set up input directories.
    cache_dir=folder
    anchor_images_path = cache_dir / "left"
    positive_images_path = cache_dir / "right"

    # We need to make sure both the anchor and positive images are loaded in
    # sorted order so we can match them together.
    anchor_images = sorted(
        [str(anchor_images_path / f) for f in os.listdir(anchor_images_path)]
    )

    positive_images = sorted(
        [str(positive_images_path / f) for f in os.listdir(positive_images_path)]
    )

    dataset_full_size = len(anchor_images) * num_negatives
    dataset_take_size = len(anchor_images) * n_repeats
    
    # Create datasets from these sorted files. These are in order and match in pairs.
    print('Matching triplets...')
    anchor_files, positive_files, negative_files = match_file_triplets(anchor_images, positive_images, num_negatives, lower_margin=lower_margin, upper_margin=upper_margin)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_files)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_files)
    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_files)
    
    # Combine datasets to form triplet images, apply shuffle to the whole
    # dataset. So upon iteration, order of triplets will be random, and
    # within triplets, the negative image will shuffle.
    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=dataset_full_size)
    dataset = dataset.take(dataset_take_size)

    # Batch (before mapping -- supposed to be faster).
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # Apply preprocessing and rotation via special mappable functions.
    dataset = dataset.map(preprocess_triplets, num_parallel_calls=tf.data.AUTOTUNE)

    if rotate:
        dataset = dataset.map(rotate_triplets, num_parallel_calls=tf.data.AUTOTUNE)

    # Divide into training and evaluation, batch and prefetch.
    train_dataset = dataset.take(round(dataset_take_size / batch_size  * 0.9))
    val_dataset = dataset.skip(round(dataset_take_size / batch_size  * 0.9))

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset