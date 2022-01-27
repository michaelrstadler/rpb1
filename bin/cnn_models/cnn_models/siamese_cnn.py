import numpy as np
import os
import random
import pickle
import scipy.ndimage as ndimage
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet

#from flymovie.load_save import load_pickle

############################################################################
def identity_block(input_tensor, kernel_size, filters, stage, block, 
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
			The kernel size of middle conv layer at main path (default=3)
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

############################################################################
def conv_block(input_tensor, kernel_size, filters, stage, block,
    	strides=(2, 2), channels_axis=-1):
    """A block of layers that has convolution at the shortcut.

	Architecture:
		- 1x1 convolution + batch norm + relu
		- (k,k) convolution + batch norm + relu
		- 1x1 convolution + batch norm
		- Residual step: 
			-(k,k) convolution of initial input + batch norm
			- add conv of input to output of previous layer
		- relu
    
	Args:
        input_tensor: input tensor
        kernel_size: int or tuple of ints
			The kernel size of middle conv layer at main path (default=3)
        filters: iterable of 3 ints 
			List of 3 integers, the number of filters for each of the 3 
			convolution layers
            **note: The last filter must equal be same as input (for adding 
			in resnet shortcut)
        stage: int
			The current stage label, used for generating layer names
        block: string
			'a','b'..., current block label, used for generating layer names
		strides: int or iterable of ints
			Strides for the two non-1x1 conv layers in the block (middle and
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

############################################################################
def make_base_cnn(image_shape=(100,100), name='base_cnn'):
    """Make a CNN network for a single image.

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


    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2,2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [512, 512, 2048], stage=4, block='a', strides=(2,2))
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='c')

    return Model(img_input, x, name=name)

############################################################################
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

############################################################################
class DistanceLayer(layers.Layer):
    """
    This layer computes the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

############################################################################
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
    

############################################################################
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

############################################################################
def make_triplet_inputs(folder, repeats=1):

    def preprocess_image(filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """
        a = str(filename)
        _,filename,_ = a.split("'")
        with open(filename, 'rb') as file:
            im = pickle.load(file)

        im = im.max(axis=0)
        im = im.astype('float32')
        im = np.expand_dims(im, axis=-1)
        # Normalize 0-1.
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        return im

    def preprocess_triplets(anchor, positive, negative):
        """
        Given the filenames corresponding to the three images, load and
        preprocess them.
        """
        return (
            preprocess_image(anchor),
            preprocess_image(positive),
            preprocess_image(negative)
        )

    #cache_dir = Path('/Users/michaelstadler/Bioinformatics/Projects/rpb1/data/simulations/keras_test')
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

    image_count = len(anchor_images)

    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

    # To generate the list of negative images, let's randomize the list of
    # available images and concatenate them together.
    rng = np.random.RandomState(seed=42)
    rng.shuffle(anchor_images)
    rng.shuffle(positive_images)

    negative_images = anchor_images + positive_images
    np.random.RandomState(seed=32).shuffle(negative_images)

    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
    negative_dataset = negative_dataset.shuffle(buffer_size=4096)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)
    def preprocess_triplets_py_function(a, p, n):
        [anchor, positive, negative] = tf.py_function(preprocess_triplets, [a,p,n], [tf.float32,tf.float32,tf.float32])
        return (anchor, positive, negative)

    dataset = dataset.map(preprocess_triplets_py_function)

    # Let's now split our dataset in train and validation.
    train_dataset = dataset.take(round(image_count * 0.8))
    val_dataset = dataset.skip(round(image_count * 0.8))

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    train_dataset = train_dataset.prefetch(8)

    val_dataset = val_dataset.batch(32, drop_remainder=False)
    val_dataset = val_dataset.prefetch(8)

    return train_dataset, val_dataset