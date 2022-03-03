import cnn_models.siamese_cnn as cn
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
import sys
import os
import argparse
from configparser import ConfigParser
import shlex
from cnn_models.siamese_cnn import preprocess_image
from tensorflow.keras import layers
from tensorflow.keras import Model

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-f", "--image_folder", dest='image_folder', required=True,
            help="Folder containing left and right folders with pickled 3d stacks.")
    parser.add_argument("-e", "--epochs", dest="epochs", required=True, type=int,
            help="Number of epochs to train for")
    parser.add_argument("-n", "--model_name", dest="model_name", default='model',
            help="Optional: model name")
    parser.add_argument("-l", "--nlayers", dest="nlayers", default=8, type=int,
            help="Optional: Number of conv layers in cnn (8, 18, or 34, default: 8)")
    parser.add_argument("-i", "--initial_lr", dest="initial_lr", default=0.0001, type=float,
            help="Optional: initial learning rate (default: 0.0001)")
    parser.add_argument("-y", "--decay_rate", dest="decay_rate", default=0.95, type=float,
            help="Optional: decay rate for learning rate (default: 0.95)")
    parser.add_argument("-s", "--decay_steps", dest="decay_steps", default=100_000, type=int,
            help="Optional: decay steps for learning rate scheduler (default: 1e5)")
    parser.add_argument("-d", action="store_true", dest="distributed",
            help="Optional: use distributed (multi-GPU) mode")
    
    args = parser.parse_args()
    return args

def get_target_shape_nparams(dir_):
    """Retrieve the image target size."""
    left = dir_ / 'left'
    imfile = os.listdir(left)[-1]
    impath = os.path.join(left, imfile)
    with open(impath, 'rb') as file:
        im = pickle.load(file)

    shape = im.shape
    nparams = len(imfile.split('_')) - 2
    return shape, nparams

def make_datasets(folder, batch_size=32):
    """Make training and validation datasets"""
    def get_files_params(subfolder):
        """Get filenames, extract parameters."""
        files = os.listdir(subfolder)
        files = [x if x[0] != '.' else None for x in files]
        n_params = len(files[0].split('_')) - 2
        params = np.zeros((0, n_params))
        for f in files:
            p = f.split('_')[1:-1]
            p = [float(x) for x in p]
            p = np.expand_dims(np.array(p), axis=0)
            params = np.vstack((params, p))
        files = [os.path.join(subfolder, x) for x in files]
        return files, list(params)

    def preprocess(file, paramset):
        """Wrapper for image preprocessing."""
        [im,] = tf.py_function(preprocess_image,[file,],[tf.float32,])
        return im, paramset

    # Set up input directories.
    cache_dir=folder
    left_path = cache_dir / "left"
    right_path = cache_dir / "right"
    
    # Get files and parameters from left and right directories.
    files_l, params_l = get_files_params(left_path)
    files_r, params_r = get_files_params(right_path)
    files = files_l + files_r
    params = params_l + params_r
    image_count = len(files)

    # Build datasets, preprocess, divide into train/val, batch
    # and prefetch.
    dataset = tf.data.Dataset.from_tensor_slices((files, params))
    dataset = dataset.shuffle(buffer_size=len(files))

    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_dataset = dataset.take(round(image_count * 0.9))
    val_dataset = dataset.skip(round(image_count * 0.9))
    
    train_dataset = train_dataset.batch(32)
    val_dataset = val_dataset.batch(32)

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset

def build_cnn(target_shape, nparams, nlayers):
    """Build cnn model."""
    base_cnn = cn.make_base_cnn_3d(target_shape, 'base_cnn', nlayers=nlayers)
    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(nparams)(dense2)
    model = Model(base_cnn.input, output, name="Model")
    return model

def main():
    # Get args.
    args = parse_args()
    image_folder = args.image_folder
    nlayers = args.nlayers
    distributed = args.distributed
    initial_learning_rate = args.initial_lr
    decay_rate = args.decay_rate
    decay_steps = args.decay_steps
    model_name = args.model_name
    epochs = args.epochs

    # Set up paths.
    cache_dir = Path(image_folder)
    final_checkpoint_path = os.path.join(image_folder, 'checkpoint_final_' + model_name)
    log_path = os.path.join(image_folder, model_name + '_log.txt')
    history_path = os.path.join(image_folder, 'history_' + model_name + '.pkl')

    # Make datasets.
    target_shape, nparams = get_target_shape_nparams(cache_dir)
    train_dataset, val_dataset = make_datasets(cache_dir)

    # Construct model.
    if distributed:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = build_cnn(target_shape, nparams, nlayers)

    else:
        model = build_cnn(target_shape, nparams, nlayers)

    # Train model.
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True
    )

    model.compile(
        loss="mean_absolute_percentage_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        shuffle=True,
    )

    # Save everything.
    model.save_weights(final_checkpoint_path)

    with open(history_path, 'wb') as history_file:
        pickle.dump(history.history, history_file)

    with open(log_path, 'w') as log_file:
        log_file.write('placeholder')

if __name__ == "__main__":
    sys.exit(main())