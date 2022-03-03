#!/usr/bin/env python

"""train_siamese_cnn.py: Train a siamense CNN image similarity detector

CNN model trained on full 3D image stacks
with multiple architectures. Curriculum learning can be implemented by restricting
the similarity of training images.

To run on savio: 
export PYTHONPATH="$PWD/rpb1/bin/cnn_models";module unload python/3.7;module load ml/tensorflow/2.5.0-py37
# Put path to cnn_models here...I don't know how to manage this with savio module system yet (facepalm).
"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2022, California, USA"

import cnn_models.siamese_cnn as cn
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
import sys
import os
from optparse import OptionParser
from time import time

def parse_options():
    parser = OptionParser()
    parser.add_option("-f", "--training_data_folder", dest="training_data_folder",
                      help="Folder containing training data in folders labeled left and right.")
    parser.add_option("-n", "--model_name", dest="model_name",
                      help="Name for model -- is appended to folder.")
    parser.add_option("-e", "--num_epochs", dest="num_epochs",
                      help="Number of epochs to train for. Either a single number or a comma-separated " +
                       "list of epochs for curriculum learning.")
    parser.add_option("-z", "--dataset_size", dest="dataset_size", default=1,
                      help="Size of the dataset used in each epoch as a multiple of the number of input files.")
    parser.add_option("-y", "--num_layers", dest="nlayers", default=18,
                      help="Number of layers in resnet 3D CNN (8, 18, or 34, default=18)")
    parser.add_option("-r", "--initial_learning_rate", dest="initial_learning_rate", default=0.0001,
                      help="Initial learning rate (default=0.0001)")
    parser.add_option("-R", "--learning_rate_exp", dest="learning_rate_exp", default=0.05,
                      help="Exponential decay factor R for learning rate lr = lr * exp(-R) (default=0.05)")
    parser.add_option("-c", "--learning_rate_constant_epochs", dest="learning_rate_constant_epochs", default=10,
                      help="Number of epochs to train at initial rate before beginning exponential decay (default=10)")
    parser.add_option("-w", "--lower_margin", dest="lower_margin", default='0',
                      help="Single number or comma-separated list of lower margins for curriculum learning, " +
                        "must be same length as epochs list (default=0)")
    parser.add_option("-u", "--upper_margin", dest="upper_margin", default='100',
                      help="Single number or comma-separated list of upper margins for curriculum learning, " +
                        "must be same length as epochs list (default=100)")
    parser.add_option("-p", "--num_neg_pairs", dest="num_neg_pairs", default=0,
                      help="Number of possible negative pairings to generate for each anchor-positive pair (default is 3 * dataset_size")
    parser.add_option("-b", "--batch_size", dest="batch_size", default=32,
                      help="Batch size")
    parser.add_option("-W", "--initial_weights_file", dest="initial_weights_file", default=None,
                      help="File containing initial weights")
    parser.add_option("-t", action="store_true", dest="rotate",
                      help="Randomly rotate images in training dataset.")
    parser.add_option("-d", action="store_true", dest="distributed",
                      help="Flag: Use distributed (multiple) GPUs.")
    
    (options, args) = parser.parse_args()
    return options

def get_target_shape(dir_):
    """Retrieve the image target size."""
    left = dir_ / 'left'
    imfile = os.listdir(left)[-1]
    impath = os.path.join(left, imfile)
    with open(impath, 'rb') as file:
        im = pickle.load(file)

    shape = im.shape
    return shape

def build_siamese_model():
    """Construct siamese CNN model."""

    base_cnn = cn.make_base_cnn_3d(image_shape=target_shape, nlayers=nlayers)

    embedding = cn.make_embedding(base_cnn)
    siamese_network = cn.make_siamese_network(embedding)
    siamese_model = cn.SiameseModel(siamese_network)
    return embedding, siamese_model

def scheduler(rel_epoch, lr, start_epoch, constant_epochs, learning_rate_exp):
    """Learning rate scheduler function."""
    true_epoch = start_epoch + rel_epoch
    new_lr = lr
    if true_epoch > constant_epochs :
        new_lr = lr * tf.math.exp(-1 * learning_rate_exp)
    return new_lr


# Process all options.
options = parse_options()
train_data_folder = options.training_data_folder
model_name = options.model_name
epoch_nums = [int(x) for x in options.num_epochs.split(',')]
dataset_size = int(options.dataset_size)
nlayers = int(options.nlayers)
initial_learning_rate = float(options.initial_learning_rate)
learning_rate_exp = float(options.learning_rate_exp)
learning_rate_constant_epochs = int(options.learning_rate_constant_epochs)
lower_margins = [float(x) for x in options.lower_margin.split(',')]
upper_margins = [float(x) for x in options.upper_margin.split(',')]
num_neg_pairs = int(options.num_neg_pairs)
batch_size = int(options.batch_size)
rotate = options.rotate
distributed = options.distributed
initial_weights_file = options.initial_weights_file

if num_neg_pairs == 0:
    num_neg_pairs = 3 * dataset_size

if (len(epoch_nums) != len(lower_margins)) or (len(lower_margins) != len(upper_margins)):
    raise ValueError('epoch_nums, lower_margins, and upper_margins must be same length.')

# Set paths and shape.
model_save_path = os.path.join(train_data_folder, 'model_' + model_name)
cache_dir = Path(train_data_folder)
target_shape = get_target_shape(cache_dir)
t1 = time()

# Construct model, set files.
if distributed:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        embedding, siamese_model = build_siamese_model()

else:
    embedding, siamese_model = build_siamese_model()

if initial_weights_file is not None:
    embedding.load_weights(initial_weights_file)

siamese_model.compile(optimizer=tf.keras.optimizers.Adam(initial_learning_rate))
final_checkpoint_path = os.path.join(train_data_folder, 'checkpoint_final_' + model_name)
best_checkpoint_path = os.path.join(train_data_folder, 'checkpoint_best_' + model_name)
log_path = os.path.join(train_data_folder, model_name + '_log.txt')
history_path = os.path.join(train_data_folder, 'history_' + model_name + '.pkl')

# Go through each iteration in epoch_nums individually.
histories = []
epoch_count = 1
for i in range(len(epoch_nums)):
    epoch_num = epoch_nums[i]
    lower_margin = lower_margins[i]
    upper_margin = upper_margins[i]

    # Generate datasets.
    train_dataset, val_dataset = cn.make_triplet_inputs(cache_dir, 
        lower_margin=lower_margin, 
        upper_margin=upper_margin, 
        num_negatives=num_neg_pairs,
        n_repeats=dataset_size,
        batch_size=batch_size,
        rotate=rotate)

    # Set up learning rate scheduler.
    scheduler_function = lambda a,b: scheduler(a, b, start_epoch=epoch_count, 
        constant_epochs=learning_rate_constant_epochs, learning_rate_exp=learning_rate_exp)

    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler_function)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

    # Train model, store history.
    history = siamese_model.fit(train_dataset, epochs=epoch_num, validation_data=val_dataset, 
        callbacks=[lr_scheduler_callback, model_checkpoint_callback], verbose=True)

    histories.append(history.history)
    epoch_count += epoch_num

# Save everything.
embedding.save_weights(final_checkpoint_path)
with open(history_path, 'wb') as history_file:
    pickle.dump(histories, history_file)

with open(log_path, 'w') as log_file:
    log_file.write(str(options))

t2 = time()
sys.stdout.write('training time: ' + str(t2 - t1) + '\n')
sys.stdout.flush()

