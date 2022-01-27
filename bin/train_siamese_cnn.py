"""


To run on savio: 
export PYTHONPATH="$PWD/rpb1/bin/cnn_models";module unload python/3.7;module load ml/tensorflow/2.5.0-py37
# Put path to cnn_models here...I don't know how to manage this with savio module system yet (facepalm).


"""

import cnn_models.siamese_cnn as cn
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import os
from optparse import OptionParser
from time import time

def parse_options():
    parser = OptionParser()
    parser.add_option("-f", "--training_data_folder", dest="training_data_folder",
                      help="Folder containing training data in folders labeled left and right.", 
                      metavar="FOLDER")
    parser.add_option("-e", "--num_epochs", dest="num_epochs",
                      help="Number of epochs to train for", 
                      metavar="NUMEPOCHS")
    parser.add_option("-n", "--model_name", dest="model_name",
                      help="Name for model -- is appended to folder.", 
                      metavar="MODELNAME")
    parser.add_option("-r", "--num_repeats", dest="num_repeats", default=1,
                      help="Number of repeats of the dataset to use. Repeats will have shuffled negative images and be rotated.", 
                      metavar="NUMREPEATS")
    parser.add_option("-m", action="store_true", dest="mip",
                      help="Flag: write maximum intensity projection (mip).")
    (options, args) = parser.parse_args()
    return options

def get_target_shape(dataset):
    iter = dataset.as_numpy_iterator()
    i = next(iter)
    shape = i[0].shape[1:-1]
    return shape

options = parse_options()
train_data_folder = options.training_data_folder
num_epochs = int(options.num_epochs)
num_repeats = int(options.num_repeats)
mip = options.mip
model_name = options.model_name

model_save_path = os.path.join(train_data_folder, 'model_' + model_name)

cache_dir = Path(train_data_folder)

sys.stdout.write('Loading training data...\n')
sys.stdout.flush()
t1 = time()

train_dataset, val_dataset = cn.make_triplet_inputs(cache_dir, num_repeats, mip=mip)

t2 = time()
sys.stdout.write('finished.\n')
sys.stdout.write('time: ' + str(t2 - t1) + '\n')
sys.stdout.flush()

target_shape = get_target_shape(val_dataset)

if mip:
    base_cnn = cn.make_base_cnn(image_shape=target_shape)

if not mip:
    base_cnn = cn.make_base_cnn_3d(image_shape=target_shape, nlayers=18)

embedding = cn.make_embedding(base_cnn)
siamese_network = cn.make_siamese_network(embedding)
siamese_model = cn.SiameseModel(siamese_network)
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))

checkpoint_path = os.path.join(train_data_folder, 'checkpoint_' + model_name)

history = siamese_model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset, 
    verbose=True)

embedding.save_weights(checkpoint_path)
#tf.keras.models.save_model(embedding, model_save_path)

t3 = time()
sys.stdout.write('training time: ' + str(t3 - t2) + '\n')
sys.stdout.flush()
