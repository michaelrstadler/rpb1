"""train_simaese_cnn_3d.py: Build and train a siamese CNN using 3d images as inputs.


To run on savio: 
export PYTHONPATH="$PWD/rpb1/bin/cnn_models" # Put path to cnn_models here...I don't know how to manage this with savio module system yet (facepalm).
module unload python/3.7
module load ml/tensorflow/2.5.0-py37
"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2022, Berkeley, California, USA"

import cnn_models.siamese_cnn_3d as cn
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import os
import pickle
from optparse import OptionParser
from time import time



def parse_options():
    parser = OptionParser()
    parser.add_option("-f", "--training_data_folder", dest="training_data_folder",
                      help="Folder containing pre-saved tensorflow dataset.", 
                      metavar="FOLDER")
    parser.add_option("-e", "--num_epochs", dest="num_epochs",
                      help="Number of epochs to train for", 
                      metavar="NUMEPOCHS")
    parser.add_option("-r", "--num_repeats", dest="num_repeats", default=1,
                      help="Number of repeats of the dataset to use. Repeats will have shuffled negative images and be rotated.", 
                      metavar="NUMREPEATS")
    (options, args) = parser.parse_args()
    return options

def load_dataset(folder):
    """Load pre-saved tensorflow dataset; get shape of input images."""
    elementspec_file = os.path.join(folder, 'elements_spec.pkl')
    with open(elementspec_file, 'rb') as file:
        elementspec = pickle.load(file) 
    shape = tuple(elementspec[0].shape.as_list()[1:-1])
    return tf.data.experimental.load(folder, elementspec, compression='GZIP'), shape

options = parse_options()
train_data_folder = options.training_data_folder
num_epochs = int(options.num_epochs)
num_repeats = int(options.num_repeats)

model_save_file = os.path.join(train_data_folder, 'model')
cache_dir = Path(train_data_folder)
target_shape = (20, 100, 100)

sys.stdout.write('Loading training data...\n')
sys.stdout.flush()
t1 = time()
train_dataset, target_shape = load_dataset(os.path.join(train_data_folder, 'training_dataset'))
val_dataset, _ = load_dataset(os.path.join(train_data_folder, 'val_dataset'))
t2 = time()
sys.stdout.write('finished.\n')
sys.stdout.write('time: ' + str(t2 - t1) + '\n')
sys.stdout.flush()
train_dataset_folder = os.path.join(train_data_folder, 'training_dataset')
val_dataset_folder = os.path.join(train_data_folder, 'val_dataset')
if not os.path.isdir(train_dataset_folder):
    os.mkdir(train_dataset_folder)
if not os.path.isdir(val_dataset_folder):
    os.mkdir(val_dataset_folder)

base_cnn = cn.make_base_cnn(image_shape=target_shape)
embedding = cn.make_embedding(base_cnn)
siamese_network = cn.make_siamese_network(embedding)
siamese_model = cn.SiameseModel(siamese_network)
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))

history = siamese_model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset, verbose=True)

tf.keras.models.save_model(embedding, model_save_file)
t3 = time()
sys.stdout.write('training time: ' + str(t3 - t2) + '\n')
sys.stdout.flush()
#tf.saved_model.save(siamese_model, model_save_file)