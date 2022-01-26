"""


To run on savio: 
export PYTHONPATH="$PWD/rpb1/bin/cnn_models" # Put path to cnn_models here...I don't know how to manage this with savio module system yet (facepalm).
module unload python/3.7
module load ml/tensorflow/2.5.0-py37
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
    parser.add_option("-r", "--num_repeats", dest="num_repeats", default=1,
                      help="Number of repeats of the dataset to use. Repeats will have shuffled negative images and be rotated.", 
                      metavar="NUMREPEATS")
    (options, args) = parser.parse_args()
    return options

options = parse_options()
train_data_folder = options.training_data_folder
num_epochs = int(options.num_epochs)
num_repeats = int(options.num_repeats)

model_save_file = os.path.join(train_data_folder, 'model')

cache_dir = Path(train_data_folder)
target_shape = (100, 100)

sys.stdout.write('Loading training data...\n')
sys.stdout.flush()
t1 = time()
train_dataset, val_dataset = cn.make_triplet_inputs(cache_dir, num_repeats)
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

base_cnn = cn.make_base_cnn(image_shape=(100,100))
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