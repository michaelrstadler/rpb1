from importlib import reload
import cnn_models.siamese_cnn as cn
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras import Model
import sys
from time import time

"""
export PYTHONPATH="$PWD/rpb1/bin/cnn_models";module unload python/3.7;module load ml/tensorflow/2.5.0-py37
"""
t1 = time()
# Add files and folders.
reals_folder = '/global/home/users/mstadler/scratch/blackbox/reals/'
sims_folder = '/global/home/users/mstadler/scratch/blackbox/sims/'
model_file = '/global/home/users/mstadler/scratch/blackbox/model3'
#reals_folder = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/data/blackbox/reals/'
#sims_folder = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/data/blackbox/sims/'
#model_file = '/Users/michaelstadler/Bioinformatics/Projects/rpb1/data/blackbox/model'
batch_size = 32
image_shape = (34,100,100)

def preprocess_images(im_sim, im_real):
    """Wraps cn.preprocess_image for TF."""
    [sim,] = tf.py_function(cn.preprocess_image,[im_sim,],[tf.float32,])
    [real,] = tf.py_function(cn.preprocess_image,[im_real,],[tf.float32,])
    return sim, real


# Make file lists from input folders.
sim_files1 = os.listdir(sims_folder)
sim_files2 = []
real_files = []
for f in sim_files1:
    if f[0] == '.':
        continue

    splits = f.split('_')
    real_file = '_'.join(splits[1:])
    sim_files2.append(os.path.join(sims_folder, f))
    real_files.append(os.path.join(reals_folder, real_file))

# Make dataset, map preprocessing, batch.
sim_ds = tf.data.Dataset.from_tensor_slices((sim_files2, real_files))
sim_ds = sim_ds.map(preprocess_images, num_parallel_calls=tf.data.AUTOTUNE)
sim_ds = sim_ds.batch(batch_size, drop_remainder=False)


num_elem = np.product(image_shape)
img_input = layers.Input(shape=image_shape + (1,)) # Channels last.
#x = layers.ZeroPadding3D(padding=(5, 10, 10), name='psf_pad')(img_input)
x = layers.Conv3D(1, (8,24,24),
        strides=(1, 1, 1),
        padding='same',
        kernel_initializer='he_normal',
        name='psf')(img_input)

x = layers.Conv3D(256, (3,3,3),
        strides=(1, 1, 1),
        padding='same',
        kernel_initializer='he_normal',
        name='conv1')(x)

x = layers.Conv3D(1, (1,1,1),
        strides=(1, 1, 1),
        padding='same',
        kernel_initializer='he_normal',
        name='conv1')(x)


output = tf.math.reduce_sum(x, axis=-1)

model = Model(img_input, output, name="Embedding")

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.mean_squared_error)
model.fit(sim_ds, epochs=20)
model.save(model_file)

t2 = time()
sys.stdout.write('training time: ' + str(t2 - t1) + '\n')
sys.stdout.flush()