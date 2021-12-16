import cnn_models.siamese_cnn as cn
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import os

train_data_folder = sys.argv[1]
model_save_file = os.path.join(train_data_folder, 'model')

cache_dir = Path(train_data_folder)
target_shape = (100, 100)
channels_axis=1
print('Loading training data...', end='')
train_dataset, val_dataset = cn.make_triplet_inputs(cache_dir)
print('finished.')
base_cnn = cn.make_base_cnn(image_shape=(100,100))
embedding = cn.make_embedding(base_cnn)
siamese_network = cn.make_siamese_network(embedding)
siamese_model = cn.SiameseModel(siamese_network)
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))
history = siamese_model.fit(train_dataset, epochs=2, validation_data=val_dataset, verbose=True)

#tf.saved_model.save(siamese_model, model_save_file)