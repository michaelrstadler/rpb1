import flymovie as fm
import numpy as np
import tensorflow as tf
from pathlib import Path

cache_dir = Path('')
target_shape = (100, 100)
channels_axis=1

train_dataset, val_dataset = fm.make_triplet_inputs(cache_dir)

base_cnn = fm.cnn_models.siamese_cnn.make_base_cnn(image_shape=(100,100))
embedding = fm.cnn_models.siamese_cnn.make_embedding(base_cnn)
siamese_network = fm.cnn_models.siamese_cnn.make_siamese_network(embedding)
siamese_model = fm.cnn_models.siamese_cnn.SiameseModel(siamese_network)
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))
history = siamese_model.fit(train_dataset, epochs=2, validation_data=val_dataset, verbose=True)
