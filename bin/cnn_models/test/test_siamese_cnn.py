import unittest
import numpy as np
import tempfile
from cnn_models.siamese_cnn import *

def count_params(variables):
    total = 0
    for v in variables:
        product = 1
        for dim in v.shape:
            product *= dim
        total += product
    return total

#---------------------------------------------------------------------------
class TestSiameseCNN(unittest.TestCase):

    def test_identity_block_2d(self):

        image_shape=(100,100)
        img_input = tf.keras.layers.Input(shape=image_shape + (1,))
        x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = tf.keras.layers.Conv2D(256, 3, (1,1), padding='same')(x)
        x = identity_block_2d(x, 3, [64, 64, 256], stage=2, block='b')
        self.assertTrue(np.array_equal(x.shape, [None, 106, 106, 256]), "Wrong shape.")
        # Wrong dimension in last position.
        self.assertRaises(ValueError, identity_block_2d, x, 3, [64, 64, 255], 2, 'b')

#---------------------------------------------------------------------------

    def test_conv_block_2d(self):

        image_shape=(100,100)
        img_input = tf.keras.layers.Input(shape=image_shape + (1,))
        x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = conv_block_2d(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.assertTrue(np.array_equal(x.shape, [None, 106, 106, 256]), "Wrong shape.")
        x = conv_block_2d(x, 3, [64, 64, 256], stage=2, block='a', strides=(2, 2))
        self.assertTrue(np.array_equal(x.shape, [None, 53, 53, 256]), "Wrong shape.")
        x = conv_block_2d(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        self.assertTrue(np.array_equal(x.shape, [None, 53, 53, 256]), "Wrong shape.")
        x = conv_block_2d(x, 3, [64, 64, 64], stage=2, block='a', strides=(2, 2))
        self.assertTrue(np.array_equal(x.shape, [None, 27, 27, 64]), "Wrong shape.")

#---------------------------------------------------------------------------

    def test_make_base_cnn_2d(self):
        base_cnn = make_base_cnn_2d(image_shape=(100,100), name='base_cnn')
        
        total = count_params(base_cnn.trainable_variables)
        
        self.assertEqual(total, 23_528_320, 'Wrong number of parameters in base_cnn_2d model.')
        self.assertTrue(np.array_equal(base_cnn.output_shape, [None, 7, 7, 2048]), "Wrong shape.")

#---------------------------------------------------------------------------

    def test_identity_block_3d(self):
        
        image_shape=(40, 100,100)
        img_input = tf.keras.layers.Input(shape=image_shape + (1,))
        x = tf.keras.layers.ZeroPadding3D(padding=(3, 3, 3), name='conv1_pad')(img_input)
        x = tf.keras.layers.Conv3D(64, 3, (1,1,1), padding='same')(x)
        x = identity_block_3d(x, 3, [64, 64], stage=2, block='b')
        self.assertTrue(np.array_equal(x.shape, [None, 46, 106, 106, 64]), "Wrong shape.")
        x = tf.keras.layers.Conv3D(128, 3, (2,2,2), padding='same')(x)
        x = identity_block_3d(x, 3, [64, 128], stage=2, block='b')
        self.assertTrue(np.array_equal(x.shape, [None, 23, 53, 53, 128]), "Wrong shape.")
        # Wrong dimension in last position.
        self.assertRaises(ValueError, identity_block_3d, x, 3, [64, 127], 2, 'b')

#---------------------------------------------------------------------------

    def test_conv_block_3d(self):

        image_shape=(20,100,100)
        img_input = tf.keras.layers.Input(shape=image_shape + (1,))
        x = tf.keras.layers.ZeroPadding3D(padding=(1, 3, 3), name='conv1_pad')(img_input)
        x = layers.Conv3D(64, (3,7,7),
                        strides=(2, 2, 2),
                        padding='valid',
                        kernel_initializer='he_normal',
                        name='conv1')(x)
        x = conv_block_3d(x, 3, [64, 64], stage=2, block='a', strides=(1, 1, 1))
        self.assertTrue(np.array_equal(x.shape, [None, 10, 50, 50, 64]), "Wrong shape.")
  
        x = conv_block_3d(x, 3, [64, 64], stage=2, block='a', strides=(2, 2, 2))
        self.assertTrue(np.array_equal(x.shape, [None, 5, 25, 25, 64]), "Wrong shape.")
        x = conv_block_3d(x, 3, [64, 128], stage=2, block='a', strides=(1, 1, 1))
        self.assertTrue(np.array_equal(x.shape, [None, 5, 25, 25, 128]), "Wrong shape.")
        x = conv_block_3d(x, 3, [64, 256], stage=2, block='a', strides=(2, 2, 2))
        self.assertTrue(np.array_equal(x.shape, [None, 3, 13, 13, 256]), "Wrong shape.")

#---------------------------------------------------------------------------

    def test_make_base_cnn_3d(self):
        base_cnn = make_base_cnn_3d(image_shape=(20,100,100), name='base_cnn')
        
        total = count_params(base_cnn.trainable_variables)
        self.assertEqual(total, 33_156_544, 'Wrong number of parameters in base_cnn_2d model.')
        self.assertTrue(np.array_equal(base_cnn.output_shape, [None, 1, 7, 7, 512]), "Wrong shape.")

        base_cnn = make_base_cnn_3d(image_shape=(20,100,100), name='base_cnn', nlayers=34)
        
        total = count_params(base_cnn.trainable_variables)
        self.assertEqual(total, 63_469_888, 'Wrong number of parameters in base_cnn_2d model.')
        self.assertTrue(np.array_equal(base_cnn.output_shape, [None, 1, 7, 7, 512]), "Wrong shape.")

#---------------------------------------------------------------------------

    def test_make_embedding(self):
        base2 = make_base_cnn_2d()
        e2 = make_embedding(base2)
        self.assertTrue(np.array_equal(e2.output_shape, [None, 256]), 'Wrong output shape')
        self.assertEqual(count_params(e2.trainable_variables), 75_107_712, "wrong number of params")

        base3 = make_base_cnn_3d(nlayers=18)
        e3 = make_embedding(base3)
        self.assertTrue(np.array_equal(e3.output_shape, [None, 256]), 'Wrong output shape')
        self.assertEqual(count_params(e3.trainable_variables), 46_200_768, "wrong number of params")

        base3 = make_base_cnn_3d(nlayers=34)
        e3 = make_embedding(base3)
        self.assertTrue(np.array_equal(e3.output_shape, [None, 256]), 'Wrong output shape')
        self.assertEqual(count_params(e3.trainable_variables), 76_514_112, "wrong number of params")



if __name__ == '__main__':
	unittest.main()