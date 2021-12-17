import unittest
import numpy as np
import pandas as pd
import os
import sys

from flymovie.fitting import *
from flymovie.load_save import load_pickle

# Workaround for github file size limits...needed to split 
# test data into multiple files. To do this I load them with
# load_test_data function in test package.
wkdir = os.getcwd()
sys.path.append(wkdir)
from .load_test_data import load_test_data

class TestData():
    def __init__(self):
        pass

test_data = load_test_data(wkdir)

#---------------------------------------------------------------------------

class TestFitting(unittest.TestCase):
	
    def test_gaussian3d(self):
        gaussian_func = gaussian3d(0, 0, 0, 1, 1, 1, 1)
        # Test several values of this simple Gaussian function.
        self.assertEqual(gaussian_func(0, 0, 0), 1, 'Should be 1')
        self.assertAlmostEqual(gaussian_func(1e6, 0, 0), 0, 'Should be 0')
        self.assertAlmostEqual(gaussian_func(1, 1, 1), 0.22313016014842982, 'Should be 0')

    def test_fitgaussian3d(self):
        input_ = np.random.rand(10, 10, 10)
        output = fitgaussian3d(input_)
        self.assertTrue(output.success, 'Fit should have succeeded')
        self.assertTrue(len(output.x) == 7, 'Should have 7 items')

    
if __name__ == '__main__':
    unittest.main()