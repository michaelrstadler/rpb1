import unittest
import numpy as np
import pandas as pd
import os
import sys

#import flymovie
from flymovie.fitting import *
from flymovie.load_save import load_pickle

# Workaround for github file size limits...needed to split 
# test data into multiple files. To do this I load them with
# load_test_data function in test package.
wkdir = os.getcwd()
sys.path.append(wkdir)
from load_test_data import load_test_data

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

    def test_gaussian_pdf(self):
        pdf = gaussian_pdf(np.arange(0, 10, 1), 5, 1)
        self.assertAlmostEqual(pdf[5], 0.39894228, 4, 'Should be equal')
        self.assertAlmostEqual(pdf[2], 0.00443185, 4, 'Should be equal')
        self.assertAlmostEqual(pdf[6], 0.24197072, 4, 'Should be equal')

        pdf = gaussian_pdf(np.arange(0, 20, 1) - 10, 0, 2)
        self.assertAlmostEqual(pdf[10], 0.19947114, 4, 'Should be equal')
        self.assertAlmostEqual(pdf[2], 0.00006692, 8, 'Should be equal')
        self.assertAlmostEqual(pdf[18], 0.00006692, 8, 'Should be equal')

    def test_n_binom_pdf(self):
        pdf = n_binom_pdf(np.arange(0, 10, 1), 5, 0.5)
        self.assertAlmostEqual(pdf[0], 0.03125, 5, 'Should be equal')
        self.assertAlmostEqual(pdf[9], 0.04364, 5, 'Should be equal')
        
        pdf = n_binom_pdf(np.arange(0, 100, 1), 10, 0.1)
        self.assertAlmostEqual(pdf[30], 0.0009, 4, 'Should be equal')
        self.assertAlmostEqual(pdf[97], 0.01196, 5, 'Should be equal')

        

    
if __name__ == '__main__':
    unittest.main()