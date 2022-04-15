#!/usr/bin/env python

"""Testing load_save.py module

Issues:
    - Need to add tests for functions loading from
    folders. 



"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2021, California, USA"

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import pickle
import sys

from flymovie.load_save import *
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

lattice_folder = os.path.join(os.getcwd(), 'test_data', 'test_lattice_data')
test_czi_file = os.path.join(os.getcwd(), 'test_data', 'czi_test.czi')
test_czi_folder = os.path.join(os.getcwd(), 'test_data')

#---------------------------------------------------------------------------

class TestLoadSave(unittest.TestCase):
	# Need to add 
    def test_read_tiff_stack(self):
        files = ['1_CamA_ch0_stack0000_4nm_0000000msec_0000123506msecAbs.tif',
        '1_CamA_ch0_stack0001_4nm_0000105msec_0000123611msecAbs.tif']
        
        output = read_tiff_stack(lattice_folder, files)
        self.assertTrue(isinstance(output, np.ndarray), 'Should be np.ndarray')
        self.assertEqual(len(output.shape), 3, 'Should be 3d array')

    def test_read_tiff_folder(self):
        output = read_tiff_folder(lattice_folder)
        self.assertTrue(isinstance(output, np.ndarray), 'Should be np.ndarray')
        self.assertEqual(len(output.shape), 3, 'Should be 3d array')

    def test_read_tiff_lattice(self):
        output = read_tiff_lattice(lattice_folder)
        self.assertTrue(isinstance(output, np.ndarray), 'Should be np.ndarray')
        self.assertTrue(np.array_equal(output.shape[0:2], [2,3]), 'Should be [2,3]')

    def test_read_czi(self):
        output = read_czi(test_czi_file)
        self.assertTrue(isinstance(output, np.ndarray), 'Should be np.ndarray')
        self.assertEqual(len(output.shape), 4, 'Should be 4d array')

    def test_read_czi_multiple(self):
        output = read_czi_multiple(['czi_test.czi', 'czi_test.czi'], test_czi_folder)
        self.assertTrue(isinstance(output, tuple), 'Should return tuple')
        self.assertEqual(len(output), 4, 'Should return 4 items')
        self.assertTrue(isinstance(output[0], np.ndarray), 'First item should be stack')
        

    def test_save_pickle(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            # Just test if this works without raising error.
            save_pickle(np.ones((2,2)), tmp_file.name)

    def test_load_pickle(self):
        with tempfile.NamedTemporaryFile() as tmp_file:
            save_pickle(np.ones((2,2)), tmp_file.name)
            output = load_pickle(tmp_file.name)    
        self.assertTrue(np.array_equal(output, np.ones((2,2))), 'Should be equal')


if __name__ == '__main__':
    unittest.main()