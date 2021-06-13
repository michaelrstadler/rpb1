import unittest
import numpy as np
import pandas as pd
import os

from flymovie.analyze import *
from flymovie.load_save import load_pickle

class TestData():
    def __init__(self):
        pass

def load_test_data(dir_):
    test_data = load_pickle(os.path.join(dir_, 'test_data', 'test_data1.pkl'))
    test_data2 = load_pickle(os.path.join(dir_, 'test_data', 'test_data2.pkl'))

    test_data.align_traces_output = test_data2.align_traces_output
    test_data.connect_nuclei_output = test_data2.connect_nuclei_output
    test_data.df = test_data2.df
    test_data.df_stack = test_data2.df_stack

    return test_data