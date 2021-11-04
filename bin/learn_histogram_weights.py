#!/usr/bin/env python

"""learn_histogram_weights.py.

The goal is to develop a comparison function for image data of the 
distributions of fluorescence signal in fly embryo nuclei. My strategy
is so make scale-space histograms of images and determine the similarities
of these histograms. To validate the approach, I'm using simulated data.

Here is the plan:

1. Start with a pool of scale-space histograms for simulated data.
2. Take a random subset of simulations, generate:
    - difference scores for the histograms (one for each sigma)
    - 


"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2021, Planet Earth"

import flymovie as fm
import scipy
import numpy as np
import pandas as pd
import dask



############################################################################