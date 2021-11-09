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
from optparse import OptionParser


def parse_options():
	parser = OptionParser()
	parser.add_option("-f", "--file", dest="filename",
					  help="Reduced bin file", metavar="FILE")
	parser.add_option("-o", "--outfolder", dest="outfolder",
					  help="Path for output files", metavar="OUTFOLDER")
	parser.add_option("-c", "--chromosome", dest="chromosome",
					  help="Chromosome", metavar="CHROMOSOME")
	parser.add_option("-w", "--windowsize", dest="windowsize", default=400,
					  help="Panel width, in bins", metavar="WINDOWSIZE")
	parser.add_option("-s", "--stepsize", dest="stepsize", default=200,
					  help="Step size, in bins", metavar="STEPSIZE")
	
	(options, args) = parser.parse_args()
	return options    

options = parse_options()