#!/usr/bin/env python

"""simdata_to_sshistogram.py.


"""
__author__      = "Michael Stadler"
__copyright__   = "Copyright 2021, Planet Earth"

import flymovie as fm
import numpy as np
from optparse import OptionParser
from time import time, process_time


def parse_options():
    parser = OptionParser()
    parser.add_option("-f", "--folder", dest="folder",
                      help="Folder containing pickled simulation data", 
                      metavar="FOLDER")
    parser.add_option("-o", "--outfile", dest="outfile",
                      help="File to save pickled data", 
                      metavar="OUTFILE")
    parser.add_option("-s", "--sigmas", dest="sigmas",
                      help="Comma-separated list of sigma values to use", 
                      metavar="SIGMAS")
    parser.add_option("-z", "--zdim", dest="zdim", default=20,
                      help="Z dimension of simulated data", metavar="ZDIM")
    parser.add_option("-i", "--idim", dest="idim", default=200,
                      help="i dimension of simulated data", metavar="IDIM")
    parser.add_option("-j", "--jdim", dest="jdim", default=200,
                      help="j dimension of simulated data", metavar="jDIM")
    parser.add_option("-r", "--radius", dest="nuc_radius", default=50,
                      help="Size of nuclear radius in simulated data", metavar="RADIUS")
    parser.add_option("-p", "--separation", dest="nuc_sep", default=100,
                      help="Separation between nuclei in simulated data", metavar="SEP")
    parser.add_option("-b", "--numbins", dest="numbins", default=100,
                      help="Number of bins in histogram", metavar="NUMBINS")
    parser.add_option("-N", "--histrange", dest="histrange", default='0,66_000',
                      help="Range of histogram in form: 0,66000", metavar="HISTRANGE")
    
    (options, args) = parser.parse_args()
    return options    
t_start = time()
options = parse_options()
folder, outfile, sigmasstring, zdim, idim, jdim, nuc_radius, nuc_sep, numbins, histrangestring = options.folder, options.outfile, options.sigmas, options.zdim, options.idim, options.jdim, options.nuc_radius, options.nuc_sep, options.numbins, options.histrange
sigmas = [float(x) for x in sigmasstring.split(',')]
histrange = [float(x) for x in histrangestring.split(',')]


mask = fm.make_dummy_mask(zdim, idim, jdim, nuc_sep, nuc_radius)
width = len(sigmas) * numbins
output = fm.sims_to_data(folder, mask, width, fm.make_scalespace_2dhist, sigmas=sigmas, numbins=numbins, histrange=histrange)
fm.save_pickle(output, outfile)
t_end = time()
print (len(data_), t_end - t_start)