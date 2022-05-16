"""
Convert 3d simulated files to maximum intensity projections.
"""

import numpy as np
import os
from optparse import OptionParser
import pickle



def parse_options():
    parser = OptionParser()
    parser.add_option("-f", "--infolder", dest="infolder",
                      help="Folder containing 3d pickled files.", 
                      metavar="INFOLDER")
    parser.add_option("-o", "--outfolder", dest="outfolder",
                      help="Folder for MIPs.", 
                      metavar="OUTFOLDER")
    (options, args) = parser.parse_args()
    return options

options = parse_options()

files = os.listdir(options.infolder)
for f in files:
    if f[-3:] != 'pkl':
        continue
    filepath = os.path.join(options.infolder, f)
    with open(filepath, 'rb') as file:
        im = pickle.load(file)
    
    mip = im.max(axis=0)
    
    outfilepath = os.path.join(options.outfolder, f)
    with open(outfilepath, 'wb') as file:
        pickle.dump(mip, file)

