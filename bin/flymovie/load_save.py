
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re
import xml.etree.ElementTree as ET
from skimage import filters, io
import pickle
import czifile

from .general_functions import concatenate_5dstacks

############################################################################
# Functions for loading reading and writing data
############################################################################

def read_tiff_stack(tif_folder, tif_files, **kwargs):
    """Read a folder of 2D or 3D TIFF files into a numpy ndarray.
    
    Args:
        tif_folder: string
            Directory containing multiple TIFF files.
        tif_files: list
            List of files in the folder to load. Must be sorted in order
            desired.
        span: tuple of ints
            Optional key-word argument specifying first and last file to 
            load, both inclusive. Example: span=(0, 5) loads the first 6 
            images, numbers 0 through 5.
    
    Returns:
        stack: ndarray
            n-dimensional image stack with the new dimension (file number) 
            in the 0 position(file_num, z, x, y) for 3D stacks, (filenum, x,
            y) for 2D stacks
            
    Raises:
        ValueError: 
            If dimensions of TIFF file don't match those of the first
            file
    """
    if 'span' in kwargs:
        first, last = (kwargs['span'])
        if (first <= last) and (last < len(tif_files)):
            tif_files = tif_files[first:(last + 1)]
        else:
            raise ValueError('Span exceeds the dimensions of the stack')

    # Create stack with dimensions from first file.
    img = io.imread(join(tif_folder, tif_files[0]))
    dims = img.shape
    num_files = len(tif_files)
    stack = np.ndarray(((num_files,) + dims), dtype=img.dtype)
    stack[0] = img
    img_num = 1
    
    # Add the rest of the files to the stack.
    for tif_file in tif_files[1:]:
        # Add image data to ndarray
        img = io.imread(join(tif_folder, tif_file))
        # Check dimensions
        if not stack[0].shape == img.shape:
            raise ValueError(f'Dimensions do not match previous files: {tif_file}')
        stack[img_num] = img
        img_num = img_num + 1
        
    return stack

############################################################################
def read_tiff_folder(tif_folder):
    """Read all TIFF files in a folder into an ndarray.
    
        Args:
            tif_folder: string
                Directory containing multiple TIFF files. Must be sortable
                asciibetically.
        
        Returns:
            stack: ndarray
                n-dimensional image stack with the new dimension (file number) 
                in the 0 position(file_num, z, x, y) for 3D stacks, (filenum, 
                x, y) for 2D stacks
                
        Raises:
            ValueError: 
                If dimensions of TIFF file don't match those of the first
                file
    """
    
    # Compile files that are files and have .tif extension (case-insensitive).
    tif_files = [f for f in listdir(tif_folder) if (isfile(join(tif_folder, f)) 
        and (os.path.splitext(f)[1][0:4].upper() == '.TIF'))]
    # Sort the files: asciibetical sorting produces files ascending in time 
    # (sorting is *in place*)
    tif_files.sort()
    return read_tiff_stack(tif_folder, tif_files)

############################################################################
def read_tiff_lattice(tif_folder, **kwargs):
    """Read all TIFF files in a lattice output folder into an ndarray.
    
        Args:
            tif_folder: string
                Directory containing multiple TIFF files with 'CamA' and 'CamB' 
                Must be equal numbers of CamA and CamB and files must be 
                sortable asciibetically.
            span: tuple of ints
                Optional key-word argument specifying first and last file to 
                load, both inclusive. Example: span=(0, 5) loads the first 6 
                images, numbers 0 through 5.
        
        Returns:
            stack: ndarray
                n-dimensional image stack with the new dimension (channel) 
                in the 0 position, e.g. (channel, t, z, x, y) for 3D stacks. 
                
        Raises:
            ValueError: 
                If dimensions of TIFF file don't match those of the first file
            ValueError: 
                If there are non-identical numbers of CamA and CamB files
    """
    
    # Compile files that are files and have .tif extension (case-insensitive).
    tif_files = [f for f in listdir(tif_folder) if (isfile(join(tif_folder, f)) 
        and (os.path.splitext(f)[1][0:4].upper() == '.TIF'))]
    # Sort files into two lists based on containing 'CamA' and 'CamB' in filename.
    regex_camA = re.compile('CamA')
    regex_camB = re.compile('CamB')
    camA_files = [*filter(regex_camA.search, tif_files)] # This syntax unpacks filter into a list.
    camB_files = [*filter(regex_camB.search, tif_files)]

    # Sort the files: asciibetical sorting produces files ascending in time 
    # (sorting is *in place*)
    camA_files.sort()
    camB_files.sort()
    # Read both sets of files, combine if they are of same dimensions.
    camA_stack = read_tiff_stack(tif_folder, camA_files, **kwargs)
    camB_stack = read_tiff_stack(tif_folder, camB_files, **kwargs)
    if camA_stack.shape == camB_stack.shape:
        return np.stack((camA_stack, camB_stack), axis=0)
    else:
        raise ValueError('Unequal number of CamA and CamB files.')

############################################################################
def read_czi(filename, trim=False, swapaxes=True, return_metadata=False,
    metadata_only=False):
    """Read a czi file into an ndarray
    
    Args:
        filename: string
            Path to czi file
        trim: bool
            If true, remove last frame if it contains blank slices
        swapaxes: bool
            If true, switches first two axes to produce a stack order ctzxy
        return_metadata: bool
            If true, returns a tuple of stack, first distance, z interval
        metadata_only: bool
            If true, just return the metadata features starting_positions
            and z_interval. Stack is returned as None.
            
    Returns:
        stack: ndarray
            Image stack in dimensions [t,c,z,x,y] (no swap) or 
            [c,t,z,x,y] (swapped)
        starting_positions: list of floats
            List of the position, in microns, of the first slice in the Z
            stack of each file, taken from czi file metadata.
        z_interval: float
            Size of Z slice, in microns, taken from czi metadata
    """
    def frame_incomplete(stack3d):
        """Determine if frame is incomplete."""
        for slice in stack3d:
            # If only value in slice is 0, frame is incomplete.
            if ((np.min(slice) == 0) & (np.max(slice) == 0)):
                return True
        return False

    if not metadata_only:
        stack = czifile.imread(filename)
        stack = np.squeeze(stack)
        # Trim off last frame if incomplete.
        if trim:
            if frame_incomplete(stack[-1,0]):
                stack = stack[:-1]
        if (swapaxes):
            stack = np.swapaxes(stack,0,1)
    else:
        stack = None

    if return_metadata:
        handle = czifile.CziFile(filename)
        metadata = handle.metadata()
        root = ET.fromstring(metadata)
        # Pull first distance and z interval, convert to microns.
        first_dist = float(root.findall('.//ZStackSetup')[0][8][0][0].text) * 1e6
        #last_dist = root.findall('.//ZStackSetup')[0][9][0][0].text
        z_interval = float(root.findall('.//ZStackSetup')[0][10][0][0].text) * 1e6
        handle.close()
        return stack, first_dist, z_interval
    else:
        return stack

############################################################################
def read_czi_multiple(czi_files, folder, metadata_only=False):
    """Read a list of 5d czi files, combine into single stack, record
    frame junctions and positions of first Z slice.
    
    Args:
        czis: list-like (iterable)
            List of filenames containing 5d .czi movies of dimension 
            [c,t,z,x,y]. Shapes must be identical except for t dimension.
        folder: path
            Path to folder containing czi files
        metadata_only: bool
            If true, just return the metadata features starting_positions
            and z_interval (no stack)
            
    Returns:
        stack: 5d ndarray
            Concatenation of input stacks
        frames: list of ints
            List of the frame numbers at which joins occur, with each entry
            representing the 0-indexed location of the first frame of a new
            stack.
        starting_positions: list of floats
            List of the position, in microns, of the first slice in the Z
            stack of each file, taken from czi file metadata.
        z_interval: float
            Size of Z slice, in microns, taken from czi metadata
    """
    
    stacks = []
    starting_positions = []
    for czi_file_ in czi_files:
        czi_file_path = os.path.join(folder, czi_file_)
        stack, first_dist, z_interval = read_czi(czi_file_path, trim=True, 
            return_metadata=True, metadata_only=metadata_only)
        stacks.append(stack)
        starting_positions.append(first_dist)
    
    if metadata_only:
        return starting_positions, z_interval
    else:
        stack, frames = concatenate_5dstacks(stacks)
        return stack, frames, starting_positions, z_interval
    
    

############################################################################
def save_pickle(obj, filename):
    """Pickel (serialize) an object into a file

    Args:
        filename: string
            Full path to save to
        obj: object
            Python object to serialize
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

############################################################################
def load_pickle(filename):
    """Load a pickled (serialized) object

    Args:
        filename: string
            Full path containing pickled object
    
    Returns:
        obj: object
            Object(s) contained in pickled file
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj