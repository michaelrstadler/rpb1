#!/usr/bin/env python

"""
Functions for fitting image data.

"""
__version__ = '1.1.0'
__author__ = 'Michael Stadler'

import numpy as np
from scipy import optimize

############################################################################
def gaussian3d(center_z, center_x, center_y, height, width_z, width_x, width_y):
    """Returns a 3D gaussian function with the given parameters
    
    Args:
        center_z, center_x, center_y: int
            Locations of the center of the gaussian
        height: int
            Height of the gaussian
        width_z, width_x, width_y: int
            Sigmas for the gaussians in each dimension

    Returns:
        Function that accepts int coordinates z,x,y and returns the value of
        the 3D gaussian function at that position based on input parameters.
    """
    width_x = float(width_x)
    width_y = float(width_y)
    width_z = float(width_z)
    return lambda z,x,y: height*np.exp(
                -(((center_z-z)/width_z)**2 + 
                  ((center_x-x)/width_x)**2 + 
                  ((center_y-y)/width_y)**2)/2)

############################################################################
def moments3d(data):
    """Estimate initial parameters of 3D gaussian fit
    
    Returns (z, x, y, height, width_z, width_x, width_y)
    the gaussian parameters of a 3D distribution by calculating its
    moments (mean for centers, standard deviation for widths) 
    
    Args:
        data: ndarray
            The 3D data to fit in shape [z,x,y]
    Returns:
        tuple of ints
            Estimates for intial fit params: (z, x, y, height, width_z,
            width_x, width_y)   
    """
    # Find total for all values in the data.
    total = data.sum()
    
    # Make index matrices.
    Z, X, Y = np.indices(data.shape)
    
    # Find mean positions in each dimension by weighted average (weight is intensity, index is position)
    z = (Z*data).sum()/total
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    
    # Estimate width in each dimension. Procedure is to fix the other two dimensions at their mean
    # and retrieve a single column in the dimension of interest through the peak. Visually, in a Z-
    # stack you would determine the X,Y position of the center of the peak, then we're drawing a line
    # (or a bar) in Z through that point. This becomes a simple 1D profile of intensity as a function
    # of Z position. Standard deviation of this 1D vector about z (mean Z position) is computed.
    z_col = data[:, int(x), int(y)] #single column through z with x and y fixed at their means.
    width_z = np.sqrt(np.abs((np.arange(z_col.size)-z)**2*z_col).sum()/z_col.sum())
    x_col = data[int(z), :, int(y)] #single column through x with z and y fixed at their means.
    width_x = np.sqrt(np.abs((np.arange(x_col.size)-x)**2*x_col).sum()/x_col.sum())
    y_col = data[int(z), int(x), :] #single column through y with z and x fixed at their means.
    width_y = np.sqrt(np.abs((np.arange(y_col.size)-y)**2*y_col).sum()/y_col.sum())
    
    # Estimator height from max value.
    height = data.max()
    return z, x, y, height, width_z, width_x, width_y

############################################################################
def fitgaussian3d(data):
    """Fit a gaussian
    
    Returns (z, x, y, height, width_z, width_x, width_y)
    the gaussian parameters of a 3D distribution found by a least squares 
    fit. Wrote for 3D, but will work for 2D.
    
    Args:
        data: ndarray
            The 3D data to fit in shape [z,x,y]
    Returns:
        opt: OptimizeResult
            opt.x: parameters of the fit: (z, x, y, height, width_z, width_x, width_y)
            opt.success: boolean: whether fit exited successfully
    """
    
    params = moments3d(data)
    # Error function is simple difference between gaussian function and data.
    errorfunction = lambda p: np.ravel(gaussian3d(*p)(*np.indices(data.shape)) -
                                 data)
    opt = optimize.least_squares(errorfunction, params, bounds=([0,0,0,-np.inf, -np.inf,-np.inf,-np.inf],[data.shape[0]-1,data.shape[1]-1,data.shape[2]-1,np.inf,np.inf,np.inf,np.inf]))
    # Make all widths positive (negative values are equally valid but less useful downstream).
    for i in range(4,7):
        opt.x[i] = abs(opt.x[i])
    return opt

def fit_viewable(data, p):
    f = gaussian3d(*p)
    x, y, z = np.indices(data.shape)
    fit = f(x, y, z)
    return fit