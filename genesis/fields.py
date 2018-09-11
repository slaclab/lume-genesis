import numpy as np
from scipy.interpolate import interpn



def interpolate_complex_slice_2d(x, y, dfl_slice, dgrid):
    """
    Interpolates 2d complex fld/dfl array at points
    
    Input
    x: list of x points to interpolate at
    y: list of y points to interpolate at
    dfl_slice: complex field grid in 2d (ncar, ncar)
    dgrid: extent of field grid from [-dgrid, dgrid] in x and y
    
    Output:
    complex field at points 
    
    """
    dat = dfl_slice
    nx = len(dat) # = ncar
    dx = 2*dgrid/(nx-1)
    xmin = -dgrid
    xlist = [dx*i + xmin for i in range(nx)]    
    # interpn only works on real data. Two calls
    xylist = np.transpose([x, y])
    redat = interpn((xlist,xlist), np.real(dat), xylist)
    imdat = interpn((xlist,xlist), np.imag(dat), xylist)
    # rejoin complex number
    return  1j*imdat + redat 

