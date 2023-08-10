# --------------
# lume-genesis genesis4 for Genesis 1.3 v4
#

import numpy as np


def load_genesis4_fields(h5):
    """
    Loads the field data into memory from an open h5 handle.

    Example usage:

    import h5py
    with h5py.File('rad_field.fld.h5', 'r') as h5:
        dfl, param = load_genesis4_fields(h5)

    Returns tuple (dfl, param) where

        dfl is a 3d complex dfl grid with shape (nx, ny, nz)

        param is a dict with:
            gridpoints:    number of gridpoints in one transverse dimension, equal to nx and ny above
            gridsize:      gridpoint spacing (meter)
            refposition:   starting position (meter)
            wavelength:    radiation wavelength (meter)
            slicecount:    number of slices
            slicespacing   slice spacing (meter)

        These params correspond to v2 params:
            gridpoints:   ncar
            gridsize:     dgrid*2 / (ncar-1)
            wavelength:   xlamds
            slicespacing: xlamds * zsep


    """

    # Get params
    param = {
        key: h5[key][0]
        for key in [
            "gridpoints",
            "gridsize",
            "refposition",
            "wavelength",
            "slicecount",
            "slicespacing",
        ]
    }

    # transverse grid points in each dimension
    nx = param["gridpoints"]

    # slice list
    slist = sorted(
        [
            g
            for g in h5
            if g.startswith("slice") and g not in ["slicecount", "slicespacing"]
        ]
    )

    # Note from Sven:
    #   The order of the 1D array of the wavefront is with the x coordinates as the inner loop. 
    #   So the order is (x1,y1),(x2,y1), ... (xn,y1),(x1,y2),(x2,y2),.....
    #   This is done int he routine getLLGridpoint in the field class.
    # Therefore the transpose is needed below
    
    dfl = np.stack(
        [
            h5[g]["field-real"][:].reshape(nx, nx).T
            + 1j * h5[g]["field-imag"][:].reshape(nx, nx).T
            for g in slist
        ],
        axis=-1,
    )

    return dfl, param
