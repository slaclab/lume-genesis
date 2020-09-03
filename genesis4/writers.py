# --------------
# lume-genesis genesis4 for Genesis 1.3 v4 
# 
import h5py

import numpy as np


from genesis.writers import fstr,  pmd_init, pmd_wavefront_init, dim_m

from scipy.constants import Planck, speed_of_light, elementary_charge

from genesis._version import __version__

import os

#------------------
# openPMD-wavefront

        
def write_wavefront_meshes_h5(h5, dfl, param, name=None):
    """
    Write genesis dfd data to an open H5 handle.
    
    dfl: 3d complex dfl grid with shape (nx, ny, nz)
    param: Genesis parameter dict. This routine extracts:
        gridpoints (ncar in v2)
        gridsize (dgrid in v2)
        wavelength (xlamds in v2)
        slicespacing (zsep in v2)
    to write the appropriate metadata.
    
    Note that the dfl file is in units of sqrt(W), 
    and needs to be divided by the grid spacing to get the openPMD-wavefront unit.
    This factor is inclueded in the unitSI factor as:
        h5['/path/to/E_real/x'].attrs['unitSI'] = 1/dx
    so that raw dfl data is not changed when writing to hdf5.
    
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
     
    # Grid
    #--------    
    nx, ny, nz = dfl.shape
    assert nx == param['gridpoints']
    assert ny == param['gridpoints']
    
    # x grid (y is the same)
    # grid goes from [-dgrid, dgrid]
        
    dx = 2*param['gridsize']/(nx-1)
    if dx == 0:
        raise ValueError('gridsize zero!!!')
        # TODO
        
    xoffset = -param['gridsize']
    
    #z grid
    dz = param['slicespacing']
    zoffset = -dz * nz / 2

    grid_attrs = {
        'geometry':'cartesian',
        'axisLabels':('x', 'y', 'z'),
        'gridSpacing': (dx, dx, dz),   
        'gridGlobalOffset': (xoffset, xoffset, zoffset),    
        'gridUnitSI': (1.0, 1.0, 1.0),
        'gridUnitDimension': (dim_m, dim_m, dim_m) 
    }
    
    # Photon energy
    #--------      
    Planck_eV = Planck/elementary_charge # Planck constant from scipy is in J
    frequency = speed_of_light/param['wavelength']
    photon_energy_eV =  Planck_eV * frequency
    
    Z0 = np.pi*119.9169832 # V^2/W exactly
    

    #grid_attrs['frequency'] = frequency
    #grid_attrs['frequencyUnitSI'] = 1.0
    #grid_attrs['frequencyUnitDimension'] = (0,0,-1,0,0,0,0)
    
    grid_attrs['photonEnergy'] = photon_energy_eV
    grid_attrs['photonEnergyUnitSI'] = elementary_charge # eV -> J
    grid_attrs['photonEnergyUnitDimension'] = (2,1,-2,0,0,0,0) # J
    
    # electricField (complex)
    #--------
    # Record
    E_complex = g.create_group('electricField')
    E_complex.attrs['unitDimension'] = (1, 1, -3, -1, 0, 0, 0) # V/m
    E_complex.attrs['timeOffset'] = 0.0
    # Add grid attrs
    for k, v in grid_attrs.items():
        E_complex.attrs[k] = v    
    # components
    E_complex['x'] = dfl
    E_complex['x'].attrs['unitSI'] = np.sqrt(2*Z0)/dx # sqrt(W) -> V/m      
    E_complex['x'].attrs['unitSymbol'] = 'V/m'
    
    
def write_openpmd_wavefront_h5(h5, dfl=None, param=None):
    """
    Writes a proper openPMD-wavefront to an open h5 handle.
    
    https://github.com/PaNOSC-ViNYL/openPMD-standard/blob/upcoming-2.0.0/EXT_WAVEFRONT.md
    

    """
    meshesPath='meshes/'
    
    pmd_init(h5,  meshesPath='meshes')    

    ii = 0 # iteration
    g = h5.create_group(f'data/{ii:06}/')    
    
    # Basic openPMD
    g.attrs['time'] = 0.0
    g.attrs['dt'] = 0.0
    g.attrs['timeUnitSI'] = 1.0
    
    write_wavefront_meshes_h5(g, dfl, param, name='meshes')  
    
    
def write_openpmd_wavefront(h5file, dfl, param, verbose=False):
    """
    Write an openPMD wavefront from the dfl
    """
        
    with h5py.File(h5file, 'w') as h5:
         write_openpmd_wavefront_h5(h5, dfl=dfl, param=param)     
    
    if verbose:
        print(f'Writing wavefront (dfl data) to file {h5file}')

    
    return h5file   