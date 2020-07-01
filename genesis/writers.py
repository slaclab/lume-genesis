import numpy as np

from scipy.constants import Planck, speed_of_light, elementary_charge

from ._version import __version__


def write_beam_file(fname, beam_columns, verbose=False):
    """
    Writes a beam file, using a dict beam_columns
    
    The header will be written as:
    ? VERSION=1.0
    ? SIZE=<length of the columns>
    ? COLUMNS <list of columns
    <data>
    
    See: genesis.parsers.parse_beam_file
    
    """
    
    # Get size
    names = list(beam_columns)
    size = len(beam_columns[names[0]])
    header=f"""? VERSION=1.0
? SIZE={size}
? COLUMNS {' '.join([n.upper() for n in names])}"""
    
    dat = np.array([beam_columns[name] for name in names]).T

    np.savetxt(fname, dat, header=header, comments='', fmt='%1.8e') # Genesis can't read format %1.16e - lines are too long?
    
    if verbose:
        print('Beam written:', fname)
    
    return header


#------------------
# openPMD-wavefront


dim_m = (1,0,0,0,0,0,0)

def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)



def pmd_init(h5, basePath='/data/%T/', meshesPath='meshes/' ):
    """
    Root attribute initialization.
    
    h5 should be the root of the file.
    """
    d = {
        'basePath':basePath,
        'dataType':'openPMD',
        'openPMD':'2.0.0',
        'openPMDextension':'wavefront',
        'meshesPath':meshesPath, 
        #
        'software':'lume-genesis',
        'softwareVersion':__version__,
#        'iterationEncoding':'groupBased'
#        'iterationFormat':???
    }
    for k,v in d.items():
        h5.attrs[k] = fstr(v)
     

    
def pmd_wavefront_init(h5, photon_energy=0):
    """
    Set openPMD-wavefront specific attributes
    
    """
    # 
    
    wavefront_series_attrs = {
        # optiona
        'beamline':'(optional) The string representation of the optical beamline',
        'temporal domain':'time',
        'spatial domain': 'r',
        'z coordinate': 0.0,
        'photon energy': photon_energy,
        'radius of curvature in x': 0.0,
        'radius of curvature in y': 0.0,
        'Delta radius of curvature in x': 0.0,
        'Delta radius of curvature in y': 0.0
    }
    for k, v in wavefront_series_attrs.items():
        h5.attrs[k] = v      
        
        
        
        
def write_wavefront_meshes_h5(h5, dfl, param, name=None):
    """
    Write genesis dfd data to an open H5 handle.
    
    dfl: 3d complex dfl grix with shape (nx, ny, nz)
    param: Genesis parameter dict. This routine extracts:
        ncar
        dgrid
        xlamds
        zsep
    to write the appropriate metadata.
    
    
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5
    
    
    # Grid
    #--------    
    nx, ny, nz = dfl.shape
    assert nx == param['ncar']
    assert ny == param['ncar']
    
    # x grid (y is the same)
    # grid goes from [-dgrid, dgrid]
        
    dx = 2*param['dgrid']/(nx-1)
    if dx == 0:
        raise ValueError('dgrid zero!!!')
        # TODO
        
    xoffset = -dx * (nx-1)/2
    
    #z grid
    dz = param['xlamds']*param['zsep']
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
    frequency = speed_of_light/param['xlamds']
    photon_energy_eV =  Planck_eV * frequency
    

    grid_attrs['frequency'] = frequency
    grid_attrs['frequencyUnitSI'] = 1.0
    grid_attrs['frequencyUnitDimension'] = (0,0,-1,0,0,0,0)
    
    grid_attrs['photonEnergy'] = photon_energy_eV
    grid_attrs['photonEnergyUnitSI'] = elementary_charge # eV -> J
    grid_attrs['photonEnergyUnitDimension'] = (2,1,-2,0,0,0,0) # J
    
    # E_real
    #--------
    # Record
    E_re = g.create_group('E_real')
    E_re.attrs['unitDimension'] = (0., 0.5, -1.5, 0., 0., 0., 0.) # (W^{1/2} / m = (kg / s^3)^{1/2}) ???
    E_re.attrs['timeOffset'] = 0.0
    # Add grid attrs
    for k, v in grid_attrs.items():
        E_re.attrs[k] = v    
    # components
    E_re['x'] = np.real(dfl)
    E_re['x'].attrs['unitSI'] = 1.0
    

    # E_imag (similar to above)
    #--------
    # Record
    E_im = g.create_group('E_imag')
    E_im.attrs['unitDimension'] = (0., 0.5, -1.5, 0., 0., 0., 0.) # (W^{1/2} / m = (kg / s^3)^{1/2}) ???
    E_im.attrs['timeOffset'] = 0.0
    # Add grid attrs
    for k, v in grid_attrs.items():
        E_im.attrs[k] = v    
    # components
    E_im['x'] = np.imag(dfl)
    E_im['x'].attrs['unitSI'] = 1.0        
    
    
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