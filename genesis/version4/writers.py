# --------------
# lume-genesis genesis4 for Genesis 1.3 v4
#

import os
import shutil
import h5py

import numpy as np


from genesis.writers import pmd_init, dim_m

from scipy.constants import Planck, speed_of_light, elementary_charge

from lume.parsers.namelist import namelist_lines

# ------------------
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
    # --------
    nx, ny, nz = dfl.shape
    assert nx == param["gridpoints"]
    assert ny == param["gridpoints"]

    # x grid (y is the same)

    dx = param["gridsize"]
    if dx == 0:
        raise ValueError("gridsize zero!!!")

    # The grid should be symmetry about zero
    xoffset = -(nx - 1) * dx / 2

    # z grid
    dz = param["slicespacing"]
    zoffset = -dz * nz / 2

    grid_attrs = {
        "geometry": "cartesian",
        "axisLabels": ("x", "y", "z"),
        "gridSpacing": (dx, dx, dz),
        "gridGlobalOffset": (xoffset, xoffset, zoffset),
        "gridUnitSI": (1.0, 1.0, 1.0),
        "gridUnitDimension": (dim_m, dim_m, dim_m),
    }

    # Photon energy
    # --------
    Planck_eV = Planck / elementary_charge  # Planck constant from scipy is in J
    frequency = speed_of_light / param["wavelength"]
    photon_energy_eV = Planck_eV * frequency

    Z0 = np.pi * 119.9169832  # V^2/W exactly

    # grid_attrs['frequency'] = frequency
    # grid_attrs['frequencyUnitSI'] = 1.0
    # grid_attrs['frequencyUnitDimension'] = (0,0,-1,0,0,0,0)

    grid_attrs["photonEnergy"] = photon_energy_eV
    grid_attrs["photonEnergyUnitSI"] = elementary_charge  # eV -> J
    grid_attrs["photonEnergyUnitDimension"] = (2, 1, -2, 0, 0, 0, 0)  # J

    # electricField (complex)
    # --------
    # Record
    E_complex = g.create_group("electricField")
    E_complex.attrs["unitDimension"] = (1, 1, -3, -1, 0, 0, 0)  # V/m
    E_complex.attrs["timeOffset"] = 0.0
    # Add grid attrs
    for k, v in grid_attrs.items():
        E_complex.attrs[k] = v
    # components
    E_complex["x"] = dfl
    E_complex["x"].attrs["unitSI"] = np.sqrt(2 * Z0) / dx  # sqrt(W) -> V/m
    E_complex["x"].attrs["unitSymbol"] = "V/m"


def write_openpmd_wavefront_h5(h5, dfl=None, param=None, meshesPath="meshes"):
    """
    Writes a proper openPMD-wavefront to an open h5 handle.

    https://github.com/PaNOSC-ViNYL/openPMD-standard/blob/upcoming-2.0.0/EXT_WAVEFRONT.md


    """
    pmd_init(h5, meshesPath=meshesPath)

    ii = 0  # iteration
    g = h5.create_group(f"data/{ii:06}/")

    # Basic openPMD
    g.attrs["time"] = 0.0
    g.attrs["dt"] = 0.0
    g.attrs["timeUnitSI"] = 1.0

    write_wavefront_meshes_h5(g, dfl, param, name="meshes")


def write_openpmd_wavefront(h5file, dfl, param, verbose=False):
    """
    Write an openPMD wavefront from the dfl
    """

    with h5py.File(h5file, "w") as h5:
        write_openpmd_wavefront_h5(h5, dfl=dfl, param=param)

    if verbose:
        print(f"Writing wavefront (dfl data) to file {h5file}")

    return h5file


# Namelist writing
#-----------------

def write_namelists(namelists, filePath, make_symlinks=False, prefixes=['file_', 'distribution'], verbose=False):
    """
    Simple function to write namelist lines to a file
    
    If make_symlinks, prefixes will be searched for paths and the appropriate links will be made.
    For Windows, make_symlinks is ignored and it is always False.See note at https://docs.python.org/3/library/os.html#os.symlink .
    """
    # With Windows 10, users need Administator Privileges or run on Developer mode
    # in order to be able to create symlinks.
    # More info: https://docs.python.org/3/library/os.html#os.symlink
    if os.name == 'nt':
        make_symlinks = False

    with open(filePath, 'w') as f:
        for key in namelists:
            namelist = namelists[key]
            
            if make_symlinks:
                # Work on a copy
                namelist = namelist.copy()
                path, _ = os.path.split(filePath)
                replacements = make_namelist_symlinks(namelist, path, prefixes=prefixes, verbose=verbose)
                namelist.update(replacements)
                
                
            lines = namelist_lines(namelist, key)
            for l in lines:
                f.write(l+'\n')
                
def write_main_input(filePath, main_list):   
    
    path, _ = os.path.split(filePath)
    
    with open(filePath, 'w') as f:
        for d in main_list:
            d = d.copy()

            name = d.pop('type')
            if name == 'setup':
                src = d['lattice'] #should be absolute
                _, file = os.path.split(src)
                dst = os.path.join(path, file)
                shutil.copy(src, dst)
                d['lattice'] = file # Local file
                
            elif name == 'profile_file':
                 write_profile_files(d, path, replace=True)
            
            f.write('\n')
            lines = namelist_lines(d, name, end='&end', strip_strings=True)
            for line in lines:
                f.write(line+'\n')    
                
                
def write_profile_files(profile_dict, path, replace=False):
    """
    Write data from profile files 
    
    If replace, will replace the original dict with 
    Genesis4 style HDF5 filegroup strings
    """
    localfile = profile_dict['label']+'.h5'
    file = os.path.join(path, localfile)
    with h5py.File(file, 'w') as h5:
        for k in ['xdata', 'ydata']:
            h5[k] = profile_dict[k]   
            if replace:
                profile_dict[k] = f'{localfile}/{k}'