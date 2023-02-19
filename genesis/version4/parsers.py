import os
import h5py
import warnings
from lume.parsers.namelist import parse_simple_namelist, parse_unrolled_namelist
from lume import tools

from pmd_beamphysics.units import unit, pmd_unit, e_charge, c_light, known_unit, mec2
# Patch these into the lookup dict.
# TODO: Add to pmd_beamphysics
known_unit['eV/m'] = pmd_unit('eV/m', e_charge, (1,1,-2,0,0,0,0))
known_unit['W/m^2'] = pmd_unit('W/m^2', 1, (1,0,-3,0,0,0,0))
known_unit['W'] = pmd_unit('W', 1, (1,2,-3,0,0,0,0))
known_unit['mec2'] = pmd_unit('m_ec^2', mec2*e_charge, 'energy')




def expand_path(file, path=None):
    if not os.path.isabs(file):
        file = os.path.join(path, file)
    assert os.path.exists(file)
    return tools.full_path(file)

def parse_main_input(filename, expand_paths = True):
    lines = parse_simple_namelist(filename, commentchar='#')
    names, dicts = parse_unrolled_namelist(lines, end='&end', commentchar='#')
    main = []
    for n, d in zip(names, dicts):
        d2 = {'type':n}
        d2.update(d)
        main.append(d2)
        
    if not expand_paths:
        return main
        
    # Expand paths
    path, _ = os.path.split(tools.full_path(filename))
    for d in main:
        if d['type'] == 'profile_file':
            for k in ['xdata', 'ydata']:
                d[k] = read_genesis4_h5filegroup(d[k], path=path)
        elif d['type'] == 'setup':
            for k in ['lattice']:
                if k in d:
                    d[k] = expand_path(d[k], path)                
    return main


def read_genesis4_h5filegroup(filegroup, path=None):
    """
    read data from filegroup, 
    See: parse_genesis4_h5filegroup
    
    Returns
    -------
    data: np.array
    
    """
    file, group = parse_genesis4_h5filegroup(filegroup, path=path)
    with h5py.File(file) as h5:
        data = h5[group][:]
    return data


def parse_genesis4_h5filegroup(filegroup, path=None):
    """
    Parses special format
        '/path/to/file.h5/g1/g2/dataset'
    into
        ('/path/to/file.h5', 'g1/g2/dataset')
    by checking if the file is HDF5.
        
    If the path to filegroup is not absolute, path must be provided
    """
    parts = filegroup.split('/') # Windows not supported?
    
    if os.path.isabs(filegroup):
        file = '/'
    else:
        file = path
    for i, s in enumerate(parts):
        file  = file + '/' + s 
        if h5py.is_hdf5(file):
            break
        
    dataset = '/'.join(parts[i+1:])
    
    return file, dataset



def try_pmd_unit(unit_str):
    """
    Form a pmd_unit from a unit string
    """
    s = unit_str.strip()
    if s == '':
        return None
    elif s == 'mc^2':
        s = 'mec2' # electrons here
    try:
        u = unit(s)
    except:
        warnings.warn(f'unknown unit {s}')
        u = None
    return u



def extract_data_and_unit(h5):
    """
    Traverses an open h5 handle and extracts a dict of datasets and units
    
    Parameters
    ----------
    h5: open h5py.File handle
    
    Returns
    -------
    data: dict of np.array
    unit: dict of str
    """
    
    data = {}
    unit = {}
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
             # node is a dataset
            key = node.name.strip('/')
            data[key] = node[:]
            if 'unit' in node.attrs:
                u = node.attrs['unit'].decode('utf-8')
                u = try_pmd_unit(u)
                if u:
                    unit[key] = u
        else:
             # node is a group
            pass  
            
    h5.visititems(visitor_func)    
    
    return data, unit

