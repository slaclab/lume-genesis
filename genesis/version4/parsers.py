import os
import math
import h5py
import warnings
from lume.parsers.namelist import parse_simple_namelist, parse_unrolled_namelist
from lume import tools

from pmd_beamphysics.units import unit, pmd_unit, e_charge, c_light, known_unit, mec2

# Patch these into the lookup dict.
known_unit["mec2"] = pmd_unit("m_ec^2", mec2 * e_charge, "energy")

for key in ['field_energy', 'pulse_energy']:
    known_unit[key] = known_unit['J']
known_unit['peak_power'] = known_unit['W']
known_unit['m^{-1}'] = pmd_unit('1/m', 1, (-1, 0, 0, 0, 0, 0, 0))
known_unit['m^{-2}'] = pmd_unit('1/m^2', 1, (-2, 0, 0, 0, 0, 0, 0))
known_unit['{s}'] = known_unit['s']
known_unit['ev'] = known_unit['eV']


def expand_path(file, path=None):
    if not os.path.isabs(file):
        file = os.path.join(path, file)
    if not os.path.exists(file):
        raise FileNotFoundError(file)
    return tools.full_path(file)


def parse_main_input(filename, expand_paths=True):
    lines = parse_simple_namelist(filename, commentchar="#")
    names, dicts = parse_unrolled_namelist(lines, end="&end", commentchar="#")
    main = []
    for n, d in zip(names, dicts):
        d2 = {"type": n}
        d2.update(d)
        main.append(d2)

    if not expand_paths:
        return main

    # Expand paths
    path, _ = os.path.split(tools.full_path(filename))
    for d in main:
        if d["type"] == "profile_file":
            for k in ["xdata", "ydata"]:
                d[k] = read_genesis4_h5filegroup(d[k], path=path)
        elif d["type"] == "setup":
            for k in ["lattice"]:
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
    parts = filegroup.split("/")  # Windows not supported?

    if os.path.isabs(filegroup):
        file = "/"
    else:
        file = path
    for i, s in enumerate(parts):
        file = file + "/" + s
        if h5py.is_hdf5(file):
            break

    dataset = "/".join(parts[i + 1 :])

    return file, dataset


def try_pmd_unit(unit_str):
    """
    Form a pmd_unit from a unit string
    """
    s = unit_str.strip()
    if s == "":
        return None
    elif s == "mc^2":
        s = "mec2"  # electrons here
    try:
        u = unit(s)
    except:
        warnings.warn(f"unknown unit '{s}'")
        u = None
    return u

EXTRA_UNITS = {"zplot": "m"}    

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
            key = node.name.strip("/")
            dat = node[:]
            if dat.shape == (1,):
                dat = dat[0]
            if isinstance(dat, bytes):
                dat = dat.decode("utf-8")
            data[key] = dat

            if "unit" in node.attrs:
                u = node.attrs["unit"].decode("utf-8")
                u = try_pmd_unit(u)
                if u:
                    unit[key] = u
        else:
            # node is a group
            pass

    # Add in extra
    for k, v in EXTRA_UNITS.items():
        unit[k] = try_pmd_unit(v)

    h5.visititems(visitor_func)

    return data, unit


def extract_aliases(output_dict):
    """
    Forms a convenient alias dict for output keys
    """
    output_alias = {}
    veto = {}
    for key in output_dict:
        ks = key.split("/")
        if len(ks) < 2:
            continue
        k = ks[-1]
        if k in veto:
            veto[k].append(key)
        if k in output_alias:
            veto[k] = [key, output_alias.pop(k)]
        else:
            output_alias[k] = key

    # Expand vetos
    for _, keys in veto.items():
        for key in keys:
            output_alias[key.replace("/", "_").lower()] = key

    return output_alias


def dumpfile_step(fname):
    """
    returns an int corresponding to the step extracted from 
    a filename that ends with '.fld.h5' or '.par.h5'
    
    If there is no int, the filename hash will be used.
    
    This is useful in sorting:
        sorted(list_of_filenames, key = lambda k: dumpfile_step(k))
    """
    _, f = os.path.split(fname)
    for suffix in ('.fld.h5', '.par.h5'):
        if f.endswith(suffix):
            f = f[:-len(suffix)]
            break
    if '.' in f:
        tail = f.split('.')[-1]
        if tail.isdigit():
            return int(tail)
        else:
            return hash(tail) 
    else:
        return hash(f)
