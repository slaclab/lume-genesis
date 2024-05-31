import keyword
import os
import re
import warnings


import h5py
import numpy as np
import pydantic.alias_generators
from lume import tools
from lume.parsers.namelist import parse_simple_namelist, parse_unrolled_namelist
from pmd_beamphysics.units import e_charge, known_unit, mec2, pmd_unit, unit

# Patch these into the lookup dict.
known_unit["mec2"] = pmd_unit("m_ec^2", mec2 * e_charge, "energy")

for key in ["field_energy", "pulse_energy"]:
    known_unit[key] = known_unit["J"]
known_unit["peak_power"] = known_unit["W"]
known_unit["m^{-1}"] = pmd_unit("1/m", 1, (-1, 0, 0, 0, 0, 0, 0))
known_unit["m^{-2}"] = pmd_unit("1/m^2", 1, (-2, 0, 0, 0, 0, 0, 0))
known_unit["{s}"] = known_unit["s"]
known_unit["ev"] = known_unit["eV"]


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
    except Exception:
        warnings.warn(f"unknown unit '{s}'")
        u = None
    return u


EXTRA_UNITS = {
    "bunching": "1",
    "zplot": "m",
    "beam_sigma_x": "m",
    "beam_sigma_y": "m",
    "beam_sigma_energy": "mc^2",
}


def extract_data(h5):
    """
    Traverses an open h5 handle and extracts a dict of datasets and units

    Parameters
    ----------
    h5: open h5py.File handle

    Returns
    -------
    data : dict of np.array
    key_map : dict of str
    """

    def convert_dataset(node: h5py.Dataset):
        # node is a dataset
        dat = node[:]
        if dat.shape == (1,):
            dat = dat[0]

        if isinstance(dat, bytes):
            return dat.decode("utf-8")
        elif isinstance(dat, np.generic):
            return dat.item()
        elif isinstance(dat, np.ndarray):
            if dat.dtype is np.str_:
                return str(dat)
            return dat
        return dat

    def convert_group(node: h5py.Group):
        data = {}
        units = {}
        key_map = {}
        for hdf_key, item in node.items():
            key = output_key_to_python_identifier(hdf_key)
            key_map[key] = hdf_key
            if isinstance(item, h5py.Group):
                data[key] = convert_group(item)
            elif isinstance(item, h5py.Dataset):
                data[key] = convert_dataset(item)

            if "unit" in item.attrs and key not in units:
                node_units = item.attrs["unit"].decode("utf-8")
                node_units = try_pmd_unit(node_units)
                if node_units:
                    units[key] = node_units
        if units:
            data["units"] = units
        if key_map:
            data["hdf_key_map"] = key_map
        return data

    return convert_group(h5)


def output_key_to_python_identifier(key: str) -> str:
    key = re.sub("[^a-zA-Z_0-9]", "_", key)
    key = pydantic.alias_generators.to_snake(key)
    if keyword.iskeyword(key):
        # global -> global_
        return f"{key}_"
    return {
        "ls_cfield": "lsc_field",
        "ss_cfield": "ssc_field",
        "one_4one": "one4one",
        "gamma_0": "gamma0",
    }.get(key, key)


def dumpfile_step(fname):
    """
    returns an int corresponding to the step extracted from
    a filename that ends with '.fld.h5' or '.par.h5'

    If there is no int, the filename hash will be used.

    This is useful in sorting:
        sorted(list_of_filenames, key = lambda k: dumpfile_step(k))
    """
    _, f = os.path.split(fname)
    for suffix in (".fld.h5", ".par.h5"):
        if f.endswith(suffix):
            f = f[: -len(suffix)]
            break
    if "." in f:
        tail = f.split(".")[-1]
        if tail.isdigit():
            return int(tail)
        else:
            return hash(tail)
    else:
        return hash(f)
