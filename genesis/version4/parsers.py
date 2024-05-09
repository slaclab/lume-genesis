import os
from typing import Any, Dict
import warnings

import h5py
import numpy as np
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

    def visitor_func(_name, node):
        if isinstance(node, h5py.Group):
            return

        if isinstance(node, h5py.Dataset):
            # node is a dataset
            key = node.name.strip("/")
            dat = node[:]
            if dat.shape == (1,):
                dat = dat[0]

            if isinstance(dat, bytes):
                data[key] = dat.decode("utf-8")
            elif isinstance(dat, np.integer):
                data[key] = int(dat)
            elif isinstance(dat, np.floating):
                data[key] = float(dat)
            elif isinstance(dat, np.bool_):
                data[key] = bool(dat)
            elif isinstance(dat, np.unicode_):
                data[key] = str(dat)
            elif isinstance(dat, np.ndarray):
                if dat.dtype is np.str_:
                    data[key] = str(dat)
                else:
                    data[key] = dat
            else:
                data[key] = dat

            if "unit" in node.attrs:
                u = node.attrs["unit"].decode("utf-8")
                u = try_pmd_unit(u)
                if u:
                    unit[key] = u

    # Add in extra
    for k, v in EXTRA_UNITS.items():
        unit[k] = try_pmd_unit(v)

    h5.visititems(visitor_func)

    return data, unit


def output_key_to_python_identifier(key: str) -> str:
    return key.replace("/", "_").lower()


def extract_aliases(output_dict: Dict[str, Any]) -> Dict[str, str]:
    """
    Forms a convenient alias dict for output keys
    """
    # Include all `start/middle/last` keys as `start_middle_last`:
    aliases = {
        output_key_to_python_identifier(key): key for key in output_dict if "/" in key
    }

    # For keys of the form:
    #   `start/middle/LAST`
    # Track all keys which have the same "LAST" part.
    # These are not unique and cannot have unqualified aliases without a
    # `start_`-like prefix.
    by_last_part = {}
    for key in output_dict:
        last_part = output_key_to_python_identifier(key.split("/")[-1])
        by_last_part.setdefault(last_part, []).append(key)

    for last_part, keys in by_last_part.items():
        if len(keys) > 1:
            continue
        (key,) = keys
        if key != last_part:
            aliases[last_part] = key
    return aliases


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
