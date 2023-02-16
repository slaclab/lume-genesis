from .tools import fstr, isotime, native_type

import numpy as np


# ----------------------------
# Basic archive metadata


def genesis_init(h5, version=None):
    """
    Set basic information to an open h5 handle

    """

    if not version:
        from genesis import __version__

        version = __version__

    d = {
        "dataType": "lume-genesis",
        "software": "lume-genesis",
        "version": version,
        "date": isotime(),
    }
    for k, v in d.items():
        h5.attrs[k] = fstr(v)


def is_genesis_archive(h5, key="dataType", value=np.string_("lume-genesis")):
    """
    Checks if an h5 handle is a lume-genesis archive
    """
    return key in h5.attrs and h5.attrs[key] == value


def find_genesis_archives(h5):
    """
    Searches one level for a valid genesis archive.
    """
    if is_genesis_archive(h5):
        return ["./"]
    else:
        return [g for g in h5 if is_genesis_archive(h5[g])]


# ------------------------------------------
# Basic tools


def write_attrs_h5(h5, data, name=None):
    """
    Simple function to write dict data to attribues in a group with name
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    for key in data:
        g.attrs[key] = data[key]
    return g


def read_attrs_h5(h5):
    """
    Simple read attributes from h5 handle
    """
    d = dict(h5.attrs)

    # Convert to native types
    for k, v in d.items():
        d[k] = native_type(v)

    return d


def write_datasets_h5(h5, labeled_data, name=None):
    """
    Write dict of arrays to h5
    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    for key, val in labeled_data.items():
        g[key] = val


def read_datasets_h5(h5):
    """
    Simple read datasets from h5 handle into numpy arrays
    """
    d = {}
    for k in h5:
        d[k] = h5[k][:]
    return d


def write_list_h5(h5, list_of_dicts, name=None):
    """
    Writes a list of dicts.

    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    for i, data in enumerate(list_of_dicts):
        write_attrs_h5(g, data, name=str(i))


def read_list_h5(h5):
    """
    Read list of dicts from h5 file.

    A list is a group of groups named with their index,
    and attributes as the data.

    The format corresponds to that written in write_list_h5
    """

    # Convert to ints for sorting
    ixlist = sorted([int(k) for k in h5])
    # Back to strings
    ixs = [str(i) for i in ixlist]
    eles = []
    for ix in ixs:
        e = read_attrs_h5(h5[ix])
        eles.append(e)
    return eles


# ------------------------------------------
# Lattice


def write_lattice_h5(h5, lattice, name="lattice"):
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    write_attrs_h5(g, lattice["param"], name="param")
    write_list_h5(g, lattice["eles"], name="eles")


def read_lattice_h5(h5):
    lat_param = read_attrs_h5(h5["param"])
    eles = read_list_h5(h5["eles"])
    return {"param": lat_param, "eles": eles}


# ------------------------------------------
# Input


def write_input_h5(h5, input, name="input"):
    """
    Write header

    Note that the filename ultimately needs to be ImpactT.in

    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    # ImpactT.in as text
    param = input["param"]
    lattice = input["lattice"]
    beam = input["beam"]

    # Main input
    write_attrs_h5(g, param, name="param")

    if lattice:
        write_lattice_h5(g, lattice, name="lattice")

    if beam:
        write_datasets_h5(g, beam, name="beam")


def read_input_h5(h5):
    """
    Read all Genesis input from h5 handle.
    """
    d = {"lattice": None, "beam": None}  # defaults

    # Required
    d["param"] = read_attrs_h5(h5["param"])

    if "lattice" in h5:
        d["lattice"] = read_lattice_h5(h5["lattice"])

    if "beam" in h5:
        d["beam"] = read_datasets_h5(h5["beam"])

    return d


# ------------------------------------------
# Output


def write_output_h5(h5, output, name="output", verbose=False):
    """
    Writes all output to an open h5 handle.

    """
    if name:
        g = h5.create_group(name)
    else:
        g = h5

    found = []
    if "param" in output:
        found.append("param")
        write_attrs_h5(g, output["param"], name="param")

    if "data" in output:
        found.append("data")
        write_datasets_h5(g, output["data"], name="data")

    if "run_info" in output:
        found.append("run_info")
        write_attrs_h5(g, output["run_info"], name="run_info")

    if verbose:
        if len(found) > 0:
            print("Archived output: " + ", ".join(found))
        else:
            print("Warning: no output found to archive")


def read_output_h5(h5, verbose=False):
    """
    Read all output from h5 handle.

    Returns a dict of:
        param
        data
    """

    d = {}

    found = []
    if "param" in h5:
        found.append("param")
        d["param"] = read_attrs_h5(h5["param"])

    if "data" in h5:
        found.append("data")
        d["data"] = read_datasets_h5(h5["data"])

    if "run_info" in h5:
        found.append("run_info")
        d["run_info"] = read_attrs_h5(h5["run_info"])

    if verbose:
        if len(found) > 0:
            print("Read output from archive: " + ", ".join(found))
        else:
            print("Warning: no output found in archive h5")

    return d
