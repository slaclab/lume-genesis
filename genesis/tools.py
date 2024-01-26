import dataclasses
import datetime
import enum
import functools
import importlib
import logging
import subprocess
import sys
import traceback
from numbers import Number

import h5py
import numpy as np
from typing import Union

from pmd_beamphysics import ParticleGroup

logger = logging.getLogger(__name__)


class OutputMode(enum.Enum):
    """Jupyter Notebook output support."""

    unknown = "unknown"
    plain = "plain"
    html = "html"


def execute(cmd, cwd=None):
    """
    Constantly print Subprocess output while process is running
    from: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running

    # Example usage:
        for path in execute(["locate", "a"]):
        print(path, end="")

    Useful in Jupyter notebook

    """
    popen = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


# Alternative execute
def execute2(cmd, timeout=None, cwd=None):
    """
    Execute with time limit (timeout) in seconds, catching run errors.
    """

    output = {"error": True, "log": ""}
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=timeout,
            cwd=cwd,
        )
        #  p = subprocess.run(' '.join(cmd), shell=True,
        # stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        # universal_newlines=True, timeout = timeout)
        output["log"] = p.stdout
        output["error"] = False
        output["why_error"] = ""
    except subprocess.TimeoutExpired as ex:
        output["log"] = ex.stdout + "\n" + str(ex)
        output["why_error"] = "timeout"
    except Exception:
        output["log"] = "unknown run error"
        error_str = traceback.format_exc()
        output["why_error"] = str(error_str)
    return output


def namelist_lines(namelist_dict, start="&name", end="/"):
    """
    Converts namelist dict to output lines, for writing to file
    """
    lines = []
    lines.append(start)
    # parse
    for key, value in namelist_dict.items():
        # if type(value) == type(1) or type(value) == type(1.): # numbers
        if isinstance(value, Number):  # numbers
            line = key + " = " + str(value)
        elif isinstance(value, list):  # lists
            liststr = ""
            for item in value:
                liststr += str(item) + " "
            line = key + " = " + liststr
        elif isinstance(value, str):  # strings
            line = (
                key + " = " + "'" + value.strip("''") + "'"
            )  # input may need apostrophes
        else:
            # Skip
            # print('skipped: key, value = ', key, value)
            continue
        lines.append(line)

    lines.append(end)
    return lines


def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)


def native_type(value):
    """
    Converts a numpy type to a native python type.
    See:
    https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998
    """
    return getattr(value, "tolist", lambda: value)()


def isotime():
    """UTC to ISO 8601 with Local TimeZone information without microsecond"""
    return (
        datetime.datetime.utcnow()
        .replace(tzinfo=datetime.timezone.utc)
        .astimezone()
        .replace(microsecond=0)
        .isoformat()
    )


@functools.cache
def get_output_mode() -> OutputMode:
    """
    Get the output mode for lume-genesis objects.

    This works by way of interacting with IPython display and seeing what
    choice it makes regarding reprs.

    Returns
    -------
    OutputMode
        The detected output mode.
    """
    if "IPython" not in sys.modules or "IPython.display" not in sys.modules:
        return OutputMode.plain

    from IPython.display import display

    class ReprCheck:
        mode: OutputMode = OutputMode.unknown

        def _repr_html_(self) -> str:
            self.mode = OutputMode.html
            return "<!-- lume-genesis detected Jupyter and will use HTML for rendering. -->"

        def __repr__(self) -> str:
            self.mode = OutputMode.plain
            return ""

    check = ReprCheck()
    display(check)
    return check.mode


def is_jupyter() -> bool:
    """Is Jupyter detected?"""
    return get_output_mode() == OutputMode.html


def import_by_name(clsname: str) -> type:
    """
    Import the given class or function by name.

    Parameters
    ----------
    clsname : str
        The module path to find the class e.g.
        ``"pcdsdevices.device_types.IPM"``

    Returns
    -------
    type
    """
    module, cls = clsname.rsplit(".", 1)
    if module not in sys.modules:
        importlib.import_module(module)

    mod = sys.modules[module]
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ImportError(f"Unable to import {clsname!r} from module {module!r}")


def store_dict_in_hdf5_file(
    h5: Union[h5py.File, h5py.Group],
    dct: dict,
) -> None:
    """
    Store a dictionary structure in an HDF5 file.

    Parameters
    ----------
    h5 : Union[h5py.File, h5py.Group]
        The file or group to store ``dct`` in.
    dct : dict
        The data to store.
    """
    for key, value in dct.items():
        if isinstance(value, dict):
            group = h5.create_group(key)
            group.attrs["python_class"] = "dict"
            store_dict_in_hdf5_file(
                group,
                dct=value,
            )
        elif isinstance(value, ParticleGroup):
            group = h5.create_group(key)
            group.attrs["python_class"] = "ParticleGroup"
            store_dict_in_hdf5_file(
                group,
                dct=value.data,
            )
        elif dataclasses.is_dataclass(value):
            group = h5.create_group(key)
            cls = type(value)
            group.attrs["python_class"] = f"{cls.__module__}.{cls.__name__}"
            store_dict_in_hdf5_file(
                group,
                dct=dataclasses.asdict(value),
            )
        elif isinstance(value, (np.ndarray, list, tuple)):
            h5[key] = value
        elif value is None:
            ...
        else:
            if isinstance(value, np.str_):
                # Known problem for h5py
                value = str(value)

            try:
                h5.attrs[key] = value
            except TypeError:
                logger.warning(
                    f"Unable to store {key} in {h5}; storing as string instead: "
                    f"{key} is of type {type(value).__name__}"
                )
                h5.attrs[key] = str(value)


def restore_from_hdf5_file(
    h5: Union[h5py.File, h5py.Group],
) -> dict:
    """
    Restore a dictionary from an HDF5 file.

    Parameters
    ----------
    h5 : Union[h5py.File, h5py.Group]
        The file or group to restore from.
    """
    result = {}

    def restore_group(attrs: dict, data: dict):
        clsname = attrs.pop("python_class", None)
        if clsname == "dict" or clsname is None:
            data.update(attrs)
            return data
        if clsname == "ParticleGroup":
            data.update(attrs)
            return ParticleGroup(data=data)

        assert isinstance(clsname, str)
        cls = import_by_name(clsname)
        return cls(**data, **attrs)

    for key, value in h5.items():
        if isinstance(value, (h5py.Group, h5py.File)):
            attrs = dict(value.attrs)
            data = restore_from_hdf5_file(value)
            result[key] = restore_group(attrs, data)
        elif isinstance(value, h5py.Dataset):
            result[key] = np.asarray(value)
        else:
            raise NotImplementedError(f"{type(value)}")

    if isinstance(h5, h5py.File):
        attrs = dict(h5.attrs)
        return restore_group(attrs, result)

    return result
