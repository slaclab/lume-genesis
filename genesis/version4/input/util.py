import uuid

from ..types import ValueType
from typing import Iterable


def python_to_namelist_value(value: ValueType) -> str:
    """
    Convert a Python value to its NameList representation.

    Parameters
    ----------
    value : ValueType
        The Python value to convert.

    Returns
    -------
    str
        Representation which can be used in a namelist as part of an input
        file.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return " ".join(v for v in value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (float, int)):
        return str(value)
    # raise NotImplementedError(type(value))
    return str(value)


def get_temporary_filename(extension: str = ".h5") -> str:
    """Get a temporary filename for use with Genesis 4 inputs."""
    random_start = str(uuid.uuid4())[:8]
    return "".join((random_start, extension))
