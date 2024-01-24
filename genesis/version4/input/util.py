import dataclasses
import uuid

from decimal import Decimal
from .types import ValueType
from typing import Iterable, Dict, List

# Genesis manual to Python attribute naming map:
renames = {
    "l": "L",
    "lambda": "lambda_",
    # Mapping to common bmad names:
    "dx": "x_offset",
    "dy": "y_offset",
}


class HiddenDecimal(Decimal):
    """
    A "Decimal" which hides itself in its representation.

    For example, ``repr(Decimal("10.0")) = 10.0``.

    Decimal is convenient for us to use to ensure that round-tripping input
    floating point values back to the original file format does not result in a
    significant change in representation.
    """

    def __repr__(self) -> str:
        dec_str = super().__str__()
        return dec_str.replace("Decimal('", "").rstrip("')")


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
    if isinstance(value, (float, int, Decimal)):
        return str(value)
    # raise NotImplementedError(type(value))
    return str(value)


def get_temporary_filename(extension: str = ".h5") -> str:
    """Get a temporary filename for use with Genesis 4 inputs."""
    random_start = str(uuid.uuid4())[:8]
    return "".join((random_start, extension))


def get_non_default_attrs(obj: object) -> Dict[str, ValueType]:
    """Dictionary of non-default parameters of an annotated [data]class."""
    data = {}
    for attr in obj.__annotations__:
        if attr.startswith("_"):
            continue
        value = getattr(obj, attr)
        default = getattr(type(obj), attr, None)
        # Whether Genesis will interpret them as the same means we
        # care about the string representation rather than the Python
        # type.
        if str(value) != str(default):
            data[attr] = value
    return data


def _indent_parameters(
    parameters: List[str], prefix: str, suffix: str, indent: int = 4
) -> str:
    full_length = sum(len(param) for param in parameters)
    if parameters and (len(parameters) > 3 or full_length > 40):
        join_by = "".join((",\n", " " * indent))
    else:
        join_by = ", "
    result = join_by.join(parameters)
    return "".join((prefix, result.rstrip(","), suffix))


def get_non_default_repr(obj: object, indent: int = 4) -> str:
    def format_value(value, indent):
        if dataclasses.is_dataclass(value):
            return get_non_default_repr(value, indent=indent + 4)
        if isinstance(value, dict):
            return _indent_parameters(
                prefix="{",
                parameters=[
                    f'"{key}": {format_value(val, indent)}'
                    for key, val in value.items()
                ],
                suffix="}",
            )
        return repr(value)

    parameters = list(
        f"{name}={format_value(value, indent)}"
        for name, value in get_non_default_attrs(obj).items()
    )
    formatted = _indent_parameters(
        prefix=f"{obj.__class__.__name__}(", parameters=parameters, suffix=")"
    )
    return formatted
