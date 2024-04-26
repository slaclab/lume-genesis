import pydantic
import pathlib
import uuid

from ..types import ValueType
from typing import Any, Callable, Iterable, Dict, List, Mapping


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


def _default_serializer(
    value: Any, type_encoders: Mapping[Any, Callable[[Any], Any]] | None = None
) -> Any:
    if isinstance(value, pathlib.Path):
        return value.as_posix()
    return value


def get_non_default_attrs(obj: pydantic.BaseModel) -> Dict[str, ValueType]:
    """Dictionary of non-default parameters of an annotated [data]class."""
    return obj.model_dump(exclude_defaults=True)


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
        if isinstance(value, pydantic.BaseModel):
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
