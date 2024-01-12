from decimal import Decimal
from .types import ValueType
from typing import Iterable


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
