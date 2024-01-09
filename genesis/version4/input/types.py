import pathlib
from decimal import Decimal

from typing import Union

AnyPath = Union[pathlib.Path, str]
Float = Union[Decimal, float]
ValueType = Union[int, Float, bool]
