from __future__ import annotations
import pathlib
from decimal import Decimal

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .core import Reference


AnyPath = Union[pathlib.Path, str]
Float = Union[Decimal, float]
ValueType = Union[int, Float, bool, str, "Reference"]
