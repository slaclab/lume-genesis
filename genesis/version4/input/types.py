from __future__ import annotations
import pathlib
from decimal import Decimal

import numpy as np
from typing import TYPE_CHECKING, Sequence, TypedDict, Union

if TYPE_CHECKING:
    from .core import Reference


AnyPath = Union[pathlib.Path, str]
Float = Union[Decimal, float]
ValueType = Union[int, Float, bool, str, "Reference"]
ArrayType = Union[Sequence[Decimal], np.ndarray]


class SerializedReference(TypedDict):
    """A serialized Reference instance as a dictionary."""

    label: str
