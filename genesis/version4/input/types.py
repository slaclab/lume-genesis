from __future__ import annotations
import pathlib

import numpy as np
from typing import TYPE_CHECKING, Sequence, TypedDict, Union

if TYPE_CHECKING:
    from .core import Reference


AnyPath = Union[pathlib.Path, str]
ValueType = Union[int, float, bool, str, "Reference"]
ArrayType = Union[Sequence[float], np.ndarray]
Float = float


class SerializedReference(TypedDict):
    """A serialized Reference instance as a dictionary."""

    label: str
