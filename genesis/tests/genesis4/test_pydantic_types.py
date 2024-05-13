import pathlib
from typing import Sequence
import pytest
import numpy as np
from pmd_beamphysics.units import pmd_unit
from pydantic import TypeAdapter
from ...version4.types import (
    PydanticNDArray,
    PydanticPmdUnit,
    Reference,
)


test_path = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "ref",
    [
        "@ref",
        Reference("@ref"),
        "ref",
        Reference("ref"),
    ],
)
def test_reference(ref: str) -> None:
    ref_type = TypeAdapter(Reference)
    assert ref_type.validate_python(ref, strict=True) == Reference("@ref")


@pytest.mark.parametrize(
    "unit",
    [
        pmd_unit("eV", 1.602176634e-19, (2, 1, -2, 0, 0, 0, 0)),
        pmd_unit("T"),
        pmd_unit("T", 1, (0, 1, -2, -1, 0, 0, 0)),
    ],
)
def test_pmd_unit(unit: pmd_unit) -> None:
    print("Unit:", repr(unit))
    adapter = TypeAdapter(PydanticPmdUnit)
    dumped = adapter.dump_json(unit)
    print("Dumped:", repr(dumped))
    deserialized = adapter.validate_json(dumped, strict=True)
    print("Deserialized:", repr(deserialized))
    assert unit == deserialized


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(10),
        np.arange(10.0),
        np.ones((5, 5)),
    ],
)
def test_nd_array(arr: np.ndarray) -> None:
    print("Array:", arr)
    adapter = TypeAdapter(PydanticNDArray)
    dumped = adapter.dump_json(arr)
    print("Dumped:", repr(dumped))
    deserialized = adapter.validate_json(dumped, strict=True)
    print("Deserialized:", repr(deserialized))
    np.testing.assert_allclose(arr, deserialized)


@pytest.mark.parametrize(
    "arr",
    [
        [0, 1, 2, 3],
        (0, 1, 2, 3),
    ],
)
def test_sequence_as_ndarray(arr: Sequence[float]) -> None:
    print("Array:", arr)
    adapter = TypeAdapter(PydanticNDArray)
    deserialized = adapter.validate_python(arr, strict=True)
    print("Deserialized:", repr(deserialized))
    np.testing.assert_allclose(arr, deserialized)
