import pathlib
from pmd_beamphysics import ParticleGroup
import pytest
import numpy as np
from pmd_beamphysics.units import pmd_unit
from pydantic import TypeAdapter
from ...version4.types import (
    PydanticNDArray,
    PydanticParticleGroup,
    PydanticPmdUnit,
)


test_path = pathlib.Path(__file__).resolve().parent


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


def test_particle_group() -> None:
    group = ParticleGroup(
        data=dict(
            x=[0],
            px=[0],
            y=[0],
            py=[0],
            z=[0],
            pz=[0],
            t=[0],
            status=[0],
            weight=[0],
            species="species",
        )
    )
    filename = test_path / "simple_particle_group.h5"
    group.write(str(filename))

    group = ParticleGroup(h5=str(filename))
    print("Group:", group)
    adapter = TypeAdapter(PydanticParticleGroup)
    dumped = adapter.dump_json(group)
    print("Dumped:", repr(dumped))
    deserialized = adapter.validate_json(dumped, strict=True)
    print("Deserialized:", repr(deserialized))
    assert group == deserialized
