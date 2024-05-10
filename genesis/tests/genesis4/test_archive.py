import pathlib
from typing import Union

import h5py
import numpy as np
import pytest

from ...version4 import Genesis4
from ...version4.input import (
    AlterSetup,
    Beam,
    Chicane,
    Corrector,
    Drift,
    Efield,
    Field,
    ImportBeam,
    ImportDistribution,
    ImportField,
    ImportTransformation,
    InitialParticles,
    Lattice,
    Line,
    Marker,
    PhaseShifter,
    ProfileArray,
    ProfileConst,
    ProfileFile,
    ProfileGauss,
    ProfilePolynom,
    ProfileStep,
    Quadrupole,
    SequenceConst,
    SequencePolynom,
    SequencePower,
    SequenceRandom,
    Setup,
    Sponrad,
    Time,
    Track,
    Undulator,
    Wake,
    Write,
)
from ...version4.types import BeamlineElement, NameList


@pytest.fixture
def hdf5_filename(
    tmp_path: pathlib.Path,
    request: pytest.FixtureRequest,
) -> pathlib.Path:
    return tmp_path / f"{request.node.name}.h5"


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(Chicane(), id="Chicane"),
        pytest.param(Corrector(), id="Corrector"),
        pytest.param(Drift(), id="Drift"),
        pytest.param(Marker(), id="Marker"),
        pytest.param(PhaseShifter(), id="PhaseShifter"),
        pytest.param(Quadrupole(), id="Quadrupole"),
        pytest.param(Undulator(), id="Undulator"),
        pytest.param(AlterSetup(), id="AlterSetup"),
        pytest.param(Beam(), id="Beam"),
        pytest.param(Efield(), id="Efield"),
        pytest.param(Field(), id="Field"),
        pytest.param(ImportBeam(), id="ImportBeam"),
        pytest.param(ImportDistribution(), id="ImportDistribution"),
        pytest.param(ImportField(), id="ImportField"),
        pytest.param(ImportTransformation(), id="ImportTransformation"),
        pytest.param(ProfileConst(label="label"), id="ProfileConst"),
        pytest.param(ProfileFile(label="label"), id="ProfileFile"),
        pytest.param(ProfileGauss(label="label"), id="ProfileGauss"),
        pytest.param(ProfilePolynom(label="label"), id="ProfilePolynom"),
        pytest.param(ProfileStep(label="label"), id="ProfileStep"),
        pytest.param(SequenceConst(label="label"), id="SequenceConst"),
        pytest.param(SequencePolynom(label="label"), id="SequencePolynom"),
        pytest.param(SequencePower(label="label"), id="SequencePower"),
        pytest.param(SequenceRandom(label="label"), id="SequenceRandom"),
        pytest.param(Setup(), id="Setup"),
        pytest.param(Sponrad(), id="Sponrad"),
        pytest.param(Time(), id="Time"),
        pytest.param(Track(), id="Track"),
        pytest.param(Wake(), id="Wake"),
        pytest.param(Write(), id="Write"),
        pytest.param(Lattice(), id="Lattice"),
        pytest.param(Line(), id="Line"),
        pytest.param(
            ProfileArray(label="label", xdata=[0.0], ydata=[0.0]), id="ProfileArray"
        ),
        pytest.param(
            InitialParticles(
                data={
                    "x": np.asarray([0.0]),
                    "y": np.asarray([0.0]),
                    "z": np.asarray([0.0]),
                    "px": np.asarray([0.0]),
                    "py": np.asarray([0.0]),
                    "pz": np.asarray([0.0]),
                    "t": np.asarray([0.0]),
                    "status": np.asarray([0.0]),
                    "weight": np.asarray([0.0]),
                    "species": "species",
                }
            ),
            id="InitialParticles-data",
        ),
        pytest.param(
            InitialParticles(filename=pathlib.Path("test.h5")),
            id="InitialParticles-file",
        ),
    ],
)
def test_round_trip_json(
    obj: Union[NameList, BeamlineElement],
) -> None:
    print("Object", obj)
    as_json = obj.model_dump_json()
    print("Dumped to JSON:")
    print(as_json)
    deserialized = obj.model_validate_json(as_json)
    print("Back to Python:")
    print(deserialized)
    assert obj == deserialized


def test_hdf_archive(
    genesis4: Genesis4,
    hdf5_filename: pathlib.Path,
) -> None:
    output = genesis4.run(raise_on_error=True)
    assert output.run.success

    genesis4.load_output()
    orig_input = genesis4.input
    orig_output = genesis4.output
    assert orig_output is not None

    genesis4.archive(hdf5_filename)
    genesis4.load_archive(hdf5_filename)
    assert genesis4.output is not None

    assert orig_input.model_dump_json() == genesis4.input.model_dump_json()
    assert orig_output.model_dump_json() == genesis4.output.model_dump_json()


def test_hdf_archive_using_group(
    genesis4: Genesis4,
    hdf5_filename: pathlib.Path,
) -> None:
    output = genesis4.run(raise_on_error=True)
    assert output.run.success

    genesis4.load_output()
    orig_input = genesis4.input
    orig_output = genesis4.output
    assert orig_output is not None

    with h5py.File(hdf5_filename, "w") as h5:
        genesis4.archive(h5)

    with h5py.File(hdf5_filename, "r") as h5:
        genesis4.load_archive(h5)

    assert genesis4.output is not None

    assert orig_input.model_dump_json() == genesis4.input.model_dump_json()
    assert orig_output.model_dump_json() == genesis4.output.model_dump_json()
