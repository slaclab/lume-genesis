import pathlib
from typing import Union

import numpy as np
import pytest

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
    MainInput,
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
from .test_run import lattice, main_input  # noqa: F401


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
    main_input: MainInput,  # noqa: F811  # this is intentional; it's a fixture
    lattice: Lattice,  # noqa: F811  # this is intentional; it's a fixture
    hdf5_filename: pathlib.Path,
) -> None:
    from .test_run import test_run_with_instances

    genesis = test_run_with_instances(main_input, lattice)
    genesis.load_output()
    orig_input = genesis.input
    orig_output = genesis.output
    genesis.archive(hdf5_filename)
    genesis.load_archive(hdf5_filename)
    assert orig_input.model_dump_json() == genesis.input.model_dump_json()
    assert orig_output.model_dump_json() == genesis.output.model_dump_json()
