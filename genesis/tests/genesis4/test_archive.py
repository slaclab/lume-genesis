import json
import pathlib
import time
from typing import Union

import h5py
import pytest
from pydantic import BaseModel

from ...version4 import (
    AlterSetup,
    Beam,
    Chicane,
    Corrector,
    Drift,
    Efield,
    Field,
    Genesis4,
    ImportBeam,
    ImportDistribution,
    ImportField,
    ImportTransformation,
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
from ..conftest import test_artifacts
from . import util


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

    t0 = time.monotonic()
    genesis4.archive(hdf5_filename)

    t1 = time.monotonic()
    genesis4.load_archive(hdf5_filename)

    t2 = time.monotonic()
    print("Took", t1 - t0, "s to archive")
    print("Took", t2 - t1, "s to restore")
    assert genesis4.output is not None

    # assert orig_input.model_dump_json(indent=True) == genesis4.input.model_dump_json(indent=True)
    # assert orig_output.model_dump_json(indent=True) == genesis4.output.model_dump_json(indent=True)
    orig_input_repr = repr(orig_input)
    restored_input_repr = repr(genesis4.input)
    assert orig_input_repr == restored_input_repr

    orig_output_repr = repr(orig_output)
    restored_output_repr = repr(genesis4.output)
    assert orig_output_repr == restored_output_repr

    if orig_input != genesis4.input:
        util.compare(orig_input, genesis4.input)
        assert False, "Verbose comparison should have failed?"
    if orig_output != genesis4.output:
        util.compare(orig_output, genesis4.output)
        assert False, "Verbose comparison should have failed?"

    with open(test_artifacts / "orig_input.json", "wt") as fp:
        print(json_for_comparison(orig_input), file=fp)
    with open(test_artifacts / "restored_input.json", "wt") as fp:
        print(json_for_comparison(genesis4.input), file=fp)
    with open(test_artifacts / "orig_output.json", "wt") as fp:
        print(json_for_comparison(orig_output), file=fp)
    with open(test_artifacts / "restored_output.json", "wt") as fp:
        print(json_for_comparison(genesis4.output), file=fp)

    assert json_for_comparison(orig_input) == json_for_comparison(genesis4.input)
    assert json_for_comparison(orig_output) == json_for_comparison(genesis4.output)


def json_for_comparison(model: BaseModel) -> str:
    # Assuming dictionary keys can't be assumed to be sorted
    data = json.loads(model.model_dump_json())
    return json.dumps(data, sort_keys=True, indent=True)


def test_hdf_archive_using_group(
    genesis4: Genesis4,
    request: pytest.FixtureRequest,
    # hdf5_filename: pathlib.Path,
) -> None:
    output = genesis4.run(raise_on_error=True)
    assert output.run.success

    genesis4.load_output()
    orig_input = genesis4.input
    orig_output = genesis4.output
    assert orig_output is not None

    hdf5_filename = test_artifacts / f"archive-{request.node.name}.h5"
    t0 = time.monotonic()
    with h5py.File(hdf5_filename, "w") as h5:
        genesis4.archive(h5)

    t1 = time.monotonic()

    with h5py.File(hdf5_filename, "r") as h5:
        genesis4.load_archive(h5)

    t2 = time.monotonic()
    print("Took", t1 - t0, "s to archive")
    print("Took", t2 - t1, "s to restore")

    assert genesis4.output is not None

    # with open("orig_output.json", "wt") as fp:
    #     print(json_for_comparison(orig_output), file=fp)
    # with open("restored_output.json", "wt") as fp:
    #     print(json_for_comparison(genesis4.output), file=fp)

    orig_input_repr = repr(orig_input)
    restored_input_repr = repr(genesis4.input)
    assert orig_input_repr == restored_input_repr

    orig_output_repr = repr(orig_output)
    restored_output_repr = repr(genesis4.output)
    assert orig_output_repr == restored_output_repr
