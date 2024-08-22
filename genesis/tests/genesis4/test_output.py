import copy
from typing import Union

import numpy as np
import pydantic
import pytest

from ... import version4 as g4
from ...tools import pretty_repr
from ...version4 import Genesis4
from ...version4.output import (
    Genesis4Output,
    OutputBeam,
    OutputField,
    OutputGlobal,
    OutputLattice,
    OutputMeta,
    OutputMetaVersion,
)
from ..conftest import test_root


@pytest.fixture(scope="function")
def output(
    genesis4: Genesis4,
) -> Genesis4Output:
    output = genesis4.run(raise_on_error=True)
    assert output.run.success
    return output


def test_update_aliases(
    output: Genesis4Output,
) -> None:
    output.update_aliases()


@pytest.mark.parametrize(
    ("alias", "expected_attr"),
    [
        ("beam_sigma_energy", "beam.stat.sigma_energy"),
        ("beam_sigma_x", "beam.stat.sigma_x"),
        ("field_intensity_farfield", "field.intensity_farfield"),
        ("field_globals_energy", "field.globals.energy"),
        ("field_stat_xsize", "field.stat.xsize"),
        ("globals_frequency", "globals.frequency"),
        ("beam_bunching", "beam.bunching"),
        ("beam_globals_energy", "beam.globals.energy"),
        ("beam_stat_sigma_x", "beam.stat.sigma_x"),
        ("lattice_ax", "lattice.ax"),
    ],
)
def test_alias(
    output: Genesis4Output,
    alias: str,
    expected_attr: str,
) -> None:
    assert output.alias[alias] == expected_attr
    output._get_array_info(alias)
    assert isinstance(output[alias], np.ndarray)


def test_field_harmonic_aliases(
    output: Genesis4Output,
    # alias: str,
    # expected_attr: str,
) -> None:
    old_aliases = dict(output.alias)
    output.field_harmonics[2] = copy.copy(output.field_harmonics[1])
    output.field_harmonics[3] = copy.copy(output.field_harmonics[1])
    output.update_aliases()
    new_aliases = dict(output.alias)
    assert len(new_aliases) > len(old_aliases)
    print("New aliases:", sorted(set(new_aliases) - set(old_aliases)))

    output.field_harmonics[2].xpointing = np.array([1, 2, 3])
    assert output.alias["field2_xpointing"] == "field_harmonics[2].xpointing"
    assert output["field2_xpointing"].tolist() == [1, 2, 3]

    output.field_harmonics[2].globals.xpointing = np.array([1, 2, 3])
    assert (
        output.alias["field2_globals_xpointing"]
        == "field_harmonics[2].globals.xpointing"
    )
    assert output["field2_globals_xpointing"].tolist() == [1, 2, 3]

    output.field_harmonics[3].xpointing = np.array([4, 2, 3])
    assert output.alias["field3_xpointing"] == "field_harmonics[3].xpointing"
    assert output["field3_xpointing"].tolist() == [4, 2, 3]


@pytest.mark.parametrize(
    "attr, model_cls",
    [
        ("beam", OutputBeam),
        ("field", OutputField),
        ("lattice", OutputLattice),
        ("globals", OutputGlobal),
        ("meta", OutputMeta),
        ("version", OutputMetaVersion),
    ],
)
def test_typed_dictionaries(
    genesis4: Genesis4,
    output: Genesis4Output,
    attr: str,
    model_cls: pydantic.BaseModel,
) -> None:
    model = getattr(output, attr)
    assert not model.model_extra

    for fld in model.model_fields:
        value = getattr(model, fld)
        print(f"Checking {model_cls.__name__}.{fld} = {pretty_repr(value)}")
        # NOTE: ssc_field may raise on Genesis4 < 4.6.6
        if isinstance(value, np.ndarray):
            assert len(value)
    assert not model.extra


def test_repr(
    output: Genesis4Output,
) -> None:
    print(repr(output))


def test_stat_beamsigma_smoke(
    output: Genesis4Output,
) -> None:
    assert output.beam.stat.sigma_x.shape
    assert output.beam.stat.sigma_y.shape
    assert output.beam.stat.sigma_energy.shape


def test_plot_smoke(
    output: Genesis4Output,
) -> None:
    fig = output.plot(return_figure=True)
    assert fig is not None
    fig.savefig(test_root / "test_output_plot_smoke.png")


def test_lattice_plot_smoke(
    output: Genesis4Output,
) -> None:
    ax = output.lattice.plot()
    assert ax is not None
    fig = ax.get_figure()
    assert fig is not None
    fig.savefig(test_root / "test_output_lattice_plot_smoke.png")


def test_mock_load_failure(
    genesis4: Genesis4,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_raises(*_, **__) -> None:
        raise ValueError("mock failure")

    monkeypatch.setattr(genesis4, "load_output", load_raises)

    with pytest.raises(ValueError) as captured:
        genesis4.run(raise_on_error=True)
    assert "mock failure" in str(captured.value)


def test_convenience_methods(
    output: Genesis4Output,
) -> None:
    output.info()
    output["beam_xsize"]

    with pytest.raises(TypeError):
        # Not a mutable mapping
        output["testing"] = np.asarray([0])


@pytest.mark.parametrize(
    ("filename", "key"),
    [
        ("test.par.h5", "test"),
        ("test.123.par.h5", 123),
        ("test.456.fld.h5", 456),
    ],
)
def test_get_file_key(
    filename: str,
    key: Union[str, int],
) -> None:
    assert g4.output.get_key_from_filename(filename) == key


@pytest.mark.parametrize(
    ("key",),
    [
        ("beam_energy",),
        ("peak_power",),
    ],
)
def test_ensure_units(
    output: Genesis4Output,
    key: str,
) -> None:
    units = output.units(key)
    print("Units for", key, "is", units)
    assert units is not None


def test_load_raw_output_smoke(
    genesis4: Genesis4,
    output: Genesis4Output,
) -> None:
    h5 = genesis4.load_raw_h5_output()
    try:
        summary = output.to_hdf_summary()
        for key in summary:
            h5[key]
    finally:
        h5.close()


def test_hdf_summary_smoke(
    output: Genesis4Output,
) -> None:
    assert "/Global" in output.to_hdf_summary()


def test_nested_load() -> None:
    meta = OutputMeta.from_hdf5_data({"beamdumps": {"intstep": 1}})
    assert meta.beamdumps.intstep.tolist() == [1]
    meta = OutputMeta.from_hdf5_data({"fielddumps": {"intstep": 1}})
    assert meta.fielddumps.intstep.tolist() == [1]


@pytest.mark.parametrize(("key",), [["power"]])
def test_unsupported_key(
    output: Genesis4Output,
    key: str,
) -> None:
    with pytest.raises(KeyError):
        output[key]
    with pytest.raises(KeyError):
        output.stat(key)
