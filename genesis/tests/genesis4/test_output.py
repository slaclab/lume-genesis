from typing import Dict, List

import numpy as np
import pydantic
import pytest

from ...version4 import Genesis4
from ...version4.output import (
    Genesis4Output,
    OutputLattice,
    OutputBeam,
    OutputGlobal,
    OutputMetaVersion,
    OutputMeta,
    OutputField,
)
from ...version4.parsers import extract_aliases
from ..conftest import test_root


@pytest.mark.parametrize(
    "keys, expected_aliases",
    [
        pytest.param(
            ["Beam/Global/xsize", "Beam/xsize", "Field/Global/xsize", "Field/xsize"],
            {
                "beam_global_xsize": "Beam/Global/xsize",
                "beam_xsize": "Beam/xsize",
                "field_global_xsize": "Field/Global/xsize",
                "field_xsize": "Field/xsize",
            },
            id="xsize",
        ),
        pytest.param(
            [
                "Beam/Global/xsize",
                "Beam/xsize",
                "Field/Global/xsize",
                "Field/xsize",
                "test/unique",
                "foobar",
            ],
            {
                "beam_global_xsize": "Beam/Global/xsize",
                "beam_xsize": "Beam/xsize",
                "field_global_xsize": "Field/Global/xsize",
                "field_xsize": "Field/xsize",
                "unique": "test/unique",
                "test_unique": "test/unique",
            },
            id="xsize-and-unique",
        ),
    ],
)
def test_extract_aliases(keys: List[str], expected_aliases: Dict[str, str]) -> None:
    assert extract_aliases(dict.fromkeys(keys)) == expected_aliases


@pytest.fixture(scope="function")
def output(
    genesis4: Genesis4,
) -> Genesis4Output:
    output = genesis4.run(raise_on_error=True)
    assert output.run.success
    return output


@pytest.mark.parametrize(
    "attr, model_cls",
    [
        ("beam", OutputBeam),
        ("field_info", OutputField),
        ("lattice", OutputLattice),
        ("global_", OutputGlobal),
        ("meta", OutputMeta),
        ("version", OutputMetaVersion),
    ],
)
def test_typed_dictionaries(
    output: Genesis4Output,
    attr: str,
    model_cls: pydantic.BaseModel,
) -> None:
    # typeddict.__pydantic_config__ = pydantic.ConfigDict(strict=True, extra="forbid")

    model = getattr(output, attr)
    assert not model.model_extra


def test_repr(
    output: Genesis4Output,
) -> None:
    print(repr(output))


def test_stat_smoke(
    output: Genesis4Output,
) -> None:
    """Smoke test calling 'stat' on all keys"""
    for key in output.data:
        value = output[key]
        if not isinstance(value, np.ndarray):
            print(f"Skipping non-array key: {key}")
            continue

        try:
            print(key, output.stat(key))
        except ValueError as ex:
            print("Failed:", key, ex)
            if "Cannot compute stat" not in str(ex):
                raise


def test_stat_beamsigma_smoke(
    output: Genesis4Output,
    key: str,
) -> None:
    assert output.beam.stat.x.shape
    assert output.beam.stat.y.shape
    assert output.beam.stat.energy.shape


def test_plot_smoke(
    output: Genesis4Output,
) -> None:
    fig = output.plot(return_figure=True)
    assert fig is not None
    fig.savefig(test_root / "test_output_plot_smoke.png")


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
