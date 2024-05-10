from typing import Dict, List

import pydantic
import pytest

from ...version4 import Genesis4
from ...version4.output import Genesis4Output
from ...version4.parsers import extract_aliases
from ...version4.types import (
    OutputLatticeDict,
    OutputBeamDict,
    OutputGlobalDict,
    OutputMetaVersionDict,
    OutputMetaDict,
    OutputFieldDict,
)
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
    "attr, typeddict",
    [
        ("beam", OutputBeamDict),
        ("field_info", OutputFieldDict),
        ("lattice", OutputLatticeDict),
        ("global_", OutputGlobalDict),
        ("meta", OutputMetaDict),
        ("version", OutputMetaVersionDict),
    ],
)
def test_typed_dictionaries(
    output: Genesis4Output,
    attr: str,
    typeddict: type,
) -> None:
    typeddict.__pydantic_config__ = pydantic.ConfigDict(strict=True, extra="forbid")

    dct = getattr(output, attr)
    new_keys = set(dct) - set(typeddict.__annotations__)
    assert (
        not new_keys
    ), f"Found extra keys {new_keys}; upstream Genesis changes perhaps"

    adapter = pydantic.TypeAdapter(typeddict)
    adapter.validate_python(dct)


def test_stat_smoke(
    output: Genesis4Output,
) -> None:
    """Smoke test calling 'stat' on all keys"""
    for key in output.data:
        try:
            print(key, output.stat(key))
        except ValueError as ex:
            print("Failed:", key, ex)
            if "Cannot compute stat" not in str(ex):
                raise


@pytest.mark.parametrize(
    "key",
    [
        "beam_sigma_x",
        "beam_sigma_y",
        "beam_sigma_energy",
        pytest.param(
            "beam_sigma_px", marks=pytest.mark.xfail(reason="Not yet implemented")
        ),
        pytest.param(
            "beam_sigma_py", marks=pytest.mark.xfail(reason="Not yet implemented")
        ),
    ],
)
def test_stat_beamsigma_smoke(
    output: Genesis4Output,
    key: str,
) -> None:
    output.stat(key)


def test_plot_smoke(
    output: Genesis4Output,
) -> None:
    fig = output.plot(return_figure=True)
    assert fig is not None
    fig.savefig(test_root / "test_plot_smoke.png")
