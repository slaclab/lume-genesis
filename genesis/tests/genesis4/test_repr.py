import textwrap
from typing import Generator, Union

import numpy as np
import pytest

from ... import tools
from ...tools import DisplayOptions
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


@pytest.fixture(
    params=[
        ("html", True),
        ("markdown", True),
        ("genesis", True),
        ("repr", True),
        ("html", False),
        ("markdown", False),
        ("genesis", False),
    ],
    ids=[
        "html-with-desc",
        "markdown-with-desc",
        "genesis-with-desc",
        "repr_pretty",
        "html-no-desc",
        "markdown-no-desc",
        "genesis-no-desc",
    ],
)
def display_options(
    request: pytest.FixtureRequest,
) -> Generator[DisplayOptions, None, None]:
    mode, desc = request.param
    opts = DisplayOptions(
        jupyter_render_mode=mode,
        console_render_mode=mode if mode != "html" else "genesis",
        include_description=desc,
    )
    tools.global_display_options = opts
    yield opts
    tools.global_display_options = DisplayOptions()


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
        # pytest.param(InitialParticles(filename='...'), id="InitialParticles-File"),
        pytest.param(Lattice(), id="Lattice"),
        pytest.param(Line(), id="Line"),
        pytest.param(
            ProfileArray(label="label", xdata=[0], ydata=[0]), id="ProfileArray"
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
    ],
)
def test_render(
    display_options: DisplayOptions, obj: Union[BeamlineElement, NameList]
) -> None:
    print("Render options", display_options)
    print("Using __str__:")
    print(str(obj))
    assert "<pre" not in str(obj)
    print("Using _repr_html_:")
    print(obj._repr_html_())
    assert "<" in obj._repr_html_()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("abcdef", "abcdef"),
        ("abcdefghijkl", "abcdef..."),
    ],
)
def test_truncate_string(
    value: str,
    expected: str,
) -> None:
    assert tools._truncated_string(value, 6) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param({}, "{}", id="empty-dict"),
        pytest.param(
            {"a": 3},
            """{'a': 3}""",
            id="simple-dict",
        ),
        pytest.param(
            {"a": 3, "b": {"c": "d"}},
            """\
            {
              'a': 3,
              'b': {'c': 'd'},
            }
            """,
            id="simple-dict2",
        ),
        pytest.param(
            {"a": 3, "b": {"c": "d", "e": [1, 2, 3, 4]}},
            """\
            {
              'a': 3,
              'b': {
                'c': 'd',
                'e': [
                  1,
                  2,
                  3,
                  4,
                ],
              },
            }
            """,
            id="dict-with-list",
        ),
        pytest.param(
            {"a": 3, "b": {"c": "d", "e": (1, 2, 3, 4)}},
            """\
            {
              'a': 3,
              'b': {
                'c': 'd',
                'e': (
                  1,
                  2,
                  3,
                  4,
                ),
              },
            }
            """,
            id="dict-with-tuple",
        ),
    ],
)
def test_repr_pretty_dict(
    value,
    expected: str,
) -> None:
    expected = textwrap.dedent(expected)
    print("Expected:")
    print("---------")
    print(expected)
    print("---------")
    repr_ = tools.pretty_repr(value, newline_threshold=0)
    print("Repr:")
    print("---------")
    print(repr_)
    print("---------")
    assert repr_.rstrip() == expected.rstrip()
    assert eval(repr_) == eval(expected)


# WIP
# @pytest.mark.parametrize(
#     "value, expected",
#     [
#         pytest.param(
#             {},
#             "{}",
#             id="empty-dict"
#         ),
#     ]
# )
# def test_repr_pretty_element(
#     value,
#     expected: str,
# ) -> None:
#     expected = textwrap.dedent(expected)
#     print("Expected:")
#     print("---------")
#     print(expected)
#     print("---------")
#     repr_ = tools.pretty_repr(value, newline_threshold=0)
#     print("Repr:")
#     print("---------")
#     print(repr_)
#     print("---------")
#     assert repr_.rstrip() == expected.rstrip()
#     assert eval(repr_) == eval(expected)
