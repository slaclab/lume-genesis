import pathlib

import pytest

from ...version4.manual import (
    Parameter,
    make_dataclasses_from_manual,
    parse_lattice_manual,
    parse_manual_parameter,
)


test_v4_root = pathlib.Path(__file__).resolve().parent
main_manual = test_v4_root / "main_input.md"
lattice_manual = test_v4_root / "lattice.md"


@pytest.mark.parametrize(
    "param, expected",
    [
        pytest.param(
            "- `l` (*double, 0, [m]*): Length of the chicane, ",
            {
                "name": "l",
                "type": "double",
                "default": 0,
                "units": "m",
                "description": "Length of the chicane,",
                "options": set(),
            },
        ),
        pytest.param(
            "- `l` (*double, 0*): Length of the chicane, ",
            {
                "name": "l",
                "type": "double",
                "default": 0,
                "units": None,
                "description": "Length of the chicane,",
                "options": set(),
            },
        ),
        pytest.param(
            r"- `field` (*string, \<empty>*): attribute name for a given element. ",
            {
                "name": "field",
                "type": "string",
                "default": "",
                "units": None,
                "description": "attribute name for a given element.",
                "options": set(),
            },
        ),
        pytest.param(
            r"- `loss` (*double, 0 or profile label*): Loss in $eV/m$. This is",
            {
                "name": "loss",
                "type": "double",
                "default": 0,
                "units": None,
                "description": "Loss in $eV/m$. This is",
                "options": {"profile_label"},
            },
        ),
        pytest.param(
            r"- `betax` (*double, 15 or matched value or profile label*): Initial beta-function in",
            {
                "name": "betax",
                "type": "double",
                "default": 15,
                "units": None,
                "description": "Initial beta-function in",
                "options": {"matched_value", "profile_label"},
            },
        ),
    ],
)
def test_parse_param(param: str, expected: Parameter) -> None:
    assert parse_manual_parameter(param) == expected


def test_parse_lattice_manual() -> None:
    print(parse_lattice_manual(lattice_manual))


def test_parse_main_manual() -> None:
    print(parse_lattice_manual(main_manual))


def test_make_dataclasses() -> None:
    print(make_dataclasses_from_manual(lattice_manual))
