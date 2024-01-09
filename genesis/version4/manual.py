from __future__ import annotations
import ast
import pathlib

import jinja2

from typing import Dict, Optional, Set, TypedDict, Union

AnyPath = Union[pathlib.Path, str]

MODULE_PATH = pathlib.Path(__file__).resolve().parent
dataclasses_template = MODULE_PATH / "dataclasses.tpl"


class Parameter(TypedDict):
    name: str
    type: str
    description: str
    default: Optional[Union[str, bool, int, float]]
    units: Optional[str]
    options: Optional[Set[str]]


class LatticeElement(TypedDict):
    parameters: Dict[str, Parameter]


class LatticeManual(TypedDict):
    elements: Dict[str, LatticeElement]


def parse_manual_parameter(line: str) -> Parameter:
    """
    Parse a single line of the manual which contains parameter information.

    Parameters
    ----------
    line : str
        The manual line.

    Returns
    -------
    Parameter
    """
    line = line.lstrip("- ")
    type_and_default = line[line.index("(*") : line.index("*)")].strip("(* ")
    desc = line.split("*)", 1)[1].lstrip(": ")
    info = type_and_default.split(",")
    units = None
    if len(info) == 2:
        type_, default = info
    elif len(info) == 3:
        type_, default, units = info
    else:
        raise ValueError(f"Unexpected parameter details: {info}")

    default = default.strip()
    if default == r"\<empty>":
        default = '""'

    options: Set[str] = set()
    for match, option in [
        (" or matched value", "matched_value"),
        (" or profile label", "profile_label"),
        (" or by existing field", "existing_field"),
    ]:
        if match in default:
            options.add(option)
            default = default.replace(match, "")
    default = default.replace("`", "").capitalize()
    if "from setup" in default:
        default = "None"

    if default == "0/1":
        default = "0"
    if default in ("Gamma0", "Lambda0"):
        default = None

    if not default:
        default = {
            "double": "0.0",
        }[type_]

    return {
        "name": line.split()[0].strip("`"),
        "type": type_,
        "default": ast.literal_eval(default),
        "units": units.strip(" []") if units else None,
        "description": desc.strip(),
        "options": options,
    }


def parse_lattice_manual(path: AnyPath) -> LatticeManual:
    """
    Parse the lattice manual for information about lattice elements.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``LATTICE.md`` manual.

    Returns
    -------
    LatticeManual
        Parsed lattice manual information.
    """
    with open(path) as fp:
        lines = fp.read().splitlines()

    section = None
    manual: LatticeManual = {"elements": {}}

    def combine_lines(start: int) -> str:
        result = []
        for line in lines[start:]:
            if not line.startswith(" ") or not line.strip():
                break
            result.append(line)
        return " ".join(result)

    for idx, line in enumerate(lines):
        if line.startswith("#"):
            section = line.lstrip("# ")
            continue

        if not section:
            continue

        if line.startswith("- ") and "(*" in line:
            # Include any next-line continuations
            line += combine_lines(idx + 1)
            element = manual["elements"].setdefault(section, {"parameters": {}})
            param = parse_manual_parameter(line)
            element["parameters"][param["name"]] = param

    return manual


def make_dataclasses_from_manual(
    path: AnyPath,
    *,
    template_filename: AnyPath = dataclasses_template,
) -> str:
    """
    Parse the manual and generate dataclass source code for it.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the manual file.

    Returns
    -------
    str
        Generated Python source code.
    """
    manual = parse_lattice_manual(path)
    with open(template_filename) as fp:
        template = fp.read()
    tpl = jinja2.Template(
        template,
    )
    return tpl.render(
        manual=manual,
        type_map={
            "string": "str",
            "double": "Float",
        },
        docstrings={
            "undulator": "Lattice beamline element: an undulator",
            "drift": "Lattice beamline element: drift",
            "quadrupole": "Lattice beamline element: quadrupole",
            "corrector": "Lattice beamline element: corrector",
            "chicane": "Lattice beamline element: chicane",
            "phaseshifter": "Lattice beamline element: phase shifter",
            "marker": "Lattice beamline element: marker",
            "line": "Lattice beamline element: line",
        },
    ).strip()
