from __future__ import annotations
import ast
import pathlib

import jinja2

from typing import Dict, Optional, Set, TypedDict, Tuple, Union

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


class ManualSection(TypedDict):
    header: Optional[str]
    parameters: Dict[str, Parameter]


class LatticeManual(TypedDict):
    elements: Dict[str, ManualSection]


def parse_manual_default(default_: str, type_: str) -> Tuple[str, Set[str]]:
    """
    Parse a single "default" value of a Genesis 4 manual parameter.

    This

    Parameters
    ----------
    default_ : str
        The default portion of the parameter.

    Returns
    -------
    str
        The default value - not yet converted to the native type.
    Set[str]
        The "options" associated with the default value.  Several are
        supported: matched_value, profile_label, existing_field.
    """
    default = default_.strip()
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

    if " or " in default:
        raise ValueError(f"Unhandled default option: {default}")
    return default, options


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
    name = line.strip("` ").split("`")[0]
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

    default, options = parse_manual_default(default, type_)

    return {
        "name": name,
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
    elements = manual["elements"]

    def combine_lines(start: int) -> str:
        result = []
        for line in lines[start:]:
            if not line.startswith(" ") or not line.strip():
                break
            result.append(line)
        return " ".join(result)

    header = []
    element: Optional[ManualSection] = None
    for idx, line in enumerate(lines):
        if line.startswith("#"):
            section = line.lstrip("# ")
            header = []
            element = {
                "header": None,
                "parameters": {},
            }
            elements[section] = element
            continue

        if element is None:
            continue

        if line.startswith("- ") and "(*" in line:
            if element["header"] is None:
                # Everything we saw before the first parameter is what
                # we'll call the header:
                element["header"] = "\n".join(header).strip()

            # Include any next-line continuations
            line += combine_lines(idx + 1)
            param = parse_manual_parameter(line)
            element["parameters"][param["name"]] = param
        else:
            header.append(line)

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
    env = jinja2.Environment()
    env.filters["repr"] = repr
    tpl = env.from_string(
        template,
    )
    return tpl.render(
        manual=manual,
        type_map={
            "string": "str",
            "double": "Float",
        },
        docstrings={
            # Lattice:
            "undulator": "Lattice beamline element: an undulator",
            "drift": "Lattice beamline element: drift",
            "quadrupole": "Lattice beamline element: quadrupole",
            "corrector": "Lattice beamline element: corrector",
            "chicane": "Lattice beamline element: chicane",
            "phaseshifter": "Lattice beamline element: phase shifter",
            "marker": "Lattice beamline element: marker",
            "line": "Lattice beamline element: line",
            # Main input:
        },
    ).strip()
