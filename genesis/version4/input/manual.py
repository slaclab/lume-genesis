from __future__ import annotations
import ast
import pathlib

import jinja2

from typing import Dict, Optional, Set, TypedDict, Tuple, Union

AnyPath = Union[pathlib.Path, str]

MODULE_PATH = pathlib.Path(__file__).resolve().parent
dataclasses_template = MODULE_PATH / "dataclasses.tpl"

renames = {
    "l": "L",
    "lambda": "lambda_",
    # Mapping to common bmad names:
    "dx": "x_offset",
    "dy": "y_offset",
    "Phaseshifter": "PhaseShifter",
    "Importdistribution": "ImportDistribution",
    "Importbeam": "ImportBeam",
    "Importfield": "ImportField",
    "Importtransformation": "ImportTransformation",
    "SequenceFilelist": "SequenceFilelist",
}


class Parameter(TypedDict):
    """A named, single parameter for an input file."""

    name: str
    python_name: str
    type: str
    description: str
    default: Optional[Union[str, bool, int, float]]
    units: Optional[str]
    options: Optional[Set[str]]


class ManualSection(TypedDict):
    """A section of the manual, with some parameters delineated."""

    header: Optional[str]
    parameters: Dict[str, Parameter]


class LatticeManual(TypedDict):
    """A representation of either the main manual or the lattice manual."""

    elements: Dict[str, ManualSection]


def parse_manual_default(default_: str, type_: str) -> Tuple[str, Set[str]]:
    """
    Parse a single "default" value of a Genesis 4 manual parameter.

    This is the string inside of (* *) but not including units, for example:

    .. code::
        (*double, 0
        (*double, 0

    Parameters
    ----------
    default_ : str
        The default portion of the parameter.

    Returns
    -------
    str
        The default value - not yet converted to the native type.
    Set[str]
        The alternate "options" associated with the default value.  Several are
        currently supported: matched_value, profile_label, existing_field,
        sequence_label.
    """
    default = default_.strip()
    if default.lstrip("\\") == "<empty>":
        default = '""'

    options: Set[str] = set()
    for match, *add_options in [
        # Lattice-held value
        (" or matched value", "matched_value"),
        # Data taken from already-existing field at the given harmonic:
        (" or by existing field", "existing_field"),
        # Reference to label name of profile
        (" or profile label", "profile_label", "reference"),
        # Reference to label name of sequence
        (" or sequence label", "sequence_label", "reference"),
    ]:
        if match in default:
            options = options.union(add_options)
            default = default.replace(match, "")
    default = default.replace("`", "").strip().capitalize()
    if "from" in default:
        default = "None"
    if default == "0/1":
        default = "0"
    if default in ("Gamma0", "Lambda0"):
        default = None

    if not default or default == "None":
        default = {
            "double": "0.0",
            "string": "''",
        }[type_]

    if default.startswith("["):
        default = "[]"
        options.add("vector")

    if " or " in default:
        raise ValueError(f"Unhandled default option: {default}")

    if type_ == "double" and "vector" not in options:
        if "e" not in default and "." not in default:
            default = f"{default}.0"

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
    type_and_default = type_and_default.replace(", or", " or")
    info = type_and_default.split(",")
    units = None
    if len(info) == 2:
        type_, default = info
    elif len(info) == 3:
        type_, default, units = info
    else:
        raise ValueError(f"Unexpected parameter details: {info}")

    default_value, options = parse_manual_default(default, type_)

    if name == "label":
        # Require a label explicitly.
        default_value = "None"

    return {
        "name": name,
        "python_name": renames.get(name, name),
        "type": type_,
        "default": ast.literal_eval(default_value),
        "units": units.strip(" []") if units else None,
        "description": desc.strip(),
        "options": options,
    }


def parse_manual(path: AnyPath) -> LatticeManual:
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


def _to_class_name(genesis_name: str) -> str:
    """Convert a Genesis 4 manual name to a dataclass name."""
    name_chars = list(genesis_name.capitalize())
    while "_" in name_chars:
        idx = name_chars.index("_")
        name_chars.pop(idx)
        if idx < len(name_chars):
            name_chars[idx] = name_chars[idx].upper()
    class_name = "".join(name_chars)
    return renames.get(class_name, class_name)


def _custom_repr(obj) -> str:
    """A tweaked ``repr`` to always return double quotes."""
    result = repr(obj)
    return result.replace("'", '"')


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
    manual = parse_manual(path)
    with open(template_filename) as fp:
        template = fp.read()
    env = jinja2.Environment()
    env.filters["repr"] = _custom_repr
    env.filters["to_class_name"] = _to_class_name
    tpl = env.from_string(template)

    if "undulator" in manual["elements"]:
        base_class = "BeamlineElement"
    elif "setup" in manual["elements"]:
        base_class = "NameList"
    else:
        raise ValueError(
            f"Unsupported manual pages; expected to see 'undulator' or 'setup' "
            f"to identify the correct page. Saw: {manual['elements']}"
        )

    return tpl.render(
        manual=manual,
        type_map={
            "string": "str",
            "double": "float",
            "integer": "int",
        },
        docstrings={
            # Lattice:
            "undulator": "Lattice beamline element: an undulator.",
            "drift": "Lattice beamline element: drift.",
            "quadrupole": "Lattice beamline element: quadrupole.",
            "corrector": "Lattice beamline element: corrector.",
            "chicane": "Lattice beamline element: chicane.",
            "phaseshifter": "Lattice beamline element: phase shifter.",
            "marker": "Lattice beamline element: marker.",
            "line": "Lattice beamline element: line.",
            # Main input:
        },
        base_class=base_class,
    ).strip()


def cli_entrypoint():
    import sys

    try:
        filename = sys.argv[1]
    except IndexError:
        print(f"Usage: {sys.argv[0]} manual-filename.md", file=sys.stderr)
        sys.exit(1)

    print(make_dataclasses_from_manual(filename))


if __name__ == "__main__":
    cli_entrypoint()
