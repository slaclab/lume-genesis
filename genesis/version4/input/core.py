from __future__ import annotations
import dataclasses
from decimal import Decimal
import pathlib

from typing import Iterable, List, Optional, Union
import typing
from .types import Float, ValueType, AnyPath

if typing.TYPE_CHECKING:
    from .generated_lattice import BeamlineElement
    from .generated_main import NameList


LineItem = Union[str, "DuplicatedLineItem", "PositionedLineItem"]


@dataclasses.dataclass
class DuplicatedLineItem:
    """
    A Genesis 4 lattice Line item which is at a certain position.

    Attributes
    ----------
    label : str
        The name/label of the line item.
    count : int
        The number of times to repeat the line item.
    """

    label: str
    count: int

    def __str__(self) -> str:
        return f"{self.count}*{self.label}"


@dataclasses.dataclass
class PositionedLineItem:
    """
    A Genesis 4 lattice Line item which is at a certain position.

    Attributes
    ----------
    label : str
        The name/label of the line item.
    position : Float
        The position of the element.
    """

    label: str
    position: Float

    def __str__(self) -> str:
        return f"{self.label}@{self.position}"


@dataclasses.dataclass
class Line:
    """
    A Genesis 4 beamline Line.

    Attributes
    ----------
    label : str
        The name/label of the line.
    elements : list of LineItem
        Elements contained in the line.
    """

    label: str
    elements: List[LineItem] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        elements = ", ".join(str(element) for element in self.elements)
        return "".join(
            (
                self.label,
                ": LINE = {",
                elements,
                "};",
            )
        )


@dataclasses.dataclass
class Lattice:
    """
    A Genesis 4 beamline Lattice configuration.

    Attributes
    ----------
    elements : list of BeamlineElement or Line
        Elements contained in the lattice.
    filename : pathlib.Path or None
        The filename from which this lattice was loaded.
    """

    elements: List[Union[BeamlineElement, Line]] = dataclasses.field(
        default_factory=list
    )
    filename: Optional[pathlib.Path] = None

    def __str__(self) -> str:
        return "\n".join(str(element) for element in self.elements)

    @classmethod
    def from_contents(
        cls, contents: str, filename: Optional[AnyPath] = None
    ) -> Lattice:
        """
        Load a lattice from its file contents.

        Parameters
        ----------
        contents : str
            The contents of the lattice file.
        filename : AnyPath or None, optional
            The filename of the lattice, if known.

        Returns
        -------
        Lattice
        """
        from .grammar import _LatticeTransformer, new_lattice_parser

        parser = new_lattice_parser()
        tree = parser.parse(contents)
        filename = filename or "unknown"
        return _LatticeTransformer(filename).transform(tree)

    @classmethod
    def from_file(cls, filename: AnyPath) -> Lattice:
        """
        Load a lattice file from disk.

        Parameters
        ----------
        filename : AnyPath
            The filename to load.

        Returns
        -------
        Lattice
        """
        with open(filename) as fp:
            contents = fp.read()
        return cls.from_contents(contents, filename=filename)


@dataclasses.dataclass
class Reference:
    """
    A Genesis 4 main input value which is a reference to another namelist or
    value.

    Attributes
    ----------
    label : str
        The reference name.
    """

    label: str

    def __str__(self) -> str:
        return f"@{self.label}"


@dataclasses.dataclass
class MainInput:
    """
    A Genesis 4 main input configuration file.

    Attributes
    ----------
    namelists : list of NameList
        Elements contained in the lattice.
    filename : pathlib.Path or None
        The filename from which this was loaded.
    """

    namelists: List[NameList] = dataclasses.field(default_factory=list)
    filename: Optional[pathlib.Path] = None

    def __str__(self) -> str:
        return "\n\n".join(str(namelist) for namelist in self.namelists)

    @classmethod
    def from_contents(
        cls, contents: str, filename: Optional[AnyPath] = None
    ) -> MainInput:
        """
        Load main input from its file contents.

        Parameters
        ----------
        contents : str
            The contents of the main input file.
        filename : AnyPath or None, optional
            The filename, if known.

        Returns
        -------
        MainInput
        """
        from .grammar import _MainInputTransformer, new_main_input_parser

        parser = new_main_input_parser()
        tree = parser.parse(contents)
        filename = filename or "unknown"
        return _MainInputTransformer(filename).transform(tree)

    @classmethod
    def from_file(cls, filename: AnyPath) -> MainInput:
        """
        Load a main input file from disk.

        Parameters
        ----------
        filename : AnyPath
            The filename to load.

        Returns
        -------
        MainInput
        """
        with open(filename) as fp:
            contents = fp.read()
        return cls.from_contents(contents, filename=filename)


def python_to_namelist_value(value: ValueType) -> str:
    """
    Convert a Python value to its NameList representation.

    Parameters
    ----------
    value : ValueType
        The Python value to convert.

    Returns
    -------
    str
        Representation which can be used in a namelist as part of an input
        file.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        return " ".join(v for v in value)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (float, int, Decimal)):
        return str(value)
    # raise NotImplementedError(type(value))
    return str(value)
