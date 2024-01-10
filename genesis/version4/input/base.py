from __future__ import annotations
import dataclasses
from decimal import Decimal
import pathlib
import textwrap

from typing import ClassVar, Dict, Iterable, List, Optional, Union
from .types import Float, ValueType, AnyPath
from ..manual import renames


@dataclasses.dataclass
class BeamlineElement:
    _lattice_to_attr_: ClassVar[Dict[str, str]] = renames
    _attr_to_lattice_: ClassVar[Dict[str, str]] = dict(
        (v, k) for k, v in _lattice_to_attr_.items()
    )

    label: str

    @property
    def parameters(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        skip = {"label"}
        data = {}
        for attr in self.__annotations__:
            if attr.startswith("_") or attr in skip:
                continue
            value = getattr(self, attr)
            default = getattr(type(self), attr, None)
            if str(value) != str(default):
                param = self._attr_to_lattice_.get(attr, attr)
                data[param] = value
        return data

    def __str__(self) -> str:
        parameters = ", ".join(
            f"{name}={value}" for name, value in self.parameters.items()
        )
        type_ = type(self).__name__.upper()
        return "".join(
            (
                self.label,
                f": {type_} = " "{",
                parameters,
                "};",
            )
        )


@dataclasses.dataclass
class DuplicatedLineItem:
    #: Duplication count.
    count: int
    #: Element name.
    label: str

    def __str__(self) -> str:
        return f"{self.count}*{self.label}"


@dataclasses.dataclass
class PositionedLineItem:
    #: Position in meters.
    position: Float
    #: Element name.
    label: str

    def __str__(self) -> str:
        return f"{self.label}@{self.position}"


LineItem = Union[str, DuplicatedLineItem, PositionedLineItem]


@dataclasses.dataclass
class Line:
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
        from ..grammar import _LatticeTransformer, new_lattice_parser

        parser = new_lattice_parser()
        tree = parser.parse(contents)
        filename = filename or "unknown"
        return _LatticeTransformer(filename).transform(tree)

    @classmethod
    def from_file(cls, filename: AnyPath) -> Lattice:
        with open(filename) as fp:
            contents = fp.read()
        return cls.from_contents(contents, filename=filename)


def python_to_namelist_value(value: ValueType) -> str:
    """
    Convert a Python-typed value to a format supported by namelists.
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


@dataclasses.dataclass
class NameList:
    _namelist_to_attr_: ClassVar[Dict[str, str]] = renames
    _attr_to_namelist_: ClassVar[Dict[str, str]] = dict(
        (v, k) for k, v in _namelist_to_attr_.items()
    )

    @property
    def parameters(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        skip = {"label"}
        data = {}
        for attr in self.__annotations__:
            if attr.startswith("_") or attr in skip:
                continue
            value = getattr(self, attr)
            default = getattr(type(self), attr, None)
            if str(value) != str(default):
                param = self._attr_to_namelist_.get(attr, attr)
                data[param] = value
        return data

    def __str__(self) -> str:
        parameters = [
            f"{name} = {python_to_namelist_value(value)}"
            for name, value in self.parameters.items()
        ]
        type_ = type(self).__name__.lower()
        return "\n".join(
            (
                f"&{type_}",
                textwrap.indent("\n".join(parameters), "  ") if parameters else "",
                "&end",
            )
        )
