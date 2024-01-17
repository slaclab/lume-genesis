from __future__ import annotations

import dataclasses
import pathlib
from decimal import Decimal
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple, Type, Union

import lark

from .generated_lattice import BeamlineElement
from .generated_main import NameList, Reference
from .types import AnyPath, Float, ValueType
from .util import HiddenDecimal

LATTICE_GRAMMAR = pathlib.Path("version4") / "input" / "lattice.lark"
MAIN_INPUT_GRAMMAR = pathlib.Path("version4") / "input" / "main_input.lark"

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

    @classmethod
    def from_string(cls, value: str) -> DuplicatedLineItem:
        count, label = value.split("*", 1)
        return cls(
            label=label.strip(),
            count=int(count),
        )

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

    @classmethod
    def from_string(cls, value: str) -> PositionedLineItem:
        label, position = value.split("@", 1)
        return cls(
            label=label.strip(),
            position=HiddenDecimal(position.strip()),
        )

    def __str__(self) -> str:
        return f"{self.label}@{self.position}"


def _fix_line_item(line_item: LineItem) -> LineItem:
    """Make the appropriate dataclass for a serialized line item, if necessary."""
    if isinstance(line_item, (DuplicatedLineItem, PositionedLineItem)):
        return line_item

    if "@" in line_item:
        return PositionedLineItem.from_string(line_item)
    if "*" in line_item:
        return DuplicatedLineItem.from_string(line_item)
    return line_item


@dataclasses.dataclass
class Line(BeamlineElement):
    """
    A Genesis 4 beamline Line.

    Attributes
    ----------
    label : str
        The name/label of the line.
    elements : list of LineItem
        Elements contained in the line.
    """

    _genesis_name_: ClassVar[str] = "line"
    label: str
    elements: List[LineItem] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.elements = [_fix_line_item(item) for item in self.elements]

    def serialize(self) -> Dict:
        """
        Get a serialized (dictionary representation) of this beamline element.
        """
        return {
            "type": "line",
            "label": self.label,
            "elements": [str(element) for element in self.elements],
        }

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

    def serialize(self) -> List[Dict]:
        """Serialize this lattice to a list of dictionaries."""
        return [element.serialize() for element in self.elements]

    @classmethod
    def deserialize(
        cls, contents: Sequence[Dict], filename: Optional[pathlib.Path] = None
    ) -> Lattice:
        """
        Load a Lattice instance from a list of serialized dictionaries.

        Parameters
        ----------
        contents : sequence of dict
            The serialized contents of the lattice.

        Returns
        -------
        Lattice
        """
        return cls(
            elements=[BeamlineElement.deserialize(dct) for dct in contents],
            filename=filename,
        )

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

    def serialize(self) -> List[Dict]:
        """Serialize this main input to a list of dictionaries."""
        return [namelist.serialize() for namelist in self.namelists]

    @classmethod
    def deserialize(
        cls, contents: Sequence[Dict], filename: Optional[pathlib.Path] = None
    ) -> MainInput:
        """
        Load main input from a list of serialized dictionaries.

        Parameters
        ----------
        contents : sequence of dict
            The serialized contents of the main input file.

        Returns
        -------
        MainInput
        """
        return cls(
            namelists=[NameList.deserialize(dct) for dct in contents],
            filename=filename,
        )

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


def new_parser(filename: AnyPath, **kwargs) -> lark.Lark:
    """
    Get a new parser for one of the packaged grammars.

    Parameters
    ----------
    filename : str
        The packaged grammar filename.
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    return lark.Lark.open_from_package(
        "genesis",
        str(filename),
        parser="lalr",
        maybe_placeholders=True,
        propagate_positions=True,
        **kwargs,
    )


def new_lattice_parser(**kwargs) -> lark.Lark:
    """
    Get a new parser for the packaged Lattice input grammar.

    Parameters
    ----------
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    return new_parser(LATTICE_GRAMMAR, **kwargs)


def new_main_input_parser(**kwargs) -> lark.Lark:
    """
    Get a new parser for the packaged main input file grammar.

    Parameters
    ----------
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    return new_parser(MAIN_INPUT_GRAMMAR, **kwargs)


def _fix_parameters(
    cls: Union[Type[BeamlineElement], Type[NameList]],
    params: Dict[str, lark.Token],
) -> Tuple[Dict[str, ValueType], Dict[str, str]]:
    """
    Fix parameters to beamline elements when transforming with
    :class:``_LatticeTransformer`.

    Parameters
    ----------
    cls : Type[BeamlineElement]
        The dataclass associated with the beamline element.  This is used to
        determine the attribute name and data type associated with the
        parameter.
    params : Dict[str, lark.Token]
        Parameter name map to lark Token value.

    Returns
    -------
    Dict[str, ValueType]
        Keyword arguments for the dataclass.
    Dict[str, str]
        Unexpected arguments for the dataclass; unsure what to do with them.
    """
    kwargs: Dict[str, ValueType] = {}
    extra: Dict[str, str] = {}
    mapping = cls._parameter_to_attr_
    for name, value in params.items():
        name = mapping.get(name, name)
        dtype = cls.__annotations__.get(name, None)
        allow_reference = "| Reference" in dtype
        dtype = dtype.replace("| Reference", "").strip()
        if dtype is None:
            extra[name] = str(value)
        elif value.startswith("@") and allow_reference:
            kwargs[name] = Reference(str(value[1:]).strip())
        elif dtype == "int":
            kwargs[name] = int(value)
        elif dtype == "Float":
            kwargs[name] = HiddenDecimal(value)
        elif dtype == "bool":
            kwargs[name] = value.lower() == "true"
        elif dtype == "str":
            kwargs[name] = value.strip()
        else:
            raise RuntimeError(f"Unexpected type annotation hit for {name}: {dtype}")
    return kwargs, extra


class _LatticeTransformer(lark.visitors.Transformer_InPlaceRecursive):
    """
    Grammar transformer which takes lark objects and makes a :class:`Lattice`.

    Attributes
    ----------
    _filename : str
        Filename source of the input.
    """

    _filename: Optional[pathlib.Path]
    # This maps, e.g., "UNDU" -> Undulator
    type_map: Dict[str, Type[BeamlineElement]] = {
        cls.__name__.upper()[:4]: cls for cls in BeamlineElement.__subclasses__()
    }

    def __init__(self, filename: AnyPath) -> None:
        super().__init__()
        self._filename = pathlib.Path(filename)

    @lark.v_args(inline=True)
    def line(
        self,
        label: lark.Token,
        line: lark.Token,
        element_list: List[LineItem],
    ) -> Line:
        return Line(
            label=str(label),
            elements=element_list,
        )

    @lark.v_args(inline=True)
    def parameter_set(
        self,
        parameter: lark.Token,
        value: lark.Token,
    ) -> Tuple[str, lark.Token]:
        return (str(parameter), value)

    def parameter_list(
        self, sets: List[Tuple[str, ValueType]]
    ) -> List[Tuple[str, ValueType]]:
        return list(sets)

    @lark.v_args(inline=True)
    def beamline_element(
        self,
        label: lark.Token,
        type_: lark.Token,
        parameter_list: List[Tuple[str, lark.Token]],
    ) -> BeamlineElement:
        cls = self.type_map[type_.upper()[:4]]
        parameters, unknown = _fix_parameters(cls, dict(parameter_list))
        if unknown:
            raise ValueError(
                f"Beamline element {label} received unexpected parameter(s): "
                f"{unknown}"
            )
        return cls(
            label=str(label),
            **parameters,
        )

    @lark.v_args(inline=True)
    def duplicate_item(
        self,
        count: lark.Token,
        label: lark.Token,
    ) -> DuplicatedLineItem:
        return DuplicatedLineItem(
            label=str(label),
            count=int(count),
        )

    @lark.v_args(inline=True)
    def positioned_item(
        self,
        label: lark.Token,
        position: lark.Token,
    ) -> PositionedLineItem:
        return PositionedLineItem(
            label=str(label),
            position=Decimal(position),
        )

    @lark.v_args(inline=True)
    def line_item(
        self, item: Union[lark.Token, DuplicatedLineItem, PositionedLineItem]
    ) -> Union[str, DuplicatedLineItem, PositionedLineItem]:
        if isinstance(item, lark.Token):
            return str(item)
        return item

    def element_list(
        self, items: List[Union[lark.Token, DuplicatedLineItem, PositionedLineItem]]
    ) -> List[Union[lark.Token, DuplicatedLineItem, PositionedLineItem]]:
        return items

    def lattice(self, elements: List[Union[BeamlineElement, Line]]) -> Lattice:
        return Lattice(
            elements,
            filename=self._filename,
        )


class _MainInputTransformer(lark.visitors.Transformer_InPlaceRecursive):
    """
    Grammar transformer which takes lark objects and makes a :class:`MainInput`.

    Attributes
    ----------
    _filename : str
        Filename source of the input.
    """

    _filename: Optional[pathlib.Path]
    # This maps, e.g., "setup" to the Setup dataclass
    type_map: Dict[str, Type[NameList]] = {
        cls._genesis_name_: cls for cls in NameList.__subclasses__()
    }

    def __init__(self, filename: AnyPath) -> None:
        super().__init__()
        self._filename = pathlib.Path(filename)

    @lark.v_args(inline=True)
    def parameter_set(
        self,
        parameter: lark.Token,
        value: lark.Token,
    ) -> Tuple[str, lark.Token]:
        return (str(parameter), value)

    def parameter_list(
        self, sets: List[Tuple[str, ValueType]]
    ) -> List[Tuple[str, ValueType]]:
        return list(sets)

    @lark.v_args(inline=True)
    def namelist(
        self,
        name: lark.Token,
        parameter_list: List[Tuple[str, lark.Token]],
        end: lark.Token,
    ) -> NameList:
        cls = self.type_map[name]
        parameters, unknown = _fix_parameters(cls, dict(parameter_list))
        if unknown:
            raise ValueError(
                f"Namelist {name} received unexpected parameter(s): " f"{unknown}"
            )
        return cls(**parameters)

    @lark.v_args(inline=True)
    def main_input(self, *namelists: NameList) -> MainInput:
        return MainInput(namelists=list(namelists), filename=self._filename)
