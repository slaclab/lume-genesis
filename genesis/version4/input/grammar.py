from __future__ import annotations
import typing
import lark
import pathlib
from decimal import Decimal

from typing import Dict, List, Optional, Tuple, Type, Union
from .types import AnyPath, ValueType
from .generated_main import NameList
from .generated_lattice import BeamlineElement

from .core import (
    Lattice,
    MainInput,
    Line,
    DuplicatedLineItem,
    PositionedLineItem,
    Reference,
)


if typing.TYPE_CHECKING:
    from .core import LineItem

LATTICE_GRAMMAR = pathlib.Path("version4") / "input" / "lattice.lark"
MAIN_INPUT_GRAMMAR = pathlib.Path("version4") / "input" / "main_input.lark"


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


class HiddenDecimal(Decimal):
    """
    A "Decimal" which hides itself in its representation.

    For example, ``repr(Decimal("10.0")) = 10.0``.

    Decimal is convenient for us to use to ensure that round-tripping input
    floating point values back to the original file format does not result in a
    significant change in representation.
    """

    def __repr__(self) -> str:
        dec_str = super().__str__()
        return dec_str.replace("Decimal('", "").rstrip("')")


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
        if dtype is None:
            extra[name] = str(value)
        elif value.startswith("@"):
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
