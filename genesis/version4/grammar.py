from __future__ import annotations
import lark
import pathlib
from decimal import Decimal

from typing import Dict, List, Optional, Tuple, Type, Union
from .input.base import LineItem
from .input.types import AnyPath, ValueType
from .input import (
    BeamlineElement,
    Chicane,
    Corrector,
    Drift,
    Marker,
    Phaseshifter,
    Quadrupole,
    Undulator,
    Lattice,
    Line,
    DuplicatedLineItem,
    PositionedLineItem,
)


LATTICE_GRAMMAR = pathlib.Path("version4") / "lattice.lark"


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


def _fix_parameters(
    cls: Type[BeamlineElement],
    params: Dict[str, lark.Token],
) -> Tuple[Dict[str, ValueType], Dict[str, str]]:
    kwargs: Dict[str, ValueType] = {}
    extra: Dict[str, str] = {}
    for name, value in params.items():
        name = cls._lattice_to_attr_.get(name, name)
        dtype = cls.__annotations__.get(name, None)
        if dtype is None:
            extra[name] = str(value)
        elif dtype == "int":
            kwargs[name] = int(value)
        elif dtype == "Float":
            kwargs[name] = Decimal(value)
        elif dtype == "bool":
            kwargs[name] = value.lower() == "true"
        else:
            raise RuntimeError(f"Unexpected type annotation hit for {name}: {dtype}")
    return kwargs, extra


class _LatticeTransformer(lark.visitors.Transformer_InPlaceRecursive):
    """
    Grammar transformer which takes lark objects and makes a :class:`Lattice`.

    Attributes
    ----------
    _filename : str
        Filename of grammar being transformed.
    """

    _filename: Optional[pathlib.Path]
    type_map: Dict[str, Type[BeamlineElement]] = {
        "UNDU": Undulator,
        "QUAD": Quadrupole,
        "DRIF": Drift,
        "CORR": Corrector,
        "CHIC": Chicane,
        "PHAS": Phaseshifter,
        "MARK": Marker,
        # "LINE": Line,
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
            count=int(count),
            label=str(label),
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
