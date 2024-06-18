from __future__ import annotations

from pydantic import BaseModel
from typing import (
    Any,
    Dict,
    Tuple,
    Type,
)
import pathlib

import lark
from ..types import (
    AnyPath,
    ValueType,
    Reference,
)

MAIN_INPUT_GRAMMAR = pathlib.Path("version4") / "input" / "main_input.lark"
LATTICE_GRAMMAR = pathlib.Path("version4") / "input" / "lattice.lark"


def fix_parameters(
    cls: Type[BaseModel],
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
        Arguments for Pydantic model validation.
    Dict[str, str]
        Unexpected arguments for the dataclass; unsure what to do with them.
    """
    from ... import tools

    kwargs: Dict[str, ValueType] = {}
    extra: Dict[str, str] = {}
    fields = {
        field.serialization_alias: field
        for field in cls.model_fields.values()
        if field.serialization_alias
    }
    fields.update(
        {field.alias: field for field in cls.model_fields.values() if field.alias}
    )
    fields.update(cls.model_fields)

    for name, value in params.items():
        field = fields[name]
        dtype = field.annotation
        allow_reference = tools.field_allows_reference(field)
        dtype = tools.get_primary_type_for_field(field)
        string_value = str(value).strip()
        if value.startswith("@") and allow_reference:
            kwargs[name] = Reference(string_value[1:].strip())
        elif dtype is int:
            try:
                kwargs[name] = int(value)
            except ValueError:
                # These may be references, let pydantic deal with them
                kwargs[name] = value
        elif dtype is float:
            try:
                kwargs[name] = float(value.rstrip("."))
            except ValueError:
                # These may be references, let pydantic deal with them
                kwargs[name] = value
        elif dtype is bool:
            kwargs[name] = string_value.lower() in ("true", "t", "1")
        elif dtype is str:
            kwargs[name] = string_value
        else:
            raise RuntimeError(f"Unexpected type annotation hit for {name}: {dtype}")
    return kwargs, extra


def new_parser(filename: AnyPath, **kwargs: Any) -> lark.Lark:
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


def new_main_input_parser(**kwargs: Any) -> lark.Lark:
    """
    Get a new parser for the packaged main input file grammar.

    Parameters
    ----------
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    return new_parser(MAIN_INPUT_GRAMMAR, **kwargs)


def new_lattice_parser(**kwargs: Any) -> lark.Lark:
    """
    Get a new parser for the packaged Lattice input grammar.

    Parameters
    ----------
    **kwargs :
        See :class:`lark.lark.LarkOptions`.
    """
    return new_parser(LATTICE_GRAMMAR, **kwargs)
