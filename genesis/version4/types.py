from __future__ import annotations

import abc
import pathlib
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pydantic
import pydantic_core
from pmd_beamphysics.units import pmd_unit

from .. import tools

try:
    from types import UnionType
except ImportError:
    # Python < 3.10
    union_types = {Union}
else:
    union_types = {UnionType, Union}

from typing_extensions import Annotated, Literal, NotRequired, TypedDict, override


class ReprTableData(TypedDict):
    """Data to use for table output."""

    obj: Union[BaseModel, Dict[str, Any]]
    descriptions: Optional[Dict[str, str]]
    annotations: Optional[Dict[str, str]]


def _check_equality(obj1: Any, obj2: Any) -> bool:
    """
    Check equality of `obj1` and `obj2`.`

    Parameters
    ----------
    obj1 : Any
    obj2 : Any

    Returns
    -------
    bool
    """
    if not isinstance(obj1, type(obj2)):
        return False

    if isinstance(obj1, pydantic.BaseModel):
        return all(
            _check_equality(
                getattr(obj1, attr),
                getattr(obj2, attr),
            )
            for attr, fld in obj1.model_fields.items()
            if not fld.exclude
        )

    if isinstance(obj1, dict):
        if set(obj1) != set(obj2):
            return False

        return all(
            _check_equality(
                obj1[key],
                obj2[key],
            )
            for key in obj1
        )

    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(
            _check_equality(obj1_value, obj2_value)
            for obj1_value, obj2_value in zip(obj1, obj2)
        )

    if isinstance(obj1, np.ndarray):
        if not obj1.shape and not obj2.shape:
            return True
        return np.allclose(obj1, obj2)

    if isinstance(obj1, float):
        return np.allclose(obj1, obj2)

    return bool(obj1 == obj2)


class BaseModel(pydantic.BaseModel, extra="forbid", validate_assignment=True):
    """
    LUME-Genesis customized pydantic BaseModel.

    Alters `dir()` handling and other things for user convenience.
    """

    def _repr_table_data_(self) -> ReprTableData:
        units = getattr(self, "units", None)
        return {
            "obj": self,
            "descriptions": None,
            "annotations": units if isinstance(units, dict) else None,
        }

    def to_table(self):
        return tools.table_output(**self._repr_table_data_())

    def _pretty_repr_(self) -> str:
        return tools.pretty_repr(self, skip_defaults=True)

    def to_string(self, mode: Literal["html", "markdown", "genesis", "repr"]) -> str:
        if mode == "html":
            return tools.html_table_repr(**self._repr_table_data_(), seen=[])
        if mode == "markdown":
            return str(tools.ascii_table_repr(**self._repr_table_data_(), seen=[]))
        if mode == "genesis":
            to_genesis = getattr(self, "to_genesis", None)
            if callable(to_genesis):
                return to_genesis()
            return self._pretty_repr_()
        if mode == "repr":
            return self._pretty_repr_()

        raise NotImplementedError(f"Render mode {mode} unsupported")

    def _repr_html_(self) -> str:
        render_mode = tools.global_display_options.jupyter_render_mode
        as_string = self.to_string(render_mode)
        if render_mode == "html":
            return as_string
        return f"<pre>{as_string}</pre>"

    @override
    def __eq__(self, other: Any) -> bool:
        return _check_equality(self, other)

    @override
    def __ne__(self, other: Any) -> bool:
        return not _check_equality(self, other)

    @override
    def __str__(self) -> str:
        return self.to_string(tools.global_display_options.console_render_mode)

    @override
    def __repr__(self) -> str:
        return self._pretty_repr_()

    @override
    def __dir__(self) -> Iterable[str]:
        full = super().__dir__()
        if not tools.global_display_options.filter_tab_completion:
            return full

        base_model = set(dir(pydantic.BaseModel))
        return [
            attr for attr in full if not attr.startswith("_") and attr not in base_model
        ]


class _PydanticPmdUnit(BaseModel):
    unitSI: float
    unitSymbol: str
    unitDimension: Tuple[int, ...]

    @staticmethod
    def _from_dict(dct: dict) -> pmd_unit:
        dct = dict(dct)
        dim = dct.pop("unitDimension", None)
        if dim is not None:
            dim = tuple(dim)
        return pmd_unit(**dct, unitDimension=dim)

    def _as_dict(self) -> Dict[str, Any]:
        return {
            "unitSI": self.unitSI,
            "unitSymbol": self.unitSymbol,
            "unitDimension": tuple(self.unitDimension),
        }

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        return pydantic_core.core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                cls._as_dict, when_used="json-unless-none"
            ),
        )

    @classmethod
    def _pydantic_validate(
        cls, value: Union[Dict[str, Any], pmd_unit, Any]
    ) -> pmd_unit:
        if isinstance(value, pmd_unit):
            return value
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise ValueError(f"No conversion from {value!r} to pmd_unit")


class _PydanticNDArray(BaseModel):
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        def serialize(obj: np.ndarray, info: pydantic.SerializationInfo):
            if not isinstance(obj, np.ndarray):
                raise ValueError(
                    f"Only supports numpy ndarray. Got {type(obj).__name__}: {obj}"
                )

            return obj.tolist()

        return pydantic_core.core_schema.with_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used="json-unless-none", info_arg=True
            ),
        )

    @classmethod
    def _pydantic_validate(
        cls,
        value: Union[Any, np.ndarray, Sequence, dict],
        info: pydantic.ValidationInfo,
    ) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, Sequence):
            return np.asarray(value)
        raise ValueError(f"No conversion from {value!r} to numpy ndarray")


class Reference(str):
    """
    A Genesis 4 main input value which is a reference to another namelist or
    value.
    """

    def __new__(cls, value: str) -> Reference:
        if not value.startswith("@"):
            value = f"@{value}"
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pydantic_core.CoreSchema:
        return pydantic_core.core_schema.no_info_after_validator_function(
            cls, pydantic_core.core_schema.str_schema()
        )


class NameList(BaseModel, abc.ABC):
    """Base class for name lists used in Genesis 4 main input files."""

    @property
    def _to_genesis_params(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        dump = self.model_dump(by_alias=True, exclude_defaults=True)
        return {attr: value for attr, value in dump.items() if attr not in {"type"}}

    def to_genesis(self) -> str:
        """Create a Genesis 4-compatible namelist from this instance."""
        from .input.util import python_to_namelist_value

        parameters = (
            f"  {name} = {python_to_namelist_value(value)}"
            for name, value in self._to_genesis_params.items()
        )
        return "\n".join(
            (
                f"&{self.type}",
                *parameters,
                "&end",
            )
        )


class ParticleData(TypedDict):
    """
    ParticleGroup raw data as a dictionary.

    The following keys are required:
    * `x`, `y`, `z` are np.ndarray in units of [m]
    * `px`, `py`, `pz` are np.ndarray momenta in units of [eV/c]
    * `t` is a np.ndarray of time in [s]
    * `status` is a status coordinate np.ndarray
    * `weight` is the macro-charge weight in [C], used for all statistical calculations.
    * `species` is a proper species name: `'electron'`, etc.
    The following keys are optional:
    * `id` is an optional np.ndarray of unique IDs
    """

    # `x`, `y`, `z` are positions in units of [m]
    x: NDArray
    y: NDArray
    z: NDArray

    # `px`, `py`, `pz` are momenta in units of [eV/c]
    px: NDArray
    py: NDArray
    pz: NDArray

    # `t` is time in [s]
    t: NDArray
    status: NDArray

    # `weight` is the macro-charge weight in [C], used for all statistical
    # calculations.
    weight: NDArray

    # `species` is a proper species name: `'electron'`, etc.
    species: str
    id: NotRequired[NDArray]


class BeamlineElement(BaseModel, abc.ABC):
    """Base class for beamline elements used in Genesis 4 lattice files."""

    label: str

    @property
    def _to_genesis_params(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        dump = self.model_dump(by_alias=True, exclude_defaults=True)
        return {attr: value for attr, value in dump.items() if attr not in {"type"}}

    def to_genesis(self) -> str:
        """Create a Genesis 4 compatible element from this instance."""
        from .input.util import python_to_namelist_value

        parameters = ", ".join(
            f"{name}={python_to_namelist_value(value)}"
            for name, value in self._to_genesis_params.items()
            if name not in {"label"}
        )
        return "".join(
            (
                self.label,
                f": {self.type} = ",
                "{",
                parameters,
                "};",
            )
        )


AnyPath = Union[pathlib.Path, str]
ValueType = Union[int, float, bool, str, Reference]
PydanticPmdUnit = Annotated[pmd_unit, _PydanticPmdUnit]
NDArray = Annotated[np.ndarray, _PydanticNDArray]


def _get_output_discriminator_value(value):
    # Note: this is a bit of a hack to instruct pydantic which type should
    # be used in the union. As someone new to custom types in Pydantic v2,
    # I'm sure there's a better way to do this - and am open to suggestions!
    if isinstance(value, np.ndarray):
        return "array"
    if isinstance(value, np.generic):
        value = value.item()
    return type(value).__name__


OutputDataType = Annotated[
    Union[
        Annotated[float, pydantic.Tag("float")],
        Annotated[int, pydantic.Tag("int")],
        Annotated[str, pydantic.Tag("str")],
        Annotated[bool, pydantic.Tag("bool")],
        Annotated[NDArray, pydantic.Tag("array")],
    ],
    pydantic.Discriminator(_get_output_discriminator_value),
]


class FieldFileParams(BaseModel, extra="allow"):
    __pydantic_extra__: Dict[str, Union[int, float, str, bool]]

    #  number of gridpoints in one transverse dimension equal to nx and ny above
    gridpoints: int
    # gridpoint spacing (meter)
    gridsize: float
    # starting position (meter)
    refposition: float
    # radiation wavelength (meter)
    wavelength: float
    # number of slices
    slicecount: int
    # slice spacing (meter)
    slicespacing: float
