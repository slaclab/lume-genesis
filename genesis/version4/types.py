from __future__ import annotations
import abc
import pathlib
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import pmd_unit
import pydantic
import pydantic_core

import numpy as np
from typing import (
    Any,
    Dict,
    TYPE_CHECKING,
    Sequence,
    Type,
    Tuple,
    Union,
)

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

if TYPE_CHECKING:
    from .input.core import Reference


class _PydanticPmdUnit(pydantic.BaseModel):
    unitSI: float
    unitSymbol: str
    unitDimension: Tuple[int, ...]

    @staticmethod
    def _from_dict(dct) -> pmd_unit:
        return pmd_unit(**dct)

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
    ) -> "pmd_unit":
        if isinstance(value, pmd_unit):
            return value
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise ValueError(f"No conversion from {value!r} to pmd_unit")


class _PydanticNDArray(pydantic.BaseModel):
    todo: float

    @staticmethod
    def _from_dict(dct) -> pmd_unit:
        raise NotImplementedError()
        return pmd_unit(**dct)

    def _as_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()
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
    def _pydantic_validate(cls, value: Union[Any, np.ndarray, Sequence]) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, Sequence):
            return np.asarray(value)
        raise ValueError(f"No conversion from {value!r} to numpy ndarray")


class _PydanticParticleGroup(pydantic.BaseModel):
    todo: float

    @staticmethod
    def _from_dict(dct) -> pmd_unit:
        return pmd_unit(**dct)

    def _as_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()
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
    ) -> "pmd_unit":
        raise NotImplementedError()
        if isinstance(value, pmd_unit):
            return value
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise ValueError(f"No conversion from {value!r} to pmd_unit")


class Reference(str):
    """
    A Genesis 4 main input value which is a reference to another namelist or
    value.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pydantic_core.CoreSchema:
        return pydantic_core.core_schema.no_info_after_validator_function(
            cls, handler(str)
        )

    def __str__(self) -> str:
        name = super().__str__()
        return f"@{name}"


class NameList(pydantic.BaseModel, abc.ABC):
    """Base class for name lists used in Genesis 4 main input files."""

    def __post_model_init__(self) -> None:
        # Fix up any references or types for convenience (and deserialization)
        # TODO/NOTE: anything more complicated than this will call for
        # msgspec / apischema / pydantic for (de)serialization
        for attr, annotation in self.__annotations__.items():
            if attr.startswith("_"):
                pass
            elif "Reference" in annotation:
                try:
                    value = Reference.from_namelist(getattr(self, attr))
                except ValueError:
                    # Not a reference type to be deserialized
                    ...
                else:
                    # Update the attribute with the Reference instance.
                    setattr(self, attr, value)

    @property
    def genesis_parameters(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        dump = self.model_dump(by_alias=True, exclude_defaults=True)
        return {attr: value for attr, value in dump.items() if attr not in {"type"}}

    def to_genesis(self) -> str:
        """Create a Genesis 4-compatible namelist from this instance."""
        from .input.util import python_to_namelist_value

        parameters = (
            f"  {name} = {python_to_namelist_value(value)}"
            for name, value in self.genesis_parameters.items()
        )
        return "\n".join(
            (
                f"&{self.type}",
                *parameters,
                "&end",
            )
        )

    def __str__(self) -> str:
        return self.to_genesis()


class BeamlineElement(pydantic.BaseModel, abc.ABC):
    """Base class for beamline elements used in Genesis 4 lattice files."""

    label: str

    @property
    def genesis_parameters(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        dump = self.model_dump(by_alias=True, exclude_defaults=True)
        return {attr: value for attr, value in dump.items() if attr not in {"type"}}

    def to_genesis(self) -> str:
        """Create a Genesis 4 compatible element from this instance."""
        from .input.util import python_to_namelist_value

        parameters = ", ".join(
            f"{name}={python_to_namelist_value(value)}"
            for name, value in self.genesis_parameters.items()
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

    def __str__(self) -> str:
        return self.to_genesis()


PydanticPmdUnit = Annotated[pmd_unit, _PydanticPmdUnit]
PydanticNDArray = Annotated[np.ndarray, _PydanticNDArray]
PydanticParticleGroup = Annotated[ParticleGroup, _PydanticParticleGroup]

AnyPath = Union[pathlib.Path, str]
ValueType = Union[int, float, bool, str, "Reference"]
ArrayType = Union[Sequence[float], PydanticNDArray]
Float = float


try:
    from types import UnionType
except ImportError:
    # Python < 3.10
    union_types = {Union}
else:
    union_types = {UnionType, Union}
