from __future__ import annotations
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
    TypedDict,
    Union,
)

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

if TYPE_CHECKING:
    from .input.core import Reference


class SerializedReference(TypedDict):
    """A serialized Reference instance as a dictionary."""

    label: str


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
    def _pydantic_validate(
        cls, value: Union[Dict[str, Any], pmd_unit, Any]
    ) -> "pmd_unit":
        raise NotImplementedError()
        if isinstance(value, pmd_unit):
            return value
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise ValueError(f"No conversion from {value!r} to pmd_unit")


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


PydanticPmdUnit = Annotated[pmd_unit, _PydanticPmdUnit]
PydanticNDArray = Annotated[np.ndarray, _PydanticNDArray]
PydanticParticleGroup = Annotated[ParticleGroup, _PydanticParticleGroup]

AnyPath = Union[pathlib.Path, str]
ValueType = Union[int, float, bool, str, "Reference"]
ArrayType = Union[Sequence[float], PydanticNDArray]
Float = float
