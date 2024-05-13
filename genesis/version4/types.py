from __future__ import annotations

import abc
import pathlib
import sys
from typing import Any, Dict, Sequence, Tuple, Type, Union

import h5py
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

try:
    from typing import Annotated, Literal, NotRequired
except ImportError:
    from typing_extensions import Annotated, Literal, NotRequired

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    # Pydantic specifically requires this for Python < 3.12
    from typing_extensions import TypedDict


class _PydanticPmdUnit(pydantic.BaseModel):
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


class _PydanticNDArray(pydantic.BaseModel):
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

            if info.context and isinstance(info.context, dict):
                if "hdf5" in info.context:
                    h5: h5py.Group = info.context["hdf5"]
                    array_prefix = info.context["array_prefix"]
                    info.context["array_index"] += 1
                    key = array_prefix + str(info.context["array_index"])
                    h5.create_dataset(name=key, data=obj)
                    full_path = f"{h5.name}/{key}" if h5.name else key
                    return H5Reference(path=full_path).model_dump()

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
        value: Union[Any, np.ndarray, Sequence, H5Reference, dict],
        info: pydantic.ValidationInfo,
    ) -> np.ndarray:
        if info.context and isinstance(info.context, dict) and "hdf5" in info.context:
            h5: h5py.Group = info.context["hdf5"]
            if isinstance(value, dict) and "path" in value:
                ref = H5Reference.model_validate(value)
                return ref.load(h5)
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


class NameList(pydantic.BaseModel, abc.ABC):
    """Base class for name lists used in Genesis 4 main input files."""

    @property
    def genesis_parameters(self) -> Dict[str, ValueType]:
        """Dictionary of parameters to pass to Genesis 4."""
        dump = self.model_dump(by_alias=True, exclude_defaults=True)
        return {attr: value for attr, value in dump.items() if attr not in {"type"}}

    def to_string(self, mode: Literal["html", "markdown", "genesis"]) -> str:
        if mode == "html":
            return tools.html_table_repr(self, [])
        if mode == "markdown":
            return str(tools.ascii_table_repr(self, []))
        if mode == "genesis":
            return self.to_genesis()
        raise NotImplementedError(f"Render mode {mode} unsupported")

    def _repr_html_(self) -> str:
        return self.to_string(tools.global_display_options.jupyter_render_mode)

    def __str__(self) -> str:
        return self.to_string(tools.global_display_options.console_render_mode)

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
    x: PydanticNDArray
    y: PydanticNDArray
    z: PydanticNDArray

    # `px`, `py`, `pz` are momenta in units of [eV/c]
    px: PydanticNDArray
    py: PydanticNDArray
    pz: PydanticNDArray

    # `t` is time in [s]
    t: PydanticNDArray
    status: PydanticNDArray

    # `weight` is the macro-charge weight in [C], used for all statistical
    # calculations.
    weight: PydanticNDArray

    # `species` is a proper species name: `'electron'`, etc.
    species: str
    id: NotRequired[PydanticNDArray]


class BeamlineElement(pydantic.BaseModel, abc.ABC):
    """Base class for beamline elements used in Genesis 4 lattice files."""

    label: str

    def to_string(self, mode: Literal["html", "markdown", "genesis"]) -> str:
        if mode == "html":
            return tools.html_table_repr(self, [])
        if mode == "markdown":
            return str(tools.ascii_table_repr(self, []))
        if mode == "genesis":
            return self.to_genesis()
        raise NotImplementedError(f"Render mode {mode} unsupported")

    def _repr_html_(self) -> str:
        return self.to_string(tools.global_display_options.jupyter_render_mode)

    def __str__(self) -> str:
        return self.to_string(tools.global_display_options.console_render_mode)

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


class H5Reference(pydantic.BaseModel):
    path: str

    def load(self, h5: h5py.Group) -> np.ndarray:
        return np.asarray(h5[self.path])


# ArrayType = Union[np.ndarray, Sequence[float], H5Reference]
ArrayType = np.ndarray
AnyPath = Union[pathlib.Path, str]
ValueType = Union[int, float, bool, str, Reference]
PydanticPmdUnit = Annotated[pmd_unit, _PydanticPmdUnit]
PydanticNDArray = Annotated[ArrayType, _PydanticNDArray]


class OutputLatticeDict(TypedDict):
    aw: PydanticNDArray
    ax: PydanticNDArray
    ay: PydanticNDArray
    chic_angle: PydanticNDArray
    chic_lb: PydanticNDArray
    chic_ld: PydanticNDArray
    chic_lt: PydanticNDArray
    cx: PydanticNDArray
    cy: PydanticNDArray
    dz: PydanticNDArray
    gradx: PydanticNDArray
    grady: PydanticNDArray
    ku: PydanticNDArray
    kx: PydanticNDArray
    ky: PydanticNDArray
    phaseshift: PydanticNDArray
    qf: PydanticNDArray
    qx: PydanticNDArray
    qy: PydanticNDArray
    slippage: PydanticNDArray
    z: PydanticNDArray
    zplot: PydanticNDArray


class OutputBeamDict(TypedDict):
    LSCfield: PydanticNDArray
    alphax: PydanticNDArray
    alphay: PydanticNDArray
    betax: PydanticNDArray
    betay: PydanticNDArray
    bunching: PydanticNDArray
    bunchingphase: PydanticNDArray
    current: PydanticNDArray
    efield: PydanticNDArray
    emax: PydanticNDArray
    emin: PydanticNDArray
    emitx: PydanticNDArray
    emity: PydanticNDArray
    energy: PydanticNDArray
    energyspread: PydanticNDArray
    pxmax: PydanticNDArray
    pxmin: PydanticNDArray
    pxposition: PydanticNDArray
    pymax: PydanticNDArray
    pymin: PydanticNDArray
    pyposition: PydanticNDArray
    wakefield: PydanticNDArray
    xmax: PydanticNDArray
    xmin: PydanticNDArray
    xposition: PydanticNDArray
    xsize: PydanticNDArray
    ymax: PydanticNDArray
    ymin: PydanticNDArray
    yposition: PydanticNDArray
    ysize: PydanticNDArray


class OutputMetaDumpsDict(TypedDict):
    ndumps: int


class OutputMetaVersionDict(TypedDict):
    Beta: float
    Build_Info: str
    Major: float
    Minor: float
    Revision: float


class OutputMetaDict(TypedDict):
    Beamdumps: OutputMetaDumpsDict
    Fielddumps: OutputMetaDumpsDict
    HOST: str
    InputFile: str
    LatticeFile: str
    TimeStamp: str
    User: str
    Version: OutputMetaVersionDict
    cwd: str
    mpisize: float


class OutputGlobalDict(TypedDict):
    frequency: PydanticNDArray
    gamma0: float
    lambdaref: float
    one4one: float
    s: PydanticNDArray
    sample: float
    scan: float
    slen: float
    time: float


OutputFieldDict = TypedDict(
    "OutputFieldDict",
    {
        "dgrid": float,
        "intensity-farfield": PydanticNDArray,
        "intensity-nearfield": PydanticNDArray,
        "ngrid": float,
        "phase-farfield": PydanticNDArray,
        "phase-nearfield": PydanticNDArray,
        "power": PydanticNDArray,
        "xdivergence": PydanticNDArray,
        "xpointing": PydanticNDArray,
        "xposition": PydanticNDArray,
        "xsize": PydanticNDArray,
        "ydivergence": PydanticNDArray,
        "ypointing": PydanticNDArray,
        "yposition": PydanticNDArray,
        "ysize": PydanticNDArray,
    },
)


class FieldFileParamDict(TypedDict):
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


class FieldFileDict(TypedDict):
    label: str
    dfl: PydanticNDArray
    param: FieldFileParamDict
