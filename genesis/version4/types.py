from __future__ import annotations

import abc
import pathlib
import pydantic
import sys
from typing import Any, Dict, Iterable, Sequence, Tuple, Type, Union

import h5py
import numpy as np
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


class BaseModel(pydantic.BaseModel):
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
                return H5Reference.model_validate(value).load(h5)
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


class H5Reference(BaseModel):
    """
    HDF5 path reference.

    Used in archiving of Genesis 4 input/output.  The _PydanticNDArray
    validator dereferences this path and seamlessly inserts an ndarray in its
    place.
    """

    path: str

    def load(self, h5: h5py.Group) -> np.ndarray:
        return np.asarray(h5[self.path])


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
    if isinstance(value, dict) and "path" in value:
        # H5Reference which is restored by NDArray during validation
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


class OutputLatticeDict(TypedDict):
    aw: NDArray
    ax: NDArray
    ay: NDArray
    chic_angle: NDArray
    chic_lb: NDArray
    chic_ld: NDArray
    chic_lt: NDArray
    cx: NDArray
    cy: NDArray
    dz: NDArray
    gradx: NDArray
    grady: NDArray
    ku: NDArray
    kx: NDArray
    ky: NDArray
    phaseshift: NDArray
    qf: NDArray
    qx: NDArray
    qy: NDArray
    slippage: NDArray
    z: NDArray
    zplot: NDArray


class OutputBeamDict(TypedDict):
    LSCfield: NDArray
    alphax: NDArray
    alphay: NDArray
    betax: NDArray
    betay: NDArray
    bunching: NDArray
    bunchingphase: NDArray
    current: NDArray
    efield: NDArray
    emax: NDArray
    emin: NDArray
    emitx: NDArray
    emity: NDArray
    energy: NDArray
    energyspread: NDArray
    pxmax: NDArray
    pxmin: NDArray
    pxposition: NDArray
    pymax: NDArray
    pymin: NDArray
    pyposition: NDArray
    wakefield: NDArray
    xmax: NDArray
    xmin: NDArray
    xposition: NDArray
    xsize: NDArray
    ymax: NDArray
    ymin: NDArray
    yposition: NDArray
    ysize: NDArray


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
    frequency: NDArray
    gamma0: float
    lambdaref: float
    one4one: float
    s: NDArray
    sample: float
    scan: float
    slen: float
    time: float


OutputFieldDict = TypedDict(
    "OutputFieldDict",
    {
        "dgrid": float,
        "intensity-farfield": NDArray,
        "intensity-nearfield": NDArray,
        "ngrid": float,
        "phase-farfield": NDArray,
        "phase-nearfield": NDArray,
        "power": NDArray,
        "xdivergence": NDArray,
        "xpointing": NDArray,
        "xposition": NDArray,
        "xsize": NDArray,
        "ydivergence": NDArray,
        "ypointing": NDArray,
        "yposition": NDArray,
        "ysize": NDArray,
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
