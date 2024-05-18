from __future__ import annotations

from functools import cached_property
import logging
import pathlib
import typing
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import h5py
import matplotlib.figure
import numpy as np
import pydantic
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import c_light, pmd_unit

from .. import tools
from . import parsers, readers
from .plot import plot_stats_with_layout
from .types import (
    AnyPath,
    BaseModel,
    FieldFileParams,
    OutputDataType,
    NDArray,
    PydanticPmdUnit,
)
from .particles import load_particle_group

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

try:
    from collections.abc import Mapping
except ImportError:
    from typing import Mapping

if typing.TYPE_CHECKING:
    from .input.core import Genesis4Input


logger = logging.getLogger(__name__)


class FieldFile(BaseModel):
    label: str
    dfl: NDArray
    param: FieldFileParams

    def write_openpmd_wavefront(
        self,
        dest: Union[AnyPath, h5py.Group],
        verbose: bool = True,
    ) -> None:
        """
        Write the field file information to the given HDF5 file in
        OpenPMD-wavefront format.

        Parameters
        ----------
        dest : str, pathlib.Path, or h5py.Group
            Filename or already-open h5py.Group to write to.
        """
        from .writers import write_openpmd_wavefront_h5, write_openpmd_wavefront

        if isinstance(dest, (str, pathlib.Path)):
            write_openpmd_wavefront(
                str(dest), self.dfl, self.param.model_dump(), verbose=verbose
            )
        elif isinstance(dest, h5py.Group):
            write_openpmd_wavefront_h5(dest, self.dfl, self.param.model_dump())
        else:
            raise ValueError(type(dest))


class RunInfo(BaseModel):
    """
    Genesis 4 run information.

    Attributes
    ----------
    error : bool
        True if an error occurred during the Genesis run.
    error_reason : str or None
        Error explanation, if `error` is set.
    run_script : str
        The command-line arguments used to run Genesis
    output_log : str
        Genesis 4 output log
    start_time : float
        Start time of the process
    end_time : float
        End time of the process
    run_time : float
        Wall clock run time of the process
    """

    error: bool = pydantic.Field(
        default=False, description="`True` if an error occurred during the Genesis run"
    )
    error_reason: Optional[str] = pydantic.Field(
        default=None, description="Error explanation, if `error` is set."
    )
    run_script: str = pydantic.Field(
        default="", description="The command-line arguments used to run Genesis"
    )
    output_log: str = pydantic.Field(
        default="", repr=False, description="Genesis 4 output log"
    )
    start_time: float = pydantic.Field(
        default=0.0, repr=False, description="Start time of the process"
    )
    end_time: float = pydantic.Field(
        default=0.0, repr=False, description="End time of the process"
    )
    run_time: float = pydantic.Field(
        default=0.0, description="Wall clock run time of the process"
    )

    @property
    def success(self) -> bool:
        """`True` if the run was successful."""
        return not self.error


class HDF5ReferenceFile(BaseModel):
    """An externally-referenced HDF5 file.."""

    key: str
    filename: pathlib.Path


class _FieldH5File(HDF5ReferenceFile):
    type: Literal["field"] = "field"

    def load(self, **kwargs) -> FieldFile:
        with h5py.File(self.filename) as h5:
            if self.type == "field":
                return load_field_file(h5, **kwargs)
        raise NotImplementedError(self.type)


class _ParticleGroupH5File(HDF5ReferenceFile):
    type: Literal["particle_group"] = "particle_group"

    def load(self, **kwargs) -> ParticleGroup:
        with h5py.File(self.filename) as h5:
            return load_particle_group(h5, **kwargs)


def _empty_ndarray() -> np.ndarray:
    return np.zeros(0)


class OutputLattice(BaseModel, extra="allow"):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict)
    aw: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ax: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ay: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    chic_angle: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    chic_lb: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    chic_ld: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    chic_lt: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    cx: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    cy: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    dz: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    gradx: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    grady: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ku: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    kx: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ky: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    phaseshift: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    qf: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    qx: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    qy: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    slippage: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    z: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    zplot: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    extra: Dict[str, OutputDataType] = pydantic.Field(default_factory=dict)


class OutputBeamStat(BaseModel):
    sigma_x: NDArray
    sigma_y: NDArray
    sigma_energy: NDArray

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
    lsc_field: NDArray
    pxmax: NDArray
    pxmin: NDArray
    pxposition: NDArray
    pymax: NDArray
    pymin: NDArray
    pyposition: NDArray
    ssc_field: NDArray
    wakefield: NDArray
    xmax: NDArray
    xmin: NDArray
    xposition: NDArray
    ymax: NDArray
    ymin: NDArray
    yposition: NDArray
    extra: Dict[str, OutputDataType]
    # TODO: px, py, emitx, emity

    @staticmethod
    def calculate_projected_sigma(
        current: np.ndarray,
        position: np.ndarray,
        size: np.ndarray,
    ) -> np.ndarray:
        # Properly calculated the projected value
        x = np.nan_to_num(position)  # <x>_islice
        x2 = np.nan_to_num(size**2)  # <x^2>_islice
        norm = np.sum(current, axis=1)
        # Total projected sigma_x
        sigma_x2 = (
            np.sum((x2 + x**2) * current, axis=1) / norm
            - (np.sum(x * current, axis=1) / norm) ** 2
        )
        return np.sqrt(sigma_x2)

    @staticmethod
    def calculate_bunching(
        current: np.ndarray,
        bunching: np.ndarray,
        bunchingphase: np.ndarray,
    ) -> np.ndarray:
        dat = np.nan_to_num(bunching)  # Convert any nan to zero for averaging.
        phase = np.nan_to_num(bunchingphase)
        return np.abs(np.sum(np.exp(1j * phase) * dat * current, axis=1)) / np.sum(
            current, axis=1
        )

    @staticmethod
    def calculate_simple_stat(
        current: np.ndarray,
        dat: np.ndarray,
    ) -> np.ndarray:
        dat = np.nan_to_num(dat)  # Convert any nan to zero for averaging.
        return np.sum(dat * current, axis=1) / np.sum(current, axis=1)

    @classmethod
    def from_output_beam(cls, beam: OutputBeam) -> OutputBeamStat:
        current = np.nan_to_num(beam.current)

        simple_stats = {
            attr: cls.calculate_simple_stat(current, getattr(beam, attr))
            for attr in set(OutputBeam.model_fields)
            - {"xsize", "ysize", "bunching", "extra", "stat", "units"}
        }
        extra = {
            key: cls.calculate_simple_stat(current, value)
            for key, value in beam.extra.items()
        }
        return OutputBeamStat(
            sigma_x=cls.calculate_projected_sigma(
                current=current,
                size=beam.xsize,
                position=beam.xposition,
            ),
            sigma_y=cls.calculate_projected_sigma(
                current=current,
                size=beam.ysize,
                position=beam.yposition,
            ),
            sigma_energy=cls.calculate_projected_sigma(
                current=current,
                size=beam.energyspread,
                position=beam.energy,
            ),
            bunching=cls.calculate_bunching(
                current=current,
                bunching=beam.bunching,
                bunchingphase=beam.bunchingphase,
            ),
            extra=extra,
            **simple_stats,
        )


class OutputBeam(BaseModel):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict)
    lsc_field: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ssc_field: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    alphax: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    alphay: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    betax: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    betay: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    bunching: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    bunchingphase: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    current: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    efield: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    emax: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    emin: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    emitx: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    emity: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    energy: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    energyspread: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    pxmax: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    pxmin: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    pxposition: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    pymax: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    pymin: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    pyposition: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    wakefield: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    xmax: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    xmin: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    xposition: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    xsize: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ymax: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ymin: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    yposition: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ysize: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    extra: Dict[str, OutputDataType] = pydantic.Field(default_factory=dict)

    @pydantic.computed_field
    @cached_property
    def stat(self) -> OutputBeamStat:
        return OutputBeamStat.from_output_beam(self)


class OutputMetaDumps(BaseModel):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict)
    ndumps: int = 0
    extra: Dict[str, OutputDataType] = pydantic.Field(default_factory=dict)


class OutputMetaVersion(BaseModel, extra="allow"):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict)
    beta: float = 0.0
    build_info: str = ""
    major: float = 0.0
    minor: float = 0.0
    revision: float = 0.0
    extra: Dict[str, OutputDataType] = pydantic.Field(default_factory=dict)


class OutputMeta(BaseModel, extra="allow"):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict)
    beamdumps: OutputMetaDumps = pydantic.Field(default_factory=OutputMetaDumps)
    fielddumps: OutputMetaDumps = pydantic.Field(default_factory=OutputMetaDumps)
    host: str = ""
    input_file: str = ""
    lattice_file: str = ""
    time_stamp: str = ""
    user: str = ""
    version: OutputMetaVersion = pydantic.Field(default_factory=OutputMetaVersion)
    cwd: str = ""
    mpisize: float = 0.0
    extra: Dict[str, OutputDataType] = pydantic.Field(default_factory=dict)


class OutputGlobal(BaseModel, extra="allow"):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict)
    frequency: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    gamma0: float = 0.0
    lambdaref: float = 0.0
    one4one: float = 0.0
    s: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    sample: float = 0.0
    scan: float = 0.0
    slen: float = 0.0
    time: float = 0.0
    extra: Dict[str, OutputDataType] = pydantic.Field(default_factory=dict)


class OutputFieldStat(BaseModel):
    xposition: NDArray
    xsize: NDArray
    yposition: NDArray
    ysize: NDArray

    @classmethod
    def from_output_field(cls, field: OutputField) -> OutputFieldStat:
        return OutputFieldStat(
            xposition=np.mean(field.xposition, axis=1),
            yposition=np.mean(field.yposition, axis=1),
            xsize=np.mean(field.xsize, axis=1),
            ysize=np.mean(field.ysize, axis=1),
        )


class OutputField(BaseModel, extra="allow"):
    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict)
    dgrid: float = 0.0
    intensity_farfield: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    intensity_nearfield: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ngrid: float = 0.0
    phase_farfield: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    phase_nearfield: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    power: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    xdivergence: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    xpointing: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    xposition: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    xsize: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ydivergence: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ypointing: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    yposition: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ysize: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    extra: Dict[str, OutputDataType] = pydantic.Field(default_factory=dict)

    @pydantic.computed_field
    @property
    def peak_power(self) -> float:
        return np.max(self.power, axis=1)


LoadableH5File = Union[
    _ParticleGroupH5File,
    _FieldH5File,
]
_T = TypeVar("_T", bound=BaseModel)


class Genesis4Output(Mapping, BaseModel, arbitrary_types_allowed=True):
    """
    Genesis 4 command output.

    Attributes
    ----------
    data : dict
        Dictionary of data from a variety of sources, including the main
        output HDF5 file, particle files, and field files.
    unit_info : Dict[str, pmd_unit]
        Unit information.
    run : RunInfo
        Execution information - e.g., how long did it take and what was
        the output from Genesis 4.
    alias : dict[str, str]
        Dictionary of aliased data keys.
    """

    beam: OutputBeam = pydantic.Field(default_factory=OutputBeam)
    field_info: OutputField = pydantic.Field(default_factory=OutputField)
    lattice: OutputLattice = pydantic.Field(default_factory=OutputLattice)
    global_: OutputGlobal = pydantic.Field(default_factory=OutputGlobal)
    meta: OutputMeta = pydantic.Field(default_factory=OutputMeta)
    version: OutputMetaVersion = pydantic.Field(default_factory=OutputMetaVersion)
    extra: Dict[str, OutputDataType] = pydantic.Field(default_factory=dict)
    field: Dict[str, FieldFile] = pydantic.Field(
        default_factory=dict,
        exclude=True,
    )
    particles: Dict[str, ParticleGroup] = pydantic.Field(
        default_factory=dict,
        exclude=True,
    )
    unit_info: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict)
    run: RunInfo = pydantic.Field(default_factory=RunInfo)
    alias: Dict[str, str] = pydantic.Field(default_factory=dict)
    field_files: Dict[str, LoadableH5File] = pydantic.Field(default_factory=dict)
    particle_files: Dict[str, LoadableH5File] = pydantic.Field(default_factory=dict)

    @pydantic.computed_field
    @property
    def field_energy(self) -> np.ndarray:
        dat = self.field_info.power

        # Integrate to get J
        nslice = dat.shape[1]
        slen = self.global_.slen
        ds = slen / nslice
        return np.sum(dat, axis=1) * ds / c_light

    pulse_energy = field_energy

    @property
    def fields(self) -> Dict[str, FieldFile]:
        return self.field

    @staticmethod
    def get_output_filename(input: Genesis4Input, workdir: AnyPath) -> pathlib.Path:
        """Get the output filename based on the input/run-related settings."""
        root_name = input.output_path or input.main.setup.rootname
        if not root_name:
            raise RuntimeError(
                "Unable to find 'rootname'; cannot determine output filename."
            )
        return pathlib.Path(workdir) / f"{root_name}.out.h5"

    @classmethod
    def from_input_settings(
        cls,
        input: Genesis4Input,
        workdir: pathlib.Path,
        load_fields: bool = False,
        load_particles: bool = False,
        smear: bool = True,
    ) -> Genesis4Output:
        """
        Load Genesis 4 output based on the configured input settings.

        Parameters
        ----------
        load_fields : bool, default=True
            After execution, load all field files.
        load_particles : bool, default=True
            After execution, load all particle files.
        smear : bool, default=True
            If set, for particles, this will smear the phase over the sample
            (skipped) slices, preserving the modulus.

        Returns
        -------
        Genesis4Output
            The output data.
        """
        output_filename = cls.get_output_filename(input, workdir)
        return cls.from_files(
            output_filename,
            load_fields=load_fields,
            load_particles=load_particles,
            smear=smear,
        )

    @classmethod
    def from_files(
        cls,
        filename: AnyPath,
        load_fields: bool = False,
        load_particles: bool = False,
        smear: bool = True,
    ) -> Genesis4Output:
        """
        Load Genesis 4 output from the given filename.

        Parameters
        ----------
        filename : pathlib.Path or str
        load_fields : bool, default=True
            After execution, load all field files.
            These are assumed to be in the same directory as the primary output
            filename.
        load_particles : bool, default=True
            After execution, load all particle files.
            These are assumed to be in the same directory as the primary output
            filename.
        smear : bool, default=True
            If set, this will smear the particle phase over the sample
            (skipped) slices, preserving the modulus.

        Returns
        -------
        Genesis4Output
            The output data.
        """
        output_root = pathlib.Path(filename).parent

        units = parsers.known_unit.copy()
        with h5py.File(filename, "r") as h5:
            data, loaded_units = parsers.extract_data_and_unit(h5)

        units.update(loaded_units)
        fields = [
            _FieldH5File(
                key=fn.name[: -len(".fld.h5")],
                filename=fn,
            )
            for fn in output_root.glob("*.fld.h5")
        ]
        particles = [
            _ParticleGroupH5File(
                key=fn.name[: -len(".par.h5")],
                filename=fn,
            )
            for fn in output_root.glob("*.par.h5")
        ]

        alias = parsers.extract_aliases(data)
        for alias_from, alias_to in alias.items():
            if alias_to in units:
                units[alias_from] = units[alias_to]

        def instantiate(cls: Type[_T], data_key: str) -> _T:
            dct = data.pop(data_key)
            extra = {key: dct.pop(key) for key in set(dct) - set(cls.model_fields)}
            return cls(
                **dct,
                extra=extra,
                units=units.pop(data_key, {}),
            )

        beam = instantiate(OutputBeam, "beam")
        field_info = instantiate(OutputField, "field")
        lattice = instantiate(OutputLattice, "lattice")
        global_ = instantiate(OutputGlobal, "global")
        meta = instantiate(OutputMeta, "meta")
        version = meta.version
        extra = data

        output = cls(
            beam=beam,
            field_info=field_info,
            lattice=lattice,
            global_=global_,
            version=version,
            extra=extra,
            unit_info=units,
            alias=alias,
            field_files={field.key: field for field in fields},
            particle_files={particle.key: particle for particle in particles},
        )

        if load_fields:
            output.load_fields()

        if load_particles:
            output.load_particles(smear=smear)

        return output

    def load_field_by_name(self, label: str) -> FieldFile:
        """
        Loads a single field file by name into a dictionary.

        Parameters
        ----------
        label : str
            The label of the particles (e.g., "end" of "end.par.h5").

        Returns
        -------
        FieldFile
        """
        lazy = self.field_files[label]
        field = lazy.load()
        self.field[label] = field
        logger.info(f"Loaded field data: '{label}'")
        return field

    def load_particles_by_name(self, label: str, smear: bool = True) -> ParticleGroup:
        """
        Loads a single particle file into openPMD-beamphysics ParticleGroup
        object.

        Parameters
        ----------
        label : str
            The label of the particles (e.g., "end" of "end.par.h5").
        smear : bool, optional, default=True
            If set, will smear the phase over the sample (skipped) slices,
            preserving the modulus.
        """
        lazy = self.particle_files[label]
        group = lazy.load(smear=smear)
        self.particles[label] = group
        logger.info(
            f"Loaded particle data: '{label}' as a ParticleGroup with "
            f"{len(group)} particles"
        )
        return group

    def load_particles(self, smear: bool = True) -> List[str]:
        """
        Loads all particle files produced.

        Parameters
        ----------
        smear : bool, default=True
            If set, for particles, this will smear the phase over the sample
            (skipped) slices, preserving the modulus.

        Returns
        -------
        list of str
            Key names of all loaded particles.
        """
        to_load = list(self.particle_files)
        for name in to_load:
            self.load_particles_by_name(name, smear=smear)
        return list(to_load)

    def load_fields(self) -> List[str]:
        """
        Loads all field files produced.

        Returns
        -------
        list of str
            Key names of all loaded fields.
        """
        to_load = list(self.field_files)
        for label in to_load:
            self.load_field_by_name(label)
        return to_load

    def units(self, key: str) -> Optional[pmd_unit]:
        """pmd_unit of a given key"""
        return self.unit_info.get(key, None)

    def get_array(self, key: str) -> np.ndarray:
        """
        Gets an array, considering aliases
        """
        if key in self.data:
            value = self.data[key]
            assert isinstance(value, np.ndarray)
            return value
        key = self.alias.get(key, key)
        if key in self.data:
            value = self.data[key]
            assert isinstance(value, np.ndarray)
            return value
        raise ValueError(f"Unknown key: {key}")

    def archive(self, h5: h5py.Group, key: str = "output") -> None:
        """
        Dump outputs into the given HDF5 group.

        Parameters
        ----------
        h5 : h5py.Group
            The HDF5 file in which to write the information.
        key : str, default="output"
            The key to use when storing the data.
        """
        tools.store_in_hdf5_file(h5, self, key=key)

    @classmethod
    def from_archive(cls, h5: h5py.Group, key: str = "output") -> Genesis4Output:
        """
        Loads output from the given HDF5 group.

        Parameters
        ----------
        h5 : str or h5py.File
            The filename or handle on h5py.File from which to load data.
        key : str, default="output"
            The key to use when restoring the data.
        """
        loaded = tools.restore_from_hdf5_file(h5, key=key)
        if not isinstance(loaded, Genesis4Output):
            raise ValueError(
                f"Loaded {loaded.__class__.__name__} instead of a "
                f"Genesis4Output instance.  Is key={key} correct?"
            )
        return loaded

    def plot(
        self,
        y: Union[str, Sequence[str]] = "field_energy",
        x="zplot",
        xlim=None,
        ylim=None,
        ylim2=None,
        yscale="linear",
        yscale2="linear",
        y2: Union[str, Sequence[str]] = (),
        nice=True,
        include_layout=True,
        include_legend=True,
        return_figure=False,
        tex=False,
        **kwargs,
    ) -> Optional[matplotlib.figure.Figure]:
        """
        Plots output multiple keys.

        Parameters
        ----------
        y : str or list of str
            List of keys to be displayed on the Y axis
        x : str
            Key to be displayed as X axis
        xlim : list
            Limits for the X axis
        ylim : list
            Limits for the Y axis
        ylim2 : list
            Limits for the secondary Y axis
        yscale: str
            one of "linear", "log", "symlog", "logit", ... for the Y axis
        yscale2: str
            one of "linear", "log", "symlog", "logit", ... for the secondary Y axis
        y2 : list
            List of keys to be displayed on the secondary Y axis
        nice : bool
            Whether or not a nice SI prefix and scaling will be used to
            make the numbers reasonably sized. Default: True
        include_layout : bool
            Whether or not to include a layout plot at the bottom. Default: True
            Whether or not the plot should include the legend. Default: True
        return_figure : bool
            Whether or not to return the figure object for further manipulation.
            Default: True
        kwargs : dict
            Extra arguments can be passed to the specific plotting function.

        Returns
        -------
        fig : matplotlib.pyplot.figure.Figure
            The plot figure for further customizations or `None` if `return_figure` is set to False.
        """

        # Expand keys
        if isinstance(y, str):
            y = y.split()
        if isinstance(y2, str):
            y2 = y2.split()

        y = list(y)
        y2 = list(y2)

        # Special case
        for yk in (y, y2):
            for i, key in enumerate(yk):
                if "power" in key:
                    yk[i] = "peak_power"

        return plot_stats_with_layout(
            self,
            ykeys=y,
            ykeys2=y2,
            xkey=x,
            xlim=xlim,
            ylim=ylim,
            ylim2=ylim2,
            yscale=yscale,
            yscale2=yscale2,
            nice=nice,
            tex=tex,
            include_layout=include_layout,
            include_legend=include_legend,
            return_figure=return_figure,
            **kwargs,
        )

    def info(self):
        info = {
            key: get_description_for_value(key, self.data[key])
            for key in sorted(self.data)
        }
        annotations = {key: str(self.unit_info.get(key, "")) for key in info}
        return tools.table_output(
            info,
            annotations=annotations,
        )

    def get_description_for_key(self, key: str) -> str:
        """
        Returns a line describing an output
        """
        value = self[key]
        units = self.unit_info.get(key, "")
        return get_description_for_value(key, value, units)

    def __getitem__(self, key: str) -> Any:
        """Support for Mapping -> easy access to data."""
        if key in self.data:
            return self.data[key]
        if key in self.alias:
            alias = self.alias[key]
            if alias in self.data:
                return self.data[alias]
        raise KeyError(key)

    def __iter__(self) -> Generator[str, None, None]:
        """Support for Mapping -> easy access to data."""
        # yield from iter(self.data)
        # TODO
        yield from []

    def __len__(self) -> int:
        """Support for Mapping -> easy access to data."""
        # return len(self.data)
        # TODO
        return 0


def get_description_for_value(key: str, value) -> str:
    """
    Returns a line describing an output
    """
    if isinstance(value, dict):
        return ""

    if isinstance(value, str):
        if len(value) > 200:
            return "long str: " + value[0:20].replace("\n", " ") + "..."
        return value
    if np.isscalar(value):
        return f"{value}"
    if isinstance(value, np.ndarray):
        return f"array: {str(value.shape):10}"
    if isinstance(value, list):
        return str(value)
    raise ValueError(f"Cannot describe {key}")


def projected_variance_from_slice_data(x2, x1, current):
    """
    Slice variance data individually removes the mean values.
    This restores that in a proper projection calc.

    Parameters
    ----------
    x2: numpy.ndarray
        2D <x^2 - <x> >_islice array
    x: numpy.ndarray
        2D <x>_islice
    current: numpy.ndarray
        1D current array

    Returns
    -------
    projected_variance

    """
    norm = np.sum(current, axis=1)

    return (
        np.sum((x2 + x1**2) * current, axis=1) / norm
        - (np.sum(x1 * current, axis=1) / norm) ** 2
    )


def load_field_file(file: Union[AnyPath, h5py.File]) -> FieldFile:
    """
    Load a single .dfl.h5 file into .output
    """
    if isinstance(file, h5py.File):
        dfl, param = readers.load_genesis4_fields(file)
        filename = file.filename
    else:
        filename = pathlib.Path(file)
        if not h5py.is_hdf5(filename):
            raise ValueError(f"Field file {filename} is not an HDF5 file")

        with h5py.File(filename, "r") as h5:
            dfl, param = readers.load_genesis4_fields(h5)

    label = pathlib.Path(filename).name
    if label.endswith("fld.h5"):
        label = label[:-7]

    return FieldFile(
        label=label,
        dfl=dfl,
        param=FieldFileParams(**param),
    )
