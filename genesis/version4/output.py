from __future__ import annotations

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
    FieldFileDict,
    FieldFileParamDict,
    OutputBeamDict,
    OutputFieldDict,
    OutputGlobalDict,
    OutputLatticeDict,
    OutputMetaDict,
    OutputMetaVersionDict,
    PydanticNDArray,
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


class RunInfo(pydantic.BaseModel):
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


class HDF5ReferenceFile(pydantic.BaseModel):
    """An externally-referenced HDF5 file.."""

    key: str
    filename: pathlib.Path


class _FieldH5File(HDF5ReferenceFile):
    type: Literal["field"] = "field"

    def load(self, **kwargs) -> FieldFileDict:
        with h5py.File(self.filename) as h5:
            if self.type == "field":
                return load_field_file(h5, **kwargs)
        raise NotImplementedError(self.type)


class _ParticleGroupH5File(HDF5ReferenceFile):
    type: Literal["particle_group"] = "particle_group"

    def load(self, **kwargs) -> ParticleGroup:
        with h5py.File(self.filename) as h5:
            return load_particle_group(h5, **kwargs)


LoadableH5File = Union[
    _ParticleGroupH5File,
    _FieldH5File,
]
DataType = Union[float, int, str, bool, PydanticNDArray]


def _split_data(data: Dict[str, DataType], prefix: str) -> Dict[str, Any]:
    res = {}

    def add_item(key: str, value: Any, parent: Dict[str, Any]) -> None:
        if "/" not in key:
            parent[key] = value
        else:
            first, rest = key.split("/", 1)
            add_item(key=rest, value=value, parent=parent.setdefault(first, {}))

    for key, value in data.items():
        if key.startswith(prefix):
            key = key[len(prefix) :].lstrip("/")
            add_item(key, value, res)
    return res


class Genesis4Output(Mapping, pydantic.BaseModel, arbitrary_types_allowed=True):
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

    data: Dict[str, DataType] = pydantic.Field(default_factory=dict)
    field: Dict[str, FieldFileDict] = pydantic.Field(
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

    def __repr__(self):
        return f"{self.__class__.__name__}(run={self.run})"

    def to_string(self, mode: Literal["html", "markdown"]) -> str:
        if mode == "html":
            return tools.html_table_repr(self, [])
        if mode == "markdown":
            return str(tools.ascii_table_repr(self, []))
        raise NotImplementedError(f"Render mode {mode} unsupported")

    @property
    def beam(self) -> OutputBeamDict:
        """Beam-related output information dictionary."""
        return typing.cast(OutputBeamDict, _split_data(self.data, "Beam/"))

    @property
    def field_info(self) -> OutputFieldDict:
        """Field-related output information dictionary."""
        return typing.cast(OutputFieldDict, _split_data(self.data, "Field/"))

    @property
    def lattice(self) -> OutputLatticeDict:
        """Lattice-related output information dictionary."""
        return typing.cast(OutputLatticeDict, _split_data(self.data, "Lattice/"))

    @property
    def global_(self) -> OutputGlobalDict:
        """Global settings-related output information dictionary."""
        return typing.cast(OutputGlobalDict, _split_data(self.data, "Global/"))

    @property
    def meta(self) -> OutputMetaDict:
        """Run meta information information dictionary."""
        return typing.cast(OutputMetaDict, _split_data(self.data, "Meta/"))

    @property
    def version(self) -> OutputMetaVersionDict:
        """Version-related information dictionary."""
        return typing.cast(
            OutputMetaVersionDict, _split_data(self.data, "Meta/Version/")
        )

    @property
    def fields(self) -> Dict[str, FieldFileDict]:
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

        output = cls(
            data=data,
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

    def load_field_by_name(self, label: str) -> FieldFileDict:
        """
        Loads a single field file by name into a dictionary.

        Parameters
        ----------
        label : str
            The label of the particles (e.g., "end" of "end.par.h5").

        Returns
        -------
        FieldFileDict
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

    def stat(self, key):
        """
        Calculate a statistic of the beam or field
        along z.
        """

        # Derived stats
        if key.startswith("beam_sigma_"):
            comp = key[11:]
            if comp not in ("x", "px", "y", "py", "energy"):
                raise NotImplementedError(f"Unsupported component for f{key}: '{comp}'")

            current = np.nan_to_num(self.beam["current"])

            if comp in ("x", "y"):
                k2 = f"{comp}size"
                k1 = f"{comp}position"
            elif comp in ("energy"):
                k2 = "energyspread"
                k1 = "energy"
            else:
                # TODO: emittance from alpha, beta, etc.
                raise NotImplementedError(f"TODO: {key}")
            x2 = np.nan_to_num(self.beam[k2] ** 2)  # <x^2>_islice
            x1 = np.nan_to_num(self.beam[k1])  # <x>_islice
            sigma_X2 = projected_variance_from_slice_data(x2, x1, current)

            return np.sqrt(sigma_X2)

        # Original stats
        key = self.alias.get(key, key)

        # Peak power
        if "power" in key.lower():
            if "field/" in key.lower():
                dat = self.data[key]
            else:
                dat = self.field_info["power"]
            return np.max(dat, axis=1)

        if key.startswith("Lattice"):
            return self.get_array(key)

        if key.startswith("Beam"):
            dat = self.get_array(key)
            skey = key.split("/")[1]

            # Average over the beam taking to account the weighting (current)
            current = np.nan_to_num(self.beam["current"])

            if skey in ("xsize", "ysize"):
                # TODO: emitx, emity
                # Properly calculated the projected value
                plane = skey[0]
                x = np.nan_to_num(self.beam[f"{plane}position"])  # <x>_islice
                x2 = np.nan_to_num(self.beam[f"{plane}size"] ** 2)  # <x^2>_islice
                norm = np.sum(current, axis=1)
                # Total projected sigma_x
                sigma_x2 = (
                    np.sum((x2 + x**2) * current, axis=1) / norm
                    - (np.sum(x * current, axis=1) / norm) ** 2
                )

                output = np.sqrt(sigma_x2)
            elif skey == "bunching":
                # The bunching calc needs to take the phase into account.
                dat = np.nan_to_num(dat)  # Convert any nan to zero for averaging.
                phase = np.nan_to_num(self.beam["bunchingphase"])
                output = np.abs(
                    np.sum(np.exp(1j * phase) * dat * current, axis=1)
                ) / np.sum(current, axis=1)

            else:
                # Simple stat
                dat = np.nan_to_num(dat)  # Convert any nan to zero for averaging.
                output = np.sum(dat * current, axis=1) / np.sum(current, axis=1)

            return output

        elif key.lower() in ["field_energy", "pulse_energy"]:
            dat = self.field_info["power"]

            # Integrate to get J
            nslice = dat.shape[1]
            slen = self.global_["slen"]
            ds = slen / nslice
            return np.sum(dat, axis=1) * ds / c_light

        elif key.startswith("Field"):
            dat = self.get_array(key)
            skey = key.split("/")[1]
            if skey in [
                "xposition",
                "xsize",
                "yposition",
                "ysize",
            ]:
                return np.mean(dat, axis=1)

        raise ValueError(f"Cannot compute stat for: '{key}'")

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

    def info(self) -> None:
        print("Output data\n")
        print("key                       value              unit")
        print(50 * "-")
        for k in sorted(self.data):
            line = self.get_description_for_key(k)
            print(line)

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
        yield from iter(self.data)

    def __len__(self) -> int:
        """Support for Mapping -> easy access to data."""
        return len(self.data)


def get_description_for_value(key: str, value, units) -> str:
    """
    Returns a line describing an output
    """
    if isinstance(value, dict):
        return ""

    if isinstance(value, str):
        if len(value) > 200:
            descrip = "long str: " + value[0:20].replace("\n", " ") + "..."
        else:
            descrip = value
    elif np.isscalar(value):
        descrip = f"{value} {units} "
    elif isinstance(value, np.ndarray):
        descrip = f"array: {str(value.shape):10}  {units}"
    elif isinstance(value, list):
        descrip = str(value)
    else:
        raise ValueError(f"Cannot describe {key}")

    line = f"{key:25} {descrip}"
    return line


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


def load_field_file(file: Union[AnyPath, h5py.File]) -> FieldFileDict:
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

    return {
        "label": label,
        "dfl": dfl,
        "param": typing.cast(FieldFileParamDict, param),
    }
