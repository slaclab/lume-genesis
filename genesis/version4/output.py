from __future__ import annotations
import dataclasses
import logging
import pathlib

# import shutil
import typing

import h5py
import numpy as np
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.genesis import genesis4_par_to_data
from pmd_beamphysics.units import pmd_unit, c_light
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from .input.types import AnyPath
from . import parsers, readers
from .plot import plot_stats_with_layout


if typing.TYPE_CHECKING:
    from .input.core import Genesis4Input


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RunInfo:
    error: bool = False
    run_script: str = ""
    output_log: str = ""
    error_reason: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    run_time: float = 0.0


LatticeDict = TypedDict(
    "LatticeDict",
    {
        "aw": np.ndarray,
        "ax": np.ndarray,
        "ay": np.ndarray,
        "chic_angle": np.ndarray,
        "chic_lb": np.ndarray,
        "chic_ld": np.ndarray,
        "chic_lt": np.ndarray,
        "cx": np.ndarray,
        "cy": np.ndarray,
        "dz": np.ndarray,
        "gradx": np.ndarray,
        "grady": np.ndarray,
        "ku": np.ndarray,
        "kx": np.ndarray,
        "ky": np.ndarray,
        "phaseshift": np.ndarray,
        "qf": np.ndarray,
        "qx": np.ndarray,
        "qy": np.ndarray,
        "slippage": np.ndarray,
        "z": np.ndarray,
        "zplot": np.ndarray,
    },
)


BeamDict = TypedDict(
    "BeamDict",
    {
        "LSCfield": np.ndarray,
        "alphax": np.ndarray,
        "alphay": np.ndarray,
        "betax": np.ndarray,
        "betay": np.ndarray,
        "bunching": np.ndarray,
        "bunchingphase": np.ndarray,
        "current": np.ndarray,
        "efield": np.ndarray,
        "emax": np.ndarray,
        "emin": np.ndarray,
        "emitx": np.ndarray,
        "emity": np.ndarray,
        "energy": np.ndarray,
        "energyspread": np.ndarray,
        "pxmax": np.ndarray,
        "pxmin": np.ndarray,
        "pxposition": np.ndarray,
        "pymax": np.ndarray,
        "pymin": np.ndarray,
        "pyposition": np.ndarray,
        "wakefield": np.ndarray,
        "xmax": np.ndarray,
        "xmin": np.ndarray,
        "xposition": np.ndarray,
        "xsize": np.ndarray,
        "ymax": np.ndarray,
        "ymin": np.ndarray,
        "yposition": np.ndarray,
        "ysize": np.ndarray,
    },
)


MetaDict = TypedDict(
    "MetaDict",
    {
        "Beamdumps/ndumps": int,
        "Fielddumps/ndumps": int,
        "HOST": str,
        "InputFile": str,
        "User": str,
        "TimeStamp": str,
        "Version/Beta": float,
        "Version/Build_Info": str,
        "Version/Major": float,
        "Version/Minor": float,
        "Version/Revision": float,
        "cwd": str,
        "mpisize": float,
    },
)


VersionDict = TypedDict(
    "VersionDict",
    {
        "Beta": float,
        "Build_Info": str,
        "Major": float,
        "Minor": float,
        "Revision": float,
    },
)


GlobalDict = TypedDict(
    "GlobalDict",
    {
        "frequency": np.ndarray,
        "gamma0": float,
        "lambdaref": float,
        "one4one": float,
        "s": np.ndarray,
        "sample": float,
        "scan": float,
        "slen": float,
        "time": float,
    },
)


FieldDict = TypedDict(
    "FieldDict",
    {
        "dgrid": float,
        "intensity-farfield": np.ndarray,
        "intensity-nearfield": np.ndarray,
        "ngrid": float,
        "phase-farfield": np.ndarray,
        "phase-nearfield": np.ndarray,
        "power": np.ndarray,
        "xdivergence": np.ndarray,
        "xpointing": np.ndarray,
        "xposition": np.ndarray,
        "xsize": np.ndarray,
        "ydivergence": np.ndarray,
        "ypointing": np.ndarray,
        "yposition": np.ndarray,
        "ysize": np.ndarray,
    },
)


FieldFileDict = TypedDict(
    "FieldFileDict",
    {
        #  number of gridpoints in one transverse dimension, equal to nx and ny above
        "gridpoints": int,
        # gridpoint spacing (meter)
        "gridsize": float,
        # starting position (meter)
        "refposition": float,
        # radiation wavelength (meter)
        "wavelength": float,
        # number of slices
        "slicecount": int,
        # slice spacing (meter)
        "slicespacing": float,
    },
)


class LazyLoadedFile:
    ...


@dataclasses.dataclass
class Genesis4Output:
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

    data: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)
    unit_info: Dict[str, pmd_unit] = dataclasses.field(default_factory=dict, repr=False)
    run: RunInfo = dataclasses.field(default_factory=lambda: RunInfo(), repr=False)
    alias: Dict[str, str] = dataclasses.field(default_factory=dict)
    particle_files: Dict[str, h5py.File] = dataclasses.field(default_factory=dict)

    @property
    def beam(self) -> BeamDict:
        """Beam-related output information dictionary."""
        return typing.cast(BeamDict, self._split_data("Beam/"))

    @property
    def field(self) -> FieldDict:
        """Field-related output information dictionary."""
        return typing.cast(FieldDict, self._split_data("Field/"))

    @property
    def lattice(self) -> LatticeDict:
        """Lattice-related output information dictionary."""
        return typing.cast(LatticeDict, self._split_data("Lattice/"))

    @property
    def global_(self) -> GlobalDict:
        """Global settings-related output information dictionary."""
        return typing.cast(GlobalDict, self._split_data("Global/"))

    @property
    def meta(self) -> MetaDict:
        """Run meta information information dictionary."""
        return typing.cast(MetaDict, self._split_data("Meta/"))

    @property
    def version(self) -> VersionDict:
        """Version-related information dictionary."""
        return typing.cast(VersionDict, self._split_data("Meta/Version/"))

    @property
    def particles(self) -> dict:
        """Loaded particles."""
        return self.data.get("particles", {})

    def _split_data(self, prefix: str) -> Dict[str, Any]:
        res = {}
        for key, value in self.data.items():
            if key.startswith(prefix):
                key = key[len(prefix) :].lstrip("/")
                res[key] = value
        return res

    @staticmethod
    def get_output_filename(input: Genesis4Input, workdir: AnyPath) -> pathlib.Path:
        """Get the output filename based on the input/run-related settings."""
        setup = input.main.get_setup()
        root_name = input.output_path or setup.rootname
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
        load_fields=False,
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
        data["particles"] = {}
        data["fields"] = dict(
            load_field_file(fn) for fn in output_root.glob("*.fld.h5")
        )

        # GOAL always load into memory and avoid accesing temporary files
        # after this point.
        # TODO user access of these things?
        particles = {
            fn.name.split(".")[0]: h5py.File(fn)
            # ParticleGroup(
            #     data=genesis4_par_to_data(str(filename), smear=smear)
            # )
            for fn in output_root.glob("*.par.h5")
        }

        alias = parsers.extract_aliases(data)
        for alias_from, alias_to in alias.items():
            if alias_to in units:
                units[alias_from] = units[alias_to]

        return cls(
            data=data,
            unit_info=units,
            alias=alias,
            particle_files=particles,
        )

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
        group = ParticleGroup(
            data=genesis4_par_to_data(
                self.particle_files[label],
                smear=smear,
            )
        )
        self.data["particles"][label] = group
        logger.info(
            f"Loaded particle data: '{label}' as a ParticleGroup with "
            f"{len(group)} particles"
        )
        return group

    def load_all_particles(self, smear: bool = True) -> List[str]:
        """
        Loads all particle files produced.
        """
        for name in self.particle_files:
            self.load_particles_by_name(name, smear=smear)
        return list(self.particle_files)

    # def load_fields(self):
    #     """
    #     Loads all field files produced.
    #     """
    #     for file in self.data["field_files"]:
    #         self.load_field_file(file)
    #
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
                dat = self.field["power"]
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
            dat = self.field["power"]

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
            return self.data[key]
        key = self.alias.get(key, key)
        if key in self.data:
            return self.data[key]
        raise ValueError(f"Unknown key: {key}")

    def archive(self, h5=None):
        """
        Dump inputs and outputs into HDF5 file.

        Parameters
        ----------
        h5 : str or h5py.File
            The filename or handle to HDF5 file in which to write the information.
            If not in informed, a new file is generated.

        Returns
        -------
        h5 : h5py.File
            Handle to the HDF5 file.
        """
        raise NotImplementedError()
        # if not h5:
        #     h5 = "genesis4_" + self.fingerprint() + ".h5"
        #
        # if isinstance(h5, str):
        #     if "outfile" in self.output:
        #         shutil.copy(self.output["outfile"], h5)
        #         self.vprint(f"Archiving to file {h5}")
        #
        #     # fname = os.path.expandvars(h5)
        #     # g = h5py.File(fname, 'w')
        #     # self.vprint(f'Archiving to file {fname}')
        # else:
        #     g = h5
        #
        # return h5

    def load_archive(self, h5, configure=True):
        """
        Loads input and output from archived h5 file.

        Parameters
        ----------
        h5 : str or h5py.File
            The filename or handle on h5py.File from which to load input and output data
        configure : bool, optional
            Whether or not to invoke the configure method after loading, by default True
        """
        raise NotImplementedError

    def plot(
        self,
        y="field_energy",
        x="zplot",
        xlim=None,
        ylim=None,
        ylim2=None,
        yscale="linear",
        yscale2="linear",
        y2=[],
        nice=True,
        include_layout=True,
        include_legend=True,
        return_figure=False,
        tex=False,
        **kwargs,
    ):
        """
        Plots output multiple keys.

        Parameters
        ----------
        y : list
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

    def output_info(self):
        print("Output data\n")
        print("key                       value              unit")
        print(50 * "-")
        for k in sorted(list(self.data)):
            line = self.get_description_for_key(k)
            print(line)

    def get_description_for_key(self, key: str) -> str:
        """
        Returns a line describing an output
        """
        value = self.data[key]
        units = self.unit_info.get(key, "")
        return get_description_for_value(key, value, units)


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


def load_field_file(filename: AnyPath) -> Tuple[str, FieldFileDict]:
    """
    Load a single .dfl.h5 file into .output
    """
    if not h5py.is_hdf5(filename):
        raise ValueError(f"Field file {filename} is not an HDF5 file")
    with h5py.File(filename, "r") as h5:
        dfl, param = readers.load_genesis4_fields(h5)

    label = pathlib.Path(filename).name
    if label.endswith("fld.h5"):
        label = label[:-7]

    result = {"dfl": dfl, **param}
    return label, result