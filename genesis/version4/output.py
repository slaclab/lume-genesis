from __future__ import annotations

import inspect
import logging
import operator
import pathlib
import typing
from functools import cached_property
from typing import (
    Any,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import h5py
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pydantic
import pydantic.alias_generators
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import c_light, pmd_unit
from typing_extensions import override

from .. import tools
from . import archive as _archive
from . import parsers
from .field import FieldFile, get_key_from_filename
from .loadable import LoadableFieldH5File, LoadableParticleGroupH5File
from .particles import Genesis4ParticleData
from .plot import PlotLimits, PlotMaybeLimits, plot_stats_with_layout
from .types import (
    AnyPath,
    BaseModel,
    FileKey,
    NDArray,
    OutputDataType,
    PydanticPmdUnit,
)

try:
    from collections.abc import Mapping
except ImportError:
    from typing import Mapping

if typing.TYPE_CHECKING:
    from .input.core import Genesis4Input


logger = logging.getLogger(__name__)


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


def _empty_ndarray():
    return np.zeros(0)


class _OutputBase(BaseModel):
    """Output model base class."""

    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
    extra: Dict[str, OutputDataType] = pydantic.Field(
        default_factory=dict,
        description=(
            "Additional Genesis 4 output data.  This is a future-proofing mechanism "
            "in case Genesis 4 changes and LUME-Genesis is not yet ready for it."
        ),
    )
    hdf_key_map: Dict[str, str] = pydantic.Field(
        default_factory=dict,
        description="Mapping of attributes to HDF5 file keys",
        repr=False,
    )

    def __init__(self, **kwargs: Any) -> None:
        extra = _split_extra(type(self), kwargs)
        super().__init__(**kwargs, extra=extra)

    @classmethod
    def from_hdf5_data(cls, dct: Dict[str, OutputDataType], **kwargs: Any):
        dct = cls._fix_scalar_data(dct)
        try:
            units: Dict[str, PydanticPmdUnit] = dct["units"]
        except KeyError:
            pass
        else:
            for key, unit_ in list(units.items()):
                if unit_ == parsers.known_unit["mec2"]:
                    if isinstance(dct[key], (float, np.ndarray, int)):
                        dct[key] = dct[key] * parsers.mec2
                        units[key] = parsers.known_unit["eV"]
        return cls(**dct, **kwargs)

    @classmethod
    def _fix_scalar_data(
        cls, dct: Dict[str, OutputDataType]
    ) -> Dict[str, OutputDataType]:
        """Fix data from HDF5 that's mistakenly scalar."""
        res = {}
        for key, value in dct.items():
            info = cls.model_fields.get(key, None)
            if info is None:
                res[key] = value
            elif info.annotation is np.ndarray and isinstance(value, (float, int)):
                res[key] = np.asarray([value])
            elif (
                inspect.isclass(info.annotation)
                and issubclass(info.annotation, _OutputBase)
                and isinstance(value, dict)
            ):
                res[key] = info.annotation._fix_scalar_data(value)
            else:
                res[key] = value
        return res


class OutputLattice(_OutputBase):
    """
    Genesis 4 lattice output information (HDF5 Group ``"/Lattice"``).

    Array indices are per step of the simulation, which relates to the Z
    position.

    The undulator strength, quadrupole field and other are resolved with the
    resolution of the requested integration step size, which is given in the
    dataset `.dz`.

    For the z-position there are two datasets. The regular one `.z` has the same
    length and describes the lattice quantities from the position ``.z[i]`` to
    ``.z[i]+.dz[i]`` of the integration step ``i``. The dataset `.zplot` is
    used for plotting the beam or field parameters along the undulator.

    Note that those are evaluated before the integration started, so that there
    can be one more entry than the lattice datasets. Also if the output is
    reduced by the output step option in the tracking command, the length of
    zplot is shorter because it has to match the length of the beam and field
    parameters.
    """

    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
    aw: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            r"The dimensionless rms undulator parameter. For planar undulator this value "
            r"is smaller by a factor $1 / \sqrt{2}$ than its K-value, while for helical "
            r"undulator rms and peak values are identical."
        ),
    )
    ax: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Offset of the undulator module in $x$ in meter.",
    )
    ay: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Offset of the undulator module in $y$ in meter.",
    )
    chic_angle: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    chic_lb: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Length of an individual dipole in meter.",
    )
    chic_ld: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            r"Drift between the outer and inner dipoles, projected onto the undulator "
            r"axis. The actual path length is longer by the factor $1/\cos\theta$, where "
            r"$\theta$ is the bending angle of an individual dipole. "
        ),
    )
    chic_lt: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    cx: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Kick angle in $x$ in units of $\gamma \beta_x$.",
    )
    cy: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Kick angle in $y$ in units of $\gamma \beta_y$.",
    )
    dz: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Step length",
    )
    gradx: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            r"Relative transverse gradient of undulator field in $x$ $\equiv (1/a_w) "
            r"\partial a_w/\partial x$."
        ),
    )
    grady: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            r"Relative transverse gradient of undulator field in $y$ $\equiv (1/a_w) "
            r"\partial a_w/\partial y$."
        ),
    )
    ku: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    kx: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            r"Roll-off parameter of the quadratic term of the undulator field in x. It "
            r"is normalized with respect to $k_u^2$."
        ),
    )
    ky: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=r"Roll-off parameter of the quadratic term of the undulator field in y.",
    )
    phaseshift: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Phase shift in radians.",
    )
    qf: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Quadrupole focusing strength in $1/m^2$",
    )
    qx: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Quadrupole offset in $x$ in meters.",
    )
    qy: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Quadrupole offset in $y$ in meters.",
    )
    slippage: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Lattice slippage."
    )
    z: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            "Step Z position in m.  The same length and describes the lattice "
            "quantities from the position ``.z[i]`` to ``.z[i]+.dz[i]`` of the "
            "i-th integration step."
        ),
    )
    zplot: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="For plotting the beam or field parameters along the undulator.",
    )
    extra: Dict[str, OutputDataType] = pydantic.Field(
        default_factory=dict,
        description=(
            "Additional Genesis 4 output data.  This is a future-proofing mechanism "
            "in case Genesis 4 changes and LUME-Genesis is not yet ready for it."
        ),
    )

    def plot(self, ax: Optional[matplotlib.axes.Axes] = None) -> matplotlib.axes.Axes:
        """
        Plot the lattice $aw$ and $qf$ versus $z$.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to insert the plot.  If not specified, a new plot will be
            generated.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots()

        assert ax is not None

        aw_color = "tab:red"
        ax.set_xlabel(r"$z$ (m)")
        ax.set_ylabel(r"$a_w$", color=aw_color)
        ax.tick_params(axis="y", labelcolor=aw_color)
        ax.step(self.z, self.aw, color=aw_color, where="post")

        ax2 = ax.twinx()
        qf_color = "tab:blue"
        ax2.set_ylabel(r"$k_1$ (m$^{-2}$)", color=qf_color)
        ax2.tick_params(axis="y", labelcolor=qf_color)
        ax2.step(self.z, self.qf, color=qf_color, where="post")
        plt.show()
        return ax


class OutputBeamStat(_OutputBase):
    """
    Output Beam statistics, based on HDF5 ``/Beam``.

    These are calculated for you by LUME-Genesis.
    """

    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
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

    xmin: NDArray
    xmax: NDArray
    xposition: NDArray
    # xsize: NDArray

    ymin: NDArray
    ymax: NDArray
    yposition: NDArray
    # ysize: NDArray
    extra: Dict[str, OutputDataType] = pydantic.Field(
        default_factory=dict,
        description=(
            "Additional Genesis 4 output data.  This is a future-proofing mechanism "
            "in case Genesis 4 changes and LUME-Genesis is not yet ready for it."
        ),
    )
    # TODO: px, py, emitx, emity

    @staticmethod
    def calculate_projected_sigma(
        current: np.ndarray,
        position: np.ndarray,
        size: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate projected sigma.

        Parameters
        ----------
        current : np.ndarray
            1D current array.
        position : np.ndarray
            2D <x>_islice
        size : np.ndarray
            2D <x^2 - <x> >_islice array.

        Returns
        -------
        np.ndarray
        """
        if not len(position) or not len(size) or not len(current):
            return _empty_ndarray()
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
        """
        Calculate bunching.

        Parameters
        ----------
        current : np.ndarray
        bunching : np.ndarray
        bunchingphase : np.ndarray

        Returns
        -------
        np.ndarray
        """
        if not len(bunching) or not len(bunchingphase) or not len(current):
            return _empty_ndarray()
        dat = np.nan_to_num(bunching)  # Convert any nan to zero for averaging.
        phase = np.nan_to_num(bunchingphase)
        return np.abs(np.sum(np.exp(1j * phase) * dat * current, axis=1)) / np.sum(
            current, axis=1
        )

    @classmethod
    def from_output_beam(cls, beam: OutputBeam) -> OutputBeamStat:
        """Calculate all statistics given an `OutputBeam` instance."""
        current = np.nan_to_num(beam.current)

        skip_attrs = {
            # xsize, ysize don't make sense to keep per cmayes
            "xsize",
            "ysize",
            # Calculated below:
            "bunching",
            # Other:
            "emin",
            "emax",
        }
        for attr in set(OutputBeam.model_fields):
            value = getattr(beam, attr)
            if not isinstance(value, np.ndarray):
                skip_attrs.add(attr)

        simple_stats = {
            attr: simple_mean_from_slice_data(getattr(beam, attr), weight=current)
            for attr in set(OutputBeam.model_fields) - skip_attrs
        }
        extra = {
            key: simple_mean_from_slice_data(value, weight=current)
            for key, value in beam.extra.items()
        }
        units = dict(beam.units)
        units["sigma_x"] = pmd_unit("m")
        units["sigma_y"] = pmd_unit("m")
        units["sigma_energy"] = pmd_unit("eV")
        return OutputBeamStat(
            units=units,
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


class OutputBeamGlobal(_OutputBase):
    """Output beam global information. (HDF5 ``/Beam/Global``)"""

    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
    energy: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="",
    )
    energyspread: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="",
    )
    xposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="",
    )
    yposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="",
    )
    xsize: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="",
    )
    ysize: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="",
    )


class OutputBeam(_OutputBase):
    """Output beam information. (HDF5 ``/Beam``)"""

    units: Dict[str, PydanticPmdUnit] = pydantic.Field(default_factory=dict, repr=False)
    globals: OutputBeamGlobal = pydantic.Field(
        default_factory=OutputBeamGlobal,
        description="",
    )

    # The following are evaluated at each integration step.
    # TODO: can be bunching_n and bunchingphase_n keys up to number of
    # harmonics
    bunching: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Evaluated at each integration step. [unitless]",
    )
    bunchingphase: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Evaluated at each integration step. [rad]",
    )
    energy: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            "Evaluated at each integration step. "
            "Genesis4 mc^2 units are automatically converted to eV in LUME-Genesis."
        ),
        # Genesis 4 units: mc^2; LUME-Genesis units: eV
    )
    energyspread: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            "Evaluated at each integration step. "
            "Genesis4 mc^2 units are automatically converted to eV in LUME-Genesis."
        ),
    )

    xsize: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Beam horizontal sigma. [m]"
    )
    ysize: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Beam horizontal sigma. [m]"
    )

    lsc_field: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Longitudinal space charge [eV/m]"
    )
    ssc_field: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Short-range space charge [eV/m]"
    )
    efield: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Efield, internally eloss+longESC [eV/m]",
    )
    wakefield: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Wakefield, internally the energy loss. [eV/m]",
    )

    emin: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            "Particle energy minimum. "
            "Genesis4 mc^2 units are automatically converted to eV in LUME-Genesis."
        ),
    )
    emax: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description=(
            "Particle energy maximum. "
            "Genesis4 mc^2 units are automatically converted to eV in LUME-Genesis."
        ),
    )

    pxmin: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle horizontal minimum momentum [rad]",
    )
    pxmax: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle horizontal maximum momentum [rad]",
    )

    pymin: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle vertical minimum momentum [rad]",
    )
    pymax: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle vertical maximum momentum [rad]",
    )

    pxposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle horizontal position in momentum space [rad]",
    )
    pyposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle vertical position in momentum space [rad]",
    )

    xmin: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle horizontal minimum position [m]",
    )
    xmax: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle horizontal maximum position [m]",
    )

    ymin: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle vertical minimum position [m]",
    )
    ymax: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Particle vertical maximum position [m]",
    )

    xposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Partical horizontal position [m]"
    )
    yposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Partical vertical position [m]"
    )

    # The following are evaluated only at the beginning of the simulation.
    alphax: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss alpha horizontal. Evaluated only at the beginning. [rad]",
    )
    alphay: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss alpha vertical. Evaluated only at the beginning. [rad]",
    )
    betax: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss beta horizontal. Evaluated only at the beginning. [m]",
    )
    betay: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Twiss beta vertical. Evaluated only at the beginning. [m]",
    )
    current: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Beam current. Evaluated only at the beginning. [A]",
    )
    emitx: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Beam horizontal emittance. Evaluated only at the beginning. [m]",
    )
    emity: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Beam vertical emittance. Evaluated only at the beginning. [m]",
    )

    extra: Dict[str, OutputDataType] = pydantic.Field(
        default_factory=dict,
        description=(
            "Additional Genesis 4 output data.  This is a future-proofing mechanism "
            "in case Genesis 4 changes and LUME-Genesis is not yet ready for it."
        ),
    )

    @pydantic.computed_field
    @cached_property
    def stat(self) -> Optional[OutputBeamStat]:
        """
        Calculate statistics for the beam.

        Returns ``None`` if the beam data is malformed and no statistics
        are available.
        """
        if not len(self.energy):
            return None
        return OutputBeamStat.from_output_beam(self)


def _split_extra(cls: Type[BaseModel], dct: dict) -> Dict[str, Any]:
    extra = dct.pop("extra", {})
    assert isinstance(extra, dict)
    # Don't let computed fields make it into 'extra':
    for fld in cls.model_computed_fields:
        dct.pop(fld, None)
    return {key: dct.pop(key) for key in set(dct) - set(cls.model_fields)}


class OutputMetaDumps(_OutputBase):
    """Dump-related output information. (HDF5 ``/Meta/*dumps``)"""

    def __init__(self, **kwargs) -> None:
        filename_keys = [key for key in kwargs if key.startswith("filename_")]
        filenames = kwargs.pop("filenames", {})
        assert isinstance(filenames, dict)
        filenames.update(
            {key[len("filename_") :]: kwargs.pop(key) for key in filename_keys}
        )
        super().__init__(filenames=filenames, **kwargs)

    filenames: Dict[str, str] = pydantic.Field(default_factory=dict)
    intstep: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    ndumps: int = pydantic.Field(default=0, description="Number of dumps")


class OutputMetaVersion(_OutputBase):
    """Version information from Genesis 4 output. (HDF5 ``/Meta/Version``)"""

    beta: bool = pydantic.Field(default=False, description="Beta version flag")
    build_info: str = pydantic.Field(default="", description="Build information string")
    major: float = pydantic.Field(default=0.0, description="Major version number")
    minor: float = pydantic.Field(default=0.0, description="Minor version number")
    revision: float = pydantic.Field(default=0.0, description="Revision numbr")


class OutputMeta(_OutputBase):
    """Meta information from Genesis 4 output. (HDF5 ``/Meta``)"""

    beamdumps: OutputMetaDumps = pydantic.Field(default_factory=OutputMetaDumps)
    fielddumps: OutputMetaDumps = pydantic.Field(default_factory=OutputMetaDumps)
    host: str = pydantic.Field(
        default="",
        description="Hostname where simulation was run",
    )
    input_file: str = pydantic.Field(default="", description="Input filename")
    lattice_file: str = pydantic.Field(default="", description="Lattice filename")
    time_stamp: str = pydantic.Field(
        default="",
        description="Timestamp when data was written",
    )
    user: str = pydantic.Field(default="", description="User who ran the simulation")
    version: OutputMetaVersion = pydantic.Field(default_factory=OutputMetaVersion)
    cwd: str = pydantic.Field(
        default="",
        description="Working directory for the simulation",
    )
    mpisize: float = pydantic.Field(default=0.0, description="Number of MPI processes")


class OutputGlobal(_OutputBase):
    """Global information from Genesis 4 output. (HDF5 ``/Global``)"""

    frequency: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Frequency [eV]",
    )
    gamma0: float = pydantic.Field(
        default=0.0,
        description="Reference energy in units of the electron rest mass",
    )
    lambdaref: float = pydantic.Field(
        default=0.0,
        description="Reference wavelength [m]",
    )
    one4one: bool = pydantic.Field(
        default=False,
        description="Flag to enable or disable simulation to resolve each electron in the simulation.",
    )
    s: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Longitudinal position [m]"
    )
    sample: float = pydantic.Field(
        default=0.0,
        description=(
            "Sample rate in units of the reference wavelength from the setup "
            "namelist, so that the number of slices is given by SLEN / "
            "LAMBDA0/SAMPLE after SLEN has been adjusted to fit the cluster "
            "size."
        ),
    )
    scan: bool = False
    slen: float = pydantic.Field(
        default=0.0,
        description="Slice length [m]",
    )
    time: bool = False


class OutputFieldStat(_OutputBase):
    """Calculated output field statistics. Mean field position and size."""

    intensity_farfield: NDArray
    intensity_nearfield: NDArray
    phase_farfield: NDArray
    phase_nearfield: NDArray
    power: NDArray
    xdivergence: NDArray
    ydivergence: NDArray
    xpointing: NDArray
    ypointing: NDArray
    xposition: NDArray
    yposition: NDArray
    xsize: NDArray
    ysize: NDArray
    energy: NDArray

    extra: Dict[str, OutputDataType] = pydantic.Field(
        default_factory=dict,
        description=(
            "Additional Genesis 4 output data.  This is a future-proofing mechanism "
            "in case Genesis 4 changes and LUME-Genesis is not yet ready for it."
        ),
    )

    @classmethod
    def from_output_field(cls, field: OutputField) -> Optional[OutputFieldStat]:
        """Calculate all statistics given an `OutputField` instance."""
        power = np.nan_to_num(field.power)

        skip_attrs = {
            "energy",  # This is already calculated
        }
        for attr in OutputField.model_fields:
            value = getattr(field, attr)
            if not isinstance(value, np.ndarray):
                skip_attrs.add(attr)

        simple_stats = {
            attr: simple_mean_from_slice_data(getattr(field, attr), weight=power)
            for attr in set(OutputField.model_fields) - skip_attrs
        }
        extra = {
            key: simple_mean_from_slice_data(value, weight=power)
            for key, value in field.extra.items()
        }

        return OutputFieldStat(
            extra=extra,
            energy=field.energy,
            **simple_stats,
        )


class OutputFieldGlobal(_OutputBase):
    """Field-global information from Genesis 4 output. (HDF5 ``/Field/Global``)"""

    energy: NDArray = pydantic.Field(default_factory=_empty_ndarray)
    intensity_farfield: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Field intensity in the far field [arb units]",
    )
    intensity_nearfield: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Field intensity in the near field [arb units]",
    )
    xdivergence: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Horizontal divergence [rad]"
    )
    ydivergence: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Vertical divergence [rad]"
    )
    xpointing: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Horizontal pointing. [rad]",
    )
    ypointing: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Vertical pointing. [rad]",
    )

    xposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Horizontal position. [m]",
    )
    yposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Vertical position. [m]",
    )

    xsize: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Horizontal sigma. [m]",
    )
    ysize: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Vertical sigma. [m]",
    )


class OutputField(_OutputBase):
    globals: OutputFieldGlobal = pydantic.Field(
        default_factory=OutputFieldGlobal,
        description="Global field information (/Field/Global)",
    )
    dgrid: float = pydantic.Field(
        default=0.0,
        description=(
            "Grid extension from the center to one edge. The whole grid is "
            "twice as large with 0 as the center position."
        ),
    )
    intensity_farfield: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Field intensity in the far field [arb units]",
    )
    intensity_nearfield: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Field intensity in the near field [arb units]",
    )
    ngrid: int = pydantic.Field(
        default=0,
        description="Number of grid points per dimension",
    )
    phase_farfield: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Far field phase [rad]"
    )
    phase_nearfield: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Near field phase [rad]"
    )
    power: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Power [W]",
    )
    xdivergence: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Horizontal divergence [rad]"
    )
    ydivergence: NDArray = pydantic.Field(
        default_factory=_empty_ndarray, description="Vertical divergence [rad]"
    )

    xpointing: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Horizontal pointing. [rad]",
    )
    ypointing: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Vertical pointing. [rad]",
    )

    xposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Horizontal position. [m]",
    )
    yposition: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Vertical position. [m]",
    )

    xsize: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Horizontal sigma. [m]",
    )
    ysize: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Vertical sigma. [m]",
    )
    energy: NDArray = pydantic.Field(
        default_factory=_empty_ndarray,
        description="Calculated by LUME-Genesis using slen from /Global.",
    )

    def __init__(self, *args, slen: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if slen is not None:
            self.energy = self.calculate_field_energy(slen)
        self.units["energy"] = parsers.known_unit["J"]
        self.units["peak_power"] = parsers.known_unit["W"]

    @pydantic.computed_field
    @property
    def peak_power(self) -> NDArray:
        """Peak power [W]."""
        if not len(self.power):
            return _empty_ndarray()
        return np.max(self.power, axis=1)

    def calculate_field_energy(self, slen: float) -> np.ndarray:
        """Calculate field energy, given global ``slen``."""
        if not len(self.power):
            return _empty_ndarray()
        # Integrate to get J
        nslice = self.power.shape[1]
        ds = slen / nslice
        return np.sum(self.power, axis=1) * ds / c_light

    @pydantic.computed_field
    @cached_property
    def stat(self) -> Optional[OutputFieldStat]:
        """
        Calculate statistics for the field.

        Returns ``None`` if the field data is malformed and no statistics
        are available.
        """
        return OutputFieldStat.from_output_field(self)


_T = TypeVar("_T", bound=_OutputBase)


class _ArrayInfo(NamedTuple):
    parent: _OutputBase
    array_attr: str
    units: Optional[PydanticPmdUnit]
    field: Union[pydantic.fields.FieldInfo, pydantic.fields.ComputedFieldInfo]
    value: Optional[np.ndarray]
    shape: Optional[Tuple[int, ...]]


def _hdf_summary(
    obj: Union[Genesis4Output, _OutputBase],
    base_attr: str,
    base_key: str,
):
    """
    Generate an HDF5 file key mapping to field information.

    Parameters
    ----------
    obj : Genesis4Output or _OutputBase
    base_attr : str
    base_key : str
    """
    res = {}
    for attr, hdf_key in obj.hdf_key_map.items():
        full_attr = ".".join(attr for attr in (base_attr, attr))
        full_key = "/".join(key for key in (base_key, hdf_key))

        if isinstance(obj, Genesis4Output) and attr == "field":
            res[full_key] = {
                "python_attr": full_attr,
                "hdf_key": full_key,
                "units": "",
                "description": "Field information",
            }
        else:
            try:
                fld = obj.model_fields[attr]
            except KeyError:
                logger.error(
                    f"Internal error: unable to find {attr} ({full_attr} from HDF key {full_key})"
                )
                continue

            if isinstance(obj, Genesis4Output):
                units = None
            else:
                units = obj.units.get(attr, None)

            desc = fld.description or ""

            path_to_remove = f"({full_key})"
            desc = desc.replace(path_to_remove, "")

            units_to_remove = f"[{str(units)}]"
            desc = desc.replace(units_to_remove, "")
            if units_to_remove == "[m_ec^2]":
                desc = desc.replace("[mc^2]", "")
                units = "mc^2"

            res[full_key] = {
                "python_attr": full_attr,
                "hdf_key": full_key,
                "units": units,
                "description": desc.strip(),
            }

        value = getattr(obj, attr)
        if isinstance(value, _OutputBase):
            for sub_key, info in _hdf_summary(
                value,
                base_attr=full_attr,
                base_key=full_key,
            ).items():
                res[sub_key] = info
    return res


class Genesis4Output(Mapping, BaseModel, arbitrary_types_allowed=True):
    """
    Genesis 4 command output.

    Attributes
    ----------
    run : RunInfo
        Execution information - e.g., how long did it take and what was
        the output from Genesis 4.
    alias : dict[str, str]
        Dictionary of aliased data keys.
    """

    run: RunInfo = pydantic.Field(
        default_factory=RunInfo,
        description="Run-related information - output text and timing.",
    )
    beam: OutputBeam = pydantic.Field(
        default_factory=OutputBeam,
        description="Genesis 4 output beam information (/Beam)",
    )
    field_harmonics: Dict[int, OutputField] = pydantic.Field(
        default_factory=dict,
        description="Genesis 4 output field information for harmonic N (/FieldN)",
    )
    lattice: OutputLattice = pydantic.Field(
        default_factory=OutputLattice,
        description="Genesis 4 output lattice information (/Lattice)",
    )
    globals: OutputGlobal = pydantic.Field(
        default_factory=OutputGlobal,
        description="Genesis 4 output global information (/Global)",
    )
    meta: OutputMeta = pydantic.Field(
        default_factory=OutputMeta,
        description="Genesis 4 output metadata (/Meta)",
    )
    version: OutputMetaVersion = pydantic.Field(
        default_factory=OutputMetaVersion,
        description="Genesis 4 version information (/Meta/Version)",
    )
    extra: Dict[str, OutputDataType] = pydantic.Field(
        default_factory=dict,
        description=(
            "Additional Genesis 4 output data which are top-level HDF5 groups. "
            "This is a future-proofing mechanism in case Genesis 4 changes and "
            "LUME-Genesis is not yet ready for it."
        ),
    )
    field3d: Dict[FileKey, FieldFile] = pydantic.Field(
        default_factory=dict,
        exclude=True,
        description="Loaded field data, keyed by filename base (e.g., 'end' of 'end.fld.h5').",
    )
    particles: Dict[FileKey, ParticleGroup] = pydantic.Field(
        default_factory=dict,
        exclude=True,
        description="Loaded particle data, keyed by integration step number or filename base.",
    )
    alias: Dict[str, str] = pydantic.Field(
        default_factory=dict,
        description=(
            "Aliases for string key-based access to data, instead of using "
            "dotted attribute names."
        ),
    )

    field_files: Dict[FileKey, LoadableFieldH5File] = pydantic.Field(
        default_factory=dict,
        description="Loadable field files, keyed by filename base (e.g., 'end' of 'end.fld.h5').",
    )
    particle_files: Dict[FileKey, LoadableParticleGroupH5File] = pydantic.Field(
        default_factory=dict,
        description="Loadable particle files, keyed by (integer) integration step number or filename base.",
    )
    hdf_key_map: Dict[str, str] = pydantic.Field(
        default_factory=dict,
        description="Mapping of attributes to HDF5 file keys",
        repr=False,
    )

    @override
    def model_post_init(self, _) -> None:
        try:
            self.update_aliases()
        except Exception:
            # If we don't catch errors here, Pydantic may fail to give us
            # an Output object back.
            logger.exception("Failed to update aliases")

    def update_aliases(self) -> None:
        """Update aliases based on available data."""
        self.alias.update(tools.make_dotted_aliases(self))

        if self.field is not None:
            self.alias.update(
                tools.make_dotted_aliases(
                    self.field,
                    existing_aliases=self.alias,
                    attr_prefix="field.",
                    alias_prefix="field_",
                )
            )

        for harmonic, field in self.field_harmonics.items():
            if harmonic > 1:
                self.alias.update(
                    tools.make_dotted_aliases(
                        field,
                        existing_aliases=self.alias,
                        attr_prefix=f"field_harmonics[{harmonic}].",
                        alias_prefix=f"field{harmonic}_",
                    )
                )

        custom_aliases = {
            # Back-compat
            "beam_sigma_energy": "beam.stat.sigma_energy",
            "beam_sigma_x": "beam.stat.sigma_x",
            "beam_sigma_y": "beam.stat.sigma_y",
        }
        for alias_from, alias_to in custom_aliases.items():
            self.alias.setdefault(alias_from, alias_to)

    @property
    def field_globals(self) -> OutputFieldGlobal:
        """Genesis 4 output 1st harmonic field global information (``/Field/Global``)."""
        return self.field.globals

    @property
    def beam_globals(self) -> Optional[OutputBeamGlobal]:
        """Genesis 4 output beam global information (``/Beam/Global``)."""
        return self.beam.globals

    @property
    def field(self) -> OutputField:
        """Genesis 4 output field information (``/Field``) - 1st harmonic."""
        return self.field_harmonics.get(1, OutputField())

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
        load_fields : bool, default=False
            After execution, load all field files.
        load_particles : bool, default=False
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

        try:
            with h5py.File(filename, "r") as h5:
                data = parsers.extract_data(h5)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Genesis 4 simulation output file not found. "
                f"Check if the simulation ran successfully and if the run settings are correct. "
                f"The expected output filename is: {filename}"
            ) from None

        fields = [
            LoadableFieldH5File(
                key=get_key_from_filename(fn.name),
                filename=fn,
            )
            for fn in output_root.glob("*.fld.h5")
        ]
        particles = [
            LoadableParticleGroupH5File(
                key=get_key_from_filename(fn.name),
                filename=fn,
            )
            for fn in output_root.glob("*.par.h5")
        ]

        def get_harmonics_keys():
            # The first harmonic is just "Field"
            yield 1, "field"
            # Harmonic 2+ are HDF5 "FieldN" -> Python "field_n"
            for key in sorted(data):
                if not key.startswith("field_"):
                    continue

                try:
                    harmonic = int(key.split("_", 1)[1])
                except ValueError:
                    logger.error(f"Unexpected field key in output: {key}")
                    data.pop(key)
                    continue

                yield harmonic, key

        globals_ = OutputGlobal.from_hdf5_data(data.pop("globals", {}))
        beam = OutputBeam.from_hdf5_data(data.pop("beam", {}))
        field_harmonics = {
            harmonic: OutputField.from_hdf5_data(data.pop(key, {}), slen=globals_.slen)
            for harmonic, key in get_harmonics_keys()
        }
        lattice = OutputLattice.from_hdf5_data(data.pop("lattice", {}))
        meta = OutputMeta.from_hdf5_data(data.pop("meta", {}))
        version = meta.version
        key_map = data.pop("hdf_key_map", {})

        for key, value in data.items():
            if not isinstance(value, (float, int, str, bool, np.ndarray)):
                logger.warning(
                    f"Ignoring unexpected output file HDF5 key: {key}.  "
                    f"This may indicate lume-genesis needs updating."
                )
                data.pop(key)

        extra = data

        output = cls(
            beam=beam,
            field_harmonics=field_harmonics,
            lattice=lattice,
            globals=globals_,
            version=version,
            extra=extra,
            meta=meta,
            alias={},
            field_files={field.key: field for field in fields},
            particle_files={particle.key: particle for particle in particles},
            hdf_key_map=key_map,
        )

        if load_fields:
            output.load_fields()

        if load_particles:
            output.load_particles(smear=smear)

        return output

    def to_hdf_summary(self):
        """Summarize the data based on HDF5 keys."""
        return _hdf_summary(self, base_attr="output", base_key="")

    def load_field_by_key(self, key: FileKey) -> FieldFile:
        """
        Loads a single field file by name into a dictionary.

        Parameters
        ----------
        key : str or int
            The label of the particles (e.g., "end" of "end.fld.h5"), or the
            integer integration step.

        Returns
        -------
        FieldFile
        """
        loadable = self.field_files[key]
        field = loadable.load()
        assert isinstance(field, FieldFile)
        self.field3d[key] = field
        logger.info(f"Loaded field data: '{key}'")
        self.update_aliases()
        return field

    def load_particles_by_key(self, key: FileKey, smear: bool = True) -> ParticleGroup:
        """
        Loads a single particle file into openPMD-beamphysics ParticleGroup
        object.

        Parameters
        ----------
        key : str or int
            The label of the particles (e.g., "end" of "end.par.h5"), or the
            integer integration step.
        smear : bool, optional, default=True
            If set, will smear the phase over the sample (skipped) slices,
            preserving the modulus.
        """
        loadable = self.particle_files[key]
        group = loadable.load(smear=smear)
        assert isinstance(group, ParticleGroup)
        self.particles[key] = group
        logger.info(
            f"Loaded particle data: '{key}' as a ParticleGroup with "
            f"{len(group)} particles"
        )
        return group

    def load_raw_particles_by_key(self, key: FileKey) -> Genesis4ParticleData:
        """
        Loads a single particle file into a raw Genesis4ParticleData object.

        Parameters
        ----------
        key : str or int
            The label of the particles (e.g., "end" of "end.par.h5"), or the
            integer integration step.
        """
        loadable = self.particle_files[key]
        particles = Genesis4ParticleData.from_filename(loadable.filename)
        logger.info(
            f"Loaded raw particle data: '{key}' with {len(particles.slices)} slices"
        )
        return particles

    def load_particles(self, smear: bool = True) -> List[FileKey]:
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

        def sort_key(key: FileKey):
            if isinstance(key, int):
                return (key, "")
            return (-1, key)

        to_load = sorted(self.particle_files, key=sort_key)
        for key in to_load:
            self.load_particles_by_key(key, smear=smear)
        return list(to_load)

    def load_fields(self) -> List[FileKey]:
        """
        Loads all field files produced.

        Returns
        -------
        list of str
            Key names of all loaded fields.
        """

        def sort_key(key: FileKey):
            if isinstance(key, int):
                return (key, "")
            return (-1, key)

        to_load = sorted(self.field_files, key=sort_key)
        for key in to_load:
            self.load_field_by_key(key)
        return to_load

    def units(self, key: str) -> Optional[pmd_unit]:
        """pmd_unit of a given key"""
        # return self.unit_info.get(key, None)
        return self._get_array_info(key).units

    def get_array(self, key: str) -> np.ndarray:
        """
        Gets an array by its string alias.

        For alias information, see `.alias`.

        Returns
        -------
        np.ndarray
        """
        return self[key]

    def archive(self, h5: h5py.Group) -> None:
        """
        Dump outputs into the given HDF5 group.

        Parameters
        ----------
        h5 : h5py.Group
            The HDF5 file in which to write the information.
        """
        _archive.store_in_hdf5_file(h5, self)

    @classmethod
    def from_archive(cls, h5: h5py.Group) -> Genesis4Output:
        """
        Loads output from the given HDF5 group.

        Parameters
        ----------
        h5 : str or h5py.File
            The key to use when restoring the data.
        """
        loaded = _archive.restore_from_hdf5_file(h5)
        if not isinstance(loaded, Genesis4Output):
            raise ValueError(
                f"Loaded {loaded.__class__.__name__} instead of a "
                f"Genesis4Output instance.  Was the HDF group correct?"
            )
        return loaded

    def plot(
        self,
        y: Union[str, Sequence[str]] = "field_energy",
        x: str = "zplot",
        xlim: Optional[PlotLimits] = None,
        ylim: Optional[PlotMaybeLimits] = None,
        ylim2: Optional[PlotMaybeLimits] = None,
        yscale: str = "linear",
        yscale2: str = "linear",
        y2: Union[str, Sequence[str]] = (),
        nice: bool = True,
        include_layout: bool = True,
        include_legend: bool = True,
        return_figure: bool = False,
        tex: bool = False,
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

    def _get_array_info(self, key: str) -> _ArrayInfo:
        dotted_attr = self.alias[key]
        parent_attr, array_attr = dotted_attr.rsplit(".", 1)
        parent: _OutputBase = operator.attrgetter(parent_attr)(self)
        try:
            field = parent.model_fields[array_attr]
        except KeyError:
            field = parent.model_computed_fields[array_attr]

        try:
            value = getattr(parent, array_attr)
        except Exception:
            value = None
            shape = None
        else:
            shape = getattr(value, "shape", None)

        return _ArrayInfo(
            parent=parent,
            array_attr=array_attr,
            units=parent.units.get(array_attr, None),
            field=field,
            value=value,
            shape=shape,
        )

    def info(self):
        """
        Get information about available string keys for the output.
        """
        array_info = {key: self._get_array_info(key) for key in sorted(self.keys())}
        shapes = {
            key: str(array_info.shape or "") for key, array_info in array_info.items()
        }
        annotations = {
            key: array_info.field.description for key, array_info in array_info.items()
        }
        table = {
            key: str(array_info.units or "") for key, array_info in array_info.items()
        }
        return tools.table_output(
            table,  # column 0 (key) and 1 (units)
            annotations=shapes,  # column 2 (description) -> shape
            descriptions=annotations,  # column 3 (type/annotation) -> description
            headers=["Key", "Units", "Shape", "Description"],
        )

    def stat(self, key: str) -> np.ndarray:
        if "power" in key:
            # TODO: this is also some back-compat
            return self.field.peak_power

        info = self._get_array_info(key)
        if isinstance(info.parent, (OutputField, OutputBeam)):
            stat = getattr(info.parent.stat, info.array_attr, None)
            if stat is not None:
                return stat

        if isinstance(info.parent, OutputBeam):
            # Check again: xsize, ysize are not stats attributes but are
            # expected to exist per many examples (beam_xsize)
            # (TODO)
            return simple_mean_from_slice_data(
                getattr(info.parent, info.array_attr),
                weight=info.parent.current,
            )
        if isinstance(info.parent, OutputLattice):
            # TODO: this is bringing forward the old functionality but
            # doesn't seem quite right
            return getattr(info.parent, info.array_attr)
        # raise ValueError(f"No stats for {key}")
        # TODO: passing the value through for now, so this would be like
        # get_array(key)
        return getattr(info.parent, info.array_attr)

    @override
    def __eq__(self, other: Any) -> bool:
        return BaseModel.__eq__(self, other)

    @override
    def __ne__(self, other: Any) -> bool:
        return BaseModel.__ne__(self, other)

    @override
    def __getitem__(self, key: str) -> Any:
        """Support for Mapping -> easy access to data."""
        dotted_attr = self.alias[key]
        # NOTE: special-casing field harmonics
        harmonics_prefix = "field_harmonics["
        if dotted_attr.startswith(harmonics_prefix):
            index = int(dotted_attr[len(harmonics_prefix) :].split("]")[0])
            field_attr = dotted_attr.split(".", 1)[1]
            return operator.attrgetter(field_attr)(self.field_harmonics[index])

        return operator.attrgetter(dotted_attr)(self)

    @override
    def __iter__(self) -> Generator[str, None, None]:
        """Support for Mapping -> easy access to data."""
        yield from self.alias

    @override
    def __len__(self) -> int:
        """Support for Mapping -> easy access to data."""
        return len(self.alias)


def simple_mean_from_slice_data(
    dat: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    """
    Calculate the mean of a 2D slice statistic array weighted by another 2D array.

    Parameters
    ----------
    dat : np.ndarray
        2D (zstep, islice) slice data array
    weight : np.ndarray
        2D (zstep, islice) weight array.

    Returns
    -------
    np.ndarray:
        mean data with shape (zstep, )

    """
    if not len(dat):
        return _empty_ndarray()
    dat = np.nan_to_num(dat)  # Convert any nan to zero for averaging.
    numerator = np.sum(dat * weight, axis=1)
    denominator = np.sum(weight, axis=1)
    return np.divide(
        numerator, denominator, where=denominator != 0, out=np.zeros_like(numerator)
    )


def projected_variance_from_slice_data(
    x2: np.ndarray,
    x1: np.ndarray,
    current: np.ndarray,
):
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
