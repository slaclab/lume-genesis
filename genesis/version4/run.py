from __future__ import annotations

import logging
import os
import pathlib
import platform
import shlex
import shutil
import traceback
from time import monotonic
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Union

import h5py
import psutil
from lume import tools as lume_tools
from lume.base import CommandWrapper
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import pmd_unit
from typing_extensions import override

from .. import tools
from ..errors import Genesis4RunFailure
from . import parsers
from .field import FieldFile
from .input import Genesis4Input, Lattice, MainInput
from .output import Genesis4Output, RunInfo
from .particles import Genesis4ParticleData
from .types import AnyPath

logger = logging.getLogger(__name__)


def find_mpirun():
    """
    Simple helper to find the mpi run command for macports and homebrew,
    as well as custom commands for Perlmutter at NERSC.
    """

    for p in [
        # Highest priority is what our PATH says:
        shutil.which("mpirun"),
        # Second, macports:
        "/opt/local/bin/mpirun",
        # Third, homebrew:
        "/opt/homebrew/bin/mpirun",
    ]:
        if p and os.path.exists(p):
            return f'"{p}"' + " -n {nproc} {command_mpi}"

    if os.environ.get("NERSC_HOST") == "perlmutter":
        srun = "srun -n {nproc} --ntasks-per-node {nproc} -c 1 {command_mpi}"
        hostname = platform.node()
        assert hostname  # This must exist
        if hostname.startswith("nid"):
            # Compute node
            return srun
        else:
            # This will work on a login node
            return "salloc -N {nnode} -C cpu -q interactive -t 04:00:00 " + srun

    # Default
    return "mpirun -n {nproc} {command_mpi}"


def find_workdir():
    if os.environ.get("NERSC_HOST") == "perlmutter":
        return os.environ.get("SCRATCH")
    else:
        return None


def _make_genesis4_input(
    input: Union[pathlib.Path, str],
    lattice_source: Union[Lattice, pathlib.Path, str],
    source_path: Optional[AnyPath] = None,
) -> Tuple[pathlib.Path, Genesis4Input]:
    input_fn, input_source = tools.read_if_path(input, source_path)
    if not input_source or not isinstance(input_source, str):
        raise ValueError(
            f"'input' must be either a Genesis4Input instance, a Genesis 4-"
            f"compatible main input, or a filename. Got: {input}"
        )

    if isinstance(lattice_source, Lattice):
        # This is nearly *too* flexible as far as supported input formats
        # go. Let's see what Chris thinks...
        if source_path is None:
            if input_fn:
                source_path = input_fn.parent
            else:
                source_path = pathlib.Path(".")

        source_path = pathlib.Path(source_path)
        return source_path, Genesis4Input(
            main=MainInput.from_contents(input_source),
            lattice=lattice_source,
            source_path=source_path,
        )

    lattice_fn, lattice_source = tools.read_if_path(lattice_source, source_path)
    logger.debug(
        "Main input: main_fn=%s contents=\n%s",
        input_fn,
        input,
    )
    if lattice_source or lattice_fn:
        logger.debug(
            "Lattice input: lattice_fn=%s contents=\n%s",
            lattice_fn,
            lattice_source,
        )

    if source_path is None:
        if input_fn:
            source_path = input_fn.parent
        elif lattice_fn:
            source_path = lattice_fn.parent
        else:
            source_path = pathlib.Path(".")

    source_path = pathlib.Path(source_path)
    return source_path, Genesis4Input.from_strings(
        input_source,
        lattice_source,
        input_filename=input_fn,
        lattice_filename=lattice_fn,
        source_path=source_path,
    )


class Genesis4(CommandWrapper):
    """
    Genesis 4 command wrapper for Python-defined configurations and lattices.

    Files will be written into a temporary directory within workdir.
    If workdir=None, a location will be determined by the system.

    Parameters
    ---------
    input : MainInput, Genesis4Input, str, or pathlib.Path
        Input settings for the Genesis 4 run.  This may be a full configuration
        (`Genesis4Input`), main input file contents, or a path to an existing
        file with main input settings (e.g., ``genesis4.in``).
    lattice : Lattice, str or pathlib.Path, optional
        Lattice file source code or path to a lattice file.
        Not required if ``Genesis4Input`` is used for the ``input`` parameter.
    command : str, default="genesis4"
        The command to run to execute Genesis 4.
    command_mpi : str, default="genesis4"
        The Genesis 4 executable to run under MPI.
    use_mpi : bool, default=False
        Enable parallel processing with MPI.
    mpi_run : str, default=""
        The template for invoking ``mpirun``. If not specified, the class
        attribute ``MPI_RUN`` is used. This is expected to be a formated string
        taking as parameters the number of processors (``nproc``) and the
        command to be executed (``command_mpi``).
    use_temp_dir : bool, default=True
        Whether or not to use a temporary directory to run the process.
    workdir : path-like, default=None
        The work directory to be used.
    verbose : bool, optional
        Whether or not to produce verbose output.
        Defaults to `global_display_options.verbose`, which is tied to the
        `LUME_VERBOSE` environment variable.
        When the variable is unset, `verbose` is `False` by default.
    timeout : float, default=None
        The timeout in seconds to be used when running Genesis.
    initial_particles : ParticleGroup or Genesis4ParticleData, optional
        Initial particles to use in the simulation, using the
        OpenPMD-beamphysics standard.
    initial_field : FieldFile, optional
        Initial Genesis 4 format field file to use in the simulation. Use
        `FieldFile.from_file` to load the file first.
    """

    COMMAND: ClassVar[str] = "genesis4"
    COMMAND_MPI: ClassVar[str] = "genesis4"
    MPI_RUN: ClassVar[str] = find_mpirun()
    WORKDIR: ClassVar[Optional[str]] = find_workdir()

    # Environmental variables to search for executables
    command_env: str = "GENESIS4_BIN"
    command_mpi_env: str = "GENESIS4_BIN"
    original_path: AnyPath

    _input: Genesis4Input
    output: Optional[Genesis4Output]

    def __init__(
        self,
        input: Optional[Union[MainInput, Genesis4Input, str, pathlib.Path]] = None,
        lattice: Union[Lattice, str, pathlib.Path] = "",
        *,
        workdir: Optional[Union[str, pathlib.Path]] = None,
        output: Optional[Genesis4Output] = None,
        alias: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, pmd_unit]] = None,
        command: Optional[str] = None,
        command_mpi: Optional[str] = None,
        use_mpi: bool = False,
        mpi_run: str = "",
        use_temp_dir: bool = True,
        verbose: bool = tools.global_display_options.verbose >= 1,
        timeout: Optional[float] = None,
        initial_particles: Optional[Union[ParticleGroup, Genesis4ParticleData]] = None,
        initial_field: Optional[FieldFile] = None,
        **kwargs: Any,
    ):
        super().__init__(
            command=command,
            command_mpi=command_mpi,
            use_mpi=use_mpi,
            mpi_run=mpi_run,
            use_temp_dir=use_temp_dir,
            workdir=workdir,
            verbose=verbose,
            timeout=timeout,
            **kwargs,
        )

        if input is None:
            input = Genesis4Input(
                main=MainInput(),
                lattice=Lattice(),
                initial_particles=initial_particles,
            )
        elif isinstance(input, MainInput):
            input = Genesis4Input.from_main_input(
                main=input,
                lattice=lattice,
                source_path=pathlib.Path(workdir or "."),
            )
        elif not isinstance(input, Genesis4Input):
            # We have either a string or a filename for our main input.
            workdir, input = _make_genesis4_input(
                input,
                lattice,
                source_path=workdir,
            )

        if (
            input.initial_particles is not initial_particles
            and initial_particles is not None
        ):
            input.initial_particles = initial_particles

        if input.initial_field is not initial_field and initial_field is not None:
            input.initial_field = initial_field

        if workdir is None:
            workdir = pathlib.Path(".")

        self.original_path = workdir
        self._input = input
        self.output = output

        # Internal
        self._units = dict(units or parsers.known_unit)
        self._alias = dict(alias or {})

        # MPI
        self.nproc = 1
        self.nnode = 1

    @property
    def input(self) -> Genesis4Input:
        """The Genesis 4 input, including namelists and lattice information."""
        return self._input

    @input.setter
    def input(self, inp: Any) -> None:
        if not isinstance(inp, Genesis4Input):
            raise ValueError(
                f"The provided input is of type {type(inp).__name__} and not `Genesis4Input`. "
                f"Please consider creating a new Genesis4 object instead with the "
                f"new parameters!"
            )
        self._input = inp

    @property
    def nproc(self):
        """
        Number of MPI processes to use.
        """
        return self._nproc

    @nproc.setter
    def nproc(self, nproc: Optional[int]):
        if nproc is None or nproc == 0:
            nproc = psutil.cpu_count(logical=False)
        elif nproc < 0:
            nproc += psutil.cpu_count(logical=False)

        if nproc <= 0:
            raise ValueError(f"Calculated nproc is invalid: {nproc}")

        self._nproc = nproc

    @override
    def configure(self):
        """
        Configure and set up for run.
        """
        self.setup_workdir(self._workdir)
        self.vprint("Configured to run in:", self.path)
        self.configured = True
        self.finished = False

    @override
    def run(
        self,
        load_fields: bool = False,
        load_particles: bool = False,
        smear: bool = True,
        raise_on_error: bool = True,
    ) -> Genesis4Output:
        """
        Execute Genesis 4 with the configured input settings.

        Parameters
        ----------
        load_fields : bool, default=False
            After execution, load all field files.
        load_particles : bool, default=False
            After execution, load all particle files.
        smear : bool, default=True
            If set, for particles, this will smear the phase over the sample
            (skipped) slices, preserving the modulus.
        raise_on_error : bool, default=True
            If Genesis 4 fails to run, raise an error. Depending on the error,
            output information may still be accessible in the ``.output``
            attribute.

        Returns
        -------
        Genesis4Output
            The output data.  This is also accessible as ``.output``.
        """
        if not self.configured:
            self.configure()

        if self.path is None:
            raise ValueError("Path (base_path) not yet set")

        self.finished = False

        runscript = self.get_run_script()

        start_time = monotonic()
        self.vprint(f"Running Genesis4 in {self.path}")
        self.vprint(runscript)

        self.write_input()

        if self.timeout:
            self.vprint(
                f"Timeout of {self.timeout} is being used; output will be "
                f"displaye after Genesis exits."
            )
            execute_result = tools.execute2(
                shlex.split(runscript),
                timeout=self.timeout,
                cwd=self.path,
            )
            self.vprint(execute_result["log"])
        else:
            log = []
            try:
                for line in tools.execute(shlex.split(runscript), cwd=self.path):
                    self.vprint(line, end="")
                    log.append(line)
            except Exception as ex:
                log.append(f"Genesis 4 exited with an error: {ex}")
                self.vprint(log[-1])
                execute_result = {
                    "log": "".join(log),
                    "error": True,
                    "why_error": "error",
                }
            else:
                execute_result = {
                    "log": "".join(log),
                    "error": False,
                    "why_error": "",
                }

        end_time = monotonic()

        self.finished = True
        run_info = RunInfo(
            run_script=runscript,
            error=execute_result["error"],
            error_reason=execute_result["why_error"],
            start_time=start_time,
            end_time=end_time,
            run_time=end_time - start_time,
            output_log=execute_result["log"].replace("\x00", ""),
        )

        success_or_failure = "Success" if not execute_result["error"] else "Failure"
        self.vprint(f"{success_or_failure} - execution took {run_info.run_time:0.2f}s.")

        try:
            self.output = self.load_output(
                load_fields=load_fields,
                load_particles=load_particles,
                smear=smear,
            )
        except Exception as ex:
            stack = traceback.format_exc()
            run_info.error = True
            run_info.error_reason = (
                f"Failed to load output file. {ex.__class__.__name__}: {ex}\n{stack}"
            )
            self.output = Genesis4Output(run=run_info)
            if hasattr(ex, "add_note"):
                # Python 3.11+
                ex.add_note(
                    f"\nGenesis output was:\n\n{execute_result['log']}\n(End of Genesis output)"
                )
            if raise_on_error:
                raise

        self.output.run = run_info
        if run_info.error and raise_on_error:
            raise Genesis4RunFailure(
                f"Genesis 4 failed to run: {run_info.error_reason}"
            )

        return self.output

    def get_executable(self):
        """
        Gets the full path of the executable from .command, .command_mpi
        Will search environmental variables:
                Genesis4.command_env='GENESIS4_BIN'
                Genesis4.command_mpi_env='GENESIS4_BIN'
        """
        if self.use_mpi:
            return lume_tools.find_executable(
                exename=self.command_mpi, envname=self.command_mpi_env
            )
        return lume_tools.find_executable(
            exename=self.command, envname=self.command_env
        )

    def get_run_prefix(self) -> str:
        """Get the command prefix to run Genesis (e.g., 'mpirun' or 'genesis4')."""
        exe = self.get_executable()

        if self.nproc != 1 and not self.use_mpi:
            self.vprint(f"Setting use_mpi = True because nproc = {self.nproc}")
            self.use_mpi = True

        if self.use_mpi:
            return self.mpi_run.format(
                nnode=self.nnode, nproc=self.nproc, command_mpi=exe
            )
        return exe

    @override
    def get_run_script(self, write_to_path: bool = True) -> str:
        """
        Assembles the run script using self.mpi_run string of the form:
            'mpirun -n {n} {command_mpi}'
        Optionally writes a file 'run' with this line to path.

        mpi_exe could be a complicated string like:
            'srun -N 1 --cpu_bind=cores {n} {command_mpi}'
            or
            'mpirun -n {n} {command_mpi}'
        """
        if self.path is None:
            raise ValueError("path (base_path) not yet set")

        runscript = shlex.join(
            [
                *shlex.split(self.get_run_prefix()),
                *self.input.arguments,
            ]
        )

        if write_to_path:
            self.write_run_script()

        return runscript

    def write_run_script(self, path: Optional[AnyPath] = None) -> pathlib.Path:
        """
        Write the 'run' script which can be used in a terminal to launch Genesis.

        This is also performed automatically in `write_input` and
        `get_run_script`.

        Parameters
        -------
        path : pathlib.Path or str
            Where to write the run script.  Defaults to `{self.path}/run`.

        Returns
        -------
        pathlib.Path
            The run script location.
        """
        path = path or self.path
        if path is None:
            raise ValueError("path (base_path) not yet set and no path specified")

        path = pathlib.Path(path)
        if path.is_dir():
            path = path / "run"

        self.input.write_run_script(
            path,
            command_prefix=self.get_run_prefix(),
        )
        logger.debug("Wrote run script to %s", path)
        return path

    @override
    def write_input(
        self,
        path: Optional[AnyPath] = None,
        write_run_script: bool = True,
    ):
        """
        Write the input parameters into the file.

        Parameters
        ----------
        path : str, optional
            The directory to write the input parameters
        """
        if not self.configured:
            self.configure()

        if path is None:
            path = self.path

        if path is None:
            raise ValueError("Path has not yet been set; cannot write input.")

        path = pathlib.Path(path)
        self.input.write(workdir=path)
        if write_run_script:
            self.write_run_script(path)

    @property
    @override
    def initial_particles(self) -> Optional[Union[ParticleGroup, Genesis4ParticleData]]:
        """Initial particles, if defined.  Property is alias for `.input.main.initial_particles`."""
        return self.input.initial_particles

    @initial_particles.setter
    def initial_particles(
        self,
        value: Optional[Union[ParticleGroup, Genesis4ParticleData]],
    ) -> None:
        self.input.initial_particles = value

    @property
    @override
    def initial_field(self) -> Optional[FieldFile]:
        """Initial field, if defined.  Property is alias for `.input.main.initial_field`."""
        return self.input.initial_field

    @initial_field.setter
    def initial_field(self, value: Optional[FieldFile]) -> None:
        self.input.initial_field = value

    def _archive(self, h5: h5py.Group):
        self.input.archive(h5.create_group("input"))
        if self.output is not None:
            self.output.archive(h5.create_group("output"))

    @override
    def archive(self, dest: Union[AnyPath, h5py.Group]) -> None:
        """
        Archive the latest run, input and output, to a single HDF5 file.

        Parameters
        ----------
        dest : filename or h5py.Group
        """
        if isinstance(dest, (str, pathlib.Path)):
            with h5py.File(dest, "w") as fp:
                self._archive(fp)
        elif isinstance(dest, (h5py.File, h5py.Group)):
            self._archive(dest)

    to_hdf5 = archive

    def _load_archive(self, h5: h5py.Group):
        self.input = Genesis4Input.from_archive(h5["input"])
        if "output" in h5:
            self.output = Genesis4Output.from_archive(h5["output"])
        else:
            self.output = None

    @override
    def load_archive(self, arch: Union[AnyPath, h5py.Group]) -> None:
        """
        Load an archive from a single HDF5 file into this Genesis4 object.

        Parameters
        ----------
        arch : filename or h5py.Group
        """
        if isinstance(arch, (str, pathlib.Path)):
            with h5py.File(arch, "r") as fp:
                self._load_archive(fp)
        elif isinstance(arch, (h5py.File, h5py.Group)):
            self._load_archive(arch)

    @override
    @classmethod
    def from_archive(cls, arch: Union[AnyPath, h5py.Group]) -> Genesis4:
        """
        Create a new Genesis4 object from an archive file.

        Parameters
        ----------
        arch : filename or h5py.Group
        """
        inst = cls()
        inst.load_archive(arch)
        return inst

    @classmethod
    def from_tao(
        cls, tao, ele_start="beginning", ele_end="end", universe=1, branch=0
    ) -> Genesis4:
        """
        Create a new Genesis4 object from a pytao.Tao instance

        Parameters
        ----------
        tao : Tao
            The Tao instance to extract elements from.
        ele_start : str, optional
            Element to start. Defaults to "beginning".
        ele_end : str, optional
            Element to end. Defaults to "end".
        branch : int, optional
            The branch index within the specified Tao universe. Defaults to 0.
        universe : int, optional
            The universe index within the Tao object. Defaults to 1.

        Returns
        -------
        Genesis4

        """
        input = MainInput.from_tao(
            tao, ele_start=ele_start, universe=universe, branch=branch
        )
        lattice = Lattice.from_tao(
            tao, ele_start=ele_start, ele_end=ele_end, universe=universe, branch=branch
        )
        return cls(input=input, lattice=lattice)

    @override
    def load_output(
        self,
        load_fields: bool = False,
        load_particles: bool = False,
        smear: bool = True,
    ) -> Genesis4Output:
        """
        Load the Genesis 4 output files from disk.

        Parameters
        ----------
        load_fields : bool, default=True
            Load all field files.
        load_particles : bool, default=True
            Load all particle files.
        smear : bool, default=True
            If set, this will smear the particle phase over the sample
            (skipped) slices, preserving the modulus.

        Returns
        -------
        Genesis4Output
        """
        if self.path is None:
            raise ValueError(
                "Cannot load the output if path is not set. "
                "Did you forget to run `.configure()`?"
            )
        return Genesis4Output.from_input_settings(
            input=self.input,
            workdir=pathlib.Path(self.path),
            load_fields=load_fields,
            load_particles=load_particles,
            smear=smear,
        )

    def load_raw_h5_output(self) -> h5py.File:
        """
        Load the unprocessed Genesis 4 HDF5 file from the last run.

        Returns
        -------
        h5py.File
        """
        if self.path is None:
            raise ValueError(
                "Cannot load the output if path is not set. "
                "Did you forget to run `.configure()`?"
            )
        filename = Genesis4Output.get_output_filename(
            self.input,
            workdir=pathlib.Path(self.path),
        )
        return h5py.File(filename)

    @override
    def plot(
        self,
        y: Union[str, Sequence[str]] = "field_energy",
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
        if self.output is None:
            raise RuntimeError(
                "Genesis 4 has not yet been run; there is no output to plot."
            )

        if not tools.is_jupyter():
            # If not in jupyter mode, return a figure by default.
            return_figure = True

        return self.output.plot(
            y=y,
            x=x,
            xlim=xlim,
            ylim=ylim,
            ylim2=ylim2,
            yscale=yscale,
            yscale2=yscale2,
            y2=y2,
            nice=nice,
            include_layout=include_layout,
            include_legend=include_legend,
            return_figure=return_figure,
            tex=tex,
            **kwargs,
        )

    def stat(self, key: str):
        """
        Calculate a statistic of the beam or field along z.
        """
        if self.output is None:
            raise RuntimeError(
                "Genesis 4 has not yet been run; there is no output to get statistics from."
            )
        return self.output.stat(key=key)

    @override
    @staticmethod
    def input_parser(path: AnyPath) -> MainInput:
        """
        Invoke the specialized main input parser and returns the `MainInput`
        instance.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the main input file.

        Returns
        -------
        MainInput
        """
        return MainInput.from_file(path)

    @staticmethod
    def lattice_parser(path: AnyPath) -> Lattice:
        """
        Invoke the specialized lattice input parser and returns the `Lattice`
        instance.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the lattice input file.

        Returns
        -------
        Lattice
        """
        return Lattice.from_file(path)

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Genesis4):
            return False
        return self.input == other.input and self.output == other.output

    @override
    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, Genesis4):
            return False
        return self.input != other.input or self.output != other.output
