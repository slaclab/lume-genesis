import logging
import os
import pathlib
import platform
import shlex
import shutil
import traceback
from time import time, monotonic
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
from lume import tools
from lume.base import CommandWrapper
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.genesis import genesis4_par_to_data
from pmd_beamphysics.units import c_light, pmd_unit

from genesis.version4.output import Genesis4Output, RunInfo

from . import parsers, readers, writers
from .input.core import Genesis4Input, MainInput
from .output import projected_variance_from_slice_data
from .plot import plot_stats_with_layout
from ..tools import is_jupyter

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


class Genesis4(CommandWrapper):
    """
    Files will be written into a temporary directory within workdir.
    If workdir=None, a location will be determined by the system.


    Parameters
    ---------
    input_file: str
        Default: None

    initial_particle: ParticleGroup
        Default: None

    command: str
        Default: "genesis4"

    command_mpi: str
        Default: "genesis4"

    use_mpi: bool
        Default: False

    mpi_run: str
        Default: ""

    use_temp_dir: bool
        Default: True

    workdir: path-like
        Default: None

    verbose: bool
        Default: False

    timeout: float
        Default: None


    """

    COMMAND = "genesis4"
    COMMAND_MPI = "genesis4"
    MPI_RUN = find_mpirun()
    WORKDIR = find_workdir()

    # Environmental variables to search for executables
    command_env = "GENESIS4_BIN"
    command_mpi_env = "GENESIS4_BIN"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Save init
        self.original_input_file = self.input_file

        # Data
        self.input = {"main": {}, "lattice": []}
        self.output = {}

        # Internal
        self._units = parsers.known_unit.copy()
        self._alias = {}

        # MPI
        self._nproc = 1
        self._nnode = 1

        # Call configure
        if self.input_file:
            infile = tools.full_path(self.input_file)
            assert os.path.exists(
                infile
            ), f"Genesis4 input file does not exist: {infile}"
            self.load_input(self.input_file)
            self.configure()

        # else:
        #    self.vprint('Using default input: 1 m drift lattice')
        #    self.input = deepcopy(DEFAULT_INPUT)
        #    self.configure()

    def configure(self):
        """
        Configure and set up for run.
        """
        self.setup_workdir(self._workdir)
        self.vprint("Configured to run in:", self.path)
        self.configured = True

    def run(self):
        """
        Execute the code.
        """
        # Auto-configure for convenience
        if not self.configured:
            self.configure()

        # Clear output
        self.output = {}

        run_info = self.output["run_info"] = {"error": False}

        # Run script, gets executables
        runscript = self.get_run_script()
        run_info["run_script"] = runscript

        t1 = time()
        run_info["start_time"] = t1

        self.vprint(f"Running Genesis4 in {self.path}")
        self.vprint(runscript)

        # Write input
        self.write_input()

        # TODO: Remove previous files

        # try:
        if self.timeout:
            res = tools.execute2(runscript.split(), timeout=self.timeout, cwd=self.path)
            log = res["log"]
            self.error = res["error"]
            run_info["error"] = self.error
            run_info["why_run_error"] = res["why_error"]

        else:
            # Interactive output, for Jupyter
            log = []
            for path in tools.execute(runscript.split(), cwd=self.path):
                self.vprint(path, end="")
                # print(f'{path.strip()}, elapsed: {time()-t1:5.1f} s')
                # # Fancy clearing of old lines
                # counter +=1
                # if self.verbose:
                #     if counter < 15:
                #         print(path, end='')
                #     else:
                #         print('\r', path.strip()+', elapsed: '+str(time()-t1), end='')
                log.append(path)
            self.vprint("Finished.")
        self.log = log

        # Load output
        self.load_output()

        # except Exception as ex:
        #    self.vprint('Exception in Genesis4:', ex)
        #    run_info['error'] = True
        #    run_info['why_run_error'] = str(ex)
        # finally:
        run_info["run_time"] = time() - t1

        self.finished = True

    def get_executable(self):
        """
        Gets the full path of the executable from .command, .command_mpi
        Will search environmental variables:
                Genesis4.command_env='GENESIS4_BIN'
                Genesis4.command_mpi_env='GENESIS4_BIN'
        """
        if self.use_mpi:
            exe = tools.find_executable(
                exename=self.command_mpi, envname=self.command_mpi_env
            )
        else:
            exe = tools.find_executable(exename=self.command, envname=self.command_env)
        return exe

    def get_run_script(self, write_to_path=False, path=None, scriptname="run"):
        """
        Assembles the run script using self.mpi_run string of the form:
            'mpirun -n {n} {command_mpi}'
        Optionally writes a file 'run' with this line to path.

        mpi_exe could be a complicated string like:
            'srun -N 1 --cpu_bind=cores {n} {command_mpi}'
            or
            'mpirun -n {n} {command_mpi}'
        """
        exe = self.get_executable()

        # Expect to run locally
        # _, infile = os.path.split(self.input_file)
        infile = "genesis4.in"

        if self.nproc != 1 and not self.use_mpi:
            self.vprint(f"Setting use_mpi = True because nproc = {self.nproc}")
            self.use_mpi = True

        if self.use_mpi:
            runscript = [
                self.mpi_run.format(
                    nnode=self.nnode, nproc=self.nproc, command_mpi=exe
                ),
                infile,
            ]
        else:
            runscript = [exe, infile]

        runscript = " ".join(runscript)

        if write_to_path:
            if path is None:
                path = self.path
            path = os.path.join(path, scriptname)
            with open(path, "w") as f:
                f.write(runscript)
            tools.make_executable(path)
        return runscript

    @property
    def nproc(self):
        """
        Number of MPI processes to use.
        """
        return self._nproc

    @nproc.setter
    def nproc(self, n):
        self._nproc = n

    @property
    def nnode(self):
        """
        Number of MPI nodes to use
        """
        return self._nnode

    @nnode.setter
    def nnode(self, n):
        self._nnode = n

    @property
    def particles(self):
        return self.output["particles"]

    @property
    def field(self):
        return self.output["field"]

    def units(self, key):
        """pmd_unit of a given key"""
        if key in self._units:
            return self._units[key]
        else:
            return None

    def expand_alias(self, key):
        if key in self._alias:
            return self._alias[key]
        else:
            return key

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

            current = np.nan_to_num(self.output["Beam/current"])

            if comp in ("x", "y"):
                k2 = f"Beam/{comp}size"
                k1 = f"Beam/{comp}position"
            elif comp in ("energy"):
                k2 = "Beam/energyspread"
                k1 = "Beam/energy"
            else:
                # TODO: emittance from alpha, beta, etc.
                raise NotImplementedError(f"TODO: {key}")
            x2 = np.nan_to_num(self.output[k2] ** 2)  # <x^2>_islice
            x1 = np.nan_to_num(self.output[k1])  # <x>_islice
            sigma_X2 = projected_variance_from_slice_data(x2, x1, current)

            return np.sqrt(sigma_X2)

        # Original stats
        key = self.expand_alias(key)

        # Peak power
        if "power" in key.lower():
            if "field" in key.lower():
                dat = self.output[key]
            else:
                dat = self.output["Field/power"]
            return np.max(dat, axis=1)

        if key.startswith("Lattice"):
            return self.get_array(key)

        if key.startswith("Beam"):
            dat = self.get_array(key)
            skey = key.split("/")[1]

            # Average over the beam taking to account the weighting (current)
            current = np.nan_to_num(self.output["Beam/current"])

            if skey in ("xsize", "ysize"):
                # TODO: emitx, emity
                # Properly calculated the projected value
                plane = skey[0]
                x = np.nan_to_num(self.output[f"Beam/{plane}position"])  # <x>_islice
                x2 = np.nan_to_num(
                    self.output[f"Beam/{plane}size"] ** 2
                )  # <x^2>_islice
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
                phase = np.nan_to_num(self.output["Beam/bunchingphase"])
                output = np.abs(
                    np.sum(np.exp(1j * phase) * dat * current, axis=1)
                ) / np.sum(current, axis=1)

            else:
                # Simple stat
                dat = np.nan_to_num(dat)  # Convert any nan to zero for averaging.
                output = np.sum(dat * current, axis=1) / np.sum(current, axis=1)

            return output

        elif key.lower() in ["field_energy", "pulse_energy"]:
            dat = self.output["Field/power"]

            # Integrate to get J
            nslice = dat.shape[1]
            slen = self.output["Global/slen"]
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

    def get_array(self, key):
        """
        Gets an array, considering aliases
        """
        # Try raw
        if key in self.output:
            return self.output[key]
        # Try alias
        key = self.expand_alias(key)
        if key in self.output:
            return self.output[key]

        raise ValueError(f"unknwon key: {key}")

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
        if not h5:
            h5 = "genesis4_" + self.fingerprint() + ".h5"

        if isinstance(h5, str):
            if "outfile" in self.output:
                shutil.copy(self.output["outfile"], h5)
                self.vprint(f"Archiving to file {h5}")

            # fname = os.path.expandvars(h5)
            # g = h5py.File(fname, 'w')
            # self.vprint(f'Archiving to file {fname}')

        return h5

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

    def write_input(self, path=None, input_filename="genesis4.in"):
        """
        Write the input parameters into the file.

        Parameters
        ----------
        input_filename : str
            The file in which to write the input parameters
        """
        if path is None:
            path = self.path

        assert os.path.exists(path)

        filePath = os.path.join(path, input_filename)

        # Write initial particles
        self.write_initial_particles()

        # Write main input file. This should come last.
        writers.write_main_input(filePath, self.input["main"])

        # Write run script
        self.get_run_script(write_to_path=True, path=path)

    @staticmethod
    def input_parser(path):
        """
        Invoke the specialized input parser and returns the
        input dictionary.

        Parameters
        ----------
        path : str
            Path to the input file

        Returns
        -------
        input : dict
            The input dictionary
        """
        d = {}
        d["main"] = parsers.parse_main_input(path)

        return d

    def write_initial_particles(
        self, update_input=True, filename="genesis4_importdistribution.h5"
    ):
        """
        Writes initial particles (ParicleGroup) if present.

        """
        p = self.initial_particles
        if not p:
            return

        fout = os.path.join(self.path, filename)

        # Particles
        p.write_genesis4_distribution(fout)
        self.vprint(f"Initial particles written to {fout}")

        # Input
        if update_input:
            d1 = {"type": "importdistribution", "file": fout, "charge": p.charge}

            main_input = self.input["main"]

            # Remove any existing 'beam', update slen
            ixpop = []
            for ix, d in enumerate(main_input):
                if d["type"] == "beam":
                    ixpop.append(ix)
                elif d["type"] == "time":
                    slen = max(c_light * p.t.ptp(), p.z.ptp())
                    d["slen"] = slen
                    self.vprint(f"Updated slen = {slen}")

            if len(ixpop) > 0:
                if len(ixpop) > 1:
                    raise NotImplementedError("Multiple 'beam' encountered")
                main_input.pop(ixpop[0])
                self.vprint(
                    "Removed 'beam' from input, will be replaced by 'importdistribution'"
                )
            # else:
            #    self.vprint('No existing beam encountered')

            # look for existing importdistribution
            for ix, d in enumerate(main_input):
                if d["type"] == "importdistribution":
                    d.update(d1)
                    self.vprint("Updated existing importdistribution")
                    return

            # Now try to insert before the first track or write statement
            for ix, d in enumerate(main_input):
                if d["type"] in ("track", "write"):
                    main_input.insert(ix, d1)
                    self.vprint(
                        f"Added new importdistribution before the first {d['type']}"
                    )
                    return

            # Just append at the end. Note that a track will still be needed!
            self.vprint("Nothing found, inserting at the end")
            main_input.append(d1)

    def load_output(self, load_fields=False, **kwargs):
        """
        Reads and load into `.output` the outputs generated by the code.
        """

        # Main ouput
        outfile = self.input["main"][0]["rootname"] + ".out.h5"
        outfile = os.path.join(self.path, outfile)
        if not os.path.exists(outfile):
            outfile = None
            self.vprint("Warning: no main output file was created. Skipping.")
        self.output["outfile"] = outfile

        # Extract all data
        if outfile:
            self.vprint(f"Loading main output: {outfile}")
            with h5py.File(outfile) as h5:
                data, unit = parsers.extract_data_and_unit(h5)
            self._units.update(unit)

            for k, v in data.items():
                self.output[k] = v

        # Find any field files
        self.output["field"] = {}
        fld_files = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if f.endswith("fld.h5")
        ]
        self.output["field_files"] = sorted(
            fld_files, key=lambda k: parsers.dumpfile_step(k)
        )

        # Find any particle files
        self.output["particles"] = {}
        par_files = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if f.endswith("par.h5")
        ]
        self.output["particle_files"] = sorted(
            par_files, key=lambda k: parsers.dumpfile_step(k)
        )

        self._alias = parsers.extract_aliases(self.output)
        for k, key in self._alias.items():
            if key in self._units:
                self._units[k] = self._units[key]

    def load_particle_file(self, filename, smear=True):
        """
        Loads a single particle file into openPMD-beamphysics
        ParticleGroup object.

        If smear = True (default), will smear the phase over
        the sample (skipped) slices, preserving the modulus.
        """
        P = ParticleGroup(data=genesis4_par_to_data(filename, smear=smear))

        file = os.path.split(filename)[1]
        if file.endswith("par.h5"):
            label = file[:-7]
        else:
            label = file

        self.output["particles"][label] = P
        self.vprint(
            f"Loaded particle data: '{label}' as a ParticleGroup with {len(P)} particles"
        )

    def load_particles(self, smear=True):
        """
        Loads all particle files produced.
        """
        for file in self.output["particle_files"]:
            self.load_particle_file(file, smear=smear)

    def load_fields(self):
        """
        Loads all field files produced.
        """
        for file in self.output["field_files"]:
            self.load_field_file(file)

    def load_field_file(self, filename):
        """
        Load a single .dfl.h5 file into .output
        """
        if not h5py.is_hdf5(filename):
            raise ValueError(f"Field file {filename} is not an HDF5 file")
        with h5py.File(filename, "r") as h5:
            dfl, param = readers.load_genesis4_fields(h5)

        file = os.path.split(filename)[1]
        if file.endswith("fld.h5"):
            label = file[:-7]
        else:
            label = file

        self.output["field"][label] = {"dfl": dfl, "param": param}
        self.vprint(f"Loaded field data: {label}")

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
        for k in sorted(list(self.output)):
            line = self.output_description_line(k)
            print(line)

    def output_description_line(self, k):
        """
        Returns a line describing an output
        """
        v = self.output[k]
        u = self.units(k)
        if u is None:
            u = ""

        if isinstance(v, dict):
            return ""

        if isinstance(v, str):
            if len(v) > 200:
                descrip = "long str: " + v[0:20].replace("\n", " ") + "..."
            else:
                descrip = v
        elif np.isscalar(v):
            descrip = f"{v} {u} "
        elif isinstance(v, np.ndarray):
            descrip = f"array: {str(v.shape):10}  {u}"
        elif isinstance(v, list):
            descrip = str(v)
        else:
            raise ValueError(f"Cannot describe {k}")

        line = f"{k:25} {descrip}"
        return line


def _make_genesis4_input(
    input: Union[pathlib.Path, str],
    lattice_source: Union[pathlib.Path, str] = "",
    source_path: pathlib.Path = pathlib.Path("."),
) -> Genesis4Input:
    def _read_if_path(input: Union[pathlib.Path, str]) -> str:
        nonlocal source_path
        path = pathlib.Path(input).resolve()
        try:
            is_path = isinstance(input, pathlib.Path) or path.exists()
        except OSError:
            is_path = False

        if not is_path:
            return str(input)

        # Update our source path; we found a file.  This is probably what
        # the user wants.
        source_path = path.absolute().parent
        with open(path) as fp:
            return fp.read()

    input = _read_if_path(input)
    lattice_source = _read_if_path(lattice_source)
    if not input or not isinstance(input, str):
        raise ValueError(
            "'input' must be either a Genesis4Input instance, a Genesis 4-"
            "compatible main input, or a filename."
        )

    if not lattice_source:
        raise ValueError(
            "When specifying a Genesis 4-compatible main input string, "
            "you must also provide a lattice as a string or filename "
            "(`lattice_source` argument)."
        )
    return Genesis4Input.from_strings(
        input,
        lattice_source,
        source_path=source_path,
    )


class Genesis4RunFailure(Exception): ...


class Genesis4Python(CommandWrapper):
    """
    Genesis 4 command wrapper for Python-defined configurations and lattices.

    Files will be written into a temporary directory within workdir.
    If workdir=None, a location will be determined by the system.

    Parameters
    ---------
    input : Genesis4Input or str

    initial_particle: ParticleGroup
        Default: None

    command: str
        Default: "genesis4"

    command_mpi: str
        Default: "genesis4"

    use_mpi: bool
        Default: False

    mpi_run: str
        Default: ""

    use_temp_dir: bool
        Default: True

    workdir: path-like
        Default: None

    verbose: bool
        Default: False

    timeout: float
        Default: None
    """

    COMMAND = "genesis4"
    COMMAND_MPI = "genesis4"
    MPI_RUN = find_mpirun()
    WORKDIR = find_workdir()

    # Environmental variables to search for executables
    command_env = "GENESIS4_BIN"
    command_mpi_env = "GENESIS4_BIN"

    input: Genesis4Input
    output: Optional[Genesis4Output]

    def __init__(
        self,
        input: Union[Genesis4Input, str, pathlib.Path],
        lattice_source: Union[str, pathlib.Path] = "",
        *,
        source_path: Union[str, pathlib.Path] = ".",
        output: Optional[Genesis4Output] = None,
        alias: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, pmd_unit]] = None,
        command=None,
        command_mpi=None,
        use_mpi=False,
        mpi_run="",
        use_temp_dir=True,
        workdir=None,
        verbose=False,
        timeout=None,
        **kwargs,
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
        if not isinstance(input, Genesis4Input):
            input = _make_genesis4_input(
                input,
                lattice_source,
                source_path=pathlib.Path(source_path),
            )

        self.input = input
        self.output = output

        # Internal
        self._units = dict(units or parsers.known_unit)
        self._alias = dict(alias or {})

        # MPI
        self.nproc = 1
        self.nnode = 1

    def configure(self):
        """
        Configure and set up for run.
        """
        self.setup_workdir(self._workdir)
        self.vprint("Configured to run in:", self.path)
        self.configured = True

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
        load_fields : bool, default=True
            After execution, load all field files.
        load_particles : bool, default=True
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
            The output data.
        """
        if not self.configured:
            self.configure()

        if self.path is None:
            raise ValueError("path (base_path) not yet set")

        runscript = self.get_run_script()

        start_time = monotonic()
        self.vprint(f"Running Genesis4 in {self.path}")
        self.vprint(runscript)

        self.write_input()

        # if self.timeout:
        res = tools.execute2(
            shlex.split(runscript),
            timeout=self.timeout,
            cwd=self.path,
        )
        # else:
        #     # TODO
        #     # Interactive output, for Jupyter
        #     log = []
        #     for line in tools.execute(shlex.split(runscript), cwd=self.path):
        #         self.vprint(line, end="")
        #         log.append(line)
        #     self.vprint("Finished.")
        #     log = "\n".join(log)

        end_time = monotonic()

        self.finished = True
        run_info = RunInfo(
            run_script=runscript,
            error=res["error"],
            error_reason=res["why_error"],
            start_time=start_time,
            end_time=end_time,
            run_time=end_time - start_time,
            output_log=res["log"],
        )

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
            if raise_on_error:
                raise

        if is_jupyter():
            print(f"Execution failed. Took {run_info.run_time}s:")
            print(run_info.output_log)

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
            return tools.find_executable(
                exename=self.command_mpi, envname=self.command_mpi_env
            )
        return tools.find_executable(exename=self.command, envname=self.command_env)

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

    def get_run_script(self) -> str:
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

        runscript = [
            *shlex.split(self.get_run_prefix()),
            *self.input.get_arguments(workdir=self.path),
        ]

        return shlex.join(runscript)

    def write_input(self, path=None):
        """
        Write the input parameters into the file.

        Parameters
        ----------
        input_filename : str
            The file in which to write the input parameters
        """
        if path is None:
            path = self.path

        if path is None:
            raise ValueError("Path has not yet been set; cannot write input.")
        return self.input.write(workdir=path)

    def archive(self, *args, **kwargs):
        """Archive the latest run, input and output, to a single HDF5 file."""
        # TODO
        # return self.output.archive(*args, **kwargs)

    to_hdf5 = archive

    def load_archive(self, *args, **kwargs):
        """Load an archive from a single HDF5 file."""
        # return Genesis4Output.from_archive(*args, **kwargs)

    def __getnewargs_ex__(self) -> Tuple[tuple, dict]:
        return (
            (),
            {
                "input": self.input,
                "output": self.output,
                "alias": self._alias,
                "units": self._units,
                "command": self.command,
                "command_mpi": self.command_mpi,
                "use_mpi": self.use_mpi,
                "mpi_run": self.mpi_run,
                "use_temp_dir": self.use_temp_dir,
                "workdir": self.workdir,
                "verbose": self.verbose,
                "timeout": self.timeout,
            },
        )

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
            raise ValueError("Cannot load the output if path is not set.")
        return Genesis4Output.from_input_settings(
            input=self.input,
            workdir=pathlib.Path(self.path),
            load_fields=load_fields,
            load_particles=load_particles,
            smear=smear,
        )

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
        if self.output is None:
            raise RuntimeError(
                "Genesis 4 has not yet been run; there is no output to plot."
            )
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

    @staticmethod
    def input_parser(path):
        """
        Invoke the specialized input parser and returns the dataclass.

        Parameters
        ----------
        path : str
            Path to the main input file.

        Returns
        -------
        MainInput
            The input dictionary
        """
        return MainInput.from_file(path)
