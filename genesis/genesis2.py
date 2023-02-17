import os
from time import time
import traceback
from copy import deepcopy

import h5py

from lume.base import CommandWrapper
import lume.tools as lume_tools

from genesis import parsers, tools, lattice, writers, archive
from genesis.particles import final_particles


class Genesis2(CommandWrapper):
    """
    Files will be written into a temporary directory within workdir.
    If workdir=None, a location will be determined by the system.
    """

    COMMAND = "genesis2"
    COMMAND_MPI = "genesis2-mpi"

    # Environmental variables to search for executables
    command_env = "GENESIS2_BIN"
    command_mpi_env = "GENESIS2_MPI_BIN"

    def __init__(self, *args, group=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Save init
        self.original_input_file = self.input_file

        self.input = {"param": None, "beam": None, "lattice": None}
        self.output = {}
        self.numprocs = 1

        # Call configure
        if self.input_file:
            infile = lume_tools.full_path(self.input_file)
            assert os.path.exists(
                infile
            ), f"Genesis2 input file does not exist: {infile}"
            self.load_input(self.input_file)

        else:
            # Use default
            self.input["param"] = parsers.MAIN_INPUT_DEFAULT.copy()
            self.vprint("Using default input")
        self.configure()

    # Conveniences
    @property
    def beam(self):
        return self.input["beam"]

    @property
    def lattice(self):
        return self.input["lattice"]

    @property
    def param(self):
        return self.input["param"]

    def configure(self):
        self.setup_workdir(self._workdir)
        self.input_file = os.path.join(self.path, "genesis.in")
        self.vprint("Configured to run in:", self.path)
        self.configured = True

    def input_parser(self, path):
        return parsers.parse_genesis2_input(path)

    def load_output(self):
        fname = os.path.join(self.path, self.param["outputfile"])
        self.output = parsers.parse_genesis2_output(fname)

    def plot(self):
        raise NotImplementedError("Plot not yet implemented.")

    def run(self):
        """
        Run Genesis2
        """

        # Clear previous output
        self.output = {}
        run_info = self.output["run_info"] = {"error": False}

        t1 = time()
        run_info["start_time"] = t1

        # Debugging
        self.vprint(f"Running genesis in {self.path}")

        # Write all input
        self.write_input()

        runscript = self.get_run_script()
        run_info["run_script"] = " ".join(runscript)

        try:
            if self.timeout:
                res = tools.execute2(runscript,
                                     timeout=self.timeout,
                                     cwd=self.path)
                log = res["log"]
                self.error = res["error"]
                run_info["why_error"] = res["why_error"]
            else:
                # Interactive output, for Jupyter
                log = []
                for path in tools.execute(runscript, cwd=self.path):
                    self.vprint(path, end="")
                    log.append(path)

            self.log = log
            self.error = False

            self.load_output()

        except Exception as ex:
            print("Run Aborted", ex)
            error_str = traceback.format_exc()
            self.error = True
            run_info["why_error"] = str(error_str)

        finally:
            run_info["run_time"] = time() - t1
            run_info["run_error"] = self.error

        self.finished = True

    def write_input(self):
        """
        Writes all input files
        """
        self.write_input_file()

        self.write_beam()
        self.write_lattice()

        # Write the run script
        self.get_run_script()

    def write_input_file(self):
        """
        Write parameters to main .in file

        """
        lines = tools.namelist_lines(self.param, start="$newrun", end="$end")

        with open(self.input_file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    def write_beam(self):
        if not self.beam:
            return

        filePath = os.path.join(self.path, os.path.split(self.param['beamfile'])[-1])
        writers.write_beam_file(filePath, self.beam, verbose=self.verbose)

    def write_lattice(self):

        if not self.lattice:
            self.vprint("Warning: no lattice to write")
            return

        else:
            filePath = os.path.join(self.path, self.param["maginfile"])
            lattice.write_lattice(filePath, self.lattice)
            self.vprint("Lattice written:", filePath)

    def write_wavefront(self, h5=None):
        """
        Write an openPMD wavefront from the dfl
        """

        if not h5:
            h5 = "genesis_wavefront_" + self.fingerprint() + ".h5"

        if isinstance(h5, str):
            fname = os.path.expandvars(h5)
            g = h5py.File(fname, "w")
            self.vprint(f"Writing wavefront (dfl data) to file {fname}")
        else:
            g = h5

        dfl = self.output["data"]["dfl"]
        param = self.output["param"]
        writers.write_openpmd_wavefront_h5(g, dfl=dfl, param=param)

        return h5

    def get_run_script(self, write_to_path=True):
        """
        Assembles the run script usg self.mpi_run string of the form:
            'mpirun -n {n} {command_mpi}'
        Optionally writes a file 'run' with this line to path.
        """

        n_procs = self.numprocs

        exe = self.get_executable()

        if self.use_mpi:
            # mpi_exe could be a complicated string like:
            # 'srun -N 1 --cpu_bind=cores {n} {command_mpi}'
            # 'mpirun -n {n} {command_mpi}'

            cmd = self.mpi_run.format(nproc=n_procs, command_mpi=exe)

        else:
            if n_procs > 1:
                raise ValueError("Error: n_procs > 1 but use_mpi = False")
            cmd = exe

        _, infile = os.path.split(self.input_file)

        runscript = cmd.split() + [infile]

        if write_to_path:
            with open(os.path.join(self.path, "run"), "w") as f:
                f.write(" ".join(runscript))

        return runscript

    def get_executable(self):
        """
        Gets the full path of the executable from .command, .command_mpi
        Will search environmental variables:
                Genesis2.command_env='GENESIS2_BIN'
                Genesis2.command_mpi_env='GENESIS2_MPI_BIN'
        """
        if self.use_mpi:
            exe = lume_tools.find_executable(
                exename=self.command_mpi, envname=self.command_mpi_env
            )
        else:
            exe = lume_tools.find_executable(
                exename=self.command, envname=self.command_env
            )
        return exe

    def archive(self, h5=None):
        """
        Archive all data to an h5 handle or filename.

        If no file is given, a file based on the fingerprint will be created.

        """
        if not h5:
            h5 = "genesis_" + self.fingerprint() + ".h5"

        if isinstance(h5, str):
            fname = os.path.expandvars(h5)
            g = h5py.File(fname, "w")
            self.vprint(f"Archiving to file {fname}")
        else:
            g = h5

        # Write basic attributes
        archive.genesis_init(g)

        # All input
        archive.write_input_h5(g, self.input, name="input")

        # All output
        archive.write_output_h5(g, self.output, name="output",
                                verbose=self.verbose)

        return h5

    def load_archive(self, h5, configure=True):
        """
        Loads input and output from archived h5 file.

        See: Genesis.archive
        """
        if isinstance(h5, str):
            fname = os.path.expandvars(h5)
            g = h5py.File(fname, "r")

            glist = archive.find_genesis_archives(g)
            n = len(glist)
            if n == 0:
                # legacy: try top level
                message = "legacy"
            elif n == 1:
                gname = glist[0]
                message = f"group {gname} from"
                g = g[gname]
            else:
                raise ValueError(
                    f"Multiple archives found in file {fname}: {glist}")

            self.vprint(f"Reading {message} archive file {h5}")
        else:
            g = h5

        self.input = archive.read_input_h5(g["input"])
        self.output = archive.read_output_h5(g["output"], verbose=self.verbose)

        self.vprint("Loaded from archive. Must reconfigure to run again.")
        self.configured = False

        if configure:
            self.configure()

    def __getitem__(self, key):
        """
        Convenience syntax to get an attribute

        See: __setitem__
        """

        if key in self.param:
            return self.param[key]

        raise ValueError(f"{key} does not exist in input param")

    def __setitem__(self, key, item):
        """
        Convenience syntax to set input parameters

        Example:

        G['ncar'] = 251

        """

        if key in self.param:
            self.param[key] = item
        else:
            raise ValueError(f"{key} does not exist in input param")

    def __str__(self):
        path = self.path
        s = ""
        if self.finished:
            s += "Genesis finished in " + path
        elif self.configured:
            s += "Genesis configured in " + path
        else:
            s += "Genesis not configured."
        return s

    def final_particles(self):
        """
        Returns a ParticleGroup object from dpa data (if present)
        """
        if "dpa" in self.output["data"]:
            return final_particles(self)
        else:
            return None
