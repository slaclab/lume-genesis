import os
import platform
import shutil
from time import time
import numpy as np
import h5py
from . import parsers, writers, readers
from .plot import plot_stats_with_layout
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.interfaces.genesis import genesis4_par_to_data
from pmd_beamphysics.units import c_light, known_unit, pmd_unit, mec2
from lume.base import CommandWrapper
from lume import tools

def find_mpirun():
    """
    Simple helper to find the mpi run command for macports and homebrew,
    as well as custom commands for Perlmutter at NERSC.
    """

    for p in ["/opt/local/bin/mpirun", "/opt/homebrew/bin/mpirun"]:
        if os.path.exists(p):
            return p + " -n {nproc} {command_mpi}"
        
    if os.environ.get('NERSC_HOST') == 'perlmutter':
        srun = 'srun -n {nproc} --ntasks-per-node {nproc} -c 1 {command_mpi}'
        hostname = platform.node()
        assert hostname # This must exist
        if hostname.startswith('nid'):
            # Compute node
            return srun
        else:
            # This will work on a login node
            return 'salloc -N {nnode} -C cpu -q interactive -t 04:00:00 ' + srun
        
      
    # Default    
    return "mpirun -n {nproc} {command_mpi}"

def find_workdir():
    if os.environ.get('NERSC_HOST') == 'perlmutter':
        return os.environ.get('SCRATCH')
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
            assert os.path.exists(infile), f"Impact input file does not exist: {infile}"
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

    def configure(self):
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
            counter = 0
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

        if self.nproc > 1 and not self.use_mpi:
            self.vprint(f"Setting use_mpi = True because nproc = {self.nproc}")
            self.use_mpi = True

        if self.use_mpi:
            runscript = [self.mpi_run.format(nnode=self.nnode, nproc=self.nproc, command_mpi=exe), infile]
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
        if key.startswith('beam_sigma_'):
            comp = key[11:]
            if comp not in ('x', 'px', 'y', 'py', 'energy'):
                raise NotImplementedError(f"Unsupported component for f{key}: '{comp}'")

            current = np.nan_to_num(self.output['Beam/current'])

            if comp in ('x', 'y'):
                k2 = f'Beam/{comp}size'
                k1 = f'Beam/{comp}position'
            elif comp in ('energy'):
                k2 = f'Beam/energyspread'
                k1 = f'Beam/energy'
            else:
                raise  NotImplementedError(f"TODO: {key}")
                
            x2 = np.nan_to_num(self.output[k2]**2) # <x^2>_islice
            x1 =  np.nan_to_num(self.output[k1]) # <x>_islice
            sigma_X2 = projected_variance_from_slice_data(x2, x1, current)
            
            return np.sqrt(sigma_X2) 

        # Original stats
        key = self.expand_alias(key)
        
        # Peak power
        if 'power' in key.lower():
            if 'field' in key.lower():
                dat = self.output[key]
            else:
                dat = self.output['Field/power']
            return np.max(dat, axis=1)
        
        if key.startswith('Lattice'):
            return self.get_array(key)
        
        if key.startswith('Beam'):
            dat = self.get_array(key)
            skey = key.split('/')[1]
            
            # Average over the beam taking to account the weighting (current)
            current = np.nan_to_num(self.output['Beam/current'])
        
            if skey in ('xsize', 'ysize'):
                # TODO: emitx, emity
                # Properly calculated the projected value
                plane = skey[0]
                x =  np.nan_to_num(self.output[f'Beam/{plane}position']) # <x>_islice
                x2 = np.nan_to_num(self.output[f'Beam/{plane}size']**2) # <x^2>_islice
                norm = np.sum(current, axis=1)
                # Total projected sigma_x
                sigma_x2 = np.sum( (x2 + x**2) * current, axis = 1)/norm - (np.sum(x * current, axis=1)/norm)**2

                output = np.sqrt(sigma_x2)
            else:
                # Simple stat
                dat = np.nan_to_num(dat) # Convert any nan to zero for averaging.
                output = np.sum(dat * current, axis=1) / np.sum(current, axis=1)

            return output
        
        elif key.lower() in ['field_energy', 'pulse_energy']:
            dat = self.output['Field/power']
            
            # Integrate to get J
            nslice = dat.shape[1]
            slen = self.output['Global/slen']            
            ds = slen/nslice
            return np.sum(dat, axis=1) * ds / c_light
            
        elif key.startswith('Field'):
            dat = self.get_array(key)
            skey = key.split('/')[1]
            if skey in ['xposition', 'xsize',
                        'yposition', 'ysize',
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
        
        raise ValueError(f'unknwon key: {key}')
        
        
    

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
        else:
            g = h5

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

    
    def write_initial_particles(self, update_input=True, filename='genesis4_importdistribution.h5'):
        """
        Writes initial particles (ParicleGroup) if present.
        
        """
        p = self.initial_particles
        if not p:
            return
        
        fout = os.path.join(self.path, filename)
        
        # Particles
        p.write_genesis4_distribution(fout)
        self.vprint(f'Initial particles written to {fout}')
        
        # Input
        if update_input:
            d1 = {'type': 'importdistribution',
               'file': fout,
               'charge': p.charge}            
            
            main_input = self.input['main']


            # Remove any existing 'beam', update slen
            ixpop = []
            for ix, d in enumerate(main_input):
                if d['type'] == 'beam':
                    ixpop.append(ix)
                elif d['type'] == 'time':
                    slen = max(c_light * p.t.ptp() , p.z.ptp() ) 
                    d['slen'] = slen
                    self.vprint(f"Updated slen = {slen}")

            if len(ixpop) > 0:
                if len(ixpop) > 1:
                    raise NotImplementedError("Multiple 'beam' encountered")                    
                main_input.pop(ixpop[0])
                self.vprint("Removed 'beam' from input, will be replaced by 'importdistribution'")
            #else:
            #    self.vprint('No existing beam encountered')

            # look for existing importdistribution
            for ix, d in enumerate(main_input):
                if d['type'] == 'importdistribution':
                    found = True
                    d.update(d1)
                    self.vprint("Updated existing importdistribution")
                    return
                
            # Now try to insert before the first track or write statement
            for ix, d in enumerate(main_input):
                if d['type'] in ('track', 'write'):
                    main_input.insert(ix, d1)          
                    self.vprint(f"Added new importdistribution before the first {d['type']}")
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
        self.output["field_files"] = sorted(fld_files, key = lambda k: parsers.dumpfile_step(k))

        # Find any particle files
        self.output["particles"] = {}
        par_files = [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if f.endswith("par.h5")
        ]        
        self.output["particle_files"] = sorted(par_files, key = lambda k: parsers.dumpfile_step(k))



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
        P = ParticleGroup(data = genesis4_par_to_data(filename, smear=smear))
        
        file = os.path.split(filename)[1]
        if file.endswith("par.h5"):
            label = file[:-7]
        else:
            label = file
            
        self.output["particles"][label] = P
        self.vprint(f"Loaded particle data: '{label}' as a ParticleGroup with {len(P)} particles")
        
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
        yscale='linear',
        yscale2='linear',        
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
                if 'power' in key:
                    yk[i] = 'peak_power'   
        
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
        print('Output data\n')
        print("key                       value              unit")
        print(50*'-')
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
            u = ''
        
        if isinstance(v, dict):
            return ''
        
        if isinstance(v, str):
            if len(v) > 200:
                descrip = 'long str: '+ v[0:20].replace('\n', ' ') + '...'
            else:
                descrip = v  
        elif np.isscalar(v):
            descrip = f'{v} {u} '
        elif isinstance(v, np.ndarray):
            descrip = f'array: {str(v.shape):10}  {u}'
        elif isinstance(v, list):
            descrip = str(v)
        else:
            raise ValueError(f'Cannot describe {k}')

            
        line = f'{k:25} {descrip}'
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
    
    return np.sum( (x2 + x1**2) * current, axis = 1)/norm - (np.sum(x1 * current, axis=1)/norm)**2


    




