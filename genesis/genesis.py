""" 
LUME-Genesis primary class

 
"""
from genesis import archive, lattice, parsers, tools, writers


import h5py
import tempfile
from time import time
import os




class Genesis:
    """
    LUME-Genesis class to parse input, run genesis, and parse output.
    
    By default, a temporary directory is created for working.
    
    """
    
        
    def __init__(self, input_file=None, 
                 genesis_bin='$GENESIS_BIN', 
                 use_tempdir=True,
                 workdir=None,
                 verbose=False
                ):
        
        # Save init
        self.original_input_file = input_file
        self.use_tempdir = use_tempdir
        self.workdir = workdir
        if workdir:
            assert os.path.exists(workdir), 'workdir does not exist: '+workdir           
        self.verbose=verbose
        
        self.genesis_bin = genesis_bin
        self.binary_prefixes = [] #  For example, ['mpirun', '-n', '2']
        self.finished = False
    
        # 
        self.output = {}
        
        #
        self.timeout = None
        
        # Run control
        self.finished = False
        self.configured = False
        
        if input_file:
            self.load_input(input_file)
            self.configure()
        else:
            self.vprint('Warning: Input file does not exist. Not configured. Please call .load_input(input_file) and .configure()')
        
    
    def configure(self):
        self.configure_genesis(workdir=self.workdir)        
        

    def configure_genesis(self, input_filePath=None, workdir=None):
        """
        Configures working directory. 
        """
        
        if input_filePath:
            self.load_input(input_filePath)
        
        # Set paths
        if self.use_tempdir:
            # Need to attach this to the object. Otherwise it will go out of scope.
            self.tempdir = tempfile.TemporaryDirectory(dir=self.workdir)
            self.path = self.tempdir.name
        else:
            if workdir:
                self.path = workdir
                self.tempdir = None
            else:
                # Work in place
                self.path = self.original_path      
     
        # Make full path
        self.input_file = os.path.join(self.path, 'genesis.in')    
        
        self.vprint('Configured to run in:', self.path)
        
        self.configured = True
    
    # Conveniences
    @property
    def beam(self):
        return self.input['beam']    
    @property
    def lattice(self):
        return self.input['lattice']
    @property
    def param(self):
        return self.input['param']    
    
    def load_input(self, filePath):
        """
        Loads existing input file, with lattice
        
        """
        
        assert os.path.exists(filePath), f'Input file does not exist: {filePath}'
        
        f = tools.full_path(filePath)
        self.original_path, self.input_file = os.path.split(f) # Get original path, name of main input
        
        self.input = {
            'beam':None
        }        
        d = self.input
        
        main = parsers.parse_main_inputfile(filePath)
        d['param'] = main
        
        if main['beamfile'] != '':
            fname = main['beamfile']
            d['beam'] = parsers.parse_beam_file(main['beamfile'], verbose=self.verbose)  
            
            # Use this new name
            main['beamfile'] = parsers.POSSIBLE_INPUT_FILES['beamfile']
        else:
            d['beam'] = None
        
        if main['maginfile'] != '':
            self.load_lattice(filePath=main['maginfile'], verbose=self.verbose)
            
            # Use this new name
            main['maginfile'] = parsers.POSSIBLE_INPUT_FILES['maginfile']  
        else:
            main['lattice'] = None


    def load_output(self, filePath=None):
        if not filePath:
            fname = os.path.join(self.path, self.param['outputfile'])
        else:
            fname = filePath
        if os.path.exists(fname):
            self.output.update(parsers.parse_genesis_out(fname))
            self.vprint('Loaded output:', fname)      
            
            
        # Final field    
        dflfile = fname+'.dfl'
        if os.path.exists(dflfile):
            self.output['data']['dfl'] = parsers.parse_genesis_dfl(dflfile, self.param['ncar'])
            self.vprint('Loaded dfl:', dflfile)
            
        # Field history
        fldfile = fname+'.fld'
        if os.path.exists(fldfile):
            # Time independent is just one slice
            if self['itdp'] == 0:
                nslice = 1
            else:
                nslice = self.param['nslice']
            self.output['data']['fld'] = parsers.parse_genesis_fld(fldfile, self.param['ncar'], nslice)
            self.vprint('Loaded fld:', fldfile)            
            
        # Final particles    
        dpafile = fname+'.dpa'
        if os.path.exists(dpafile):
            self.output['data']['dpa'] = parsers.parse_genesis_dpa(dpafile, self.param['npart'])
            self.vprint('Loaded dpa:', dpafile)            
            
        # Particle history
        parfile = fname+'.par'
        if os.path.exists(parfile):
            self.output['data']['par'] = parsers.parse_genesis_dpa(parfile, self.param['npart'])
            self.vprint('Loaded par:', parfile)                   
            
            
        #    
        
    def load_lattice(self, filePath=None, verbose=False):
        """
        loads an original Genesis-style lattice into a standard_lattice
        """
        if not filePath:
            fname = os.path.join(self.path, self.param['maginfile'])
        else:
            fname = filePath
            
        self.vprint('loading lattice: ', fname)    
        
        lat = parsers.parse_genesis_lattice(fname)
        # Standardize
        lat['eles'] = lattice.standard_eles_from_eles(lat['eles'])
        self.input['lattice'] = lat
        
    
    def write_beam(self, filePath=None):  
        if not self.beam:        
            return
        
        if not filePath:
            filePath = os.path.join(self.path, self.param['beamfile'])
        
        writers.write_beam_file(filePath, self.beam, verbose=self.verbose)
            
    
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
        lines = tools.namelist_lines(self.param, start='$newrun', end='$end')
        
        with open(self.input_file, 'w') as f:
            for line in lines:
                f.write(line+'\n')
                
    def write_lattice(self):
    
        if not self.lattice:
            print('Error, no lattice to write')
            return
    
        else:
            filePath = os.path.join(self.path, self.param['maginfile'])
            print(self.path, self.param['maginfile'])
            lattice.write_lattice(filePath, self.lattice)
            self.vprint('Lattice written:', filePath)
            
            
    def write_wavefront(self, h5=None):
        """
        Write an openPMD wavefront from the dfl
        """
        
        if not h5:
            h5 = 'genesis_wavefront_'+self.fingerprint()+'.h5'
         
        if isinstance(h5, str):
            fname = os.path.expandvars(h5)
            g = h5py.File(fname, 'w')
            self.vprint(f'Writing wavefront (dfl data) to file {fname}')
        else:
            g = h5        
        
        dfl = self.output['data']['dfl']
        param = self.output['param']
        writers.write_openpmd_wavefront_h5(g, dfl=dfl, param=param)        
        
        return h5
            
    def get_run_script(self, write_to_path=True):
        """
        Assembles the run script. Optionally writes a file 'run' with this line to path.
        """
        
        _, infile = os.path.split(self.input_file)
        
        runscript = [self.genesis_bin, infile]

        # Allow for MPI commands
        if len(self.binary_prefixes) > 0:
            runscript = self.binary_prefixes + runscript
            
        if write_to_path:
            filename = os.path.join(self.path, 'run')
            with open(filename, 'w') as f:
                f.write(' '.join(runscript))
            tools.make_executable(filename)
            
        return runscript
        
            
    def run(self):
        if not self.configured:
            print('not configured to run')
            return
        self.run_genesis(verbose=self.verbose, timeout=self.timeout)    

        
    def run_genesis(self, verbose=False, parse_output=True, timeout=None):
        
        # Check that binary exists
        self.genesis_bin = tools.full_path(self.genesis_bin)
        assert os.path.exists(self.genesis_bin), 'Genesis binary does not exist: '+ self.genesis_bin
        
        # Clear old output
        self.output = {}
        
        run_info = self.output['run_info'] = {}
        t1 = time()
        run_info['start_time'] = t1

        # Move to local directory

        # Save init dir
        init_dir = os.getcwd()
        self.vprint('init dir: ', init_dir)
        
        os.chdir(self.path)
        # Debugging
        self.vprint('Running genesis in '+os.getcwd())

        # Write all input
        self.write_input()
        
        runscript = self.get_run_script()
        run_info['run_script'] = ' '.join(runscript)
        
        try:
            if timeout:
                res = tools.execute2(runscript, timeout=timeout)
                log = res['log']
                self.error = res['error']
                run_info['why_error'] = res['why_error']    
            else:
                # Interactive output, for Jupyter
                log = []
                for path in tools.execute(runscript):
                    self.vprint(path, end="")
                    log.append(path)
    
            self.log = log
            self.error = False   

            if parse_output:
                self.load_output()
                
        except Exception as ex:
            print('Run Aborted', ex)
            self.error = True
            run_info['why_error'] = str(ex)
            
        finally:
            run_info['run_time'] = time() - t1
            run_info['run_error'] = self.error
            
            # Return to init_dir
            os.chdir(init_dir)                        
        
        self.finished = True        

        
    def fingerprint(self):
        """
        Data fingerprint using the input. 
        """
        return tools.fingerprint(self.input)        
    
    def vprint(self, *args, **kwargs):
        # Verbose print
        if self.verbose:
            print(*args, **kwargs)   
           
        
    def input_twiss(self):
        
        betax = self['rxbeam']**2 * self['gamma0'] / self['emitx']
        betay = self['rybeam']**2 * self['gamma0'] / self['emity']
        alphax = self['alphax']
        alphay = self['alphay']
        
        return {'betax':betax, 'betay':betay, 'alphax':alphax, 'alphay':alphay}   
    
    
    
    
    def archive(self, h5=None):
        """
        Archive all data to an h5 handle or filename.
        
        If no file is given, a file based on the fingerprint will be created.
        
        """
        if not h5:
            h5 = 'genesis_'+self.fingerprint()+'.h5'
         
        if isinstance(h5, str):
            fname = os.path.expandvars(h5)
            g = h5py.File(fname, 'w')
            self.vprint(f'Archiving to file {fname}')
        else:
            g = h5
            
        # Write basic attributes
        archive.genesis_init(g)            
                        
        # All input
        archive.write_input_h5(g, self.input, name='input')

        # All output
        archive.write_output_h5(g, self.output, name='output', verbose=self.verbose) 
        
        return h5

    
    def load_archive(self, h5, configure=True):
        """
        Loads input and output from archived h5 file.
        
        See: Genesis.archive
        """
        if isinstance(h5, str):
            fname = os.path.expandvars(h5)
            g = h5py.File(fname, 'r')
            
            glist = archive.find_genesis_archives(g)
            n = len(glist)
            if n == 0:
                # legacy: try top level
                message = 'legacy'
            elif n == 1:
                gname = glist[0]
                message = f'group {gname} from'
                g = g[gname]
            else:
                raise ValueError(f'Multiple archives found in file {fname}: {glist}')
            
            self.vprint(f'Reading {message} archive file {h5}')
        else:
            g = h5
        
        self.input  = archive.read_input_h5(g['input'])
        self.output = archive.read_output_h5(g['output'], verbose=self.verbose)

        self.vprint('Loaded from archive. Note: Must reconfigure to run again.')
        self.configured = False     
        
        if configure:    
            self.configure()          
    
    
    def copy(self):
        """
        Returns a deep copy of this object.
        
        If a tempdir is being used, will clear this and deconfigure. 
        """
        G2 = deepcopy(self)
        # Clear this 
        if G2.use_tempdir:
            G2.path = None
            G2.configured = False
        
        return G2
    
    def __getitem__(self, key):
        """
        Convenience syntax to get an attribute
        
        See: __setitem__
        """        
        
        if key in self.param:
            return self.param[key]
        
        raise ValueError(f'{key} does not exist in input param')
        
        
    def __setitem__(self, key, item):
        """
        Convenience syntax to set input parameters
        
        Example:
        
        G['ncar'] = 251
        
        """
        
        if key in self.param:
            self.param[key] = item
        else:   
            raise ValueError(f'{key} does not exist in input param')
    
    def __str__(self):
        path = self.path
        s = ''
        if self.finished:
            s += 'Genesis finished in '+path
        elif self.configured:
            s += 'Genesis configured in '+path
        else:
            s += 'Genesis not configured.'
        return s    
    
    