""" 
LUME-Genesis primary class

 
"""
from genesis import tools, parsers, lattice
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

        else:
            # Load a default for testing
            # input params
            # param descriptions here http://genesis.web.psi.ch/download/documentation/genesis_manual.pdf
            self.input = DEFAULT_INPUT_PARAMS        
            self.original_input_file = 'genesis.in'
            self.lattice = DEFAULT_LATTICE
            self.lattice_params = DEFAULT_LATTICE_PARMS
            self.finished = False
            
        # Call configure
        self.configure()

    
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
            # Work in place
            self.path = self.original_path        
     
        # Make full path
        self.input_file = os.path.join(self.path, self.original_input_file)    
        
        self.vprint('Configured to run in:', self.path)
        
        self.configured = True
        

    def load_input(self, filePath):
        """
        Loads existing input file, with lattice
        
        """
        f = tools.full_path(filePath)
        self.original_path, _ = os.path.split(f) # Get original path
        self.input = parsers.parse_inputfile(f) 
        
        # Lattice
        latfile =  self.input['maginfile']
        if not os.path.isabs(latfile):
            latfile = os.path.join(self.original_path, latfile)
        

    def load_output(self, filePath=None):
        if not filePath:
            fname = os.path.join(self.path, self.input['outputfile'])
        else:
            fname = filePath
        if os.path.exists(fname):
            self.output = parsers.parse_genesis_out(fname)    
            self.vprint('Loaded output:', fname)      
        
    def load_lattice(self, filePath=None, verbose=False):
        """
        loads an original Genesis-style lattice into a standard_lattice
        """
        if not filePath:
            fname = os.path.join(self.path, self.input['maginfile'])
        else:
            fname = filePath
            
        self.vprint('loading lattice: ', fname)    
        eles, params = parsers.parse_genesis_lattice(fname)
        
        self.lattice = lattice.standard_lattice_from_eles(eles)
        self.lattice_params = params
    
        
    def write_lattice(self):
    
        if not self.lattice:
            print('Error, no lattice to write')
            return
    
        else:
            filePath = os.path.join(self.path, self.input['maginfile'])
            lattice.write_lattice(filePath, self.lattice, self.lattice_params['unitlength'])
    
    
    def write_input_file(self):
        """
        Write parameters to main .in file
        
        """    
        lines = tools.namelist_lines(self.input, start='$newrun', end='$end')
        
        with open(self.input_file, 'w') as f:
            for line in lines:
                f.write(line+'\n')
            
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
            with open(os.path.join(self.path, 'run'), 'w') as f:
                f.write(' '.join(runscript))
            
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
        
        run_info = {}
        t1 = time()
        run_info['start_time'] = t1
        
        # Move to local directory

        # Save init dir
        init_dir = os.getcwd()
        self.vprint('init dir: ', init_dir)
        
        os.chdir(self.path)
        # Debugging
        self.vprint('running genesis in '+os.getcwd())

        # Write input file from internal dict
        self.write_input_file()
        self.write_lattice()
        
        runscript = self.get_run_script()
    
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
            
            # Add run_info
            self.output.update(run_info)
            
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
        
        betax = self.input['rxbeam']**2 * self.input['gamma0'] / self.input['emitx']
        betay = self.input['rybeam']**2 * self.input['gamma0'] / self.input['emity']
        alphax = self.input['alphax']
        alphay = self.input['alphay']
        
        return {'betax':betax, 'betay':betay, 'alphax':alphax, 'alphay':alphay}   
    
    
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
    
    
# Defaults for testing    

DEFAULT_LATTICE_PARMS = {'version': 1, 'unitlength': 0.03}    
    
DEFAULT_LATTICE = [{'type': 'QF', 'strength': 12.84, 'L': 5.0, 's': 5.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 130.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 130.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 135.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 260.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 260.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 265.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 390.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 390.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 395.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 520.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 520.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 525.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 650.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 650.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 655.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 780.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 780.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 785.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 910.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 910.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 915.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1040.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1040.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 1045.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1170.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1170.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 1175.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1300.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1300.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 1305.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1430.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1430.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 1435.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1560.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1560.0}]    
    
    
    
    
    
    
# Defaults for testing 
    
    
DEFAULT_INPUT_PARAMS = {'aw0'   :  2.473180,
    'xkx'   :  0.000000E+00,
    'xky'   :  1.000000E+00,
    'wcoefz':  [7.500000E-01,   0.000000E+00,   1.000000E+00],
    'xlamd' :  3.000000E-02,
    'fbess0':  0.000000E+00,
    'delaw' :  0.000000E+00,
    'iertyp':    0,
    'iwityp':    0,
    'awd'   :  2.473180,
    'awx'   :  0.000000E+00,
    'awy'   :  0.000000E+00,
    'iseed' :   10,
    'npart' :   2048,
    'gamma0':  6.586752E+03,
    'delgam':  2.600000E+00,
    'rxbeam':  2.846500E-05,
    'rybeam':  1.233100E-05,
    'alphax':  0,
    'alphay': -0,
    'emitx' :  4.000000E-07,
    'emity' :  4.000000E-07,
    'xbeam' :  0.000000E+00,
    'ybeam' :  0.000000E+00,
    'pxbeam':  0.000000E+00,
    'pybeam':  0.000000E+00,
    'conditx' :  0.000000E+00,
    'condity' :  0.000000E+00,
    'bunch' :  0.000000E+00,
    'bunchphase' :  0.000000E+00,
    'emod' :  0.000000E+00,
    'emodphase' :  0.000000E+00,
    'xlamds':  2.472300E-09,
    'prad0' :  2.000000E-04,
    'pradh0':  0.000000E+00,
    'zrayl' :  3.000000E+01,
    'zwaist':  0.000000E+00,
    'ncar'  :  251,
    'lbc'   :    0,
    'rmax0' :  1.100000E+01,
    'dgrid' :  7.500000E-04,
    'nscr'  :    1,
    'nscz'  :    0,
    'nptr'  :   40,
    'nwig'  :  112,
    'zsep'  :  1.000000E+00,
    'delz'  :  1.000000E+00,
    'nsec'  :    1,
    'iorb'  :    0,
    'zstop' :  3.195000E+11, # note: this is huge
    'magin' :    1,
    'magout':    0,
    'quadf' :  1.667000E+01,
    'quadd' : -1.667000E+01,
    'fl'    :  8.000000E+00,
    'dl'    :  8.000000E+00,
    'drl'   :  1.120000E+02,
    'f1st'  :  0.000000E+00,
    'qfdx'  :  0.000000E+00,
    'qfdy'  :  0.000000E+00,
    'solen' :  0.000000E+00,
    'sl'    :  0.000000E+00,
    'ildgam':    9,
    'ildpsi':    1,
    'ildx'  :    2,
    'ildy'  :    3,
    'ildpx' :    5,
    'ildpy' :    7,
    'itgaus':    1,
    'nbins' :    8,
    'igamgaus' :    1,
    'inverfc' :    1,
    'lout'  : [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
    'iphsty':    1,
    'ishsty':    1,
    'ippart':    0,
    'ispart':    2,
    'ipradi':    0,
    'isradi':    0,
    'idump' :    0,
    'iotail':    1,
    'nharm' :    1,
    'iallharm' :    1,
    'iharmsc' :    0,
    'curpeak':  4.500000E+03,
    'curlen':  0.000000E+00,
    'ntail' :    0,
    'nslice': 4129,
    'shotnoise':  1.000000E+00,
    'isntyp':    1,
    'iall'  :    1,
    'itdp'  :    0,
    'ipseed':    1,
    'iscan' :    0,
    'nscan' :    3,
    'svar'  :  1.000000E-02,
    'isravg':    0,
    'isrsig':    1,
    'cuttail': -1.000000E+00,
    'eloss' :  0.000000E+00,
    'version':  1.000000E-01,
    'ndcut' :  150,
    'idmpfld':    0,
    'idmppar':    0,
    'ilog'  :    0,
    'ffspec':    1,
    'convharm':    1,
    'ibfield':  0.000000E+00,
    'imagl':    0.000000E+00,
    'idril':    0.000000E+00,
    'alignradf':    0,
    'offsetradf':    0,
    'multconv':    0,
    'igamref':  0.000000E+00,
    'rmax0sc':  0.000000E+00,
    'iscrkup':    0,
    'trama':    0,
    'itram11':  1.000000E+00,
    'itram12':  0.000000E+00,
    'itram13':  0.000000E+00,
    'itram14':  0.000000E+00,
    'itram15':  0.000000E+00,
    'itram16':  0.000000E+00,
    'itram21':  0.000000E+00,
    'itram22':  1.000000E+00,
    'itram23':  0.000000E+00,
    'itram24':  0.000000E+00,
    'itram25':  0.000000E+00,
    'itram26':  0.000000E+00,
    'itram31':  0.000000E+00,
    'itram32':  0.000000E+00,
    'itram33':  1.000000E+00,
    'itram34':  0.000000E+00,
    'itram35':  0.000000E+00,
    'itram36':  0.000000E+00,
    'itram41':  0.000000E+00,
    'itram42':  0.000000E+00,
    'itram43':  0.000000E+00,
    'itram44':  1.000000E+00,
    'itram45':  0.000000E+00,
    'itram46':  0.000000E+00,
    'itram51':  0.000000E+00,
    'itram52':  0.000000E+00,
    'itram53':  0.000000E+00,
    'itram54':  0.000000E+00,
    'itram55':  1.000000E+00,
    'itram56':  0.000000E+00,
    'itram61':  0.000000E+00,
    'itram62':  0.000000E+00,
    'itram63':  0.000000E+00,
    'itram64':  0.000000E+00,
    'itram65':  0.000000E+00,
    'itram66':  1.000000E+00,
    'outputfile' : 'genesis.out',
    'maginfile' : 'genesis.lat',
    'distfile': None,
    'filetype':'ORIGINAL'}    

   

DEFAULT_LATTICE_PARMS = {'version': 1, 'unitlength': 0.03}    
    
DEFAULT_LATTICE = [{'type': 'QF', 'strength': 12.84, 'L': 5.0, 's': 5.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 130.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 130.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 135.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 260.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 260.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 265.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 390.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 390.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 395.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 520.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 520.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 525.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 650.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 650.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 655.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 780.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 780.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 785.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 910.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 910.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 915.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1040.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1040.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 1045.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1170.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1170.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 1175.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1300.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1300.0},
     {'type': 'QF', 'strength': 12.84, 'L': 10.0, 's': 1305.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1430.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1430.0},
     {'type': 'QF', 'strength': -12.64, 'L': 10.0, 's': 1435.0},
     {'type': 'AW', 'strength': 2.47318, 'L': 110.0, 's': 1560.0},
     {'type': 'AD', 'strength': 0.29, 'L': 20.0, 's': 1560.0}]    