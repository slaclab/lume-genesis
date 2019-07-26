""" 
Genesis wrapper for use in LUME

 
"""
from __future__ import print_function # python 2.7 compatibility
from genesis import parsers, lattice


import os, errno, random, string, subprocess, copy
import numpy as np
import subprocess
import numbers


MY_GENESIS_BIN = os.path.expandvars('$HOME/bin/genesis')
MY_WORKDIR = os.path.expandvars('$HOME/work/')

class Genesis:
    """ This class allows us to write inputs, run genesis, return data, and clean up genesis junk."""
    
    def __del__(self):
        if  self.auto_cleanup:
            self.clean() # clean directory before deleting
        
    def __init__(self, genesis_bin=MY_GENESIS_BIN, workdir=MY_WORKDIR, input_filePath=None):
        self.genesis_bin = genesis_bin
        self.binary_prefixes = [] #  For example, ['mpirun', '-n', '2']
        self.finished = False
        
        # For loading an existing input file
        if input_filePath:
            # Separate path and filename
            self.sim_path, self.sim_input_file = os.path.split(input_filePath)
            self.load_inputfile(input_filePath)
            
            self.load_lattice()
            self.load_outputfile()
            self.finished = True
            self.auto_cleanup = False          
        else:
            self.sim_id = 'genesis_run_' + randomword(10)
            self.sim_path =  workdir + self.sim_id + '/'
            mkdir_p(self.sim_path)

            # input params
            # param descriptions here http://genesis.web.psi.ch/download/documentation/genesis_manual.pdf
            self.input_params = DEFAULT_INPUT_PARAMS        
        
            # some file paths (more in self.input_params just below)
            self.sim_input_file = 'genesis.in'
        
            self.lattice = None
            self.output = None
            self.auto_cleanup = True
            self.sim_log_file = 'genesis.log'
  
        # Option for cleaning on exit
       
        

    def load_outputfile(self, filePath=None):
        if not filePath:
            fname = os.path.join(self.sim_path,self.input_params['outputfile'])
        else:
            fname = filePath
        if os.path.exists(fname):
            self.output = parsers.parse_genesis_out(fname)    

    def load_inputfile(self, filePath):
        """
        Loads an inputfile. 
        
        """
        self.input_params = parsers.parse_inputfile(filePath)    
        
    
    def load_lattice(self, filePath=None, verbose=False):
        """
        loads an original Genesis-style lattice into a standard_lattice
        """
        if not filePath:
            fname = os.path.join(self.sim_path, self.input_params['maginfile'])
        else:
            fname = filePath
            
        if verbose: print('loading lattice: ', fname)    
        eles, params = parsers.parse_genesis_lattice(fname)
        
        self.lattice = lattice.standard_lattice_from_eles(eles)
        self.lattice_params = params

    def write_lattice(self):
    
        if not self.lattice:
            # use old routine
            self.old_write_lattice()
    
        else:
            filePath = os.path.join(self.sim_path, self.input_params['maginfile'])
            lattice.write_lattice(filePath, self.lattice, self.lattice_params['unitlength'])
            
               
    def input_twiss(self):
        
        betax = self.input_params['rxbeam']**2 * self.input_params['gamma0'] / self.input_params['emitx']
        betay = self.input_params['rybeam']**2 * self.input_params['gamma0'] / self.input_params['emity']
        alphax = self.input_params['alphax']
        alphay = self.input_params['alphay']
        
        return {'betax':betax, 'betay':betay, 'alphax':alphax, 'alphay':alphay} 
    
    def clean(self):
        os.system('rm -rf ' + self.sim_path)
    
    
      
    def write_input(self):
        """
        Write parameters to main .in file
        
        """    
        with open(self.sim_path + self.sim_input_file, "w") as f:
            f.write("$newrun\n")
         
            # parse
            for key, value in self.input_params.items():
                #if type(value) == type(1) or type(value) == type(1.): # numbers
                if isinstance(value,numbers.Number): # numbers
                    f.write(key + ' = ' + str(value) + '\n')
                elif type(value) == type([]): # lists
                    liststr = ''
                    for item in value:
                        liststr += str(item) + ' '
                    f.write(key + ' = ' + liststr + '\n')
                elif type(value) == type('a'): # strings
                    f.write(key + ' = ' + "'" + value + "'" + '\n') # genesis input may need apostrophes
                else:
                    #print 'skipped: key, value = ', key, value
                    pass
            
            f.write("$end\n")
            
            f.close()
    

        
        
    def run_genesis(self, parseOutput=True):
        # Save init dir
        print('init dir: ', os.getcwd())
        init_dir = os.getcwd()
        os.chdir(self.sim_path)
        # Debugging
        print('running genesis in '+os.getcwd())
        self.write_input()
        
        self.write_lattice()


        runscript = [self.genesis_bin, self.sim_input_file]

        # Allow for MPI commands
        if len(self.binary_prefixes) > 0:
            runscript = self.binary_prefixes + runscript
    
        log = []
        for path in execute(runscript):
            print(path, end="")
            log.append(path)
        with open(self.sim_log_file, 'w') as f:
            for line in log:
                f.write(line)
    
        if parseOutput:
            self.load_outputfile()


        self.finished = True
        
        
        # Return to init_dir
        os.chdir(init_dir)
        


#------------------------------------
# To deprecate :


    # write the magnetic lattice file for Genesis 1.3 v2
    def old_write_lattice(self):
        # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        
        #self.quad_grads, self.und_Ks
        
        # input lattice
        # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        quad_grads = DEFAULT_QUAD_GRADS #= 6*[12.84,-12.64] # 6 FODO
        # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        und_Ks = DEFAULT_UND_Ks #= 2*[np.sqrt(2.) * 2.473180]
        
    
        
        quads =quad_grads
        Ks = und_Ks / np.sqrt(2.) # change peak to rms
        
        nquad = len(quads)
        nund = len(Ks)
        nund = min(nquad,nund)
        
        f = open(self.sim_path + self.input_params['maginfile'], "w")
        
        f.write("? VERSION = 1.0" + '\n')
        f.write("? UNITLENGTH = " + str(self.input_params['xlamd']) + '\n')
        f.write('\n')
        f.write("QF " + str(quads[0]) + " 5 0" + '\n') # half of first quad
        f.write('\n')
        
        # parse
        for i in range(nund):
            f.write("AW " + str(Ks[i]) + " 110 20" + '\n')
            f.write("AD " + str(0.29) + " 20 110" + '\n')
            try:
                f.write("QF " + str(quads[i+1]) + " 10 120" + '\n\n')
            except:
                #if i >= nund-1:  # this will never be true
                print(str(self.__class__) + '.write_lattice - WARNING: ran out of quads for lattice...')
                break
        
        f.close()        
        
#----------------
# Helper routines


def execute(cmd):
    """
    
    Constantly print Subprocess output while process is running
    from: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    
    # Example usage:
        for path in execute(["locate", "a"]):
        print(path, end="")
        
    Useful in Jupyter notebook
    
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def randomword(length):
   letters = string.ascii_letters + string.digits
   return ''.join(random.choice(letters) for i in range(length))


  
DEFAULT_QUAD_GRADS = 6*[12.84,-12.64] # 6 FODO  
  
DEFAULT_UND_Ks = 12*[np.sqrt(2.) * 2.473180]     
    
    
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
