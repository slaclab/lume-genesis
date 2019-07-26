# -*- coding: iso-8859-1 -*-

""" 


 Adapted from: 
 Joe Duris / jduris@slac.stanford.edu / 2018-07-31
 
 Genesis - Genesis 1.3 v2 interface for Python
 Grants (dubious?) knowledge of the FEL.
 Manages Genesis simulations
 
 serpent - controller of the Genesis
 Named after the manipulating serpent in the book of Genesis.
 Manages Genesis to execute and clean Genesis sims.
 Also allows optimizing the detuning for time independent sims.
 
 TODO: parallelize serpent routines
 TODO: calculate gradients and hessians
 TODO: read in/out particle distributions and radiation files
 
"""
from __future__ import print_function # python 2.7 compatibility
from genesis import parsers, lattice


import os, errno, random, string, subprocess, copy
import numpy as np
import subprocess



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

#genesis_bin = 'genesis' # genesis command
#genesis_bin = '/u/ra/jduris/bin/genesis'        # genesis command works from OPIs
#genesis_bin = '/u/ra/jduris/bin/genesis_single' # genesis command works from AFS


MY_GENESIS_BIN = os.path.expandvars('$HOME/bin/genesis')
MY_WORKDIR = os.path.expandvars('$HOME/work/')

class Genesis:
    """ This class allows us to write inputs, run genesis, return data, and clean up genesis junk."""
    
    def __del__(self):
        if  self.auto_cleanup:
            self.clean() # clean the crap before deleting
        
    def __init__(self, genesis_bin=MY_GENESIS_BIN, workdir=MY_WORKDIR):
        self.class_name = 'Genesis'

        self.genesis_bin = genesis_bin
        #self.init_dir = os.getcwd() # save initial directory to return to once cleaned
        
        # make simulation directory
        self.sim_id = 'genesis_run_' + randomword(10)
        self.sim_path =  workdir + self.sim_id + '/'
        mkdir_p(self.sim_path)
        
        
        self.lattice = None
        
        # some file paths (more in self.input_params just below)
        self.sim_input_file = 'genesis.in'
        self.sim_log_file = 'genesis.log'
  
        # Option for cleaning on exit
        self.auto_cleanup = True
        
        # input lattice
        # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        self.quad_grads = DEFAULT_QUAD_GRADS #= 6*[12.84,-12.64] # 6 FODO
        # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        self.und_Ks = DEFAULT_UND_Ks #= 2*[np.sqrt(2.) * 2.473180]
        
        # input params
        # param descriptions here http://genesis.web.psi.ch/download/documentation/genesis_manual.pdf
        self.input_params = DEFAULT_INPUT_PARAMS
    
    
    def load_lattice(self, filePath):
        """
        loads an original Genesis-style lattice into a standard_lattice
        """

        eles, params = parsers.parse_genesis_lattice(filePath)
        
        self.lattice = lattice.standard_lattice_from_eles(eles)
        self.lattice_params = params

    def write_lattice(self):
    
        if not self.lattice:
            # use old routine
            self.old_write_lattice()
    
        else:
            unitlength = self.lattice_params['unitlength']
            lines = lattice.genesis_lattice_from_standard_lattice(self.lattice, unitlength=unitlength)
            with open(self.sim_path + self.input_params['maginfile'], "w") as f:
                for l in lines:
                    f.write(l+'\n')
            

    
    def input_twiss(self):
        
        betax = self.input_params['rxbeam']**2 * self.input_params['gamma0'] / self.input_params['emitx']
        betay = self.input_params['rybeam']**2 * self.input_params['gamma0'] / self.input_params['emity']
        alphax = self.input_params['alphax']
        alphay = self.input_params['alphay']
        
        return {'betax':betax, 'betay':betay, 'alphax':alphax, 'alphay':alphay} 
    
    def clean(self):
        # run genesis executable
        #p = subprocess.Popen(['rm', '-rf', self.sim_path], bufsize=2048, stdin=subprocess.PIPE)
        #p.stdin.write('e')
        #p.wait()
        #if p.returncode == 0:
            #pass # put code that must only run if successful here.
        #else:
            #print self.class_name + ' - ERROR: Could not remove directory ' + self.sim_path
        os.system('rm -rf ' + self.sim_path)
        # os.chdir(self.init_dir)
        
    def write_input(self):
        
        f = open(self.sim_path + self.sim_input_file, "w")
        
        f.write("$newrun\n")
        
        import numbers # so many numbers, so list time
        
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
    
    

    
    # write the magnetic lattice file for Genesis 1.3 v2
    def old_write_lattice(self):
        # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        
        #self.quad_grads, self.und_Ks
        
        quads = self.quad_grads
        Ks = self.und_Ks / np.sqrt(2.) # change peak to rms
        
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
                print(self.class_name + '.write_lattice - WARNING: ran out of quads for lattice...')
                break
        
        f.close()
        
    def run_genesis_and_read_output(self, column=0, waitQ=True):
        
        self.run_genesis()
        
        return self.read_output(column)
        
    def run_genesis(self, waitQ=True):
        # Save init dir
        print('init dir: ', os.getcwd())
        init_dir = os.getcwd()
        os.chdir(self.sim_path)
        # Debugging
        print('running genesis in '+os.getcwd())
        self.write_input()
        
        self.write_lattice()

        runscript = [self.genesis_bin, self.sim_input_file]
        log = []
        for path in execute(runscript):
            print(path, end="")
            log.append(path)
        with open('genesis.log', 'w') as f:
            for line in log:
                f.write(line)
        
        # Return to init_dir
        os.chdir(init_dir)
        

    # based on https://raw.githubusercontent.com/archman/felscripts/master/calpulse/calpulse.py
    def read_output(self, column=0, stat=np.max):
        # column => columns of genesis.out; column 0 is power

        #filename1 = self.outputfilename # TDP output filename defined by external parameter
        filename1 = self.sim_path+self.input_params['outputfile']
        #slicetoshow = int(sys.argv[2]) # slice number to show as picture
        #zrecordtoshow = int(sys.argv[2])# z-record num
        idx = column

        #open files
        f1 = open(filename1, 'r')

        #extract z, au, QF [BEGIN]
        while not f1.readline().strip().startswith('z[m]'):pass
        zaq   = []
        line  = f1.readline().strip()
        count = 0
        while line:
            zaq.append(line)
            line = f1.readline().strip()
            count += 1
        #print "%d lines have been read!" %count
        #count: total z-record number
        #extraxt z, au, QF [END]


        #find where to extract power ...
        slicenum = 0 # count read slice num
        data=[]
        while True:
            while not f1.readline().strip().startswith('power'):pass
            data.append([])
            slicenum += 1
            line = f1.readline().strip()
            while line: 
        #        data[slicenum-1].append(["%2.6E" %float(x) for x in line])
                data[slicenum-1].append(line)
                line = f1.readline().strip()
        #    print 'Read slice %05d' %slicenum
            if not f1.readline():break

        f1.close()
        #data extraction end, close opened file

        #print sys.getsizeof(zaq)
        #raw_input()

        #cmd1 = "/bin/grep -m1 sepe " + filename1 + " | awk '{print $1}'"
        cmd2 = "grep xlamd "    + filename1 + " | grep -v xlamds | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd3 = "grep delz "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd4 = "grep zsep "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd5 = "grep iphsty "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd6 = "grep ishsty "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"
        cmd7 = "grep xlamds "     + filename1 + " | awk -F'=' '{print $NF}' | sed 's/[D,d]/e/g'"

        try:
            import subprocess
            #dels  = float(subprocess.check_output(cmd1, shell = True))
            xlamd = float(subprocess.check_output(cmd2, shell = True))
            delz  = float(subprocess.check_output(cmd3, shell = True))
            zsep = float(subprocess.check_output(cmd4, shell = True))
            iphsty = float(subprocess.check_output(cmd5, shell = True))
            ishsty = float(subprocess.check_output(cmd6, shell = True))
            xlamds = float(subprocess.check_output(cmd7, shell = True))
        except AttributeError:
            import os
            #dels  = float(os.popen4(cmd1)[1].read())
            xlamd = float(os.popen4(cmd2)[1].read())
            delz  = float(os.popen4(cmd3)[1].read())
            zsep = float(os.popen4(cmd4)[1].read())
            iphsty = float(os.popen4(cmd5)[1].read())
            ishsty = float(os.popen4(cmd6)[1].read())
            xlamds = float(os.popen4(cmd7)[1].read())

        c0 = 299792458.0

        dz = xlamd * delz * iphsty
        ds = xlamds * zsep * ishsty
        dt = ds / c0

        import numpy as np
        x  =  np.arange(count)
        s  =  np.arange(slicenum)
        z  =  np.array([float(zaq[i].split()[0]) for i in x])
        #p1 =  [data[slicetoshow][i].split()[0] for i in x]
        ##ps =  [data[i][zrecordtoshow].split()[0] for i in s]
        ##plot(s,ps,'r-')
        ##plot(z,p1,'r-')

        j=0
        pe=[]
        pmax = 0
        #idx = int(sys.argv[4]) # moved up
        """
        idx = 0  # fundamental power
        idx = 15 # 3rd harmonic power
        idx = 23 # 5th harmonic power
        """
        ssnaps = [] # list of list for a constant s (beam coordinate)
        #while j < count:
        for j in range(count): # loop over s slices (beam coords)
                psi =  [data[i][j].split()[idx] for i in s]
                ssnap = [float(x) for x in psi]
                ssnaps += [ssnap]
                
                #ptmp = max(ssnap) # max seen at this s-coord
                #if ptmp > pmax:
                        #pmax = ptmp
                #pe.append(sum([float(x) for x in psi])*dt)
        #maxpe = max(pe)
        #psmax = [data[i][pe.index(maxpe)].split()[0] for i in s]
        #print "Pulse Energy: ", maxpe*1e9, "nJ @ z= ", pe.index(maxpe)*dz
        #print "Max Power: ",pmax, "W"

        #print 'count = ', count
        #print 'np.shape(ssnaps) = ', np.shape(ssnaps)
        #print 'np.shape(data) = ', np.shape(data)
        #print 'np.shape(x) = ', np.shape(x)
        #print 'np.shape(s) = ', np.shape(s)
        #print 'np.shape(z) = ', np.shape(z)
        #print 'z = ', z

        #print 'np.shape(np.mean(ssnaps,axis=1)) = ', np.shape(np.mean(ssnaps,axis=1))
        #print 'np.mean(ssnaps,axis=1) = ', np.mean(ssnaps,axis=1)
        #print 'stat(ssnaps,axis=1) = ', stat(ssnaps,axis=1)
        
        stat_vs_z = stat(ssnaps,axis=1)
        
        self.output = (z, stat_vs_z)
        
        return self.output
        
        
        
class serpent():
    """ This class allows us to control Genesis runs."""
    
    def __init__(self):
        # list of sims
        self.Genesiss = []
        
        # input lattice
        # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        #self.quad_half_length_xlamd_units = 5 # multiply by 2*xlamd for full quad length
        self.quad_grads = [12.84,-12.64,12.84,-12.64,12.84,-12.64,12.84,-12.64,12.84,-12.64,12.84,-12.64] # 6 FODO
        # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        self.und_Ks = [np.sqrt(2.) * K for K in [2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180,2.473180]]
        
        # input params for the sims
        # param descriptions here http://genesis.web.psi.ch/download/documentation/genesis_manual.pdf
        self.input_params = DFEAULT_INPUT_PARAMS 

        #self.input_params = None # if we want to use defaults..
        
    # stub - fcn to calculate the twiss given a lattice
    def matched_twiss(self):
        pass
    
    def input_twiss(self):
        
        betax = self.input_params['rxbeam']**2 * self.input_params['gamma0'] / self.input_params['emitx']
        betay = self.input_params['rybeam']**2 * self.input_params['gamma0'] / self.input_params['emity']
        alphax = self.input_params['alphax']
        alphay = self.input_params['alphay']
        
        return {'betax':betax, 'betay':betay, 'alphax':alphax, 'alphay':alphay} 
    
    def run_genesis_for_twiss(self, betax=None, betay=None, alphax=None, alphay=None):
        
        ff = Genesis()
        
        ff.input_params = copy.deepcopy(self.input_params)
        ff.quad_grads = copy.deepcopy(self.quad_grads) # quads are gradients in Tesla/meter (use a negative gradient to defocus)
        ff.und_Ks = copy.deepcopy(self.und_Ks) # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
        
        #########################################################################
        ## WARNING: this should be changed:
        #ff.input_params['zstop'] = 30.6
        #########################################################################
                
        gamma0 = ff.input_params['gamma0'] # beam energy
        emitx = ff.input_params['emitx'] # normalized emit
        emity = ff.input_params['emity']
        
        if type(betax) is not type(None):
            ff.input_params['rxbeam'] = np.sqrt(betax * emitx / gamma0)
        
        if type(betay) is not type(None):
            ff.input_params['rybeam'] = np.sqrt(betay * emity / gamma0)
        
        if type(alphax) is not type(None):
            ff.input_params['alphax'] = alphax
            
        if type(alphay) is not type(None):
            ff.input_params['alphay'] = alphay
        
        # return tuple of lists (zs, powers)
        ffout = ff.run_genesis_and_read_output()
        
        #del ff
        
        return ffout
        
        #self.Genesiss += [ff]
        
    # when running time independent sims (itdp=0), need to optimize detuning
    def optimize_detuning(self, relative_range = 0.05, nsteps = 21):
        
        # calculate with the resonance condition (ignoring emittance) # should also add angular spread prob.
        xlamds_guess = self.input_params['xlamd'] / 2. / self.input_params['gamma0']**2 * (1. + 0.5 * self.und_Ks[0]**2)
        
        xlamds_list = xlamds_guess * (1. + relative_range*np.linspace(-1,1,nsteps)) # list of xlamds to try
        maxps = []
        
        for xlamds in xlamds_list:
        
            ff = Genesis()
            
            ff.input_params = copy.deepcopy(self.input_params)
            ff.quad_grads = copy.deepcopy(self.quad_grads) # quads are gradients in Tesla/meter (use a negative gradient to defocus)
            ff.und_Ks = copy.deepcopy(self.und_Ks) # Ks are the list of peak undulator strengths (NOT RMS) since epics gives us peak
            
            ff.input_params['xlamds'] = xlamds
            ff.input_params['zstop'] = 4. # ~1 undulator just for the detuning optimization
            
            (zs, ps) = ff.run_genesis_and_read_output()
            
            maxps += [ps[-1]]
            
            del ff
            
        # interpolate to find maximum
        
        maxps = np.array(maxps) # convert to numpy array
        
        #print (xlamds_list,maxps) # maybe try to make a plot of the scan result (print to a file; not to a window)
        #from matplotlib import pyplot as plt
        
        from scipy import interpolate
        
        interp = interpolate.interp1d(xlamds_list, maxps, kind='cubic')
        
        xlamds_list_finer = xlamds_guess * (1. + relative_range*np.linspace(-1,1,nsteps*100+1))
        maxps_finer = np.array([interp(x) for x in xlamds_list_finer])
        
        xlamds_best = xlamds_list_finer[maxps_finer == max(maxps_finer)]
        
        xlamds0 = self.input_params['xlamds'] # save old xlamds
        
        self.input_params['xlamds'] = xlamds_best # automatically overwrite xlamds with the optimum
        
        print('Guessed resonant wavelength of ', xlamds_guess, ' m. Changed xlamds from ', xlamds0, ' m to ', xlamds_best, ' m')
        
        return xlamds_best
        
    # stub 
    def hessian(self):
        pass
        
    # gradient 
    def gradient(self):
        pass
  
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
