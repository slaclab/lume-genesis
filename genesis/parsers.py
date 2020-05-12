import numpy as np

from collections import OrderedDict
import re
import os

def isfloat(value):
      try:
            float(value)
            return True
      except ValueError:
            return False
def try_int(x):
    if x == int(x):
        return int(x)
    else:
        return x

    
    
# Simple function to try casting to a float or int
def number(x):
    z = x.replace('D', 'E') # Some floating numbers use D
    if isfloat(z):
        val =  try_int(float(z))
    else:
        # must be a string. Strip quotes.
        val = x.strip().strip('\'').strip('\"')
    return val
    
    
#-------------------------------------------------
# All input
# P

def parse_input(filePath):
    """
    Parses the main input file, and then parses:
        
        
    Returns dict of:
        'main': the main input dict
        'lattice': the lattice, if any.
        'lattice_params'
        'beam': beam description, if any. 
    
    """
    d = {
        'beam':None
    }
    
    d['main'] = parse_main_inputfile(filePath)
    
    if d['beamfile']:
        d['beam'] = parse_beam_file(d['beamfile'])
    
    if d['maginfile']:        
        eles, params = parsers.parse_genesis_lattice(d['maginfile'])
        d['lattice'] = standard_lattice_from_eles(eles)
        d['lattice_params'] = params
              
    return d

#-------------------------------------------------
# Main Input file

def parse_main_inputfile(filePath, expand_paths=True, strict=True, fill_defaults=True):
    """
    Parses the main input file into a dict.
    
    See: genesis.parsers.parse_main_input
    """
    with open(filePath) as f:
        rawtext = f.read()
    dat =  parse_main_input(rawtext, strict=strict, fill_defaults=fill_defaults)
    
    
    
    if expand_paths:
        root, _ = os.path.split(os.path.abspath(filePath))
        for key in POSSIBLE_INPUT_FILES:        
            path = dat[key]
            # Skip None
            if not path:
                continue
           
            path = os.path.expandvars(path)
            if not os.path.isabs(path):
                path = os.path.join(root, path)
            assert os.path.exists(path), f'Path for {key} does not exist: {path}'

            dat[key] = path
            
    return dat

def parse_main_input(rawtext, strict=True, fill_defaults=True):
    """
    Parses the main input text into a dict.
    
    If strict, will ensure that all keys are.
    
    Vectors (lout, wcoef) are flattend:
        wcoefz = 4 5 6
    becomes:
        wcoefz(1) = 4
        wcoefz(2) = 5
        wcoefz(3) = 6
    
    If fill_defaults, the default parameters that Genesis uses will be filled.
    
    """
    # Look for text between $newrun and $end
    text = re.search(r'\$((?i)newrun)\n.*?\$((?i)end)', rawtext, re.DOTALL).group()
    input_parameters = OrderedDict() # Maintain order for readability
    # Split on newline: \n and comma: 
    for line in re.split('\n|,',text)[1:-1]:
        if line.strip() =='':
            continue
        x = line.split('=')
        key = x[0].strip().lower() # force all keys to be lower case
        

        # Look for vectors
        if len(x[1].split()) == 1:
            input_parameters[key] = number(x[1])
        else:
            # OLD: input_parameters[key] = [number(z) for z in x[1].split()]
            # Expand key(1), key(2), etc.
            for i, z in enumerate(x[1].split()):
                nkey = f'{key}({i+1})'
                input_parameters[nkey] = number(z)
            
    # Check that all keys are in MAIN_INPUT_DEFAULT
    if strict:
        bad_keys = []
        for k in input_parameters:
            if k not in MAIN_INPUT_DEFAULT:
                bad_keys.append(k)
        if len(bad_keys) > 0:
            raise ValueError(f'Bad keys found in main input: {bad_keys}')
            
    if not fill_defaults:
        return input_parameters
    
    # Fill defaults.
    d = MAIN_INPUT_DEFAULT.copy()
    for k, v in input_parameters.items():
        d[k] = v
    
    return d
    

    


#-------------------------------------------------
# Lattice


def parse_outfile_lattice(latticetext):
    lines = latticetext.split('\n')
    lines = [x.split() for x in lines if len(x.strip())>0] # Remove empty lines
    rdat = [map(float, x) for x in lines] # Cast to floats
    rdat = list(map(list, zip(*rdat))) # Transpose
    
    header = ['z', 'aw', 'qfld'] #was: s1.split() from below
    data = {}
    # Populate column data
    for i in range(len(header)):
        data[header[i]] = np.array(rdat[i])
        
        
    return data    
    


#-------------------------------------------------
# Slices

def parse_slice(slicetext):
    lines = slicetext.split('\n')
    lines = [x for x in lines if len(x.strip())>0] # Remove empty lines
    header = lines[3].split()
    d = {'index':int(lines[0]),
         'current':float(lines[2].split()[0]),
        }
    rdat = [map(float, x.split()) for x in lines[4:]]
    rdat = list(map(list, zip(*rdat))) # Transpose
    data = {}
    # Populate column data
    for i, key in enumerate(header):
        data[key] = np.array(rdat[i])
    d['data'] = data

    return d
    
    
#-------------------------------------------------
# Full .out file
 
    
def old_parse_genesis_out(fname, save_raw=False):
    """
    Old routine. See the new routine:
    
    parse_genesis_output
    """
    with open(fname, 'r') as f:
        rawdat = f.read()  
    d = {}
    if save_raw:
        d['raw'] = rawdat # Save this just in case something was missed
    
    # magic strings to search for
    s1 = '    z[m]          aw            qfld '
    s2 = '********** output: slice'
    header, dat = rawdat.split(s1)
    sdat = dat.split(s2)
    latticetext = sdat[0]
    slices = sdat[1:]
    
    d['param'] = parse_main_input(header, fill_defaults=False)
    d['lattice'] = parse_outfile_lattice(latticetext)
    d['slice_data'] = [parse_slice(s) for s in slices]
        
    return d    
    
   
# New labels
RELABEL_OUT = {
    'power': 'field_power',
    'p_mid': 'field_power_density_on_axis',
    'phi_mid': 'field_phase_on_axis',
    'r_size':'field_sigma_r',
    'angle': 'field_sigma_diffraction_angle',
    'energy': 'beam_delta_gamma',
    'error': 'error_energy_conservation',
    'xrms':'beam_sigma_x',
    'yrms':'beam_sigma_y',
    '<x>':'beam_mean_x',
    '<y>':'beam_mean_y',
    'e-spread':'beam_sigma_gamma',
    'far_field':'field_far_field_intensity'
}
def new_outfile_label(key):
    if key in RELABEL_OUT:
        return RELABEL_OUT[key]
    return key  
    
    
    
def parse_genesis_out(fname, save_raw=False, relabel=False):
    """
    
    """
    with open(fname, 'r') as f:
        rawdat = f.read()  
    d = {}
    if save_raw:
        d['raw'] = rawdat # Save this just in case something was missed
    
    # magic strings to search for
    s1 = '    z[m]          aw            qfld '
    s2 = '********** output: slice'
    header, dat = rawdat.split(s1)
    sdat = dat.split(s2)
    latticetext = sdat[0]
    slices = sdat[1:]
    
    d['param'] = parse_main_input(header, fill_defaults=False)
    
    # Load lattice readback, as well as slice data, into data
    d['data'] = parse_outfile_lattice(latticetext)
    d2 = d['data']
    
    sdata = [parse_slice(s) for s in slices]
    # Form 1D data
    for key in ['index', 'current']:
        d2[key] = np.array([sli[key] for sli in sdata]) 
    
    # Form 2D data
    skeys = list(sdata[0]['data'])
    for key in skeys:
        if relabel:
            newkey=new_outfile_label(key)
        else:
            newkey=key
        d2[newkey] = np.array([sli['data'][key] for sli in sdata])
    
        
        
    return d       

#-------------------------------------------------
# .lat file
def parse_genesis_lattice(filePath):
    """
    
    """
    
    
    
    with open(filePath, 'r') as f:
        lines = f.readlines()
        lattice = parse_genesis_lattice_lines(lines)
    return lattice
    
    
def parse_genesis_lattice_lines(lines):
    """
    Parses a Genesis style into a list of ele dicts
    Will unfold loops 
    
    returns dict of
    eles: list of elements, as dicts with keys: type, strength, L, d   
    param: parameter dicts identifed with ? <key> = <value>
    """
    commentchar =  '#'
    inLoop = False
    eles = []
    param = {}
    
    for line in lines:
        x = line.strip()
        if len(x) ==0:
            continue
            
        # Parameter, starts with ?
        if x[0] == '?':
            a = x[1:].split('=')
            key = a[0].strip().lower()
            # Strip off comments:
            val = a[1].split(commentchar)[0]
            val = number(val)    
            param[key] = val
            continue
        # Comment line
        if x[0] == commentchar:
            ele = {'type':'comment', 'text':line.strip('\n'), 'zend':0}
            eles.append(ele)
            continue    
            
        # Loop commands: ! LOOP = <integer>, and ! ENDLOOP
        if x[0] == '!':
            command = x[1:].split('=')
            if command[0].strip().upper()=='LOOP':
                nloop = int(command[1])
                inLoop = True
                loopeles = []
            elif command[0].strip().upper() == 'ENDLOOP':
                inLoop = False
                for e in nloop*loopeles:
                    eles.append(e.copy())
            continue
        
        # must be an ele
        ele = {}
           
        # Look for inline comment
        y = x.split('#', 1)
        if len(y) > 1:
            ele['comment'] =  y[1]
            
        # element: type, strength, L, d
        x = x.split()
        ele['type'] = x[0].upper()
        ele['strength'] = float(x[1])
        ele['L'] = float(x[2])
        ele['d'] = float(x[3]) 
        
        if inLoop:
            loopeles.append(ele)
        else:
            eles.append(ele)
            
    return {'eles':eles, 'param':param}


    
    
#-------------------------------------------------
# beam file
#    Beam Description File
    
    

VALID_BEAM_COLUMNS = {'zpos',
 'tpos',
 'gamma0',
 'delgam',
 'emitx',
 'emity',
 'rxbeam',
 'betax',
 'rybeam',
 'betay',
 'xbeam',
 'ybeam',
 'pxbeam',
 'pybeam',
 'alphax',
 'alphay',
 'curpeak',
 'eloss'}


#LINES = open().readlines()

def parse_beam_file_header(fname):
    """
    Parses a Genesis beam file header. 
    
    Helper routine for parse_beam_file
    """
    
    params = {}
    with open(fname) as f:
        i = 0
        for line in f:
            i += 1
            x = line.strip()
            # Skip comments
            if x.startswith('#') or x == '':
                continue
            # parameter 
            
            if x.startswith('?'):
                x = x[1:].strip()
                if '=' in x:
                    a = x.split('=')
                    params[a[0].strip().lower()] = a[1].strip()
                else:
                    # Should be columns
                    col = [a.lower() for a in x.split()]
                    assert col[0] == 'columns', f'Unknown parameter: {x}'
                    params['columns'] = col[1:]
            else:
                #print(x)
                # Finished. 
                params['n_headerlines'] = i-1
                if 'size' in params:
                    params['size'] = int(params['size'])
                return params
            
def parse_beam_file(fname, verbose=False):
    """
    Parses a Gensis beam file. 
    
    Asserts that the version is 1.0.
    
    SIZE is optional, but will check.
    
    Returns a dict of the columns.
    
    See: genesis.writers.write_beam_file
    
    """
    
    params = parse_beam_file_header(fname)
    
    dat = np.loadtxt(fname, skiprows=params['n_headerlines'])
    
    size = dat.shape[0]
    
    # Check version
    if 'version' in params:
        v = float(params['version'])
        assert v == 1.0 # This is the only version allowed
    
    # Check size
    if 'size' in params:
        assert size == params['size'], f"Mismatch with SIZE = {params['size']} and found column size {size}"   
    
    cdat = {}
    for i, name in enumerate(params['columns']):
        assert name in  VALID_BEAM_COLUMNS, f'{name} is not a valid beam column'
            
        cdat[name] = dat[:, i]
    
    if verbose:
        print(f'Parsed beam file: {fname} with {len(cdat)} columns')
    
    return cdat    
    
#-------------------------------------------------
#.dfl file
#    Dump file at the end
#    complex numbers, output in a loop over nslices, ny, nx
    
def parse_genesis_dfl(fname, nx):
    """
    fname: filename
    nx: grid size in x and y. Same as Genesis 'ncar'
    
    returnsReturns numpy.array:
    
    [z, x, y]
    
    """
    dat = np.fromfile(fname, dtype=np.complex).astype(np.complex)
    npoints = dat.shape[0] 
    
    # Determine number of slices
    ny = nx
    nz =  npoints / ny /nx
    assert (nz % 1 == 0) # 
    nz = int(nz)   
    dat = dat.reshape(nz, ny, nx)    
    dat = np.moveaxis(dat, [1,2], [2,1]) # exchange x and y 
    
    return dat
    


#-------------------------------------------------
#.fld file
#    history file
#    output in a loop over histories, nslices, real/imaginary, ny, nx
    
def parse_genesis_fld(fname, nx, nz):
    """
    fname: filename
    ncar: grid size in x and y
    nx: grid size in x and y. Same as Genesis 'ncar'  
    nz: number of slices
    
    The number of histories can be computed from these. Returns numpy.array:
    
    [history, z, x, y]
    
    """
  
    # Real and imaginary parts are separated
    dat = np.fromfile(fname, dtype=np.float).astype(float)
    npoints = dat.shape[0]
    # Determine number of slices
    ny = nx
    
    nhistories =  npoints / nz / 2 / ny / nx # 
    assert (nhistories % 1 == 0) # 
    nhistories = int(nhistories)   
    
    # real and imaginary parts are written separately. 
    dat = dat.reshape(nhistories, nz, 2,  ny, nx) # 
    dat =  np.moveaxis(dat, 2, 4) # Move complex indices to the end
    dat =  np.moveaxis(dat, [2,3], [3,2]) # exchange x and y
    # Reform complex numbers:
    dat = 1j*dat[:,:,:,:,1] + dat[:,:,:,:,0]

    
    return dat        
    
    
    
    
    
def parse_genesis_dpa(fname, npart):
    """
    Parses .dpa and .par files
    
    
    """
    pdat = np.fromfile(fname, dtype=np.float64) #.astype(float)
    nbunch = int(len(pdat)/6/npart)
    
    # gamma, phase, x, y, px/mc, py/mc
    # .par file: phase = psi  = kw*z + field_phase
    # .dpa file: phase = kw*z 
    
    
    bunch = pdat.reshape(nbunch,6,npart)
    return bunch    






    
#-------------------------------------------------
# Input defaults
# 

# Possible input files and recommended names
POSSIBLE_INPUT_FILES = {
 'maginfile': 'genesis_lattice.in',
 'beamfile':  'genesis_beam.in',
 'radfile':   'genesis_rad.in',      
 'distfile':  'genesis_dist.in',
 'fieldfile': 'genesis_field.in',
 'partfile':  'genesis_part.in'
}

MAIN_INPUT_DEFAULT = {
 # Possible Inputs files
 'maginfile': '',
 'beamfile': '',
 'radfile': '',      
 'distfile': '',
 'fieldfile': '',
 'partfile': '',

 # Output files
 'magoutfile': 'genesis_lattice.out',
 'outputfile': 'genesis.out',    
    
 # from template.in that Genesis creates  
 'aw0': 0.735,
 'xkx': 0,
 'xky': 1,
 'wcoefz(1)': 0,
 'wcoefz(2)': 0,
 'wcoefz(3)': 0,
 'xlamd': 0.0205,
 'fbess0': 0,
 'delaw': 0,
 'iertyp': 0,
 'iwityp': 0,
 'awd': 0.735,
 'awx': 0,
 'awy': 0,
 'iseed': -1,
 'npart': 8192,
 'gamma0': 35.2,
 'delgam': 0.005,
 'rxbeam': 0.0001121,
 'rybeam': 0.0001121,
 'alphax': 0,
 'alphay': 0,
 'emitx': 2e-06,
 'emity': 2e-06,
 'xbeam': 0,
 'ybeam': 0,
 'pxbeam': 0,
 'pybeam': 0,
 'conditx': 0,
 'condity': 0,
 'bunch': 0,
 'bunchphase': 0,
 'emod': 0,
 'emodphase': 0,
 'xlamds': 1.2852e-05,
 'prad0': 10,
 'pradh0': 0,
 'zrayl': 0.5,
 'zwaist': 0,
 'ncar': 151,
 'lbc': 0,
 'rmax0': 9,
 'dgrid': 0,
 'nscr': 0,
 'nscz': 0,
 'nptr': 40,
 'nwig': 98,
 'zsep': 1,
 'delz': 1,
 'nsec': 1,
 'iorb': 0,
 'zstop': -1,
 'magin': 0,
 'magout': 0,
 'quadf': 1.23,
 'quadd': 0,
 'fl': 98,
 'dl': 0,
 'drl': 0,
 'f1st': 0,
 'qfdx': 0,
 'qfdy': 0,
 'solen': 0,
 'sl': 0,
 'ildgam': 5,
 'ildpsi': 7,
 'ildx': 1,
 'ildy': 2,
 'ildpx': 3,
 'ildpy': 4,
 'itgaus': 1,
 'nbins': 4,
 'igamgaus': 1,
 'inverfc': 0,
 'lout(1)': 1,
 'lout(2)': 1,
 'lout(3)': 1,
 'lout(4)': 1,
 'lout(5)': 1,
 'lout(6)': 0,
 'lout(7)': 1,
 'lout(8)': 1,
 'lout(9)': 1,
 'lout(10)': 1,
 'lout(11)': 1,
 'lout(12)': 0,
 'lout(13)': 0,
 'lout(14)': 0,
 'lout(15)': 0,
 'lout(16)': 0,
 'lout(17)': 0,
 'lout(18)': 0,
 'lout(19)': 0,
 'lout(20)': 0,    # Note that lout(20:24) are missing in the default template.in
 'lout(21)': 0,
 'lout(22)': 0,
 'lout(23)': 0,
 'lout(24)': 0,    
 'iphsty': 1,
 'ishsty': 1,
 'ippart': 0,
 'ispart': 0,
 'ipradi': 0,
 'isradi': 0,
 'idump': 0,
 'iotail': 0,
 'nharm': 1,
 'iallharm': 0,
 'iharmsc': 0,
 'curpeak': 250,
 'curlen': 0.001,
 'ntail': -253,
 'nslice': 408,
 'shotnoise': 1,
 'isntyp': 0,
 'iall': 0,
 'itdp': 0,
 'ipseed': -1,
 'iscan': 0,
 'nscan': 3,
 'svar': 0.01,
 'isravg': 0,
 'isrsig': 0,
 'cuttail': -1,
 'eloss': 0,
 'version': 0.1,
 'ndcut': -1,
 'idmpfld': 0,
 'idmppar': 0,
 'ilog': 0,
 'ffspec': 0,
 'convharm': 1,
 'ibfield': 0,
 'imagl': 0,
 'idril': 0,
 'alignradf': 0,
 'offsetradf': 0,
 'multconv': 0,
 'igamref': 0,
 'rmax0sc': 0,
 'iscrkup': 0,
 'trama': 0,
 'itram11': 1,
 'itram12': 0,
 'itram13': 0,
 'itram14': 0,
 'itram15': 0,
 'itram16': 0,
 'itram21': 0,
 'itram22': 1,
 'itram23': 0,
 'itram24': 0,
 'itram25': 0,
 'itram26': 0,
 'itram31': 0,
 'itram32': 0,
 'itram33': 1,
 'itram34': 0,
 'itram35': 0,
 'itram36': 0,
 'itram41': 0,
 'itram42': 0,
 'itram43': 0,
 'itram44': 1,
 'itram45': 0,
 'itram46': 0,
 'itram51': 0,
 'itram52': 0,
 'itram53': 0,
 'itram54': 0,
 'itram55': 1,
 'itram56': 0,
 'itram61': 0,
 'itram62': 0,
 'itram63': 0,
 'itram64': 0,
 'itram65': 0,
 'itram66': 1,
 'filetype': 'ORIGINAL'
}