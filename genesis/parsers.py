

from collections import OrderedDict
import re
import numpy as np




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



#-------------------------------------------------
# Input file


# Simple function to try casting to a float or int
def number(x):
    z = x.replace('D', 'E') # Some floating numbers use D
    if isfloat(z):
        val =  try_int(float(z))
    else:
        # must be a string. Strip quotes.
        val = x.strip().strip('\'').strip('\"')
    return val


def parse_inputfile(filePath):
    with open(filePath) as f:
        rawtext = f.read()
    return parse_input(rawtext)



def parse_input(rawtext):
    # Look for text between $newrun and $end
    text = re.search(r'\$((?i)newrun)\n.*?\$((?i)end)', rawtext, re.DOTALL).group()
    input_parameters = OrderedDict() # Maintain order for readability
    # Split on newline: \n and comma: 
    for line in re.split('\n|,',text)[1:-1]:
        if line.strip() =='':
            continue
        x = line.split('=')
        key = x[0].strip().lower() # force all keys to be lower case
        if len(x[1].split()) == 1:
            input_parameters[key] = number(x[1])
        else:
            input_parameters[key] = [number(z) for z in x[1].split()]
    return input_parameters
    
    
    


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
    for i in range(len(header)):
        data[header[i]] = np.array(rdat[i])
    d['data'] = data

    return d
    
    
#-------------------------------------------------
# Full .out file
 
    
def parse_genesis_out(fname, save_raw=False):
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
    
    d['input_parameters'] = parse_input(header)
    d['lattice'] = parse_outfile_lattice(latticetext)
    d['slice_data'] = [parse_slice(s) for s in slices]
        
    return d    
    

#-------------------------------------------------
# .lat file
def parse_genesis_lattice(filePath):
    """
    
    """
    with open(filePath, 'r') as f:
        lines = f.readlines()
        eles, params = parse_genesis_lattice_lines(lines)
    return eles, params  
    
    
def parse_genesis_lattice_lines(lines):
    """
    Parses a Genesis style into a list of ele dicts
    Will unfold loops 
    
    returns 
    eles: list of elements, as dicts with keys: type, strength, L, d   
    params: parameter dicts identifed with ? <key> = <value>
    """
    commentchar =  '#'
    inLoop = False
    eles = []
    params = {}
    
    for line in lines:
        x = line.strip()
        if len(x) ==0:
            continue
        # Parameter
        if x[0] == '?':
            a = x[1:].split('=')
            params[a[0].strip().lower()] = number(a[1])
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
            
    return eles, params


    
    
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
            
def parse_beam_file(fname):
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
