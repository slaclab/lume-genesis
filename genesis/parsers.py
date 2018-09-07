

from collections import OrderedDict as odict
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
        val = x
    return val

def parse_inputfile(rawtext):
    text = re.search(r'\$newrun\n.*?\$end', rawtext, re.DOTALL).group()
    input_parameters = odict()
    for line in text.split('\n')[1:-1]:
        x = line.split('=')
        key = x[0].strip()
        if len(x[1].split()) == 1:
            input_parameters[key] = number(x[1])
        else:
            input_parameters[key] = [number(z) for z in x[1].split()]
    return input_parameters
    
    
    
    


#-------------------------------------------------
# Lattice


def parse_lattice(latticetext):
    lines = latticetext.split('\n')
    lines = [x.split() for x in lines if len(x.strip())>0] # Remove empty lines
    rdat = [map(float, x) for x in lines] # Cast to floats
    rdat = list(map(list, zip(*rdat))) # Transpose
    
    header = ['z', 'aw', 'qfld'] #was: s1.split() from below
    data = {}
    # Populate column data
    for i in range(len(header)):
        data[header[i]] = rdat[i]
        
        
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
        data[header[i]] = rdat[i]
    d['data'] = data

    return d
    
    
#-------------------------------------------------
# Full .out file
 
    
def parse_genesis_out(fname):
    with open(fname, 'r') as f:
        rawdat = f.read()  
    d = {}
    d['raw'] = rawdat # Save this just in case something was missed
    
    # magic strings to search for
    s1 = '    z[m]          aw            qfld '
    s2 = '********** output: slice'
    header, dat = rawdat.split(s1)
    sdat = dat.split(s2)
    latticetext = sdat[0]
    slices = sdat[1:]
    
    d['input_parameters'] = parse_inputfile(header)
    d['lattice'] = parse_lattice(latticetext)
    d['slice_data'] = [parse_slice(s) for s in slices]
        
    return d    
    
    
    
    
    
#-------------------------------------------------
#.dfl file
#    Dump file at the end
#    complex numbers, output in a loop over nslices, nx, ny
    
def parse_genesis_dfl(fname, nx):
    """
    fname: filename
    nx: grid size in x and y. Same as Genesis 'ncar'
    
    returns grid  [z, x, y]
    
    """
    dat = np.fromfile(fname, dtype=np.complex).astype(np.complex)
    npoints = dat.shape[0] 
    
    # Determine number of slices
    nz =  npoints / nx /nx
    assert (nz % 1 == 0) # 
    nz = int(nz)   
    dat = dat.reshape(nz, nx, nx)    
    
    
    return dat
    


#-------------------------------------------------
#.fld file
#    history file
#    output in a loop over histories, nslices, real/imaginary, nx, ny
    
def parse_genesis_fld(fname, nx, nz):
    """
    fname: filename
    ncar: grid size in x and y
    nx: grid size in x and y. Same as Genesis 'ncar'  
    nz: number of slices
    
    The number of histories can be computed from these. Returns array:
    
    [history, z, x, y]
    
    """
  
    # Real and imaginary parts are separated
    dat = np.fromfile(fname, dtype=np.float).astype(float)
    npoints = dat.shape[0]
    # Determine number of slices
    nhistories =  npoints / nz / 2 / nx / nx # 
    assert (nhistories % 1 == 0) # 
    nhistories = int(nhistories)   
    
    # real and imaginary parts are written separately. 
    dat = dat.reshape(nhistories, nz, 2,  nx, nx) # 
    dat =  np.moveaxis(dat, 2, 4) # Move complex indices to the end
    # Reform complex numbers:
    dat = 1j*dat[:,:,:,:,1] + dat[:,:,:,:,0]

    
    return dat        
    
    
    
    