

from collections import OrderedDict as odict
import re





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
        if len(x[1].split()) == 1:
            input_parameters[x[0]] = number(x[1])
        else:
            input_parameters[x[0]] = [number(z) for z in x[1].split()]
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
# Slices
 
    
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