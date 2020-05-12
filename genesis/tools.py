""" 


 
"""
import numpy as np
from hashlib import blake2b
from numbers import Number
import subprocess
import datetime
import json
import os

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
        
# Alternative execute
def execute2(cmd, timeout=None):
    """
    Execute with time limit (timeout) in seconds, catching run errors. 
    """
    
    output = {'error':True, 'log':''}
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout = timeout)
        output['log'] = p.stdout
        output['error'] = False
        output['why_error'] =''
    except subprocess.TimeoutExpired as ex:
        output['log'] = ex.stdout+'\n'+str(ex)
        output['why_error'] = 'timeout'
    except:
        output['log'] = 'unknown run error'
        output['why_error'] = 'unknown'
    return output


def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))



def make_executable(path):
    """
    https://stackoverflow.com/questions/12791997/how-do-you-do-a-simple-chmod-x-from-within-python
    """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)

class NpEncoder(json.JSONEncoder):
    """
    See: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def fingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data. 
    Used JSON dumps to form strings, and the blake2b algorithm to hash.
    
    """
    h = blake2b(digest_size=16)
    for key in sorted(keyed_data.keys()):
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
    return h.hexdigest()  


def namelist_lines(namelist_dict, start='&name', end='/'):
    """
    Converts namelist dict to output lines, for writing to file
    """
    lines = []
    lines.append(start)
    # parse
    for key, value in namelist_dict.items():
        #if type(value) == type(1) or type(value) == type(1.): # numbers
        if isinstance(value, Number): # numbers
            line= key + ' = ' + str(value) 
        elif type(value) == type([]): # lists
            liststr = ''
            for item in value:
                liststr += str(item) + ' '
            line = key + ' = ' + liststr 
        elif type(value) == type('a'): # strings
            line = key + ' = ' + "'" + value.strip("''") + "'"  # input may need apostrophes
        else:
            # Skip
            #print('skipped: key, value = ', key, value)
            continue
        lines.append(line)
    
    lines.append(end)
    return lines



def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)



def native_type(value):
    """
    Converts a numpy type to a native python type.
    See:
    https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998
    """
    return getattr(value, 'tolist', lambda: value)()    


"""UTC to ISO 8601 with Local TimeZone information without microsecond"""
def isotime():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat()    
    
