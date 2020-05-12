import numpy as np



def write_beam_file(fname, beam_columns, verbose=False):
    """
    Writes a beam file, using a dict beam_columns
    
    The header will be written as:
    ? VERSION=1.0
    ? SIZE=<length of the columns>
    ? COLUMNS <list of columns
    <data>
    
    See: genesis.parsers.parse_beam_file
    
    """
    
    # Get size
    names = list(beam_columns)
    size = len(beam_columns[names[0]])
    header=f"""? VERSION=1.0
? SIZE={size}
? COLUMNS {' '.join([n.upper() for n in names])}"""
    
    dat = np.array([beam_columns[name] for name in names]).T

    np.savetxt(fname, dat, header=header, comments='', fmt='%1.8e') # Genesis can't read format %1.16e - lines are too long?
    
    if verbose:
        print('Beam written:', fname)
    
    return header