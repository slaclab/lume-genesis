# ------------------------------
#
# Routines to work with Genesis lattices.
# C
#



#-------------------------------------------------
# Helper functions

def ele_types(eles):
    """
    Returns a list of unique types in eles
    """
    return list(set([e['type'] for e in eles] ))

def eles_by_type(eles):
    """
    Separated eles by type, returns a dict of:
    <type>:[<eles>]
    """
    tlist = ele_types(eles) 
    tlat = {}
     # initialize
    for t in tlist:
        tlat[t] = []
    for e in eles:
        t = e['type']
        tlat[t].append(e)
    
    return tlat


def s0(ele):
    # Returns beginning s-position of an element
    return ele['s'] - ele['L']


def zsort(eles):
    """
    Sorts elements by 's'
    """
    return sorted(eles, key = lambda e: e['s'])

#-------------------------------------------------
# Standard lattice

def standard_eles_from_eles(eles, remove_zero_strengths=True):
    """
    Converts raw ele dicts to an ordered list of elements, with absolute positions s
    s is at the end of the element
    Comments are dropped. 
    """
    lat = []
    z0 = {}
    for t in ele_types(eles):
        z0[t] = 0
    for ele in eles:
        e = ele.copy()
        
        t = e['type']
        if t == 'comment': 
            continue
   
        zbeg = z0[t] + e['d']
        zend = e['L']+zbeg
        z0[t] = zend
        e['s'] = zend
        e.pop('d') # Remove d
        
        if remove_zero_strengths and e['strength'] == 0.0:
            continue 
            
        lat.append(e) 
    return zsort(lat )
    


#-------------------------------------------------
# Utilities  
def create_names(eles):
    """
    Invents names for elements
    """
    counter = {}
    for t in ele_types(eles):
        counter[t] = 0
    for ele in eles:
        t = ele['type']
        counter[t] = counter[t]+1
        ele['name'] = ele['type']+'_'+str(counter[t])


def make_dummies_for_single_type(eles, smax):
    """
    Finds the gaps in a lattice and makes dummy (zero-strength) elements
    """   
    
    types = ele_types(eles)
    assert len(types) == 1, 'Only one type of element allowed'
    my_type = types[0]
    
    lat = zsort(eles)
    
    ref = lat[0] # End of previous ele
    dummies = []
    for i in range(1, len(lat)):
        # next ele
        ele = lat[i]
        # Distance from this ele to the previous ele end 
        zbeg = ref['s']
        zend = s0(ele)
        L = zend - zbeg
        assert L >= 0, 'Overlapping eles!'# + ref['name']+' overlaps '+ele['name']+' by '+str(L)
                
        dummy = {'type': my_type, 'strength':0, 'L':L, 's':zend}
        dummies.append(dummy)
        
        # next z
        ref = ele

    if ele['s'] < smax:
        # Make final dummy
        L = smax - ele['s']
        dummy = {'type': my_type, 'strength':0, 'L':L, 's':smax}
        dummies.append(dummy)
        
    return dummies

def lattice_dummies(eles):
    """
    Makes dummy elements to fill in gaps
    """
    # Separate by types
    tlat = eles_by_type(eles)
    smax  = max([e['s'] for e in eles
                if e['type'] not in ['comment']])
    #print(smax)
    dummies = []
    for t in tlat:
        eles2 = tlat[t]
        dummies.extend(make_dummies_for_single_type(eles2, smax))
    return dummies
        

#-------------------------------------------------
# Export genesis lattice
    
def genesis_lattice_from_standard_lattice(standard_lattice,include_name=False, include_comment=False):
    """
    Forms lines of a Genesis lattice file from a standard lattice
    
    Pads all types with zero strength dummy elements
    
    """
    
    unitlength = standard_lattice['param']['unitlength']
    version = standard_lattice['param']['version']
    
    # Make copy
    eles = [e.copy() for e in standard_lattice['eles']]
    
    tlist = ele_types(eles)
    # Add dummies    
    eles = eles + lattice_dummies(eles)
    
    # Sort 
    
    eles = zsort(eles)
    # Separate lattice by types
    glat = {} # lattice
    z = {}    # z at end of each type
    # initialize
    for t in tlist:
        glat[t] = []
        z[t] = 0
    
    for ele in eles:
        t = ele['type']
        if t in ['comment', 'drift']: # Skip these
            continue
        d = s0(ele) - z[t] # distance from previous element of the same type
        z[t] = ele['s']
        line = str(ele['type']) + ' ' + str(ele['strength']) + ' ' + str(ele['L']) + ' ' + str(d) 
        if include_name and 'name' in ele:
            line += ' #'+ele['name']
        if include_comment and 'comment' in ele:
            line += ' # '+ele['comment']
        glat[t].append(line)
        
    # header
    outlines = ['? VERSION = '+ str(version), '? UNITLENGTH = '+str(unitlength) +' # meters']
    for t in tlist:
        if t in ['comment', 'drift']: # Skip these
            continue
        outlines.append('')
        outlines.append('#------------')
        outlines.append('# '+ t)
        for line in glat[t]:
            outlines.append(line)
        
        
    return outlines


def write_lattice(filePath, standard_lattice):
    lines = genesis_lattice_from_standard_lattice(standard_lattice)
    with open(filePath, 'w') as f:
        for l in lines:
            f.write(l+'\n')


#-------------------------------------------------
# Print

def print_ele(e):
    line = ''
    if e['type']=='comment':
        c = e['comment']
        if c == '!':
            print('')
        else:
            #pass
            print(c)
        return
    
    if 'name' in e:
        name = e
    else:
        name = ''
    
    line = name+': '+e['type']
    l = len(line)
    for key in e:
        if key in ['s', 'name', 'type', 'original']: 
            continue
        val = str(e[key])
        s =  key+'='+val
        l += len(s)
        if l > 100:
            append = ',\n      '+s
            l = len(append)
        else:
            append = ', '+s
        line = line + append
    print(line)

def join_eles(eles1, eles2):


    zlist = [e['s'] for e in eles1]
    zmax = max(zlist)
    for ele in eles2:
        ele['s'] += zmax
    merged = eles1 + eles2

    return merged

    
    
        
