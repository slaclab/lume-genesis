# ------------------------------
#
# Routines to work with Genesis lattices.
# C
#



# Helper functions

def ele_types(eles):
    """
    Returns a list of unique types in eles
    """
    return list(set([e['type'] for e in eles] ))

def s0(ele):
    # Returns beginning s-position of an element
    return ele['s'] - ele['L']


def zsort(eles):
    """
    Sorts elements by 's'
    """
    return sorted(eles, key = lambda e: e['s'])


def standard_lattice_from_eles(eles):
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
        lat.append(e) 
    return zsort(lat )
    
    
def create_names(lat):
    """
    Invents names for elements
    """
    counter = {}
    for t in ele_types(lat):
        counter[t] = 0
    for ele in lat:
        t = ele['type']
        counter[t] = counter[t]+1
        ele['name'] = ele['type']+'_'+str(counter[t])


def make_drifts(lat):
    """
    Finds the gaps in a lattice and makes drifts
    
    returns a list of 'drift' elements
    """   
    ref = lat[0] # End of previous ele
    drifts = []
    for i in range(1, len(lat)):
        # next ele
        ele = lat[i]
        # Distance from this ele to the previous ele end 
        zbeg = ref['s']
        zend = s0(ele)
        L = zend - zbeg
        if L < 0 :
            print('Warning , overlapping eles!', ref['name'], 'overlaps', ele['name'], 'by ', L)
                
        drift = {'type': 'drift', 'strength':0, 'L':L, 's':zend}
        drifts.append(drift)
        
        # next z
        ref = ele

    return [d for d in drifts if d['L'] != 0]
    
    
def genesis_lattice_from_standard_lattice(lat, unitlength = 1, version = '1.0', include_name=False, include_comment=False):
    """
    Forms lines of a Genesis lattice file from a standard lattice
    
    """
    tlist = ele_types(lat)
    lat = zsort(lat)
    # Separate lattice by types
    glat = {} # lattice
    z = {}    # z at end of each type
    # initialize
    for t in tlist:
        glat[t] = []
        z[t] = 0
    
    for ele in lat:
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



    
    
        