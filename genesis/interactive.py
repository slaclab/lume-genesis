#
# Interactive plotting using bokeh
#
from genesis import parsers

from bokeh import palettes, colors
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models.widgets import Slider
from bokeh.layouts import column

import numpy as np
import os


pal = palettes.Viridis[256]

def interactive_field_history(doc, fld=None, slice=0, dgrid=0):
    """         
    
    Use with code similar to:
    
    my_fld =  parsers.parse_genesis_fld(fld_fname, g.input_params['ncar'], nslice)
    
    from bokeh.plotting import show, output_notebook
    output_notebook()
    
    def app(doc):
       return interactive_field_history(doc, fld=my_fld, dgrid=dgrid, slice = 0)
    show(app)
    
    
    """
    
    
    nhist = len(fld)
    
    ihist = nhist-1
    
    fdat = fld[ihist][slice]
    
    d = np.angle(fdat)
    ds = ColumnDataSource(data=dict(image=[d]))
    
    xyrange = (-1000*dgrid, 1000*dgrid)
    p = figure(x_range=xyrange, y_range=xyrange, title='Phase',  
           plot_width=500, plot_height=500, x_axis_label='x (mm)', y_axis_label='y (mm)')
    p.image(image='image', source=ds, 
            x=-dgrid*1000, y=-dgrid*1000, dw=2*dgrid*1000, dh=2*dgrid*1000, palette=pal)
    
    slider = Slider(start=0, end=nhist-1, value=ihist, step=1, title='History')

    def handler(attr, old, new):
        fdat = fld[new][slice]
        d = np.angle(fdat)
        ds.data = ColumnDataSource(data=dict(image=[d])).data
    
    slider.on_change('value', handler)

    doc.add_root(column(slider, p))
    
def genesis_interactive_field_history(doc, genesis=None):
    """
    Convenience routine to pass the whole genesis object to
    """
    
    # Parameters
    p = genesis.input
    
    # Check for time dependence
    if p['itdp'] == 0:
        nslice = 1
    else:
        nslice = p['nslice']
    
    fld_fname = os.path.join(genesis.path, p['outputfile']+'.fld')
    my_fld =  parsers.parse_genesis_fld(fld_fname, p['ncar'], nslice)
    
    return interactive_field_history(doc, fld=my_fld, slice=0, dgrid=p['dgrid'] )  
    