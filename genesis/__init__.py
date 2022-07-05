import sys
assert sys.version_info >= (3, 7), 'Python 3.7 or greater required.'
from warnings import warn

from . import _version
__version__ = _version.get_versions()['version']

from .genesis2 import Genesis2

# from .genesis import Genesis
def Genesis(*args, **kwargs):
    warn("The Genesis class has been renamed to Genesis2.")
    return Genesis2(*args, **kwargs)

