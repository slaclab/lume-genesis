from warnings import warn
from .genesis2 import Genesis2
from .version4.genesis4 import Genesis4

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    'Genesis2',
    'Genesis4',
]

def Genesis(*args, **kwargs):
    warn("The Genesis class has been renamed to Genesis2.")
    return Genesis2(*args, **kwargs)
