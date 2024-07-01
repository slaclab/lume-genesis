from warnings import warn

from .genesis2 import Genesis2
from .tools import global_display_options
from .version4 import Genesis4

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "Genesis2",
    "Genesis4",
    "global_display_options",
]


def Genesis(*args, **kwargs):
    warn("The Genesis class has been renamed to Genesis2.")
    return Genesis2(*args, **kwargs)
