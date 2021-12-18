"""Top-level package for gmpda."""

from .data_generator import DataGenerator
from .gmpda import GMPDA

__all__ = (
    "__version__",
    "__author__",
    "__email__",
    "DataGenerator",
    "GMPDA",
)
__author__ = "Olga Kaiser"
__email__ = "olga@nnaisense.com"

try:
    from .__version import __version__ as __version__
except ImportError:
    import sys

    print(
        "Please install the package to ensure correct behavior.\nFrom root folder:\n\tpip install -e .", file=sys.stderr
    )
    __version__ = "undefined"
