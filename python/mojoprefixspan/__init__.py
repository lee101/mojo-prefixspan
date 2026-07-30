"""Sequential pattern mining with PrefixSpan's Python API and a Mojo core."""

from ._lib import build
from .prefixspan import PrefixSpan

__version__ = "0.1.0"
__all__ = ["PrefixSpan", "build"]
