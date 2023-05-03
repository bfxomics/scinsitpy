from importlib.metadata import version

from . import io, pl, pp, tl

__all__ = ["io", "pl", "pp", "tl"]

__version__ = version("scinsitpy-0.0.2")
