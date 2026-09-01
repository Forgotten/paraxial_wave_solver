"""A JAX-based solver for the paraxial wave equation in 3D.

The public API is defined in `paraxial_wave_solver.src` and re-exported here,
so that `import paraxial_wave_solver as pws` gives access to everything.
"""

from .src import *  # noqa: F401,F403
from .src import __all__ as _src_all

__all__ = list(_src_all)
