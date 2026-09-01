"""Public API of the paraxial wave solver."""

import jax

from .config import Field, PMLConfig, SimulationConfig, SolverConfig
from .pml import PMLData, StretchFields, generate_pml_profile
from .solvers import ParaxialWaveSolver, propagate
from .utils import (
  gaussian_beam,
  get_analytical_beam,
  get_hermite_gaussian_analytical,
  get_laguerre_gaussian_analytical,
  hermite_gaussian_beam,
  laguerre_gaussian_beam,
  random_medium,
  random_medium_spectral,
)


def enable_x64(enable: bool = True) -> None:
  """Switches JAX to 64-bit floating point.

  JAX defaults to float32, which caps the achievable accuracy of a
  propagation. Call this before creating any arrays - configuration changes do
  not affect arrays that already exist.

  Args:
    enable: True for float64, False to return to float32.
  """
  jax.config.update("jax_enable_x64", enable)


__all__ = [
  "SimulationConfig",
  "SolverConfig",
  "PMLConfig",
  "PMLData",
  "StretchFields",
  "Field",
  "ParaxialWaveSolver",
  "propagate",
  "generate_pml_profile",
  "enable_x64",
  "gaussian_beam",
  "laguerre_gaussian_beam",
  "hermite_gaussian_beam",
  "get_analytical_beam",
  "get_laguerre_gaussian_analytical",
  "get_hermite_gaussian_analytical",
  "random_medium",
  "random_medium_spectral",
]
