"""Perfectly Matched Layer (PML) profile generation."""

from dataclasses import dataclass

import jax.numpy as jnp

from .config import Field, PMLConfig, SimulationConfig


@dataclass(frozen=True, slots=True)
class StretchFields:
  """Complex coordinate-stretching factors and their derivatives.

  The x-fields have shape (nx, 1) and the y-fields (1, ny); they broadcast
  against an (nx, ny) field without being materialized.

  Attributes:
    sx: Stretch factor s_x(x) = 1 + 1j * sigma_x.
    sy: Stretch factor s_y(y) = 1 + 1j * sigma_y.
    sx_prime: d(s_x)/dx.
    sy_prime: d(s_y)/dy.
  """
  sx: Field
  sy: Field
  sx_prime: Field
  sy_prime: Field

  def as_dict(self) -> dict[str, Field]:
    """Returns the fields in the form expected by the Laplacian operators."""
    return {
      'sx': self.sx,
      'sy': self.sy,
      'sx_prime': self.sx_prime,
      'sy_prime': self.sy_prime,
    }


@dataclass(frozen=True, slots=True)
class PMLData:
  """Precomputed PML data.

  Attributes:
    sigma: Absorbing potential sigma(x, y) of shape (nx, ny), applied as a
           -sigma * psi damping term. Always the real profile, so that methods
           which cannot use coordinate stretching (the spectral solver) can
           fall back to it.
    stretch: Complex coordinate-stretching fields, or None when stretching was
             not requested.
  """
  sigma: Field
  stretch: StretchFields | None = None


def _sigma_1d(
  coord: Field,
  start: float,
  end: float,
  width_idx: int,
  d_step: float,
  strength: float,
  order: int,
) -> tuple[Field, Field]:
  """Builds a 1D polynomial PML profile and its derivative.

  The profile is zero in the physical domain and grows as
  strength * (d / l_pml)**order inside each boundary layer, where d is the
  distance into the layer.

  Args:
    coord: 1D array of physical coordinates.
    start: Coordinate where the interior region begins.
    end: Coordinate where the interior region ends.
    width_idx: PML width in grid points; zero disables the layer.
    d_step: Grid spacing along this axis.
    strength: Peak absorption strength.
    order: Polynomial order of the profile.

  Returns:
    A tuple (sigma, sigma_prime) of 1D arrays matching coord's shape.
  """
  if width_idx == 0 or strength == 0.0:
    zeros = jnp.zeros_like(coord)
    return zeros, zeros

  # Distance into the PML, measured from the interior boundary.
  d = jnp.maximum(jnp.maximum(0.0, start - coord), jnp.maximum(0.0, coord - end))
  l_pml = width_idx * d_step
  d_norm = d / l_pml

  sigma = strength * d_norm**order

  # d(d)/dx is -1 on the left layer, +1 on the right, 0 in the interior.
  grad_d = jnp.where(coord < start, -1.0, jnp.where(coord > end, 1.0, 0.0))

  if order == 0:
    sigma_prime = jnp.zeros_like(coord)
  elif order == 1:
    sigma_prime = strength * grad_d / l_pml
  else:
    # Guard the power so that d_norm == 0 gives exactly zero rather than 0**0.
    d_norm_pow = jnp.where(d_norm > 0, d_norm ** (order - 1), 0.0)
    sigma_prime = strength * order * d_norm_pow * grad_d / l_pml

  return sigma, sigma_prime


def generate_pml_profile(
  sim_config: SimulationConfig,
  pml_config: PMLConfig,
) -> PMLData:
  """Generates the Perfectly Matched Layer absorption data.

  The profile is zero in the physical domain and increases polynomially in the
  PML regions at the boundaries.

  Args:
    sim_config: Simulation configuration containing grid details.
    pml_config: PML configuration containing width, strength and order.

  Returns:
    A PMLData holding the absorbing potential and, when
    `use_complex_stretching` is set, the coordinate-stretching fields as well.
    The caller decides which to apply: a solver using the stretched operator
    must not also apply the potential.
  """
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy

  x_start = pml_config.width_x * sim_config.dx
  x_end = sim_config.lx - pml_config.width_x * sim_config.dx
  y_start = pml_config.width_y * sim_config.dy
  y_end = sim_config.ly - pml_config.width_y * sim_config.dy

  sigma_x, sigma_x_prime = _sigma_1d(
    x, x_start, x_end, pml_config.width_x, sim_config.dx,
    pml_config.strength, pml_config.order,
  )
  sigma_y, sigma_y_prime = _sigma_1d(
    y, y_start, y_end, pml_config.width_y, sim_config.dy,
    pml_config.strength, pml_config.order,
  )

  # The separable 1D profiles add to give the (nx, ny) absorbing potential.
  sigma_grid = sigma_x[:, None] + sigma_y[None, :]

  if not pml_config.use_complex_stretching:
    return PMLData(sigma=sigma_grid)

  # Complex stretch factors s = 1 + 1j * sigma, so s' = 1j * sigma'. Kept in
  # broadcast shape: they are only ever used elementwise against (nx, ny).
  stretch = StretchFields(
    sx=(1.0 + 1j * sigma_x)[:, None],
    sy=(1.0 + 1j * sigma_y)[None, :],
    sx_prime=(1j * sigma_x_prime)[:, None],
    sy_prime=(1j * sigma_y_prime)[None, :],
  )
  return PMLData(sigma=sigma_grid, stretch=stretch)
