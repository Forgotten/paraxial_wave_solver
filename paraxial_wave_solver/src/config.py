"""Configuration dataclasses for the paraxial wave solver.

Conventions used throughout the package:

  * The physical field is E(x, y, z) = psi(x, y, z) * exp(1j * k * z), with
    k = 2 * pi * n0 / wavelength. The solver propagates the slowly varying
    envelope psi, never the full field.
  * Refractive index is supplied as the *perturbation* delta_n = n - n0, not
    as n itself. Vacuum is delta_n = 0.
  * The envelope obeys
        d(psi)/dz = (1j / (2 * k0 * n0)) * lap_perp(psi)
                    + 1j * k0 * delta_n * psi
                    - sigma * psi
    where sigma is the PML absorption profile.
"""

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax

# Custom type alias for 2D/3D fields.
Field: TypeAlias = jax.Array

FD_ORDERS = (2, 4, 6)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
  """Configuration for the simulation grid and domain.

  Attributes:
    nx: Number of grid points in the x-direction.
    ny: Number of grid points in the y-direction.
    dx: Grid spacing in the x-direction (physical units).
    dy: Grid spacing in the y-direction (physical units).
    dz: Step size for propagation in the z-direction (physical units).
    nz: Number of steps to propagate in the z-direction.
    wavelength: Wavelength of the optical field in vacuum.
    n0: Background refractive index (vacuum/atmosphere = 1.0, water = 1.33).
  """
  nx: int
  ny: int
  dx: float
  dy: float
  dz: float
  nz: int
  wavelength: float
  n0: float = 1.0

  def __post_init__(self) -> None:
    for name in ('nx', 'ny', 'nz'):
      value = getattr(self, name)
      if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    for name in ('dx', 'dy', 'dz', 'wavelength', 'n0'):
      value = getattr(self, name)
      if not value > 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")

  @property
  def k0(self) -> float:
    """Vacuum wavenumber, 2 * pi / wavelength."""
    return 2 * math.pi / self.wavelength

  @property
  def k(self) -> float:
    """Wavenumber in the background medium, 2 * pi * n0 / wavelength."""
    return 2 * math.pi * self.n0 / self.wavelength

  @property
  def lx(self) -> float:
    """Total physical length of the domain in the x-direction."""
    return self.nx * self.dx

  @property
  def ly(self) -> float:
    """Total physical length of the domain in the y-direction."""
    return self.ny * self.dy

  @property
  def lz(self) -> float:
    """Total physical propagation distance in the z-direction."""
    return self.nz * self.dz


@dataclass(frozen=True, slots=True)
class PMLConfig:
  """Configuration for Perfectly Matched Layers (PML).

  Attributes:
    width_x: Number of grid points for the PML region at the x-boundaries
             (both sides). Zero disables the PML in x.
    width_y: Number of grid points for the PML region at the y-boundaries
             (both sides). Zero disables the PML in y.
    strength: Maximum absorption strength of the PML profile.
    order: Polynomial order of the PML absorption profile (typically 2).
    profile_type: Type of PML profile (currently only 'polynomial').
    use_complex_stretching: If True, absorption is applied through complex
                            coordinate stretching inside the Laplacian rather
                            than as an absorbing potential. Finite-difference
                            methods only; the spectral method always uses the
                            absorbing potential.
  """
  width_x: int
  width_y: int
  strength: float = 1.0
  order: int = 2
  profile_type: Literal['polynomial'] = 'polynomial'
  use_complex_stretching: bool = False

  def __post_init__(self) -> None:
    for name in ('width_x', 'width_y'):
      value = getattr(self, name)
      if not isinstance(value, int) or value < 0:
        raise ValueError(
          f"{name} must be a non-negative integer, got {value!r}."
        )
    if self.strength < 0:
      raise ValueError(f"strength must be non-negative, got {self.strength!r}.")
    if not isinstance(self.order, int) or self.order < 0:
      raise ValueError(
        f"order must be a non-negative integer, got {self.order!r}."
      )
    if self.profile_type != 'polynomial':
      raise ValueError(
        f"Unsupported profile_type {self.profile_type!r}; "
        "only 'polynomial' is supported."
      )


@dataclass(frozen=True, slots=True)
class SolverConfig:
  """Configuration for the numerical solver method.

  Attributes:
    method: Spatial discretization ('finite_difference' or 'spectral').
    fd_order: Order of accuracy for the finite difference method (2, 4 or 6).
              Ignored when method is 'spectral' or when compact is True.
    compact: Use the isotropic compact 9-point stencil. Requires
             method='finite_difference' and dx == dy.
    stepper: Z-propagation scheme ('rk4' or 'split_step'). 'split_step'
             requires method='spectral'.
  """
  method: Literal['finite_difference', 'spectral']
  fd_order: int = 2
  compact: bool = False
  stepper: Literal['rk4', 'split_step'] = 'rk4'

  def __post_init__(self) -> None:
    if self.method not in ('finite_difference', 'spectral'):
      raise ValueError(
        f"Unsupported method {self.method!r}; expected 'finite_difference' "
        "or 'spectral'."
      )
    if self.stepper not in ('rk4', 'split_step'):
      raise ValueError(
        f"Unsupported stepper {self.stepper!r}; expected 'rk4' or "
        "'split_step'."
      )
    if self.stepper == 'split_step' and self.method != 'spectral':
      raise ValueError(
        "stepper='split_step' requires method='spectral', got "
        f"method={self.method!r}. Use stepper='rk4' for finite differences."
      )
    if self.compact and self.method != 'finite_difference':
      raise ValueError(
        "compact=True requires method='finite_difference', got "
        f"method={self.method!r}."
      )
    if (self.method == 'finite_difference' and not self.compact
        and self.fd_order not in FD_ORDERS):
      raise ValueError(
        f"Unsupported fd_order {self.fd_order!r}; expected one of "
        f"{FD_ORDERS}."
      )
