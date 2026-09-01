"""Configuration dataclasses for the paraxial wave solver.

The equation
------------

Starting from the scalar Helmholtz equation for a monochromatic field E,

    lap(E) + k0**2 * n(x, y, z)**2 * E = 0,        k0 = 2 * pi / wavelength

factor out the carrier along the propagation axis, E = psi * exp(1j * k * z)
with k = k0 * n0. The result is still exact:

    d2(psi)/dz2 + 2*1j*k * d(psi)/dz + lap_perp(psi)
        + k0**2 * (n**2 - n0**2) * psi = 0

Two approximations reduce it to what this package integrates. The paraxial
approximation drops d2(psi)/dz2, which turns the problem into an initial-value
march in z and discards the backward-travelling wave. Weak index contrast,
n = n0 + delta_n with delta_n << n0, linearizes n**2 - n0**2 to 2*n0*delta_n.
Solving for the z-derivative and adding the optional terms:

    d(psi)/dz = (1j / (2 * k0 * n0)) * L_perp(psi)
                + 1j * k0 * (delta_n(x, y, z) + n2 * |psi|**2) * psi
                - sigma(x, y) * psi

    L_perp   transverse Laplacian, discretized per `SolverConfig.method`
    delta_n  index perturbation n - n0; complex values give absorption or gain
    n2       Kerr coefficient, zero for the linear problem
    sigma    PML absorption profile, zero in the interior

Under complex coordinate stretching L_perp becomes
(1/s_x) d/dx((1/s_x) d/dx) + (1/s_y) d/dy((1/s_y) d/dy) with s = 1 + 1j*sigma,
and the sigma term above is dropped, since the absorption then lives in the
operator instead.

With `propagator='wide_angle'` the paraxial approximation is not made: the
diffraction operator is the exact one-way root 1j*(sqrt(k**2 + lap_perp) - k),
of which the paraxial form is the leading term.

Conventions
-----------

  * The solver works in the envelope psi, never the full field E.
  * Refractive index is supplied as the *perturbation* delta_n = n - n0, so
    vacuum is zero, not one.
  * Without a PML and with real delta_n the equation conserves sum(|psi|**2).
"""

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax

# Custom type alias for 2D/3D fields.
Field: TypeAlias = jax.Array

FD_ORDERS = (2, 4, 6)
SPLITTING_ORDERS = (2, 4)


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
    n2: Kerr coefficient. Non-zero adds an intensity-dependent index
        n2 * |psi|**2 to the medium, turning the propagation into a nonlinear
        Schroedinger equation. Zero (the default) is the linear problem.
  """
  nx: int
  ny: int
  dx: float
  dy: float
  dz: float
  nz: int
  wavelength: float
  n0: float = 1.0
  n2: float = 0.0

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
    splitting_order: Order of accuracy in z for the split-step composition.
                     2 is Strang splitting; 4 is a Yoshida composition of
                     three Strang steps, so it costs three times as much per
                     step but usually permits a far larger dz. Requires
                     stepper='split_step'.
    propagator: Diffraction operator. 'paraxial' is the usual
                exp(-1j * dz * k_perp**2 / (2 * k)) small-angle form.
                'wide_angle' uses the exact square-root operator
                exp(1j * dz * (sqrt(k**2 - k_perp**2) - k)), which stays
                accurate at large angles and lets evanescent components decay
                rather than mis-propagating them. In Fourier space the exact
                root is a diagonal multiplier, so no Pade approximation is
                needed and it costs the same as the paraxial form. Requires
                stepper='split_step'.
    dealias: Apply the 2/3 rule to the split-step propagator, zeroing the
             upper third of each transverse wavenumber axis. Matters once n2
             is non-zero, where the cubic term aliases energy back onto the
             grid. Requires stepper='split_step'.
  """
  method: Literal['finite_difference', 'spectral']
  fd_order: int = 2
  compact: bool = False
  stepper: Literal['rk4', 'split_step'] = 'rk4'
  splitting_order: int = 2
  propagator: Literal['paraxial', 'wide_angle'] = 'paraxial'
  dealias: bool = False

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
    if self.splitting_order not in SPLITTING_ORDERS:
      raise ValueError(
        f"Unsupported splitting_order {self.splitting_order!r}; expected one "
        f"of {SPLITTING_ORDERS}."
      )
    if self.propagator not in ('paraxial', 'wide_angle'):
      raise ValueError(
        f"Unsupported propagator {self.propagator!r}; expected 'paraxial' or "
        "'wide_angle'."
      )
    if self.stepper != 'split_step':
      # These three all act on the split-step propagator, which RK4 and the
      # finite difference operators do not build.
      for name, default in (('splitting_order', 2),
                            ('propagator', 'paraxial'),
                            ('dealias', False)):
        if getattr(self, name) != default:
          raise ValueError(
            f"{name}={getattr(self, name)!r} requires stepper='split_step', "
            f"got stepper={self.stepper!r}."
          )
