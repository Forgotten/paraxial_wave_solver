"""Configuration dataclasses for the paraxial wave solver.

The equation
------------

Starting from the scalar Helmholtz equation for a monochromatic field E,

    lap(E) + k0**2 * n(x, y, z)**2 * E = 0,        k0 = 2 * pi / wavelength

factor out the carrier along the propagation axis, E = psi * exp(1j * k * z)
with k = k0 * n0. The result is still exact:

    d2(psi)/dz2 + 2*1j*k * d(psi)/dz + lap_perp(psi)
        + k0**2 * (n**2 - n0**2) * psi = 0

Two approximations give the default paraxial model. The paraxial
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

With `propagator='wide_angle'`, homogeneous diffraction uses the exact one-way
root 1j*(sqrt(k**2 + lap_perp) - k). Heterogeneous refraction remains a separate
local term. Negative eigenvalues of the root's argument give evanescent decay.

The experimental `stepper='pade', propagator='helmholtz'` instead approximates
exp(1j*h*(sqrt(H)-k)), H = L_perp + k0**2*(n0+delta_n)**2, with a real index
frozen at each midpoint. It requires float64, n2=0, and no absorber. This
full-index one-way model differs from both the paraxial model and sqrt(-L_perp).

Conventions
-----------

  * The solver works in the envelope psi, never the full field E.
  * Refractive index is supplied as the *perturbation* delta_n = n - n0, so
    vacuum is zero, not one.
  * The periodic paraxial equation conserves sum(|psi|**2) for real index
    without absorption. Discrete errors and filtering can change that norm;
    square-root models additionally permit evanescent losses.
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
    stepper: 'rk4', 'split_step', 'crank_nicolson', 'pade', or 'etdrk4'.
             ETDRK4 requires spectral paraxial diffraction and supports Kerr
             and complex index; local terms retain explicit stability limits.
             Split-step
             requires spectral discretization. The implicit methods support
             both discretizations and currently require n2=0. Padé requires
             propagator='helmholtz', float64, and no PML.
    splitting_order: Order of accuracy in z for the split-step composition.
                     2 is Strang splitting; 4 is a Yoshida composition of
                     three Strang steps, so it costs three times as much per
                     step. Accuracy gains require smooth media; an outer
                     PML split limits the overall order to two. Requires
                     stepper='split_step'.
    propagator: Diffraction operator. 'paraxial' is the usual
                exp(-1j * dz * k_perp**2 / (2 * k)) small-angle form.
                'wide_angle' uses the exact square-root operator
                exp(1j * dz * (sqrt(k**2 - k_perp**2) - k)), which stays
                accurate at large angles and lets evanescent components decay
                rather than mis-propagating them. In Fourier space the exact
                root is a diagonal multiplier, so no Pade approximation is
                needed and it costs the same as the paraxial form. Requires
                stepper='split_step'. 'helmholtz' puts the full index inside
                the operator root and requires stepper='pade'.
    dealias: Apply the 2/3 rule to the split-step propagator, zeroing the
             upper third of each transverse wavenumber axis. Matters once n2
             is non-zero, where the cubic term aliases energy back onto the
             grid. This is filtering, not exact cubic dealiasing. Requires
             stepper='split_step'.
    pade_order: Degree of the rotated rational whole-step approximation, 1–6.
    pade_rotation: Branch rotation in radians, between zero and pi. Accuracy
                   depends on this value and the discrete operator spectrum.
    linear_rtol: Required relative residual for implicit solves.
    linear_maxiter: Maximum GMRES restart cycles.
    linear_restart: Krylov dimension per cycle.
  """
  method: Literal['finite_difference', 'spectral']
  fd_order: int = 2
  compact: bool = False
  stepper: Literal['rk4', 'split_step', 'crank_nicolson', 'pade', 'etdrk4'] = 'rk4'
  splitting_order: int = 2
  propagator: Literal['paraxial', 'wide_angle', 'helmholtz'] = 'paraxial'
  dealias: bool = False
  pade_order: int = 4
  pade_rotation: float = math.pi / 6
  linear_rtol: float = 1e-6
  linear_maxiter: int = 50
  linear_restart: int = 20

  def __post_init__(self) -> None:
    if self.method not in ('finite_difference', 'spectral'):
      raise ValueError(
        f"Unsupported method {self.method!r}; expected 'finite_difference' "
        "or 'spectral'."
      )
    if self.stepper not in ('rk4', 'split_step', 'crank_nicolson', 'pade', 'etdrk4'):
      raise ValueError(
        f"Unsupported stepper {self.stepper!r}; expected 'rk4' or "
        "'split_step', 'crank_nicolson', 'pade' or 'etdrk4'."
      )
    if self.stepper == 'etdrk4' and self.method != 'spectral':
      raise ValueError("stepper='etdrk4' requires method='spectral'.")
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
    if self.propagator not in ('paraxial', 'wide_angle', 'helmholtz'):
      raise ValueError(
        f"Unsupported propagator {self.propagator!r}; expected 'paraxial' or "
        "'wide_angle' or 'helmholtz'."
      )
    if self.propagator == 'wide_angle' and self.splitting_order == 4:
      raise ValueError(
        "wide_angle requires splitting_order=2: negative Yoshida steps "
        "amplify evanescent modes."
      )
    if (self.stepper == 'pade') != (self.propagator == 'helmholtz'):
      raise ValueError("stepper='pade' requires propagator='helmholtz' and vice versa.")
    if not isinstance(self.pade_order, int) or not 1 <= self.pade_order <= 6:
      raise ValueError("pade_order must be an integer between one and six.")
    if not 0 < self.pade_rotation < math.pi:
      raise ValueError("pade_rotation must lie strictly between zero and pi.")
    if not 0 < self.linear_rtol < 1:
      raise ValueError("linear_rtol must lie strictly between zero and one.")
    for name in ('linear_maxiter', 'linear_restart'):
      if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
        raise ValueError(f"{name} must be a positive integer.")
    if self.stepper != 'split_step':
      # These three all act on the split-step propagator, which RK4 and the
      # finite difference operators do not build.
      for name, default in (('splitting_order', 2),
                            ('propagator', 'paraxial'),
                            ('dealias', False)):
        if name == 'propagator' and self.stepper == 'pade':
          continue
        if getattr(self, name) != default:
          raise ValueError(
            f"{name}={getattr(self, name)!r} requires stepper='split_step', "
            f"got stepper={self.stepper!r}."
          )
