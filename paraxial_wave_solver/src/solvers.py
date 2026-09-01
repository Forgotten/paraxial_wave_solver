"""Propagation kernels and the paraxial wave solver.

The envelope psi obeys

    d(psi)/dz = (1j / (2 * k0 * n0)) * lap_perp(psi)
                + 1j * k0 * delta_n(x, y, z) * psi
                - sigma(x, y) * psi

where delta_n = n - n0 is the refractive index *perturbation* and sigma is the
PML absorption profile. See `config.py` for the full set of conventions.

Everything that does not vary with z - the diffraction operator, the PML
attenuation, the wavenumber grids - is built once in `ParaxialWaveSolver`
and handed to the kernels as arrays, so the scan body contains only the work
that genuinely changes from step to step.
"""

import functools
import math
import warnings
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .config import (
  SPLITTING_ORDERS,
  Field,
  PMLConfig,
  SimulationConfig,
  SolverConfig,
)
from .operators import (
  get_spectral_k_grids,
  laplacian_fd,
  laplacian_fd_9point,
  laplacian_spectral,
)
from .pml import generate_pml_profile

# A refractive index perturbation callable: (z, medium) -> delta_n. The return
# value must broadcast against an (nx, ny) field; a scalar is fine for a
# homogeneous medium.
DeltaNFn = Callable[[Any, Any], Any]

# Operator arrays handed to the kernels. A dict, so it is a pytree and its
# leaves are traced as ordinary arguments rather than baked in as constants.
Operators = dict[str, Any]


# Peak magnitude of the 1D discrete second-derivative symbol, in units of
# 1/h**2, reached at the grid Nyquist wavenumber.
_STENCIL_SPECTRAL_RADIUS = {2: 4.0, 4: 16.0 / 3.0, 6: 272.0 / 45.0}

# Extent of the RK4 stability region along the imaginary axis.
_RK4_IMAGINARY_STABILITY_LIMIT = 2.0 * math.sqrt(2.0)


def _laplacian_spectral_radius(
  solver_config: SolverConfig,
  sim_config: SimulationConfig,
) -> float:
  """Returns the largest |eigenvalue| of the discrete transverse Laplacian."""
  inv_dx2 = 1.0 / sim_config.dx**2
  inv_dy2 = 1.0 / sim_config.dy**2

  if solver_config.method == 'spectral':
    return math.pi**2 * (inv_dx2 + inv_dy2)
  if solver_config.compact:
    # Dxx + Dyy + (h^2 / 6) Dxx Dyy at the corner of the Brillouin zone.
    return 8.0 * inv_dx2 - (sim_config.dx**2 / 6.0) * (4.0 * inv_dx2)**2
  radius = _STENCIL_SPECTRAL_RADIUS[solver_config.fd_order]
  return radius * (inv_dx2 + inv_dy2)


def _warn_if_unstable(
  solver_config: SolverConfig,
  sim_config: SimulationConfig,
) -> None:
  """Warns when the RK4 step size exceeds the linear stability limit.

  The paraxial diffraction operator is purely imaginary, so RK4 is stable only
  while |lambda| * dz stays inside the imaginary-axis stability interval.
  Exceeding it makes the solution grow without bound, which otherwise shows up
  only as inf or nan in the returned field.
  """
  if solver_config.stepper != 'rk4':
    return
  radius = _laplacian_spectral_radius(solver_config, sim_config)
  growth = radius * sim_config.dz / (2 * sim_config.k0 * sim_config.n0)
  if growth > _RK4_IMAGINARY_STABILITY_LIMIT:
    dz_max = (
      _RK4_IMAGINARY_STABILITY_LIMIT * 2 * sim_config.k0 * sim_config.n0
      / radius
    )
    warnings.warn(
      f"RK4 step size is above the stability limit: |lambda| * dz = "
      f"{growth:.3g} exceeds {_RK4_IMAGINARY_STABILITY_LIMIT:.3g}. The "
      f"propagation will diverge. Reduce dz below {dz_max:.3g}, coarsen the "
      "transverse grid, or use method='spectral' with stepper='split_step', "
      "which has no step size restriction.",
      RuntimeWarning,
      stacklevel=3,
    )


def _make_laplacian_fn(
  solver_config: SolverConfig,
  sim_config: SimulationConfig,
  use_stretch: bool,
) -> Callable[[Field, Operators], Field]:
  """Builds the transverse Laplacian for the configured method.

  Grid spacings are captured as Python floats rather than passed as traced
  arguments, so stencils may branch on them at build time and so the constants
  they form are folded at trace time.

  Args:
    solver_config: Solver configuration.
    sim_config: Simulation configuration.
    use_stretch: Whether complex coordinate stretching is active.

  Returns:
    A callable (field, operators) -> Laplacian.

  Raises:
    ValueError: If the compact stencil is requested with dx != dy.
    NotImplementedError: If stretching is requested for the compact stencil.
  """
  dx, dy = sim_config.dx, sim_config.dy

  if solver_config.method == 'spectral':
    def laplacian(field: Field, operators: Operators) -> Field:
      return laplacian_spectral(field, operators['kx'], operators['ky'])
    return laplacian

  if solver_config.compact:
    if use_stretch:
      raise NotImplementedError(
        "Complex coordinate stretching is not implemented for the compact "
        "9-point stencil. Set use_complex_stretching=False, or compact=False."
      )
    if dx != dy:
      raise ValueError(
        f"compact=True requires dx == dy, got dx={dx!r}, dy={dy!r}."
      )

    def laplacian(field: Field, operators: Operators) -> Field:
      return laplacian_fd_9point(field, dx, dy)
    return laplacian

  order = solver_config.fd_order

  if use_stretch:
    def laplacian(field: Field, operators: Operators) -> Field:
      return laplacian_fd(field, dx, dy, order, operators['stretch'])
    return laplacian

  def laplacian(field: Field, operators: Operators) -> Field:
    return laplacian_fd(field, dx, dy, order)
  return laplacian


def _dealias_mask(sim_config: SimulationConfig) -> Field:
  """Returns the 2/3-rule mask, zero on the upper third of each k axis.

  A cubic nonlinearity spreads energy to three times the wavenumber of its
  input, and anything past the Nyquist limit folds back onto the grid as
  spurious low-frequency content. Zeroing the top third of each axis leaves no
  aliased product inside the retained band.

  Args:
    sim_config: Simulation configuration.

  Returns:
    A real (nx, ny) array of ones and zeros.
  """
  kx, ky = get_spectral_k_grids(
    sim_config.nx, sim_config.ny, sim_config.dx, sim_config.dy
  )
  cutoff_x = (2.0 / 3.0) * jnp.pi / sim_config.dx
  cutoff_y = (2.0 / 3.0) * jnp.pi / sim_config.dy
  return ((jnp.abs(kx) <= cutoff_x) & (jnp.abs(ky) <= cutoff_y)).astype(
    jnp.result_type(float)
  )


def _diffraction_operator(
  sim_config: SimulationConfig,
  kx: Field,
  ky: Field,
  step: float,
  propagator: str,
  mask: Field | None,
) -> Field:
  """Builds the Fourier-space diffraction multiplier for one sub-step.

  The paraxial form is the usual exp(-1j * h * k_perp**2 / (2 k)). The
  wide-angle form uses the exact square root, exp(1j * h * (kz - k)) with
  kz = sqrt(k**2 - k_perp**2); because the operator is diagonal in Fourier
  space the root needs no Pade approximation and costs the same to apply.
  Beyond the light line, kz turns imaginary and the multiplier decays, which
  is the correct treatment of evanescent components.

  Args:
    sim_config: Simulation configuration.
    kx: Transverse wavenumber grid in x.
    ky: Transverse wavenumber grid in y.
    step: Propagation distance for this sub-step; may be negative.
    propagator: 'paraxial' or 'wide_angle'.
    mask: Optional dealiasing mask folded into the multiplier.

  Returns:
    A complex (nx, ny) array.
  """
  k = sim_config.k0 * sim_config.n0
  k_perp_squared = kx**2 + ky**2

  if propagator == 'wide_angle':
    # Cast before the root so that evanescent components (k_perp > k) give a
    # decaying imaginary branch instead of nan.
    kz = jnp.sqrt((k**2 - k_perp_squared).astype(
      jnp.result_type(k_perp_squared, jnp.complex64)
    ))
    operator = jnp.exp(1j * step * (kz - k))
  else:
    operator = jnp.exp(-1j * step * k_perp_squared / (2 * k))

  if mask is not None:
    operator = operator * mask
  return operator


def splitting_weights(order: int) -> tuple[float, ...]:
  """Returns the sub-step fractions of a symmetric splitting composition.

  Order 2 is a single Strang step. Order 4 is the Yoshida composition
  S(w1 h) S(w0 h) S(w1 h) with w1 = 1 / (2 - 2**(1/3)) and w0 = 1 - 2 w1, in
  which the middle sub-step runs backwards. The fractions sum to one, so the
  composition advances by exactly h.

  Args:
    order: Order of accuracy in z; one of SPLITTING_ORDERS.

  Returns:
    The sub-step fractions, in application order.
  """
  if order == 2:
    return (1.0,)
  if order == 4:
    w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    return (w1, 1.0 - 2.0 * w1, w1)
  raise ValueError(
    f"Unsupported splitting_order {order!r}; expected one of "
    f"{SPLITTING_ORDERS}."
  )


def _make_split_step_kernel(
  sim_config: SimulationConfig,
  delta_n_fn: DeltaNFn | None,
  has_pml: bool,
  weights: tuple[float, ...] = (1.0,),
) -> Callable[[Field, Any, Operators, Any], Field]:
  """Builds a symmetric split-step Fourier kernel.

  Each sub-step is a half-step of the refraction operator, a full step of
  diffraction in Fourier space, and a second half-step; `weights` composes
  several such sub-steps to raise the order in z. Diffraction operators arrive
  precomputed in `operators`, one per sub-step.

  The PML attenuation is applied once per full step, symmetrically around the
  whole composition, rather than inside each sub-step. A composition of order
  four runs its middle sub-step backwards, and a backwards sub-step through a
  damping term would amplify rather than absorb; keeping the absorber outside
  leaves it monotone. For order two the two placements coincide exactly, so
  this is not a behaviour change.

  In vacuum without a PML the potential half-steps vanish and are omitted at
  build time, leaving only the transforms.

  Args:
    sim_config: Simulation configuration.
    delta_n_fn: Refractive index perturbation, or None for vacuum.
    has_pml: Whether the PML profile is non-trivial.
    weights: Sub-step fractions from `splitting_weights`.

  Returns:
    A callable (psi, z, operators, medium) -> psi at z + dz.
  """
  k0 = sim_config.k0
  dz = sim_config.dz
  n2 = sim_config.n2
  kerr = n2 != 0.0
  has_potential = delta_n_fn is not None or kerr

  def potential(psi: Field, z: Any, half_h: float, medium: Any) -> Any:
    """exp(1j k0 (delta_n + n2 |psi|^2) * half_h) for one half sub-step."""
    index = 0.0
    if delta_n_fn is not None:
      index = delta_n_fn(z, medium)
    if kerr:
      index = index + n2 * jnp.abs(psi)**2
    return jnp.exp((1j * k0 * half_h) * index)

  def step(psi: Field, z: Any, operators: Operators, medium: Any) -> Field:
    if has_pml:
      psi = psi * operators['pml_half']

    offset = 0.0
    for index, fraction in enumerate(weights):
      sub_h = fraction * dz
      half_h = 0.5 * sub_h
      # The medium is sampled at the midpoint of the sub-step.
      z_mid = z + offset + half_h

      if has_potential:
        first = potential(psi, z_mid, half_h, medium)
        psi = psi * first

      psi = jnp.fft.ifft2(jnp.fft.fft2(psi) * operators['linear'][index])

      if has_potential:
        # The Kerr phase is intensity dependent, and diffraction has changed
        # the intensity, so the trailing half-step is re-evaluated. Without
        # Kerr the potential is unchanged and the first factor is reused.
        second = potential(psi, z_mid, half_h, medium) if kerr else first
        psi = psi * second

      offset += sub_h

    if has_pml:
      psi = psi * operators['pml_half']
    return psi

  return step


def _make_rk4_kernel(
  sim_config: SimulationConfig,
  laplacian_fn: Callable[[Field, Operators], Field],
  delta_n_fn: DeltaNFn | None,
  apply_sigma: bool,
) -> Callable[[Field, Any, Operators, Any], Field]:
  """Builds a 4th-order Runge-Kutta kernel for the paraxial RHS.

  Args:
    sim_config: Simulation configuration.
    laplacian_fn: Transverse Laplacian, as built by `_make_laplacian_fn`.
    delta_n_fn: Refractive index perturbation, or None for vacuum.
    apply_sigma: Whether to apply the absorbing potential term. False when
                 absorption is carried by complex coordinate stretching.

  Returns:
    A callable (psi, z, operators, medium) -> psi at z + dz.
  """
  k0 = sim_config.k0
  dz = sim_config.dz
  n2 = sim_config.n2
  kerr = n2 != 0.0
  diffraction_coeff = 1j / (2 * k0 * sim_config.n0)

  def rhs(psi: Field, z: Any, operators: Operators, medium: Any) -> Field:
    out = diffraction_coeff * laplacian_fn(psi, operators)
    if delta_n_fn is not None:
      out = out + (1j * k0) * delta_n_fn(z, medium) * psi
    if kerr:
      out = out + (1j * k0 * n2) * jnp.abs(psi)**2 * psi
    if apply_sigma:
      out = out - operators['sigma'] * psi
    return out

  def step(psi: Field, z: Any, operators: Operators, medium: Any) -> Field:
    k1 = rhs(psi, z, operators, medium)
    k2 = rhs(psi + 0.5 * dz * k1, z + 0.5 * dz, operators, medium)
    k3 = rhs(psi + 0.5 * dz * k2, z + 0.5 * dz, operators, medium)
    k4 = rhs(psi + dz * k3, z + dz, operators, medium)
    return psi + (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

  return step


@functools.partial(
  jax.jit, static_argnames=('step_fn', 'return_history')
)
def _propagate(
  psi_0: Field,
  z_blocks: Field,
  operators: Operators,
  medium: Any,
  step_fn: Callable[[Field, Any, Operators, Any], Field],
  return_history: bool,
) -> tuple[Field, Field | None]:
  """Runs the propagation as a nested scan over blocks of steps.

  The outer scan emits one field per block and the inner scan advances through
  the block without emitting, so the history array holds one entry per saved
  plane rather than one per step.

  Args:
    psi_0: Initial field.
    z_blocks: Propagation coordinates, shaped (n_blocks, save_every).
    operators: Precomputed z-independent operator arrays.
    medium: Auxiliary data forwarded to the refractive index callable.
    step_fn: Single-step kernel (static).
    return_history: Whether to accumulate the field history (static).

  Returns:
    A tuple (psi_final, psi_history); psi_history is None when
    return_history is False.
  """
  def advance(psi: Field, z: Any) -> tuple[Field, None]:
    return step_fn(psi, z, operators, medium), None

  def block(psi: Field, z_block: Field) -> tuple[Field, Field | None]:
    emitted = psi if return_history else None
    psi, _ = lax.scan(advance, psi, z_block)
    return psi, emitted

  return lax.scan(block, psi_0, z_blocks)


class ParaxialWaveSolver:
  """Solver for the paraxial wave equation.

  Encapsulates the simulation configuration, solver method, PML settings and
  refractive index perturbation, and propagates an initial envelope through
  the medium.

  The refractive index is supplied as a perturbation delta_n = n - n0 and is
  called as `delta_n_fn(z, medium)`. Passing the medium through `solve` rather
  than closing over it keeps it a traced argument, so swapping media - across
  chunks, or across realizations of a turbulent ensemble - reuses the compiled
  computation instead of triggering a fresh trace.
  """

  def __init__(
    self,
    sim_config: SimulationConfig,
    solver_config: SolverConfig,
    pml_config: PMLConfig,
    delta_n_fn: DeltaNFn | None = None,
  ):
    """Initializes the solver and precomputes every z-independent operator.

    Args:
      sim_config: Simulation configuration.
      solver_config: Solver configuration.
      pml_config: PML configuration.
      delta_n_fn: Callable (z, medium) -> delta_n, where delta_n broadcasts
                  against an (nx, ny) field. None means vacuum, and lets the
                  solver drop the refraction term entirely.

    Raises:
      ValueError: If the PML is wider than half the grid, or if the solver
                  configuration is incompatible with the grid.
    """
    if 2 * pml_config.width_x >= sim_config.nx:
      raise ValueError(
        f"PML width_x={pml_config.width_x} leaves no interior domain for "
        f"nx={sim_config.nx}; it must be smaller than nx / 2."
      )
    if 2 * pml_config.width_y >= sim_config.ny:
      raise ValueError(
        f"PML width_y={pml_config.width_y} leaves no interior domain for "
        f"ny={sim_config.ny}; it must be smaller than ny / 2."
      )

    _warn_if_unstable(solver_config, sim_config)

    self.sim_config = sim_config
    self.solver_config = solver_config
    self.pml_config = pml_config
    self.delta_n_fn = delta_n_fn

    pml_data = generate_pml_profile(sim_config, pml_config)

    # Coordinate stretching only makes sense for the finite difference
    # operators; the spectral method falls back to the absorbing potential.
    use_stretch = (
      pml_data.stretch is not None
      and solver_config.method == 'finite_difference'
    )
    has_pml = bool(pml_config.width_x or pml_config.width_y) and (
      pml_config.strength > 0.0
    )

    self.pml_profile = pml_data.sigma
    operators: Operators = {}

    if solver_config.stepper == 'split_step':
      kx, ky = get_spectral_k_grids(
        sim_config.nx, sim_config.ny, sim_config.dx, sim_config.dy
      )
      mask = (
        _dealias_mask(sim_config) if solver_config.dealias else None
      )
      # Precomputed once: the diffraction operators do not depend on z. One
      # per sub-step of the splitting composition.
      weights = splitting_weights(solver_config.splitting_order)
      operators['linear'] = [
        _diffraction_operator(
          sim_config, kx, ky, fraction * sim_config.dz,
          solver_config.propagator, mask,
        )
        for fraction in weights
      ]
      if has_pml:
        operators['pml_half'] = jnp.exp(-pml_data.sigma * (sim_config.dz / 2))
      self._step_fn = _make_split_step_kernel(
        sim_config, delta_n_fn, has_pml, weights
      )
    else:
      if solver_config.method == 'spectral':
        kx, ky = get_spectral_k_grids(
          sim_config.nx, sim_config.ny, sim_config.dx, sim_config.dy
        )
        operators['kx'], operators['ky'] = kx, ky
      if use_stretch:
        operators['stretch'] = pml_data.stretch.as_dict()
      apply_sigma = has_pml and not use_stretch
      if apply_sigma:
        operators['sigma'] = pml_data.sigma
      laplacian_fn = _make_laplacian_fn(
        solver_config, sim_config, use_stretch
      )
      self._step_fn = _make_rk4_kernel(
        sim_config, laplacian_fn, delta_n_fn, apply_sigma
      )

    self._operators = operators

  def solve(
    self,
    psi_0: Field,
    z_0: float = 0.0,
    medium: Any = None,
    save_every: int = 1,
    return_history: bool = True,
  ) -> tuple[Field, Field | None]:
    """Propagates the initial envelope psi_0 through the medium.

    Args:
      psi_0: Initial complex field amplitude at z = z_0.
      z_0: Initial z position. The solver takes nz steps of size dz from here,
           so it finishes at z_0 + nz * dz.
      medium: Auxiliary data forwarded as the second argument of delta_n_fn.
              Passing it here rather than closing over it keeps the compiled
              computation reusable across media.
      save_every: Store the field every `save_every` steps. Must divide nz.
      return_history: If False, no history is accumulated and only the final
                      field is returned. For long runs this is the difference
                      between allocating an (nz, nx, ny) complex array and
                      allocating nothing.

    Returns:
      A tuple (psi_final, psi_history) where psi_final is the field at
      z_0 + nz * dz, and psi_history has shape (nz // save_every, nx, ny) with
      psi_history[j] the field at z_0 + j * save_every * dz - so index 0 is
      psi_0 itself. psi_history is None when return_history is False.

    Raises:
      ValueError: If save_every is not a positive divisor of nz.
    """
    nz = self.sim_config.nz
    if save_every < 1:
      raise ValueError(f"save_every must be positive, got {save_every}.")
    if nz % save_every:
      raise ValueError(
        f"save_every={save_every} must divide nz={nz}."
      )

    # Exact dz spacing: linspace(z_0, lz, nz) would step by (lz - z_0)/(nz - 1)
    # and would ignore z_0 in the span entirely.
    zs = z_0 + self.sim_config.dz * jnp.arange(nz)
    z_blocks = zs.reshape(nz // save_every, save_every)

    return _propagate(
      psi_0,
      z_blocks,
      self._operators,
      medium,
      step_fn=self._step_fn,
      return_history=return_history,
    )


def propagate(
  psi_0: Field,
  z_0: float,
  sim_config: SimulationConfig,
  solver_config: SolverConfig,
  pml_config: PMLConfig,
  delta_n_fn: DeltaNFn | None = None,
  medium: Any = None,
  save_every: int = 1,
  return_history: bool = True,
) -> tuple[Field, Field | None]:
  """Builds a solver and propagates psi_0 in one call.

  Convenient for one-off runs. Prefer constructing a `ParaxialWaveSolver` when
  propagating repeatedly, so the compiled computation is reused.

  Args:
    psi_0: Initial field at z = z_0.
    z_0: Initial z position.
    sim_config: Simulation configuration.
    solver_config: Solver configuration.
    pml_config: PML configuration.
    delta_n_fn: Callable (z, medium) -> delta_n, or None for vacuum.
    medium: Auxiliary data forwarded to delta_n_fn.
    save_every: Store the field every `save_every` steps.
    return_history: Whether to accumulate the field history.

  Returns:
    A tuple (psi_final, psi_history); see `ParaxialWaveSolver.solve`.
  """
  solver = ParaxialWaveSolver(
    sim_config, solver_config, pml_config, delta_n_fn
  )
  return solver.solve(
    psi_0, z_0, medium=medium, save_every=save_every,
    return_history=return_history,
  )
