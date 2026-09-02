"""Analytical beam profiles and random media generators.

All beams are returned as slowly varying envelopes in the convention fixed in
`config.py`: the physical field is psi * exp(1j * k * z). Passing
`envelope_only=False` (the default) multiplies the carrier back in.

The beams are built from the complex beam parameter

    q(z) = z - 1j * z_R,        z_R = k * w0**2 / 2,

for which the fundamental mode is exactly (q(0) / q(z)) * exp(1j*k*r^2/(2q)).
This form is free of the removable singularities that w(z), R(z) and the Gouy
phase have individually, so the generators contain no branches on z and can be
jitted and vmapped over the propagation distance.
"""

import math

import jax
import jax.numpy as jnp

from .config import Field, SimulationConfig


def _hermite_h(n: int, x: Field) -> Field:
  """Physicists' Hermite polynomial H_n(x) by upward recurrence.

  Evaluated as H_{i+1} = 2x H_i - 2i H_{i-1} rather than through explicit
  polynomial coefficients, which reach ~4e13 by n=20 and lose most of the
  float32 mantissa to cancellation.

  Args:
    n: Polynomial degree (a Python int; not traced).
    x: Evaluation points.

  Returns:
    H_n(x), same shape as x.
  """
  if n < 0:
    raise ValueError(f"Hermite degree must be non-negative, got {n}.")
  if n == 0:
    return jnp.ones_like(x)
  h_prev, h = jnp.ones_like(x), 2 * x
  for i in range(1, n):
    h_prev, h = h, 2 * x * h - 2 * i * h_prev
  return h


def _laguerre_l(p: int, alpha: int, x: Field) -> Field:
  """Generalized Laguerre polynomial L_p^alpha(x) by upward recurrence.

  Uses (i+1) L_{i+1} = (2i + 1 + alpha - x) L_i - (i + alpha) L_{i-1}.

  Args:
    p: Radial order (a Python int; not traced).
    alpha: Generalized order, |l| for a Laguerre-Gaussian mode.
    x: Evaluation points.

  Returns:
    L_p^alpha(x), same shape as x.
  """
  if p < 0:
    raise ValueError(f"Laguerre order must be non-negative, got {p}.")
  if p == 0:
    return jnp.ones_like(x)
  l_prev, l_cur = jnp.ones_like(x), 1.0 + alpha - x
  for i in range(1, p):
    l_prev, l_cur = (
      l_cur,
      ((2 * i + 1 + alpha - x) * l_cur - (i + alpha) * l_prev) / (i + 1),
    )
  return l_cur


class _BeamFrame:
  """Grid geometry and beam parameters shared by every analytical beam.

  Attributes:
    dx: x-offset from the beam centre, shape (nx, ny).
    dy: y-offset from the beam centre, shape (nx, ny).
    r2: Squared transverse radius, shape (nx, ny).
    k: Wavenumber in the background medium.
    z_R: Rayleigh range.
    w_z: Beam radius at z.
    gouy: Unit phasor exp(-1j * zeta(z)) for one unit of Gouy phase.
    gaussian: exp(1j * k * r2 / (2 * q)), i.e. the transverse Gaussian
              envelope together with the wavefront curvature.
    amplitude: The 1/w(z) amplitude prefactor.
  """

  __slots__ = ('dx', 'dy', 'r2', 'k', 'z_R', 'w_z', 'gouy', 'gaussian',
               'amplitude')

  def __init__(
    self,
    sim_config: SimulationConfig,
    w0: float,
    z: float,
    x0: float | None,
    y0: float | None,
  ):
    if x0 is None:
      x0 = sim_config.lx / 2.0
    if y0 is None:
      y0 = sim_config.ly / 2.0

    x = jnp.arange(sim_config.nx) * sim_config.dx
    y = jnp.arange(sim_config.ny) * sim_config.dy
    self.dx = (x - x0)[:, None]
    self.dy = (y - y0)[None, :]
    self.r2 = self.dx**2 + self.dy**2

    self.k = sim_config.k
    self.z_R = self.k * w0**2 / 2.0

    # Complex beam parameter. q(0) = -1j * z_R gives a decaying Gaussian for
    # the d(psi)/dz = (1j / 2k) lap(psi) convention used by the solver.
    q0 = -1j * self.z_R
    q = z + q0
    q_ratio = q0 / q                       # (w0 / w_z) * exp(-1j * zeta)

    self.w_z = w0 * jnp.abs(q) / self.z_R
    self.gouy = q_ratio / jnp.abs(q_ratio)  # exp(-1j * zeta)
    self.gaussian = jnp.exp(1j * self.k * self.r2 / (2 * q))
    self.amplitude = 1.0 / self.w_z


def _apply_power(psi: Field, sim_config: SimulationConfig,
                 power: float | None) -> Field:
  """Rescales psi to carry the requested total power, if one was given."""
  if power is None:
    return psi
  current = jnp.sum(jnp.abs(psi)**2) * sim_config.dx * sim_config.dy
  return psi * jnp.sqrt(power / current)


def _apply_carrier(psi: Field, k: float, z: float,
                   envelope_only: bool) -> Field:
  """Multiplies the carrier exp(1j * k * z) back in unless suppressed."""
  if envelope_only:
    return psi
  return psi * jnp.exp(1j * k * z)


def gaussian_beam(
  sim_config: SimulationConfig,
  w0: float,
  z: float = 0.0,
  power: float | None = None,
  x0: float | None = None,
  y0: float | None = None,
  kx0: float = 0.0,
  ky0: float = 0.0,
  envelope_only: bool = False,
) -> Field:
  """Generates an analytical fundamental Gaussian (TEM_00) beam profile.

  Peak-normalized: |psi| = 1 on axis at z = 0 unless `power` is given. The
  Laguerre- and Hermite-Gaussian generators are power-normalized instead; see
  their docstrings.

  Args:
    sim_config: Simulation configuration.
    w0: Beam waist radius (1/e^2 intensity radius) at z=0.
    z: Propagation distance along the z-axis.
    power: Optional total optical power to normalize the beam to.
    x0: Centre x-position (default: lx / 2).
    y0: Centre y-position (default: ly / 2).
    kx0: Transverse wavenumber in x (tilt).
    ky0: Transverse wavenumber in y (tilt).
    envelope_only: If True, omit the carrier phase exp(1j * k * z).

  Returns:
    psi: Complex field array of shape (nx, ny).
  """
  frame = _BeamFrame(sim_config, w0, z, x0, y0)

  psi = (w0 * frame.amplitude) * frame.gouy * frame.gaussian

  if kx0 != 0.0 or ky0 != 0.0:
    x = jnp.arange(sim_config.nx) * sim_config.dx
    y = jnp.arange(sim_config.ny) * sim_config.dy
    psi = psi * jnp.exp(1j * (kx0 * x[:, None] + ky0 * y[None, :]))

  psi = _apply_carrier(psi, frame.k, z, envelope_only)
  return _apply_power(psi, sim_config, power)


def laguerre_gaussian_beam(
  sim_config: SimulationConfig,
  w0: float,
  p: int = 0,
  l: int = 0,
  z: float = 0.0,
  power: float | None = None,
  x0: float | None = None,
  y0: float | None = None,
  envelope_only: bool = False,
) -> Field:
  """Computes the analytical Laguerre-Gaussian (LG_pl) beam.

  Normalized to unit total power when `power` is not given.

  Args:
    sim_config: Simulation configuration.
    w0: Beam waist radius (1/e^2 intensity radius) at z=0.
    p: Radial mode index (p >= 0).
    l: Azimuthal mode index (topological charge / OAM).
    z: Propagation distance along the z-axis.
    power: Optional target total optical power.
    x0: Centre x-position (default: lx / 2).
    y0: Centre y-position (default: ly / 2).
    envelope_only: If True, omit the carrier phase exp(1j * k * z).

  Returns:
    psi: Complex field array of shape (nx, ny).
  """
  frame = _BeamFrame(sim_config, w0, z, x0, y0)
  abs_l = abs(l)

  arg = 2 * frame.r2 / frame.w_z**2
  radial = (jnp.sqrt(2 * frame.r2) / frame.w_z)**abs_l
  norm = math.sqrt(
    2 * math.factorial(p) / (math.pi * math.factorial(p + abs_l))
  )

  phi = jnp.arctan2(frame.dy, frame.dx)

  psi = (
    norm * frame.amplitude
    * radial
    * _laguerre_l(p, abs_l, arg)
    * frame.gaussian
    * frame.gouy**(2 * p + abs_l + 1)
    * jnp.exp(1j * l * phi)
  )

  psi = _apply_carrier(psi, frame.k, z, envelope_only)
  return _apply_power(psi, sim_config, power)


def hermite_gaussian_beam(
  sim_config: SimulationConfig,
  w0: float,
  n: int = 0,
  m: int = 0,
  z: float = 0.0,
  power: float | None = None,
  x0: float | None = None,
  y0: float | None = None,
  envelope_only: bool = False,
) -> Field:
  """Computes the analytical Hermite-Gaussian (HG_nm) beam.

  Normalized to unit total power when `power` is not given.

  Args:
    sim_config: Simulation configuration.
    w0: Beam waist radius at z=0.
    n: Mode index in the x-direction (n >= 0).
    m: Mode index in the y-direction (m >= 0).
    z: Propagation distance along the z-axis.
    power: Optional target total optical power.
    x0: Centre x-position (default: lx / 2).
    y0: Centre y-position (default: ly / 2).
    envelope_only: If True, omit the carrier phase exp(1j * k * z).

  Returns:
    psi: Complex field array of shape (nx, ny).
  """
  frame = _BeamFrame(sim_config, w0, z, x0, y0)

  root2_over_w = jnp.sqrt(2.0) / frame.w_z
  h_n = _hermite_h(n, root2_over_w * frame.dx)
  h_m = _hermite_h(m, root2_over_w * frame.dy)

  norm = math.sqrt(
    2.0 / (math.pi * 2**(n + m) * math.factorial(n) * math.factorial(m))
  )

  psi = (
    norm * frame.amplitude
    * h_n * h_m
    * frame.gaussian
    * frame.gouy**(n + m + 1)
  )

  psi = _apply_carrier(psi, frame.k, z, envelope_only)
  return _apply_power(psi, sim_config, power)


# Aliases kept for backward compatibility.
get_laguerre_gaussian_analytical = laguerre_gaussian_beam
get_hermite_gaussian_analytical = hermite_gaussian_beam
get_analytical_beam = gaussian_beam


def _k_grids_3d(sim_config: SimulationConfig, real_z: bool = False):
  """Returns broadcastable (kx, ky, kz) wavenumber grids for the 3D volume.

  Kept in broadcast shape - (nx,1,1), (1,ny,1), (1,1,nz) - rather than
  materialized with meshgrid, which would allocate three full 3D arrays.

  Args:
    sim_config: Simulation configuration.
    real_z: If True, kz uses rfftfreq (for use with irfftn).

  Returns:
    A tuple of three broadcastable arrays.
  """
  kx = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nx, d=sim_config.dx)
  ky = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.ny, d=sim_config.dy)
  freq_z = jnp.fft.rfftfreq if real_z else jnp.fft.fftfreq
  kz = 2 * jnp.pi * freq_z(sim_config.nz, d=sim_config.dz)
  return kx[:, None, None], ky[None, :, None], kz[None, None, :]


def random_medium(
  sim_config: SimulationConfig,
  correlation_length: float,
  strength: float,
  key: jax.Array,
) -> Field:
  """Generates a Gaussian random refractive index perturbation.

  The field is sampled directly in the Fourier domain: the transform of white
  noise is white noise, so the forward FFT that a filter-the-noise formulation
  would need is unnecessary. Combined with a real-output inverse transform this
  roughly halves both the transform count and the peak memory.

  Args:
    sim_config: Simulation configuration.
    correlation_length: Correlation length of the medium (physical units).
    strength: Target standard deviation of the fluctuation (delta_n).
    key: JAX random key for reproducibility.

  Returns:
    delta_n: Real array of shape (nx, ny, nz).
  """
  kx, ky, kz = _k_grids_3d(sim_config, real_z=True)
  k_squared = kx**2 + ky**2 + kz**2

  # Gaussian correlation -> Gaussian power spectrum exp(-k^2 L^2 / 4); the
  # amplitude filter is its square root.
  amplitude = jnp.exp(-k_squared * correlation_length**2 / 8.0)

  key_r, key_i = jax.random.split(key)
  shape = amplitude.shape
  noise_k = (
    jax.random.normal(key_r, shape) + 1j * jax.random.normal(key_i, shape)
  )

  delta_n = jnp.fft.irfftn(
    noise_k * amplitude, s=(sim_config.nx, sim_config.ny, sim_config.nz)
  )
  return delta_n * (strength / jnp.std(delta_n))


def phase_screen(
  key: jax.Array,
  sim_config: SimulationConfig,
  correlation_length: float,
  strength: float,
) -> Field:
  """Generates one transverse refractive index screen, without a volume.

  Statistically this reproduces a single xy slice of `random_medium`. The 3D
  spectrum there is exp(-(kx^2 + ky^2 + kz^2) L^2 / 4); integrating it over kz
  to get the marginal seen by one slice leaves exp(-(kx^2 + ky^2) L^2 / 4)
  times a constant, which is the same functional form. So the transverse
  correlation length and variance match.

  What does *not* carry over is correlation along z. `random_medium` filters
  isotropically in three dimensions, so consecutive slices of it are
  correlated over the same length scale; screens drawn here from independent
  keys are not. This is the standard thin-screen model and is the right choice
  when the step size already exceeds the longitudinal correlation length, but
  it is an approximation rather than a drop-in replacement. Where longitudinal
  correlation matters, generate the volume.

  The point of it is memory: a run needs one (nx, ny) screen at a time rather
  than an (nx, ny, nz) volume, which at 256x256x400 is 105 MB that never has
  to exist.

  Args:
    key: JAX random key. Use `jax.random.fold_in(key, step)` inside a
      propagation loop to get a reproducible screen per step.
    sim_config: Simulation configuration.
    correlation_length: Transverse correlation length (physical units).
    strength: Standard deviation of the refractive index fluctuation.

  Returns:
    A real (nx, ny) array.
  """
  kx = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nx, d=sim_config.dx)
  ky = 2 * jnp.pi * jnp.fft.rfftfreq(sim_config.ny, d=sim_config.dy)
  k_squared = kx[:, None]**2 + ky[None, :]**2

  amplitude = jnp.exp(-k_squared * correlation_length**2 / 8.0)

  key_real, key_imag = jax.random.split(key)
  shape = amplitude.shape
  noise = (
    jax.random.normal(key_real, shape)
    + 1j * jax.random.normal(key_imag, shape)
  )
  screen = jnp.fft.irfft2(
    noise * amplitude, s=(sim_config.nx, sim_config.ny)
  )
  return screen * (strength / jnp.std(screen))


def random_medium_spectral(
  sim_config: SimulationConfig,
  Cn2: float,
  L0: float,
  l0: float,
  key: jax.Array,
) -> Field:
  """Generates a refractive index perturbation with a Von Karman spectrum.

  Args:
    sim_config: Simulation configuration.
    Cn2: Structure constant for refractive index fluctuations.
    L0: Outer scale of turbulence.
    l0: Inner scale of turbulence.
    key: JAX random key for reproducibility.

  Returns:
    A real array of shape (nx, ny, nz).
  """
  kx, ky, kz = _k_grids_3d(sim_config)
  kappa_squared = kx**2 + ky**2 + kz**2

  d_volume_k = (
    (2 * jnp.pi)**3
    / (sim_config.nx * sim_config.dx
       * sim_config.ny * sim_config.dy
       * sim_config.nz * sim_config.dz)
  )

  kappa0 = 2 * jnp.pi / L0
  kappam = 5.92 / l0

  # Von Karman spectrum.
  phi_n = (
    0.033 * Cn2
    * jnp.exp(-kappa_squared / kappam**2)
    / (kappa_squared + kappa0**2)**(11 / 6)
  )

  sigma_k2 = jnp.where(kappa_squared == 0, 0.0, phi_n * d_volume_k)

  key_r, key_i = jax.random.split(key)
  shape = sigma_k2.shape
  xi = jax.random.normal(key_r, shape) + 1j * jax.random.normal(key_i, shape)

  # jnp.fft.ifftn carries a 1/N normalization; undo it so that the sampled
  # spectral amplitudes carry the intended variance.
  n_total = sim_config.nx * sim_config.ny * sim_config.nz
  v_hat = n_total * jnp.sqrt(sigma_k2 / 2) * xi

  return jnp.real(jnp.fft.ifftn(v_hat)) * jnp.sqrt(2)
