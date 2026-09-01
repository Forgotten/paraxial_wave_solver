"""Beam diagnostics.

Every function here is a pure JAX function of a transverse field, so any of
them can be passed to `ParaxialWaveSolver.solve` as an `observable_fn` and
evaluated inside the propagation loop. Doing so records a handful of numbers
per plane instead of an (nx, ny) complex field, which is usually the
difference between a few kilobytes and a few gigabytes.

Unless a mask is passed, quantities are computed over the whole grid. When a
PML is present its absorbing region is part of that grid, so one needs to 
trim the field before measuring as the PML may bias the result.
"""

from typing import Any

import jax.numpy as jnp

from .config import Field, SimulationConfig


def _coordinates(sim_config: SimulationConfig) -> tuple[Field, Field]:
  """Returns broadcastable transverse coordinate grids (x, y)."""
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  return x[:, None], y[None, :]


def _wavenumbers(sim_config: SimulationConfig) -> tuple[Field, Field]:
  """Returns broadcastable transverse wavenumber grids (kx, ky)."""
  kx = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nx, d=sim_config.dx)
  ky = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.ny, d=sim_config.dy)
  return kx[:, None], ky[None, :]


def total_power(psi: Field, sim_config: SimulationConfig) -> Field:
  """Returns the total optical power, the integral of |psi|^2 over the plane.

  Args:
    psi: Complex field of shape (nx, ny).
    sim_config: Simulation configuration, for the cell area.

  Returns:
    A scalar.
  """
  return jnp.sum(jnp.abs(psi)**2) * sim_config.dx * sim_config.dy


def peak_intensity(psi: Field) -> Field:
  """Returns the largest |psi|^2 on the grid.

  Args:
    psi: Complex field of shape (nx, ny).

  Returns:
    A scalar.
  """
  return jnp.max(jnp.abs(psi)**2)


def centroid(psi: Field, sim_config: SimulationConfig) -> tuple[Field, Field]:
  """Returns the intensity-weighted centre of mass (x_bar, y_bar).

  Args:
    psi: Complex field of shape (nx, ny).
    sim_config: Simulation configuration.

  Returns:
    A tuple of two scalars, in the same units as dx and dy.
  """
  x, y = _coordinates(sim_config)
  intensity = jnp.abs(psi)**2
  total = jnp.sum(intensity)
  return jnp.sum(x * intensity) / total, jnp.sum(y * intensity) / total


def second_moments(
  psi: Field,
  sim_config: SimulationConfig,
) -> tuple[Field, Field, Field]:
  """Returns the central second moments (sigma_xx, sigma_yy, sigma_xy).

  These are variances, not widths; `beam_width` converts them.

  Args:
    psi: Complex field of shape (nx, ny).
    sim_config: Simulation configuration.

  Returns:
    A tuple (sigma_xx, sigma_yy, sigma_xy) of scalars.
  """
  x, y = _coordinates(sim_config)
  intensity = jnp.abs(psi)**2
  total = jnp.sum(intensity)
  x_bar = jnp.sum(x * intensity) / total
  y_bar = jnp.sum(y * intensity) / total
  dx = x - x_bar
  dy = y - y_bar
  return (
    jnp.sum(dx**2 * intensity) / total,
    jnp.sum(dy**2 * intensity) / total,
    jnp.sum(dx * dy * intensity) / total,
  )


def beam_width(
  psi: Field,
  sim_config: SimulationConfig,
) -> tuple[Field, Field]:
  """Returns the ISO 11146 D4-sigma widths (w_x, w_y).

  For a fundamental Gaussian of waist w0 this returns 2 * w0, since the
  D4-sigma diameter of such a beam is twice its 1/e^2 intensity radius.

  Args:
    psi: Complex field of shape (nx, ny).
    sim_config: Simulation configuration.

  Returns:
    A tuple of two scalars.
  """
  sigma_xx, sigma_yy, _ = second_moments(psi, sim_config)
  return 4 * jnp.sqrt(sigma_xx), 4 * jnp.sqrt(sigma_yy)


def rms_radius(psi: Field, sim_config: SimulationConfig) -> Field:
  """Returns the intensity-weighted RMS radius about the centroid.

  Args:
    psi: Complex field of shape (nx, ny).
    sim_config: Simulation configuration.

  Returns:
    A scalar.
  """
  sigma_xx, sigma_yy, _ = second_moments(psi, sim_config)
  return jnp.sqrt(sigma_xx + sigma_yy)


def m_squared(
  psi: Field,
  sim_config: SimulationConfig,
) -> tuple[Field, Field]:
  """Returns the beam quality factors (M2_x, M2_y).

  Uses the full space-frequency covariance,

      M2 = 2 * sqrt(sigma_xx * sigma_kk - sigma_xk**2),

  where sigma_xk is the correlation between position and local transverse
  wavenumber. Dropping that cross term - as taking the product of a real-space
  and a Fourier-space width does - gives the right answer only at a waist,
  and overestimates M2 everywhere else.

  A diffraction-limited fundamental Gaussian gives 1.0 at any propagation
  distance.

  Args:
    psi: Complex field of shape (nx, ny).
    sim_config: Simulation configuration.

  Returns:
    A tuple (M2_x, M2_y) of scalars.
  """
  x, y = _coordinates(sim_config)
  kx, ky = _wavenumbers(sim_config)

  intensity = jnp.abs(psi)**2
  total = jnp.sum(intensity)
  x_bar = jnp.sum(x * intensity) / total
  y_bar = jnp.sum(y * intensity) / total
  dx = x - x_bar
  dy = y - y_bar

  # Spatial variances.
  sigma_xx = jnp.sum(dx**2 * intensity) / total
  sigma_yy = jnp.sum(dy**2 * intensity) / total

  # Wavenumber variances, from Parseval on the transform.
  spectrum = jnp.fft.fft2(psi)
  spectral_intensity = jnp.abs(spectrum)**2
  spectral_total = jnp.sum(spectral_intensity)
  kx_bar = jnp.sum(kx * spectral_intensity) / spectral_total
  ky_bar = jnp.sum(ky * spectral_intensity) / spectral_total
  sigma_kx = jnp.sum((kx - kx_bar)**2 * spectral_intensity) / spectral_total
  sigma_ky = jnp.sum((ky - ky_bar)**2 * spectral_intensity) / spectral_total

  # Position-wavenumber correlation, from the local wavevector
  # Im(conj(psi) d(psi)/dx), evaluated spectrally.
  d_psi_dx = jnp.fft.ifft2(1j * kx * spectrum)
  d_psi_dy = jnp.fft.ifft2(1j * ky * spectrum)
  local_kx = jnp.imag(jnp.conj(psi) * d_psi_dx)
  local_ky = jnp.imag(jnp.conj(psi) * d_psi_dy)
  # Weighting by the centred coordinate already subtracts x_bar * kx_bar, so
  # these are the central covariances.
  sigma_xk = jnp.sum(dx * local_kx) / total
  sigma_yk = jnp.sum(dy * local_ky) / total

  # The variances are guaranteed non-negative analytically; clamp so that
  # round-off on a near-diffraction-limited beam cannot produce a nan.
  m2_x = 2 * jnp.sqrt(jnp.maximum(sigma_xx * sigma_kx - sigma_xk**2, 0.0))
  m2_y = 2 * jnp.sqrt(jnp.maximum(sigma_yy * sigma_ky - sigma_yk**2, 0.0))
  return m2_x, m2_y


def strehl_ratio(psi: Field, reference: Field) -> Field:
  """Returns the peak intensity of psi relative to a reference beam.

  The usual aberration figure of merit: 1.0 for an unaberrated beam, lower
  when wavefront distortion spreads the energy out of the core. Both fields
  should carry the same total power for the comparison to be meaningful.

  Being a ratio of peaks, it assumes the reference is the diffraction-limited
  version of the same beam and that the aberration only ever spreads energy.
  Neither holds for a strongly speckled field, where a random hot spot can be
  brighter than the unaberrated peak and the ratio exceeds 1, nor for a
  multi-lobed mode whose peak is not on axis. Prefer `overlap` when tracking
  how much of a specific mode survives.

  Args:
    psi: Complex field of shape (nx, ny).
    reference: Diffraction-limited field of the same shape.

  Returns:
    A scalar.
  """
  return jnp.max(jnp.abs(psi)**2) / jnp.max(jnp.abs(reference)**2)


def overlap(psi: Field, reference: Field) -> Field:
  """Returns the normalized mode overlap |<psi|ref>|^2 / (<psi|psi><ref|ref>).

  Insensitive to the amplitude and global phase of either field, so it
  measures how much of the beam is still in the reference mode. 1.0 for a
  perfect match, 0.0 for orthogonal modes.

  Args:
    psi: Complex field of shape (nx, ny).
    reference: Field of the same shape.

  Returns:
    A scalar in [0, 1].
  """
  inner = jnp.abs(jnp.sum(jnp.conj(reference) * psi))**2
  norms = jnp.sum(jnp.abs(psi)**2) * jnp.sum(jnp.abs(reference)**2)
  return inner / norms


def scintillation_index(psi: Field, mask: Field | None = None) -> Field:
  """Returns the spatial scintillation index over the transverse plane.

  Defined as <I^2> / <I>^2 - 1, with the averages taken over the grid.

  Pass a `mask` unless the beam fills the grid. Averaged over the whole plane,
  a beam surrounded by dark background gives a large value regardless of how
  speckled it is, because <I> collapses towards zero while <I^2> is still set
  by the bright core; a beam covering a tenth of the grid reads a scintillation
  index of order ten even in a clear medium. Restricting to the illuminated
  region, for example `jnp.abs(reference)**2 > 0.01 * peak`, measures the beam
  rather than the background.

  Note also that the standard turbulence quantity averages over an *ensemble*
  of realizations at a fixed point; use `ensemble_scintillation_index` for
  that. The spatial version here is the single-realization analogue.

  Args:
    psi: Complex field of shape (nx, ny).
    mask: Optional boolean or 0/1 array selecting the region to average over.

  Returns:
    A scalar.
  """
  intensity = jnp.abs(psi)**2
  if mask is None:
    mean = jnp.mean(intensity)
    return jnp.mean(intensity**2) / mean**2 - 1.0
  weights = jnp.asarray(mask).astype(intensity.dtype)
  total = jnp.sum(weights)
  mean = jnp.sum(weights * intensity) / total
  mean_square = jnp.sum(weights * intensity**2) / total
  return mean_square / mean**2 - 1.0


def ensemble_scintillation_index(intensities: Field) -> Field:
  """Returns the pointwise scintillation index across an ensemble.

  Args:
    intensities: Array of shape (n_realizations, ...) holding |psi|^2 for each
      realization, for example the output of a vmapped propagation.

  Returns:
    An array shaped like a single realization, holding
    <I^2> / <I>^2 - 1 at each point.
  """
  mean = jnp.mean(intensities, axis=0)
  mean_square = jnp.mean(intensities**2, axis=0)
  return mean_square / mean**2 - 1.0


def encircled_power(
  psi: Field,
  sim_config: SimulationConfig,
  radius: float,
) -> Field:
  """Returns the fraction of the total power within `radius` of the centroid.

  Args:
    psi: Complex field of shape (nx, ny).
    sim_config: Simulation configuration.
    radius: Bucket radius, in the same units as dx.

  Returns:
    A scalar in [0, 1].
  """
  x, y = _coordinates(sim_config)
  intensity = jnp.abs(psi)**2
  total = jnp.sum(intensity)
  x_bar = jnp.sum(x * intensity) / total
  y_bar = jnp.sum(y * intensity) / total
  inside = ((x - x_bar)**2 + (y - y_bar)**2) <= radius**2
  return jnp.sum(jnp.where(inside, intensity, 0.0)) / total


def beam_diagnostics(
  psi: Field,
  sim_config: SimulationConfig,
  mask: Field | None = None,
) -> dict[str, Any]:
  """Returns the common diagnostics as a dict, ready to use as an observable.

  Intended to be partially applied and handed to
  `ParaxialWaveSolver.solve(..., observable_fn=...)`:

      from functools import partial
      observable = lambda psi, z: beam_diagnostics(psi, sim_config)
      psi_final, history = solver.solve(psi_0, observable_fn=observable)

  `history` is then a dict of arrays, one entry per saved plane, rather than a
  stack of fields.

  Args:
    psi: Complex field of shape (nx, ny).
    sim_config: Simulation configuration.
    mask: Optional region for the scintillation index; see
      `scintillation_index` for why the unmasked value is dominated by dark
      background when the beam does not fill the grid.

  Returns:
    A dict with keys 'power', 'peak_intensity', 'centroid_x', 'centroid_y',
    'width_x', 'width_y', 'rms_radius', 'm2_x', 'm2_y' and
    'scintillation_index'.
  """
  x_bar, y_bar = centroid(psi, sim_config)
  width_x, width_y = beam_width(psi, sim_config)
  m2_x, m2_y = m_squared(psi, sim_config)
  return {
    'power': total_power(psi, sim_config),
    'peak_intensity': peak_intensity(psi),
    'centroid_x': x_bar,
    'centroid_y': y_bar,
    'width_x': width_x,
    'width_y': width_y,
    'rms_radius': rms_radius(psi, sim_config),
    'm2_x': m2_x,
    'm2_y': m2_y,
    'scintillation_index': scintillation_index(psi, mask),
  }
