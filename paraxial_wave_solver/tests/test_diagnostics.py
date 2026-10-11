"""Tests for the beam diagnostics, against closed-form values where possible."""

import jax
import jax.numpy as jnp
import pytest

from paraxial_wave_solver.src import diagnostics
from paraxial_wave_solver.src.config import (
  PMLConfig,
  SimulationConfig,
  SolverConfig,
)
from paraxial_wave_solver.src.solvers import ParaxialWaveSolver
from paraxial_wave_solver.src.utils import (
  gaussian_beam,
  hermite_gaussian_beam,
  laguerre_gaussian_beam,
)

NO_PML = PMLConfig(width_x=0, width_y=0, strength=0.0)
SPLIT = SolverConfig(method='spectral', stepper='split_step')


@pytest.fixture
def grid():
  return SimulationConfig(
    nx=512, ny=512, dx=0.05, dy=0.05, dz=0.02, nz=50, wavelength=1.0
  )


# --------------------------------------------------------------------------
# Moments
# --------------------------------------------------------------------------

def test_total_power_matches_direct_integral(grid, x64):
  psi = gaussian_beam(grid, w0=2.0)
  expected = jnp.sum(jnp.abs(psi)**2) * grid.dx * grid.dy
  assert diagnostics.total_power(psi, grid) == pytest.approx(float(expected))


def test_centroid_finds_a_displaced_beam(grid, x64):
  psi = gaussian_beam(grid, w0=2.0, x0=8.0, y0=15.0)
  x_bar, y_bar = diagnostics.centroid(psi, grid)
  assert float(x_bar) == pytest.approx(8.0, abs=1e-4)
  assert float(y_bar) == pytest.approx(15.0, abs=1e-4)


@pytest.mark.parametrize("w0", [1.0, 2.0, 3.0])
def test_d4sigma_width_of_a_gaussian_is_twice_the_waist(w0, grid, x64):
  """The D4-sigma diameter of a Gaussian is 2 * w0 by definition."""
  psi = gaussian_beam(grid, w0=w0)
  width_x, width_y = diagnostics.beam_width(psi, grid)
  assert float(width_x) == pytest.approx(2 * w0, rel=1e-3)
  assert float(width_y) == pytest.approx(2 * w0, rel=1e-3)


def test_rms_radius_of_a_gaussian(grid, x64):
  """sigma_xx = sigma_yy = w0**2 / 4, so the RMS radius is w0 / sqrt(2)."""
  w0 = 2.0
  psi = gaussian_beam(grid, w0=w0)
  expected = w0 / jnp.sqrt(2.0)
  assert float(diagnostics.rms_radius(psi, grid)) == pytest.approx(
    float(expected), rel=1e-3
  )


def test_second_moments_cross_term_vanishes_for_a_circular_beam(grid, x64):
  psi = gaussian_beam(grid, w0=2.0)
  _, _, sigma_xy = diagnostics.second_moments(psi, grid)
  assert abs(float(sigma_xy)) < 1e-6


# --------------------------------------------------------------------------
# Beam quality
# --------------------------------------------------------------------------

def test_m_squared_of_a_fundamental_gaussian_is_one(grid, x64):
  psi = gaussian_beam(grid, w0=2.0)
  m2_x, m2_y = diagnostics.m_squared(psi, grid)
  assert float(m2_x) == pytest.approx(1.0, abs=1e-3)
  assert float(m2_y) == pytest.approx(1.0, abs=1e-3)


def test_m_squared_is_invariant_under_propagation(grid, x64):
  """M2 is a propagation invariant, which is what the cross term buys.

  Taking the product of a real-space and a Fourier-space width instead would
  grow without bound away from the waist.
  """
  w0 = 2.0
  for z in (0.0, 4.0, 8.0):
    psi = gaussian_beam(grid, w0=w0, z=z, envelope_only=True)
    m2_x, _ = diagnostics.m_squared(psi, grid)
    assert float(m2_x) == pytest.approx(1.0, abs=2e-3), f"at z={z}"


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_m_squared_of_hermite_gaussian_modes(n, grid, x64):
  """HG_n0 has M2_x = 2n + 1, the textbook result."""
  psi = hermite_gaussian_beam(grid, w0=2.0, n=n, m=0)
  m2_x, m2_y = diagnostics.m_squared(psi, grid)
  assert float(m2_x) == pytest.approx(2 * n + 1, abs=0.02)
  assert float(m2_y) == pytest.approx(1.0, abs=0.02)


def test_m_squared_of_laguerre_gaussian_modes(x64):
  """LG_pl has M2 = 2p + |l| + 1 in both axes."""
  grid = SimulationConfig(
    nx=512, ny=512, dx=0.04, dy=0.04, dz=0.02, nz=10, wavelength=1.0
  )
  for p, l in ((0, 1), (1, 0), (0, 2)):
    psi = laguerre_gaussian_beam(grid, w0=2.0, p=p, l=l)
    m2_x, m2_y = diagnostics.m_squared(psi, grid)
    expected = 2 * p + abs(l) + 1
    assert float(m2_x) == pytest.approx(expected, rel=0.03), f"LG_{p}{l}"
    assert float(m2_y) == pytest.approx(expected, rel=0.03), f"LG_{p}{l}"


# --------------------------------------------------------------------------
# Comparisons against a reference beam
# --------------------------------------------------------------------------

def test_overlap_is_one_for_identical_fields_and_zero_for_orthogonal(grid, x64):
  fundamental = gaussian_beam(grid, w0=2.0)
  first_order = hermite_gaussian_beam(grid, w0=2.0, n=1, m=0)
  assert float(diagnostics.overlap(fundamental, fundamental)) == pytest.approx(
    1.0, abs=1e-9)
  assert abs(float(diagnostics.overlap(first_order, fundamental))) < 1e-8


def test_overlap_ignores_amplitude_and_global_phase(grid, x64):
  psi = gaussian_beam(grid, w0=2.0)
  scaled = 3.7 * jnp.exp(1.3j) * psi
  assert float(diagnostics.overlap(scaled, psi)) == pytest.approx(1.0, abs=1e-9)


def test_strehl_ratio_drops_under_aberration(grid, x64):
  """Adding wavefront curvature spreads the focus and lowers the peak."""
  reference = gaussian_beam(grid, w0=2.0)
  x = jnp.arange(grid.nx) * grid.dx - grid.lx / 2
  y = jnp.arange(grid.ny) * grid.dy - grid.ly / 2
  aberration = jnp.exp(1j * 0.5 * (x[:, None]**2 + y[None, :]**2))
  # Propagate both, so the aberration has somewhere to act.
  solver = ParaxialWaveSolver(grid, SPLIT, NO_PML)
  clean, _ = solver.solve(reference, return_history=False)
  aberrated, _ = solver.solve(reference * aberration, return_history=False)

  assert float(diagnostics.strehl_ratio(clean, clean)) == pytest.approx(1.0)
  assert float(diagnostics.strehl_ratio(aberrated, clean)) < 1.0


# --------------------------------------------------------------------------
# Scintillation and encircled power
# --------------------------------------------------------------------------

def test_scintillation_index_of_a_uniform_field_is_zero(x64):
  flat = jnp.ones((64, 64), dtype=complex)
  assert abs(float(diagnostics.scintillation_index(flat))) < 1e-12


def test_scintillation_index_grows_with_speckle(x64):
  key = jax.random.PRNGKey(0)
  smooth = jnp.ones((128, 128), dtype=complex)
  speckled = (jax.random.normal(key, (128, 128))
              + 1j * jax.random.normal(jax.random.PRNGKey(1), (128, 128)))
  assert (diagnostics.scintillation_index(speckled)
          > diagnostics.scintillation_index(smooth))


def test_ensemble_scintillation_index_of_fully_developed_speckle(x64):
  """Circular Gaussian speckle has an ensemble scintillation index of 1."""
  keys = jax.random.split(jax.random.PRNGKey(0), 2)
  real = jax.random.normal(keys[0], (4000, 16, 16))
  imaginary = jax.random.normal(keys[1], (4000, 16, 16))
  intensities = real**2 + imaginary**2
  index = diagnostics.ensemble_scintillation_index(intensities)
  assert float(jnp.mean(index)) == pytest.approx(1.0, abs=0.1)


def test_encircled_power_of_a_gaussian(grid, x64):
  """A Gaussian carries 1 - exp(-2) of its power inside r = w0."""
  w0 = 2.0
  psi = gaussian_beam(grid, w0=w0)
  fraction = float(diagnostics.encircled_power(psi, grid, w0))
  assert fraction == pytest.approx(float(1 - jnp.exp(-2.0)), abs=3e-3)
  assert float(diagnostics.encircled_power(psi, grid, 100.0)) == pytest.approx(
    1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Use inside the propagation loop
# --------------------------------------------------------------------------

def test_observable_fn_records_diagnostics_without_the_field_history(x64):
  """The history holds the observable's output instead of (nz, nx, ny)."""
  grid = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.02, nz=40, wavelength=1.0
  )
  psi_0 = gaussian_beam(grid, w0=2.0)
  solver = ParaxialWaveSolver(grid, SPLIT, NO_PML)

  def observable(psi, z):
    return diagnostics.beam_diagnostics(psi, grid)

  psi_final, history = solver.solve(psi_0, observable_fn=observable)
  psi_reference, fields = solver.solve(psi_0)

  assert jnp.allclose(psi_final, psi_reference)
  assert isinstance(history, dict)
  assert history['power'].shape == (grid.nz,)
  assert history['m2_x'].shape == (grid.nz,)
  # Index 0 is the initial plane, matching the field history's convention.
  assert float(history['power'][0]) == pytest.approx(
    float(diagnostics.total_power(psi_0, grid)), rel=1e-6)
  assert jnp.allclose(
    history['width_x'],
    jax.vmap(lambda f: diagnostics.beam_width(f, grid)[0])(fields),
    rtol=1e-5,
  )


def test_observable_fn_honours_save_every(x64):
  grid = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.02, nz=20, wavelength=1.0
  )
  psi_0 = gaussian_beam(grid, w0=2.0)
  solver = ParaxialWaveSolver(grid, SPLIT, NO_PML)
  _, history = solver.solve(
    psi_0, save_every=5,
    observable_fn=lambda psi, z: diagnostics.total_power(psi, grid),
  )
  assert history.shape == (4,)


def test_observable_fn_receives_the_plane_z(x64):
  grid = SimulationConfig(
    nx=32, ny=32, dx=0.2, dy=0.2, dz=0.1, nz=5, wavelength=1.0
  )
  psi_0 = gaussian_beam(grid, w0=2.0)
  solver = ParaxialWaveSolver(grid, SPLIT, NO_PML)
  _, zs = solver.solve(psi_0, z_0=3.0, observable_fn=lambda psi, z: z)
  assert jnp.allclose(zs, 3.0 + 0.1 * jnp.arange(5))


def test_diagnostics_are_jittable_and_vmappable(x64):
  """Every diagnostic is a pure function of the field, so both work."""
  grid = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.1, nz=4, wavelength=1.0
  )
  fields = jnp.stack([gaussian_beam(grid, w0=w) for w in (1.0, 2.0, 3.0)])
  widths = jax.jit(
    jax.vmap(lambda f: diagnostics.beam_width(f, grid)[0])
  )(fields)
  assert widths.shape == (3,)
  assert bool(jnp.all(jnp.diff(widths) > 0))


def test_scintillation_index_mask_restricts_the_average(x64):
  """The masked branch measures the beam, not the dark background around it.

  Without a mask the index is dominated by empty grid: <I> collapses towards
  zero while <I^2> is still set by the bright core, so a clean beam reads a
  large value. The mask is the whole reason the argument exists, and it was
  otherwise unexercised by any test.
  """
  grid = SimulationConfig(
    nx=128, ny=128, dx=0.05, dy=0.05, dz=0.1, nz=4, wavelength=1.0
  )
  psi = gaussian_beam(grid, w0=1.0)          # fills a small part of the grid
  illuminated = jnp.abs(psi)**2 > 0.01 * jnp.max(jnp.abs(psi)**2)

  whole_grid = float(diagnostics.scintillation_index(psi))
  masked = float(diagnostics.scintillation_index(psi, mask=illuminated))

  assert whole_grid > 5 * masked, (
    f"masking should cut the background contribution: {whole_grid} vs {masked}"
  )
  # A mask covering everything must reproduce the unmasked value.
  everywhere = jnp.ones_like(illuminated)
  assert diagnostics.scintillation_index(psi, mask=everywhere) == pytest.approx(
    whole_grid, rel=1e-9
  )


# --------------------------------------------------------------------------
# Ensemble statistics
# --------------------------------------------------------------------------

def _displaced_gaussians(grid, offsets, w0=1.0):
  """Returns an ensemble of identical Gaussians displaced along x."""
  x = jnp.arange(grid.nx) * grid.dx
  y = jnp.arange(grid.ny) * grid.dy
  centre_y = grid.ly / 2

  def one(offset):
    r2 = (x[:, None] - (grid.lx / 2 + offset))**2 + (y[None, :] - centre_y)**2
    return jnp.exp(-r2 / w0**2).astype(complex)

  return jax.vmap(one)(offsets)


def test_long_exposure_intensity_is_the_mean_of_the_intensities(x64):
  """The long-exposure beam is <|psi|^2>, averaged over realizations."""
  grid = SimulationConfig(
    nx=32, ny=32, dx=0.1, dy=0.1, dz=0.1, nz=4, wavelength=1.0
  )
  fields = _displaced_gaussians(grid, jnp.array([-0.3, 0.0, 0.5]))

  mean = diagnostics.long_exposure_intensity(fields)

  assert mean.shape == (32, 32)
  assert jnp.allclose(mean, jnp.mean(jnp.abs(fields)**2, axis=0), atol=1e-14)


def test_beam_wander_recovers_a_known_displacement(x64):
  """Wander of beams displaced by known offsets is the spread of the offsets.

  The centroid of realization n sits at its own displacement, so the standard
  deviation across realizations must reproduce std(offsets) exactly.
  """
  grid = SimulationConfig(
    nx=128, ny=128, dx=0.05, dy=0.05, dz=0.1, nz=4, wavelength=1.0
  )
  offsets = jnp.array([-0.4, -0.15, 0.0, 0.2, 0.35])
  fields = _displaced_gaussians(grid, offsets)

  wander_x, wander_y = diagnostics.beam_wander(fields, grid)

  assert float(wander_x) == pytest.approx(float(jnp.std(offsets)), rel=1e-6)
  # Nothing was displaced along y.
  assert float(wander_y) == pytest.approx(0.0, abs=1e-12)


def test_long_exposure_is_broader_than_short_exposure(x64):
  """Displacement widens the average beam by the law of total variance.

  Each realization has second moment w0^2 / 4; displacing them adds the
  variance of the displacements, so the long-exposure D4-sigma width is
  4 * sqrt(w0^2 / 4 + var(offsets)) while the short-exposure width stays at
  2 * w0. This pins the sign of the effect and its magnitude.
  """
  grid = SimulationConfig(
    nx=256, ny=256, dx=0.04, dy=0.04, dz=0.1, nz=4, wavelength=1.0
  )
  w0 = 1.0
  offsets = jnp.array([-0.6, -0.3, 0.0, 0.3, 0.6])
  fields = _displaced_gaussians(grid, offsets, w0=w0)

  stats = diagnostics.ensemble_statistics(fields, grid)
  variance = float(jnp.var(offsets))

  assert float(stats['short_exposure_width_x']) == pytest.approx(
    2 * w0, rel=1e-4
  )
  assert float(stats['long_exposure_width_x']) == pytest.approx(
    4 * jnp.sqrt(w0**2 / 4 + variance), rel=1e-3
  )
  assert stats['long_exposure_width_x'] > stats['short_exposure_width_x']


def test_mutual_coherence_of_a_deterministic_ensemble_is_one(x64):
  """Identical realizations are perfectly coherent everywhere."""
  grid = SimulationConfig(
    nx=32, ny=32, dx=0.1, dy=0.1, dz=0.1, nz=4, wavelength=1.0
  )
  one = gaussian_beam(grid, w0=1.0)
  fields = jnp.stack([one] * 6)

  mu = diagnostics.mutual_coherence(fields)

  assert jnp.allclose(mu, 1.0, atol=1e-10)
  assert jnp.isinf(diagnostics.coherence_radius(fields, grid))


def test_mutual_coherence_decays_for_independent_phases(x64):
  """Independent random phase per point destroys coherence off the peak.

  The reference point stays at 1 by construction; every other point averages
  an independent unit phasor over the ensemble and so decays as
  1 / sqrt(n_realizations).
  """
  grid = SimulationConfig(
    nx=16, ny=16, dx=0.1, dy=0.1, dz=0.1, nz=4, wavelength=1.0
  )
  n_realizations = 400
  key = jax.random.PRNGKey(0)
  phases = jax.random.uniform(
    key, (n_realizations, grid.nx, grid.ny), maxval=2 * jnp.pi
  )
  fields = jnp.exp(1j * phases)

  mu = diagnostics.mutual_coherence(fields)
  index_x, index_y = jnp.unravel_index(
    jnp.argmax(jnp.mean(jnp.abs(fields)**2, axis=0)), (grid.nx, grid.ny)
  )

  assert float(mu[index_x, index_y]) == pytest.approx(1.0, rel=1e-10)
  off_peak = jnp.delete(mu.ravel(), index_x * grid.ny + index_y)
  # Mean |mu| off the peak is of order 1 / sqrt(N) ~ 0.05, not order 1.
  assert float(jnp.mean(off_peak)) < 10 / jnp.sqrt(n_realizations)
  # The decorrelation is resolved at the first neighbouring cell.
  assert float(diagnostics.coherence_radius(fields, grid)) == pytest.approx(
    grid.dx, rel=1e-6
  )


def test_on_axis_scintillation_index_of_speckle_is_one(x64):
  """Fully developed speckle has a unit scintillation index on axis.

  Circular complex Gaussian field amplitude gives exponentially distributed
  intensity, for which <I^2> / <I>^2 = 2.
  """
  grid = SimulationConfig(
    nx=8, ny=8, dx=0.1, dy=0.1, dz=0.1, nz=4, wavelength=1.0
  )
  key = jax.random.PRNGKey(1)
  real, imaginary = jax.random.normal(key, (2, 20000, grid.nx, grid.ny))
  fields = (real + 1j * imaginary) / jnp.sqrt(2)

  stats = diagnostics.ensemble_statistics(fields, grid)

  assert float(stats['on_axis_scintillation_index']) == pytest.approx(
    1.0, rel=0.1
  )


def test_ensemble_statistics_returns_every_documented_key(x64):
  """The bundle's contract: scalars are scalars and maps are maps."""
  grid = SimulationConfig(
    nx=32, ny=32, dx=0.1, dy=0.1, dz=0.1, nz=4, wavelength=1.0
  )
  fields = _displaced_gaussians(grid, jnp.array([-0.2, 0.0, 0.3, 0.1]))

  stats = diagnostics.ensemble_statistics(fields, grid)

  scalars = (
    'on_axis_scintillation_index', 'wander_x', 'wander_y', 'coherence_radius',
    'long_exposure_width_x', 'long_exposure_width_y',
    'short_exposure_width_x', 'short_exposure_width_y',
  )
  maps = ('mean_intensity', 'scintillation_index')
  assert set(stats) == set(scalars) | set(maps)
  for name in scalars:
    assert jnp.ndim(stats[name]) == 0, name
  for name in maps:
    assert stats[name].shape == (32, 32), name


def test_ensemble_statistics_is_jittable(x64):
  """The bundle compiles, so it can be evaluated inside a traced function.

  Uses a grid wide enough for the w0 = 1 beam to decay before the boundary;
  on a narrower one, truncation biases the centroid and the wander check
  below would be measuring the grid rather than the displacement.
  """
  grid = SimulationConfig(
    nx=128, ny=128, dx=0.05, dy=0.05, dz=0.1, nz=4, wavelength=1.0
  )
  fields = _displaced_gaussians(grid, jnp.array([-0.2, 0.0, 0.3]))

  compiled = jax.jit(lambda f: diagnostics.ensemble_statistics(f, grid))
  stats = compiled(fields)

  assert float(stats['wander_x']) == pytest.approx(
    float(jnp.std(jnp.array([-0.2, 0.0, 0.3]))), rel=1e-6
  )
