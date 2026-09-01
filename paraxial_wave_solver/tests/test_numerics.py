"""Tests for the optional numerical schemes.

Covers the higher-order splitting composition, the wide-angle propagator, the
Kerr term, complex refractive index, and dealiasing. Each also checks that the
default configuration is unchanged, since all of these are opt-in.
"""

import jax
import jax.numpy as jnp
import pytest

from paraxial_wave_solver.src.config import (
  PMLConfig,
  SimulationConfig,
  SolverConfig,
)
from paraxial_wave_solver.src.operators import get_spectral_k_grids
from paraxial_wave_solver.src.solvers import (
  ParaxialWaveSolver,
  splitting_weights,
)
from paraxial_wave_solver.src.utils import gaussian_beam

NO_PML = PMLConfig(width_x=0, width_y=0, strength=0.0)
SPLIT = SolverConfig(method='spectral', stepper='split_step')


def _gaussian(sim_config, width=2.0):
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  r2 = ((x[:, None] - sim_config.lx / 2)**2
        + (y[None, :] - sim_config.ly / 2)**2)
  return jnp.exp(-r2 / width**2).astype(complex)


def _second_moment_width(sim_config, psi):
  """Returns the intensity-weighted RMS radius, a diffraction-free width."""
  x = jnp.arange(sim_config.nx) * sim_config.dx - sim_config.lx / 2
  y = jnp.arange(sim_config.ny) * sim_config.dy - sim_config.ly / 2
  intensity = jnp.abs(psi)**2
  total = jnp.sum(intensity)
  r2 = x[:, None]**2 + y[None, :]**2
  return float(jnp.sqrt(jnp.sum(r2 * intensity) / total))


# --------------------------------------------------------------------------
# Backwards compatibility
# --------------------------------------------------------------------------

def test_new_options_default_to_previous_behaviour():
  """Every added field defaults to the scheme that shipped before them."""
  cfg = SolverConfig(method='spectral', stepper='split_step')
  assert cfg.splitting_order == 2
  assert cfg.propagator == 'paraxial'
  assert cfg.dealias is False
  assert SimulationConfig(
    nx=8, ny=8, dx=1.0, dy=1.0, dz=1.0, nz=2, wavelength=1.0
  ).n2 == 0.0


def test_simulation_config_still_accepts_positional_n0():
  """n2 was appended, so existing positional construction is unaffected."""
  cfg = SimulationConfig(8, 8, 0.1, 0.1, 0.1, 4, 1.0, 1.33)
  assert cfg.n0 == 1.33
  assert cfg.n2 == 0.0


def test_splitting_order_two_matches_a_plain_strang_step():
  """Order 2 is a single Strang step, unchanged from the original kernel."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.02, nz=20, wavelength=1.0
  )
  psi_0 = _gaussian(sim_config)
  explicit = SolverConfig(method='spectral', stepper='split_step',
                          splitting_order=2)
  a, _ = ParaxialWaveSolver(sim_config, SPLIT, NO_PML).solve(
    psi_0, return_history=False)
  b, _ = ParaxialWaveSolver(sim_config, explicit, NO_PML).solve(
    psi_0, return_history=False)
  assert jnp.array_equal(a, b)


# --------------------------------------------------------------------------
# Higher-order splitting
# --------------------------------------------------------------------------

def test_splitting_weights_sum_to_one():
  assert splitting_weights(2) == (1.0,)
  weights = splitting_weights(4)
  assert len(weights) == 3
  assert weights[0] == weights[2]
  assert weights[1] < 0.0          # The middle sub-step runs backwards.
  assert sum(weights) == pytest.approx(1.0, abs=1e-12)


def test_splitting_weights_rejects_unknown_order():
  with pytest.raises(ValueError, match="Unsupported splitting_order"):
    splitting_weights(3)


def test_yoshida_composition_is_fourth_order(x64):
  """Order 4 converges as dz**4, against an independently computed reference."""
  nx = ny = 64
  dx = dy = 0.2
  lz = 20.0

  def delta_n_fn(z, medium):
    x = jnp.arange(nx) * dx
    y = jnp.arange(ny) * dy
    return (0.5 * jnp.cos(x[:, None] * 0.8) * jnp.cos(y[None, :] * 0.8)
            * jnp.cos(0.3 * z))

  def make(nz):
    return SimulationConfig(nx=nx, ny=ny, dx=dx, dy=dy, dz=lz / nz, nz=nz,
                            wavelength=1.0)

  psi_0 = _gaussian(make(1))

  def run(nz, order):
    cfg = SolverConfig(method='spectral', stepper='split_step',
                       splitting_order=order)
    return ParaxialWaveSolver(make(nz), cfg, NO_PML, delta_n_fn).solve(
      psi_0, return_history=False)[0]

  # Reference uses order 2 at a very fine step, so it is independent of the
  # scheme under test.
  reference = run(20480, 2)
  errors = [
    float(jnp.linalg.norm(run(nz, 4) - reference)
          / jnp.linalg.norm(reference))
    for nz in (320, 640, 1280)
  ]
  for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
    rate = jnp.log2(coarse / fine)
    assert rate > 3.2, f"measured order {rate}, errors {errors}"


def test_yoshida_beats_strang_at_the_same_step(x64):
  """At a step where both are in their asymptotic regime, order 4 is ahead."""
  nx = ny = 64
  dx = dy = 0.2
  lz = 20.0

  def delta_n_fn(z, medium):
    x = jnp.arange(nx) * dx
    y = jnp.arange(ny) * dy
    return 0.5 * jnp.cos(x[:, None] * 0.8) * jnp.cos(y[None, :] * 0.8)

  def make(nz):
    return SimulationConfig(nx=nx, ny=ny, dx=dx, dy=dy, dz=lz / nz, nz=nz,
                            wavelength=1.0)

  psi_0 = _gaussian(make(1))

  def run(nz, order):
    cfg = SolverConfig(method='spectral', stepper='split_step',
                       splitting_order=order)
    return ParaxialWaveSolver(make(nz), cfg, NO_PML, delta_n_fn).solve(
      psi_0, return_history=False)[0]

  reference = run(20480, 2)
  norm = jnp.linalg.norm(reference)
  # Far enough into the asymptotic regime that the extra two orders show;
  # at coarser steps the larger constant of the composition still dominates.
  err2 = float(jnp.linalg.norm(run(1280, 2) - reference) / norm)
  err4 = float(jnp.linalg.norm(run(1280, 4) - reference) / norm)
  assert err4 < err2 / 20, f"order 2 {err2:.3e}, order 4 {err4:.3e}"


# --------------------------------------------------------------------------
# Wide-angle propagator
# --------------------------------------------------------------------------

def test_wide_angle_matches_paraxial_for_a_narrow_beam(x64):
  """The two agree when the beam is genuinely paraxial.

  The domain has to be wide enough that the beam does not touch the periodic
  boundary: wrap-around puts energy at high transverse wavenumbers, where the
  two propagators legitimately disagree, and that swamps the comparison.
  """
  sim_config = SimulationConfig(
    nx=256, ny=256, dx=0.1, dy=0.1, dz=0.02, nz=50, wavelength=1.0
  )
  psi_0 = gaussian_beam(sim_config, w0=4.0)   # theta ~ 0.08 rad
  assert float(jnp.abs(psi_0)[0, :].max()) < 1e-4, "beam must not reach the edge"

  wide = SolverConfig(method='spectral', stepper='split_step',
                      propagator='wide_angle')
  a, _ = ParaxialWaveSolver(sim_config, SPLIT, NO_PML).solve(
    psi_0, return_history=False)
  b, _ = ParaxialWaveSolver(sim_config, wide, NO_PML).solve(
    psi_0, return_history=False)
  assert float(jnp.linalg.norm(b - a) / jnp.linalg.norm(a)) < 1e-4


def test_wide_angle_departs_from_paraxial_as_theta_to_the_fourth(x64):
  """The correction is the next term of the square root, so it scales as θ⁴."""
  sim_config = SimulationConfig(
    nx=256, ny=256, dx=0.05, dy=0.05, dz=0.02, nz=50, wavelength=1.0
  )
  wide = SolverConfig(method='spectral', stepper='split_step',
                      propagator='wide_angle')

  differences = []
  for w0 in (0.8, 1.2, 1.6):
    psi_0 = gaussian_beam(sim_config, w0=w0)
    a, _ = ParaxialWaveSolver(sim_config, SPLIT, NO_PML).solve(
      psi_0, return_history=False)
    b, _ = ParaxialWaveSolver(sim_config, wide, NO_PML).solve(
      psi_0, return_history=False)
    differences.append(
      float(jnp.linalg.norm(b - a) / jnp.linalg.norm(a))
    )

  # Divergence angle scales as 1/w0, so each widening should shrink the
  # difference by roughly (w0_ratio)**4.
  assert differences[0] > differences[1] > differences[2]
  assert differences[0] / differences[1] > 2.0
  assert differences[1] / differences[2] > 2.0


def test_wide_angle_decays_evanescent_components(x64):
  """Past the light line the square root is imaginary, so modes must decay."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=2.0, dy=2.0, dz=1.0, nz=1, wavelength=8.0
  )
  wide = SolverConfig(method='spectral', stepper='split_step',
                      propagator='wide_angle')
  solver = ParaxialWaveSolver(sim_config, wide, NO_PML)
  operator = solver._operators['linear'][0]

  kx, ky = get_spectral_k_grids(
    sim_config.nx, sim_config.ny, sim_config.dx, sim_config.dy
  )
  k = sim_config.k0 * sim_config.n0
  evanescent = (kx**2 + ky**2) > k**2

  assert bool(jnp.any(evanescent)), "test grid must reach past the light line"
  assert bool(jnp.all(jnp.isfinite(jnp.abs(operator))))
  # Propagating modes keep unit magnitude; evanescent ones are damped.
  assert jnp.allclose(jnp.abs(operator[~evanescent]), 1.0, atol=1e-6)
  assert float(jnp.max(jnp.abs(operator[evanescent]))) < 1.0


# --------------------------------------------------------------------------
# Kerr nonlinearity
# --------------------------------------------------------------------------

def test_kerr_is_inert_when_n2_is_zero():
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.02, nz=20, wavelength=1.0
  )
  explicit = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.02, nz=20, wavelength=1.0, n2=0.0
  )
  psi_0 = _gaussian(sim_config)
  a, _ = ParaxialWaveSolver(sim_config, SPLIT, NO_PML).solve(
    psi_0, return_history=False)
  b, _ = ParaxialWaveSolver(explicit, SPLIT, NO_PML).solve(
    psi_0, return_history=False)
  assert jnp.array_equal(a, b)


def test_kerr_conserves_power(x64):
  """The Kerr term is a pure phase, so it cannot change the total power."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.01, nz=50, wavelength=1.0, n2=0.05
  )
  psi_0 = _gaussian(sim_config)
  psi_final, _ = ParaxialWaveSolver(sim_config, SPLIT, NO_PML).solve(
    psi_0, return_history=False)
  power_in = jnp.sum(jnp.abs(psi_0)**2)
  power_out = jnp.sum(jnp.abs(psi_final)**2)
  assert jnp.abs(power_in - power_out) / power_in < 1e-10


@pytest.mark.parametrize("stepper,extra", [
  ('split_step', {}),
  ('rk4', {}),
])
def test_kerr_self_focuses_and_self_defocuses(stepper, extra, x64):
  """Positive n2 contracts the beam relative to linear diffraction; negative
  n2 expands it. Both steppers implement the same term."""
  def build(n2):
    return SimulationConfig(
      nx=128, ny=128, dx=0.1, dy=0.1, dz=0.002, nz=250, wavelength=1.0, n2=n2
    )

  cfg = SolverConfig(method='spectral', stepper=stepper, **extra)
  widths = {}
  for label, n2 in (('focus', 0.02), ('linear', 0.0), ('defocus', -0.02)):
    sim_config = build(n2)
    psi_0 = _gaussian(sim_config, width=1.5) * 2.0   # enough intensity to act
    psi_final, _ = ParaxialWaveSolver(sim_config, cfg, NO_PML).solve(
      psi_0, return_history=False)
    widths[label] = _second_moment_width(sim_config, psi_final)

  assert widths['focus'] < widths['linear'] < widths['defocus'], widths


# --------------------------------------------------------------------------
# Complex refractive index
# --------------------------------------------------------------------------

def test_complex_delta_n_absorbs_at_the_analytic_rate(x64):
  """A uniform imaginary index gives power decay exp(-2 k0 * Im(dn) * L)."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.02, nz=50, wavelength=1.0
  )
  absorption = 0.02
  psi_0 = _gaussian(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, NO_PML, lambda z, medium: 1j * absorption
  )
  psi_final, _ = solver.solve(psi_0, return_history=False)

  ratio = float(jnp.sum(jnp.abs(psi_final)**2) / jnp.sum(jnp.abs(psi_0)**2))
  expected = float(jnp.exp(-2 * sim_config.k0 * absorption * sim_config.lz))
  assert ratio == pytest.approx(expected, rel=1e-6)


def test_complex_delta_n_can_provide_gain(x64):
  """The opposite sign amplifies rather than absorbs."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.02, nz=20, wavelength=1.0
  )
  psi_0 = _gaussian(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, NO_PML, lambda z, medium: -0.005j
  )
  psi_final, _ = solver.solve(psi_0, return_history=False)
  assert jnp.sum(jnp.abs(psi_final)**2) > jnp.sum(jnp.abs(psi_0)**2)


def test_complex_delta_n_works_with_rk4(x64):
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.005, nz=100, wavelength=1.0
  )
  cfg = SolverConfig(method='finite_difference', fd_order=4, stepper='rk4')
  psi_0 = _gaussian(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, cfg, NO_PML, lambda z, medium: 0.02j
  )
  psi_final, _ = solver.solve(psi_0, return_history=False)
  assert jnp.sum(jnp.abs(psi_final)**2) < jnp.sum(jnp.abs(psi_0)**2)
  assert bool(jnp.all(jnp.isfinite(jnp.abs(psi_final))))


# --------------------------------------------------------------------------
# Dealiasing
# --------------------------------------------------------------------------

def test_dealias_removes_out_of_band_energy(x64):
  """The 2/3 rule empties the upper third of each wavenumber axis."""
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.02, nz=10, wavelength=1.0
  )
  kx, ky = get_spectral_k_grids(
    sim_config.nx, sim_config.ny, sim_config.dx, sim_config.dy
  )
  out_of_band = ((jnp.abs(kx) > (2 / 3) * jnp.pi / sim_config.dx)
                 | (jnp.abs(ky) > (2 / 3) * jnp.pi / sim_config.dy))

  # White noise: most of its energy is out of band.
  psi_0 = jax.random.normal(
    jax.random.PRNGKey(0), (sim_config.nx, sim_config.ny)
  ).astype(complex)

  def out_of_band_fraction(psi):
    spectrum = jnp.fft.fft2(psi)
    return float(jnp.linalg.norm(spectrum[out_of_band])
                 / jnp.linalg.norm(spectrum))

  assert out_of_band_fraction(psi_0) > 0.5

  dealiased = SolverConfig(method='spectral', stepper='split_step',
                           dealias=True)
  psi_plain, _ = ParaxialWaveSolver(sim_config, SPLIT, NO_PML).solve(
    psi_0, return_history=False)
  psi_clean, _ = ParaxialWaveSolver(sim_config, dealiased, NO_PML).solve(
    psi_0, return_history=False)

  assert out_of_band_fraction(psi_plain) > 0.5
  assert out_of_band_fraction(psi_clean) < 1e-12


# --------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
  dict(method='spectral', stepper='rk4', splitting_order=4),
  dict(method='spectral', stepper='rk4', propagator='wide_angle'),
  dict(method='spectral', stepper='rk4', dealias=True),
  dict(method='finite_difference', stepper='rk4', dealias=True),
])
def test_split_step_only_options_are_rejected_elsewhere(kwargs):
  with pytest.raises(ValueError, match="requires stepper='split_step'"):
    SolverConfig(**kwargs)


def test_unsupported_splitting_order_rejected():
  with pytest.raises(ValueError, match="Unsupported splitting_order"):
    SolverConfig(method='spectral', stepper='split_step', splitting_order=3)


def test_unsupported_propagator_rejected():
  with pytest.raises(ValueError, match="Unsupported propagator"):
    SolverConfig(method='spectral', stepper='split_step', propagator='pade')
