
import jax
import jax.numpy as jnp
import pytest
from scipy.special import genlaguerre, hermite

from paraxial_wave_solver.src.config import (
  PMLConfig,
  SimulationConfig,
  SolverConfig,
)
from paraxial_wave_solver.src.solvers import ParaxialWaveSolver
from paraxial_wave_solver.src.utils import (
  _hermite_h,
  _laguerre_l,
  gaussian_beam,
  hermite_gaussian_beam,
  laguerre_gaussian_beam,
)

NO_PML = PMLConfig(width_x=0, width_y=0, strength=0.0)


def _power(psi, sim_config):
  return jnp.sum(jnp.abs(psi)**2) * sim_config.dx * sim_config.dy


# --------------------------------------------------------------------------
# Special function evaluation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 5, 10, 20])
def test_hermite_recurrence_matches_scipy(n, x64):
  """The recurrence agrees with scipy, which the coefficient form stops doing.

  hermite(20) has coefficients reaching 4e13; evaluating through them loses
  most of the float32 mantissa to cancellation.
  """
  x = jnp.linspace(-3.0, 3.0, 101)
  expected = jnp.polyval(jnp.asarray(hermite(n).coef), x)
  assert jnp.allclose(_hermite_h(n, x), expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("p,alpha", [(0, 0), (1, 0), (3, 2), (5, 1), (8, 4)])
def test_laguerre_recurrence_matches_scipy(p, alpha, x64):
  x = jnp.linspace(0.0, 8.0, 101)
  expected = jnp.polyval(jnp.asarray(genlaguerre(p, alpha).coef), x)
  assert jnp.allclose(_laguerre_l(p, alpha, x), expected, rtol=1e-8, atol=1e-8)


def test_hermite_high_order_is_stable_in_float32():
  """High-order HG modes stay accurate in float32 via the recurrence."""
  import numpy as np

  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0
  )
  psi32 = np.asarray(hermite_gaussian_beam(sim_config, w0=3.0, n=20, m=0))

  jax.config.update("jax_enable_x64", True)
  try:
    psi64 = np.asarray(hermite_gaussian_beam(sim_config, w0=3.0, n=20, m=0))
  finally:
    jax.config.update("jax_enable_x64", False)

  rel = (np.linalg.norm(psi32.astype(np.complex128) - psi64)
         / np.linalg.norm(psi64))
  assert rel < 1e-5, f"float32 HG_20 differs from float64 by {rel}"


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

@pytest.mark.parametrize("w0", [1.0, 2.0, 3.0])
def test_lg_and_hg_carry_unit_power(w0, x64):
  """LG and HG modes are power-normalized independently of the waist.

  They used to carry w0**2 instead of 1, because the amplitude prefactor was
  written w0 / w(z) rather than 1 / w(z).
  """
  sim_config = SimulationConfig(
    nx=512, ny=512, dx=0.05, dy=0.05, dz=0.1, nz=10, wavelength=1.0
  )
  for psi in (laguerre_gaussian_beam(sim_config, w0=w0, p=0, l=0),
              hermite_gaussian_beam(sim_config, w0=w0, n=0, m=0)):
    assert jnp.isclose(_power(psi, sim_config), 1.0, rtol=1e-4)


@pytest.mark.parametrize("mode", [(1, 0), (0, 2), (2, 3)])
def test_hg_higher_modes_carry_unit_power(mode, x64):
  n, m = mode
  sim_config = SimulationConfig(
    nx=512, ny=512, dx=0.05, dy=0.05, dz=0.1, nz=10, wavelength=1.0
  )
  psi = hermite_gaussian_beam(sim_config, w0=2.0, n=n, m=m)
  assert jnp.isclose(_power(psi, sim_config), 1.0, rtol=1e-3)


@pytest.mark.parametrize("mode", [(0, 1), (1, 0), (1, 2), (2, -3)])
def test_lg_higher_modes_carry_unit_power(mode, x64):
  p, l = mode
  sim_config = SimulationConfig(
    nx=512, ny=512, dx=0.05, dy=0.05, dz=0.1, nz=10, wavelength=1.0
  )
  psi = laguerre_gaussian_beam(sim_config, w0=2.0, p=p, l=l)
  assert jnp.isclose(_power(psi, sim_config), 1.0, rtol=1e-3)


def test_gaussian_beam_is_peak_normalized():
  """gaussian_beam keeps its peak-1 convention; the docstring says so."""
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0
  )
  psi = gaussian_beam(sim_config, w0=2.0)
  assert jnp.isclose(jnp.abs(psi).max(), 1.0, rtol=1e-5)


def test_gaussian_beam_power_scaling():
  """Requesting a total power rescales the beam to exactly that power."""
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0
  )
  psi = gaussian_beam(sim_config, w0=1.0, power=2.5)
  assert jnp.isclose(_power(psi, sim_config), 2.5, rtol=1e-5)


def test_laguerre_gaussian_power_normalization():
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.05, dy=0.05, dz=0.1, nz=10,
    wavelength=0.6328e-6, n0=1.33,
  )
  psi = laguerre_gaussian_beam(
    sim_config, w0=0.5, p=1, l=2, z=1.0, power=0.05
  )
  assert jnp.isclose(_power(psi, sim_config), 0.05, rtol=1e-5)


# --------------------------------------------------------------------------
# Mode relationships
# --------------------------------------------------------------------------

def test_lg_p0_l0_matches_gaussian(x64):
  """LG_00 and the fundamental Gaussian agree once both are power-normalized."""
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0
  )
  kwargs = dict(w0=1.5, z=3.0, power=1.0, envelope_only=True)
  psi_lg = laguerre_gaussian_beam(sim_config, p=0, l=0, **kwargs)
  psi_gauss = gaussian_beam(sim_config, **kwargs)
  assert jnp.allclose(psi_lg, psi_gauss, atol=1e-8)


def test_hermite_gaussian_n0_m0_matches_gaussian(x64):
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0
  )
  kwargs = dict(w0=1.5, z=2.5, power=1.0, envelope_only=True)
  psi_hg = hermite_gaussian_beam(sim_config, n=0, m=0, **kwargs)
  psi_gauss = gaussian_beam(sim_config, **kwargs)
  assert jnp.allclose(psi_hg, psi_gauss, atol=1e-8)


def test_lg00_and_hg00_agree_without_power_normalization(x64):
  """The two families share one amplitude convention, not two."""
  sim_config = SimulationConfig(
    nx=256, ny=256, dx=0.08, dy=0.08, dz=0.1, nz=10, wavelength=1.0
  )
  psi_lg = laguerre_gaussian_beam(sim_config, w0=2.0, p=0, l=0, z=1.0)
  psi_hg = hermite_gaussian_beam(sim_config, w0=2.0, n=0, m=0, z=1.0)
  assert jnp.allclose(psi_lg, psi_hg, atol=1e-10)


def test_envelope_only_removes_carrier():
  """envelope_only omits exactly exp(1j * k * z)."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0, n0=1.5
  )
  z = 2.0
  psi_full = laguerre_gaussian_beam(sim_config, w0=1.0, p=0, l=1, z=z)
  psi_env = laguerre_gaussian_beam(
    sim_config, w0=1.0, p=0, l=1, z=z, envelope_only=True
  )
  reconstructed = psi_env * jnp.exp(1j * sim_config.k * z)
  assert jnp.max(jnp.abs(psi_full - reconstructed)) < 1e-6


def test_beams_are_continuous_through_z_zero(x64):
  """The q-parameter form has no special case at z = 0."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0
  )
  eps = 1e-9
  for beam, kwargs in [
    (gaussian_beam, dict(w0=1.5)),
    (laguerre_gaussian_beam, dict(w0=1.5, p=1, l=2)),
    (hermite_gaussian_beam, dict(w0=1.5, n=1, m=2)),
  ]:
    at_zero = beam(sim_config, z=0.0, envelope_only=True, **kwargs)
    just_after = beam(sim_config, z=eps, envelope_only=True, **kwargs)
    assert jnp.all(jnp.isfinite(jnp.abs(at_zero)))
    assert jnp.allclose(at_zero, just_after, atol=1e-6)


def test_beams_are_jittable_and_vmappable_over_z():
  """No Python branching on z, so the generators trace and vectorize."""
  sim_config = SimulationConfig(
    nx=32, ny=32, dx=0.2, dy=0.2, dz=0.1, nz=10, wavelength=1.0
  )

  def make(z):
    return laguerre_gaussian_beam(
      sim_config, w0=1.5, p=0, l=1, z=z, envelope_only=True
    )

  zs = jnp.linspace(0.0, 2.0, 5)
  stacked = jax.vmap(make)(zs)
  assert stacked.shape == (5, 32, 32)
  assert jnp.allclose(jax.jit(make)(1.0), make(1.0), atol=1e-6)
  assert jnp.allclose(stacked[0], make(0.0), atol=1e-6)


# --------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [(0, 1), (1, 4), (0, -6), (1, 9)])
def test_lg_propagation_matches_solver(mode, x64):
  """Every LG mode propagated in vacuum matches its analytical envelope."""
  p, l = mode
  sim_config = SimulationConfig(
    nx=256, ny=256, dx=0.08, dy=0.08, dz=0.05, nz=40, wavelength=1.0
  )
  w0 = 1.5
  psi_0 = laguerre_gaussian_beam(sim_config, w0=w0, p=p, l=l, z=0.0)

  solver = ParaxialWaveSolver(
    sim_config, SolverConfig(method='spectral', stepper='split_step'),
    PMLConfig(width_x=20, width_y=20, strength=2.0),
  )
  psi_final, _ = solver.solve(psi_0, return_history=False)

  psi_analytical = laguerre_gaussian_beam(
    sim_config, w0=w0, p=p, l=l, z=sim_config.lz, envelope_only=True
  )
  rel_err = (jnp.linalg.norm(psi_final - psi_analytical)
             / jnp.linalg.norm(psi_analytical))
  assert rel_err < 1e-3, f"LG_{p},{l} rel err {rel_err}"


def test_lg_propagation_at_physical_wavelength(x64):
  """A wavelength where a spurious carrier would not cancel.

  With lambda = 1 and an integer propagation distance, exp(1j * k0 * lz) is a
  multiple of 2 pi and hides an incorrect refractive index convention. This
  configuration does not.
  """
  sim_config = SimulationConfig(
    nx=256, ny=256, dx=0.015, dy=0.015, dz=1.0, nz=250,
    wavelength=632.8e-7,
  )
  w0 = 3.0e-1
  psi_0 = laguerre_gaussian_beam(sim_config, w0, p=0, l=1, z=0.0)

  solver = ParaxialWaveSolver(
    sim_config, SolverConfig(method='spectral', stepper='split_step'),
    PMLConfig(width_x=20, width_y=20, strength=2.0),
  )
  psi_final, _ = solver.solve(psi_0, return_history=False)

  psi_analytical = laguerre_gaussian_beam(
    sim_config, w0, p=0, l=1, z=sim_config.lz, envelope_only=True
  )
  rel_err = (jnp.linalg.norm(psi_final - psi_analytical)
             / jnp.linalg.norm(psi_analytical))
  assert rel_err < 1e-4, f"rel err {rel_err}"


def test_hg_propagation_matches_solver(x64):
  sim_config = SimulationConfig(
    nx=256, ny=256, dx=0.08, dy=0.08, dz=0.05, nz=40, wavelength=1.0
  )
  w0 = 1.5
  psi_0 = hermite_gaussian_beam(sim_config, w0=w0, n=1, m=1, z=0.0)

  solver = ParaxialWaveSolver(
    sim_config, SolverConfig(method='spectral', stepper='split_step'),
    PMLConfig(width_x=20, width_y=20, strength=2.0),
  )
  psi_final, _ = solver.solve(psi_0, return_history=False)

  psi_analytical = hermite_gaussian_beam(
    sim_config, w0=w0, n=1, m=1, z=sim_config.lz, envelope_only=True
  )
  rel_err = (jnp.linalg.norm(psi_final - psi_analytical)
             / jnp.linalg.norm(psi_analytical))
  assert rel_err < 1e-3
