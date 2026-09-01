import jax
import jax.numpy as jnp
import pytest

from paraxial_wave_solver.src.config import SimulationConfig, SolverConfig, PMLConfig
from paraxial_wave_solver.src.solvers import ParaxialWaveSolver
from paraxial_wave_solver.src.utils import (
  gaussian_beam,
  laguerre_gaussian_beam,
  hermite_gaussian_beam,
)


def test_gaussian_beam_properties():
  """Test Gaussian beam generation and power scaling."""
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0, n0=1.0
  )
  w0 = 1.0
  target_power = 2.5
  psi = gaussian_beam(sim_config, w0=w0, power=target_power)

  calculated_power = jnp.sum(jnp.abs(psi)**2) * sim_config.dx * sim_config.dy
  assert jnp.isclose(calculated_power, target_power, rtol=1e-5)


def test_laguerre_gaussian_power_normalization():
  """Test that power normalization correctly scales Laguerre-Gaussian beam."""
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.05, dy=0.05, dz=0.1, nz=10, wavelength=0.6328e-6, n0=1.33
  )
  w0 = 0.5
  target_power = 0.05
  psi = laguerre_gaussian_beam(
    sim_config, w0=w0, p=1, l=2, z=1.0, power=target_power
  )

  calculated_power = jnp.sum(jnp.abs(psi)**2) * sim_config.dx * sim_config.dy
  assert jnp.isclose(calculated_power, target_power, rtol=1e-5)


def test_laguerre_gaussian_envelope_vs_full():
  """Test that envelope_only removes the fast carrier phase exp(i*k*z)."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0, n0=1.5
  )
  w0 = 1.0
  z = 2.0
  k = sim_config.k

  psi_full = laguerre_gaussian_beam(sim_config, w0=w0, p=0, l=1, z=z, envelope_only=False)
  psi_env = laguerre_gaussian_beam(sim_config, w0=w0, p=0, l=1, z=z, envelope_only=True)

  # Check that psi_full == psi_env * exp(i * k * z).
  reconstructed_full = psi_env * jnp.exp(1j * k * z)
  diff = jnp.max(jnp.abs(psi_full - reconstructed_full))
  assert diff < 1e-6


def test_lg_p0_l0_matches_gaussian():
  """Test that LG_{0,0} matches fundamental Gaussian beam up to constant factor."""
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0, n0=1.0
  )
  w0 = 1.5
  z = 3.0
  power = 1.0

  psi_lg = laguerre_gaussian_beam(
    sim_config, w0=w0, p=0, l=0, z=z, power=power, envelope_only=True
  )
  psi_gauss = gaussian_beam(
    sim_config, w0=w0, z=z, power=power, envelope_only=True
  )

  # The normalized intensity profiles must match.
  diff = jnp.max(jnp.abs(jnp.abs(psi_lg)**2 - jnp.abs(psi_gauss)**2))
  assert diff < 1e-5


def test_hermite_gaussian_n0_m0_matches_gaussian():
  """Test that HG_{0,0} matches fundamental Gaussian beam."""
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0, n0=1.0
  )
  w0 = 1.5
  z = 2.5
  power = 1.0

  psi_hg = hermite_gaussian_beam(
    sim_config, w0=w0, n=0, m=0, z=z, power=power, envelope_only=True
  )
  psi_gauss = gaussian_beam(
    sim_config, w0=w0, z=z, power=power, envelope_only=True
  )

  diff = jnp.max(jnp.abs(jnp.abs(psi_hg)**2 - jnp.abs(psi_gauss)**2))
  assert diff < 1e-5


def test_lg_propagation_matches_solver():
  """Test that LG analytical solution matches numerical solver in vacuum."""
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.1, nz=20, wavelength=1.0, n0=1.0
  )
  pml_config = pws_pml = PMLConfig(width_x=10, width_y=10, strength=2.0)
  solver_config = SolverConfig(method='spectral', stepper='split_step')

  w0 = 1.5
  p, l = 0, 1

  psi_0 = laguerre_gaussian_beam(sim_config, w0=w0, p=p, l=l, z=0.0)

  def n_ref_fn(z):
    return jnp.ones((sim_config.nx, sim_config.ny))

  solver = ParaxialWaveSolver(sim_config, solver_config, pws_pml, n_ref_fn)
  psi_final, _ = solver.solve(psi_0)

  z_final = sim_config.lz
  psi_analytical = laguerre_gaussian_beam(
    sim_config, w0=w0, p=p, l=l, z=z_final, envelope_only=True
  )

  rel_err = jnp.linalg.norm(psi_final - psi_analytical) / jnp.linalg.norm(psi_analytical)
  assert rel_err < 1e-3
