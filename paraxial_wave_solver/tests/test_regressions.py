"""Analytical checks for coupled effects and physical parameter overrides."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from paraxial_wave_solver import PMLConfig, SimulationConfig, SolverConfig
from paraxial_wave_solver.src.pml import generate_pml_profile
from paraxial_wave_solver.src.solvers import ParaxialWaveSolver

NO_PML = PMLConfig(0, 0, 0.0)


@pytest.mark.parametrize('absorption', [-0.01, 0.0, 0.03])
def test_kerr_with_complex_index_exact_local_flow(absorption, x64):
  sim = SimulationConfig(8, 8, 1., 1., 0.2, 5, 1., n2=0.07)
  solver = ParaxialWaveSolver(
    sim, SolverConfig(method='spectral', stepper='split_step'), NO_PML,
    lambda z, m: 0.02 + 1j*m,
  )
  initial = jnp.full((8, 8), 1.2+0.3j)
  out, _ = solver.solve(initial, medium=absorption, return_history=False)
  a = sim.k0 * absorption
  integral = sim.lz if a == 0 else -np.expm1(-2*a*sim.lz)/(2*a)
  phase = sim.k0 * (0.02*sim.lz + sim.n2 * abs(1.2+0.3j)**2 * integral)
  expected = initial * np.exp(-a*sim.lz + 1j*phase)
  np.testing.assert_allclose(out, expected, atol=1e-11)
  derivative = jax.grad(lambda m: jnp.real(solver.solve(
    initial, medium=m, return_history=False)[0][0, 0]))(absorption)
  assert jnp.isfinite(derivative)


@pytest.mark.parametrize('stepper', ['split_step', 'rk4', 'crank_nicolson'])
def test_overrides_match_fresh_solver_in_z_dependent_medium(stepper, x64):
  sim = SimulationConfig(8, 8, 1., 1., 0.02, 4, 1.)
  config = SolverConfig(method='spectral', stepper=stepper, linear_rtol=1e-11)
  def medium_fn(z, m):
    return m * (1 + z)
  solver = ParaxialWaveSolver(sim, config, NO_PML, medium_fn)
  new = dataclasses.replace(sim, wavelength=0.8, n0=1.2, dz=0.03)
  fresh = ParaxialWaveSolver(new, config, NO_PML, medium_fn)
  initial = jnp.exp(2j*jnp.pi*jnp.arange(8)[:, None]/8)*jnp.ones((1, 8))
  ops = solver.build_operators({'k0': new.k0, 'n0': new.n0, 'dz': new.dz})
  out, zs = solver.solve(initial, z_0=0.4, medium=0.03, operators=ops,
                         observable_fn=lambda f, z: z)
  expected, _ = fresh.solve(initial, z_0=0.4, medium=0.03)
  np.testing.assert_allclose(out, expected, atol=1e-11)
  np.testing.assert_allclose(zs, 0.4+new.dz*np.arange(new.nz), atol=1e-12)


def test_wide_angle_optical_phase_survives_float32():
  sim = SimulationConfig(8, 8, 0.01, 0.01, 100., 1, 632.8e-9)
  solver = ParaxialWaveSolver(
    sim, SolverConfig(method='spectral', stepper='split_step',
                      propagator='wide_angle'), NO_PML,
  )
  q = 2*np.pi/sim.lx
  expected = np.exp(-1j*sim.dz*q*q/(np.sqrt(sim.k**2-q*q)+sim.k))
  assert abs(expected-1) > 0.01
  np.testing.assert_allclose(solver._operators['linear'][0][1, 0], expected,
                            atol=1e-6)


def test_unsafe_wide_angle_composition_is_rejected():
  with pytest.raises(ValueError, match='evanescent'):
    SolverConfig(method='spectral', stepper='split_step',
                 propagator='wide_angle', splitting_order=4)


def test_order_zero_pml_leaves_interior_untouched():
  sim = SimulationConfig(16, 16, 1., 1., 0.1, 2, 1.)
  profile = generate_pml_profile(sim, PMLConfig(3, 3, 2., order=0)).sigma
  assert jnp.all(profile[3:13, 3:13] == 0)
  assert profile[0, 8] == 2.
