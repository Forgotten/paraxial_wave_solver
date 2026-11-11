"""ETDRK4 accuracy against independent analytic and ODE references."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from paraxial_wave_solver import PMLConfig, SimulationConfig, SolverConfig
from paraxial_wave_solver.src.exponential import phi_functions
from paraxial_wave_solver.src.solvers import ParaxialWaveSolver

NO_PML = PMLConfig(0, 0, 0.)
ETD = SolverConfig(method='spectral', stepper='etdrk4')


@pytest.mark.parametrize('dtype,tolerance', [(jnp.complex64, 1e-6),
                                           (jnp.complex128, 1e-12)])
def test_phi_functions_against_augmented_matrix(dtype, tolerance, x64):
  values = np.array([0, 1e-12j, -1e-4j, -0.99j, -1.01j, -3j, -100j])
  actual = phi_functions(jnp.asarray(values, dtype=dtype))
  for i, value in enumerate(values):
    matrix = np.diag(np.ones(3), 1).astype(complex)
    matrix[0, 0] = value
    reference = expm(matrix)[0, 1:]
    np.testing.assert_allclose([a[i] for a in actual], reference,
                               atol=tolerance, rtol=tolerance)
  derivative = jax.jacfwd(lambda t: jnp.stack(phi_functions(t+0j)))(0.)
  np.testing.assert_allclose(derivative, [1/2, 1/6, 1/24], atol=1e-12)


def test_vacuum_exact_beyond_rk4_stability_limit(x64):
  sim = SimulationConfig(8, 8, .1, .1, 1., 3, 1.)
  psi = jnp.exp(2j*jnp.pi*3*jnp.arange(8)[:, None]/8)*jnp.ones((1, 8))
  out, _ = ParaxialWaveSolver(sim, ETD, NO_PML).solve(psi)
  q = 2*np.pi*3/sim.lx
  np.testing.assert_allclose(out, psi*np.exp(-1j*sim.lz*q*q/(2*sim.k)), atol=1e-11)


@pytest.mark.parametrize('kerr,absorber', [(False, False), (True, False),
                                         (True, True)])
def test_fourth_order_with_range_variation_and_absorption(kerr, absorber, x64):
  n, length = 6, .8
  base = SimulationConfig(n, n, 1., 1., .1, 8, 1., n2=.03 if kerr else 0.)
  x = np.arange(n)[:, None]
  medium = np.broadcast_to(.12*np.cos(2*np.pi*x/n), (n, n)).copy()
  initial = ((1+.2*np.cos(2*np.pi*x/n))*np.ones((1, n))).astype(complex)
  pml = PMLConfig(1, 1, .2) if absorber else NO_PML
  sigma = np.asarray(ParaxialWaveSolver(base, ETD, pml).pml_profile)
  q = 2*np.pi*np.fft.fftfreq(n)
  symbol = -(q[:, None]**2 + q[None, :]**2)

  def rhs(z, flat):
    field = flat.reshape(n, n)
    lap = np.fft.ifft2(symbol*np.fft.fft2(field))
    return (1j*lap/(2*base.k)
            + 1j*base.k0*(medium*np.cos(z)+.01j
                         + base.n2*np.abs(field)**2)*field - sigma*field).ravel()

  reference = solve_ivp(rhs, (0., length), initial.ravel(), method='DOP853',
                        rtol=2e-13, atol=2e-14).y[:, -1]
  errors = []
  for nz in (4, 8, 16):
    sim = dataclasses.replace(base, dz=length/nz, nz=nz)
    solver = ParaxialWaveSolver(sim, ETD, pml,
                               lambda z, m: m*jnp.cos(z)+.01j)
    out, _ = solver.solve(jnp.asarray(initial), medium=jnp.asarray(medium),
                           return_history=False)
    errors.append(np.linalg.norm(np.asarray(out).ravel()-reference)
                  / np.linalg.norm(reference))
  assert np.all(np.log2(np.array(errors[:-1])/errors[1:]) > 3.7), errors


def test_zero_diffraction_recovers_rk4(x64):
  sim = SimulationConfig(4, 4, 1., 1., .1, 1, 1.)
  solver = ParaxialWaveSolver(sim, ETD, NO_PML, lambda z, m: m)
  psi = jnp.ones((4, 4), dtype=complex)
  index = .2+.03j
  out, _ = solver.solve(psi, medium=index)
  a = 1j*sim.k0*index*sim.dz
  expected = sum(a**j/math.factorial(j) for j in range(5))
  np.testing.assert_allclose(out, psi*expected, atol=1e-12)


def test_gradient_overrides_history_and_batching(x64):
  sim = SimulationConfig(8, 8, 1., 1., .1, 4, 1., n2=.02)
  solver = ParaxialWaveSolver(sim, ETD, NO_PML, lambda z, m: m*(1+z))
  psi = (1+.2*jnp.cos(2*jnp.pi*jnp.arange(8)[:, None]/8))*jnp.ones((1, 8))
  psi = psi.astype(complex)

  def objective(a):
    ops = solver.build_operators({'k0': a*sim.k0, 'n0': a, 'dz': a*sim.dz})
    out, zs = solver.solve(psi, medium=.1*a, operators=ops, save_every=2,
                           observable_fn=lambda f, z: z)
    return jnp.real(out[0, 0])+jnp.sum(zs)

  eps = 1e-5
  exact = (objective(1.+eps)-objective(1.-eps))/(2*eps)
  assert float(jax.grad(objective)(1.)) == pytest.approx(float(exact), rel=1e-7)
  np.testing.assert_allclose(jax.vmap(objective)(jnp.array([1., 1.1])),
                             [objective(1.), objective(1.1)], atol=1e-12)
  _, zs = solver.solve(psi, medium=.1,
                       operators=solver.build_operators({'dz': .2}),
                       observable_fn=lambda f, z: z)
  np.testing.assert_allclose(zs, [.0, .2, .4, .6], atol=1e-12)


@pytest.mark.parametrize('kwargs', [dict(method='finite_difference'),
                                   dict(propagator='wide_angle'),
                                   dict(splitting_order=4), dict(dealias=True)])
def test_unsupported_etd_combinations(kwargs):
  with pytest.raises(ValueError):
    SolverConfig(**{'method': 'spectral', 'stepper': 'etdrk4', **kwargs})


def test_etd_rejects_stretching():
  sim = SimulationConfig(8, 8, 1., 1., .1, 1, 1.)
  with pytest.raises(ValueError, match='stretching'):
    ParaxialWaveSolver(sim, ETD, PMLConfig(1, 1, 1., use_complex_stretching=True))


def test_default_precision_with_complex_index():
  sim = SimulationConfig(8, 8, 1., 1., .01, 20, 1.)
  solver = ParaxialWaveSolver(sim, ETD, NO_PML, lambda z, m: .1+.02j)
  psi = jnp.ones((8, 8), dtype=jnp.complex64)
  out, _ = solver.solve(psi, return_history=False)
  exact = np.exp(1j*sim.k0*(.1+.02j)*sim.lz)
  np.testing.assert_allclose(out, exact, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize('name', ['k0', 'n0', 'dz'])
def test_parameter_override_matches_fresh_solver(name, x64):
  sim = SimulationConfig(8, 8, 1., 1., .05, 4, 1., n2=.02)
  updated = {'k0': dataclasses.replace(sim, wavelength=.8),
             'n0': dataclasses.replace(sim, n0=1.3),
             'dz': dataclasses.replace(sim, dz=.08)}[name]

  def medium(z, m):
    return m*(1+z)

  solver = ParaxialWaveSolver(sim, ETD, NO_PML, medium)
  fresh = ParaxialWaveSolver(updated, ETD, NO_PML, medium)
  psi = jnp.exp(2j*jnp.pi*jnp.arange(8)[:, None]/8)*jnp.ones((1, 8))
  ops = solver.build_operators({name: getattr(updated, name)})
  out, _ = solver.solve(psi, medium=.1, operators=ops)
  expected, _ = fresh.solve(psi, medium=.1)
  np.testing.assert_allclose(out, expected, atol=1e-12)


def test_initial_field_gradient_and_checkpoint(x64):
  sim = SimulationConfig(8, 8, 1., 1., .05, 8, 1., n2=.03)
  solver = ParaxialWaveSolver(sim, ETD, PMLConfig(1, 1, .1))
  psi = jnp.ones((8, 8), dtype=complex)
  direction = jax.random.normal(jax.random.PRNGKey(3), (8, 8))*(1+.3j)

  def objective(field, checkpoint):
    out, _ = solver.solve(field, return_history=False, checkpoint=checkpoint)
    return jnp.real(out[2, 3])

  gradient = jax.grad(objective)(psi, True)
  plain = jax.grad(objective)(psi, False)
  np.testing.assert_allclose(gradient, plain, atol=1e-12)
  eps = 1e-5
  numeric = (objective(psi+eps*direction, True)
             - objective(psi-eps*direction, True))/(2*eps)
  assert float(jnp.real(jnp.sum(gradient*direction))) == pytest.approx(
    float(numeric), rel=1e-7, abs=1e-9)
