"""Hermitian square-root actions, evanescent branches, and two refinements."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

import paraxial_wave_solver as pws


def test_square_root_dense_mixed_spectrum(x64):
  rng = np.random.default_rng(4)
  q, _ = np.linalg.qr(rng.normal(size=(12, 12))+1j*rng.normal(size=(12, 12)))
  values = np.linspace(-5., 8., 12)
  matrix = (q*values)@q.conj().T
  field = rng.normal(size=12)+1j*rng.normal(size=12)
  exact = (q*np.sqrt(values+0j))@q.conj().T@field
  result = jax.jit(lambda v: pws.square_root_action(
    lambda u: jnp.asarray(matrix)@u, v, krylov_dim=12, rtol=1e-10))(field)
  assert result.converged
  np.testing.assert_allclose(result.solution, exact, atol=2e-13)


def test_krylov_dimension_convergence(x64):
  values = np.linspace(.1, 30., 30)
  field = jnp.ones(30, dtype=complex)
  errors = []
  for dimension in (4, 8, 16, 30):
    result = pws.square_root_action(lambda u: jnp.asarray(values)*u, field,
                                    krylov_dim=dimension, rtol=1e-10)
    errors.append(np.linalg.norm(np.asarray(result.solution)-np.sqrt(values)))
  assert np.all(np.diff(errors) < 0), errors
  assert errors[-1] < 1e-12, errors
  assert result.converged


@pytest.mark.parametrize('eigenvalue', [-4., 0., 9.])
def test_invariant_and_zero_field(eigenvalue, x64):
  for field in (jnp.ones((4, 4), dtype=complex), jnp.zeros((4, 4), dtype=complex)):
    result = pws.square_root_action(lambda v: eigenvalue*v, field, krylov_dim=4)
    assert result.converged
    np.testing.assert_allclose(result.solution, np.sqrt(eigenvalue+0j)*field,
                               atol=1e-13)


def test_insufficient_dimension_and_nonhermitian_projection(x64):
  field = jnp.ones(8, dtype=complex)
  result = pws.square_root_action(lambda v: jnp.arange(8)*v, field, krylov_dim=1)
  assert not result.converged
  result = pws.square_root_action(lambda v: 1j*v, field, krylov_dim=4)
  assert not result.converged


@pytest.mark.parametrize('method', ['spectral', 'finite_difference'])
def test_propagating_and_evanescent_modes(method, x64):
  for mode in (1, 2):
    sim = pws.SimulationConfig(6, 6, .2, .2, .1, 3, 1.)
    config = pws.SolverConfig(method=method, stepper='lanczos',
                              propagator='helmholtz', krylov_dim=8)
    field = jnp.exp(2j*jnp.pi*mode*jnp.arange(6)[:, None]/6)*jnp.ones((1, 6))
    solver = pws.ParaxialWaveSolver(sim, config, pws.PMLConfig(0, 0))
    out, _ = solver.solve(field, return_history=False)
    q = 2*np.pi*mode/sim.lx
    lap = -q*q if method == 'spectral' else -4*np.sin(q*sim.dx/2)**2/sim.dx**2
    exact = field*np.exp(1j*sim.lz*(np.sqrt(sim.k**2+lap+0j)-sim.k))
    np.testing.assert_allclose(out, exact, atol=2e-12)
    assert np.linalg.norm(out) <= np.linalg.norm(field)*(1+1e-12)


def test_range_dependent_midpoint_convergence(x64):
  n, length = 4, 1.
  sim = pws.SimulationConfig(n, n, 1., 1., .1, 10, 1.)
  config = pws.SolverConfig(method='spectral', stepper='lanczos',
                            propagator='helmholtz', krylov_dim=n*n, krylov_rtol=1e-10)
  x = np.arange(n)[:, None]
  medium = np.broadcast_to(.15*np.cos(2*np.pi*x/n), (n, n)).copy()
  initial = np.broadcast_to(1+.2*np.sin(2*np.pi*x/n), (n, n)).astype(complex)
  q = 2*np.pi*np.fft.fftfreq(n)
  symbol = -(q[:, None]**2+q[None, :]**2)
  basis = np.eye(n*n).reshape(-1, n, n)
  lap = np.stack([np.fft.ifft2(symbol*np.fft.fft2(v)).ravel()
                  for v in basis], axis=1)

  def rhs(z, field):
    h = lap+np.diag((sim.k0**2*(sim.n0+medium*np.cos(2*z))**2).ravel())
    values, vectors = eigh(h)
    return 1j*((vectors*np.sqrt(values+0j))@vectors.conj().T@field-sim.k*field)

  reference = solve_ivp(rhs, (0, length), initial.ravel(), method='DOP853',
                        rtol=2e-13, atol=2e-14).y[:, -1]
  errors = []
  for nz in (4, 8, 16):
    obj = pws.ParaxialWaveSolver(dataclasses.replace(sim, dz=length/nz, nz=nz),
                                 config, pws.PMLConfig(0, 0),
                                 lambda z, m: m*jnp.cos(2*z))
    out, _ = obj.solve(jnp.asarray(initial), medium=jnp.asarray(medium),
                       return_history=False)
    errors.append(np.linalg.norm(np.asarray(out).ravel()-reference)
                  / np.linalg.norm(reference))
  rates = np.log2(np.array(errors[:-1])/errors[1:])
  assert np.all(np.abs(rates-2) < .1), (errors, rates)


def test_rejected_models_and_unresolved_action(x64):
  sim = pws.SimulationConfig(6, 6, 1., 1., .1, 2, 1.)
  config = pws.SolverConfig(method='spectral', stepper='lanczos',
                            propagator='helmholtz', krylov_dim=1)
  for pml in (pws.PMLConfig(1, 1),
              pws.PMLConfig(0, 0, use_complex_stretching=True)):
    with pytest.raises(ValueError):
      pws.ParaxialWaveSolver(sim, config, pml)
  with pytest.raises(ValueError, match='n2=0'):
    pws.ParaxialWaveSolver(dataclasses.replace(sim, n2=.1), config,
                           pws.PMLConfig(0, 0))
  field = jnp.ones((6, 6), dtype=complex)
  obj = pws.ParaxialWaveSolver(sim, config, pws.PMLConfig(0, 0), lambda z, m: m)
  for medium in (.1j, .1*jnp.arange(6)[:, None]):
    out, _ = obj.solve(field, medium=medium)
    assert np.all(np.isnan(out))


def test_parameter_override_matches_constant_medium_phase(x64):
  sim = pws.SimulationConfig(4, 4, 1., 1., .1, 3, 1.)
  obj = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='lanczos',
                          propagator='helmholtz'),
    pws.PMLConfig(0, 0), lambda z, m: m,
  )
  out, zs = obj.solve(jnp.ones((4, 4), dtype=complex), medium=.2, z_0=.3,
                      operators=obj.build_operators({'k0': 3., 'n0': 1.5, 'dz': .2}),
                      observable_fn=lambda f, z: z)
  np.testing.assert_allclose(out, np.exp(1j*3*.2*.6), atol=1e-13)
  np.testing.assert_allclose(zs, [.3, .5, .7], atol=1e-13)


def test_fractional_laplacian_plane_wave(x64):
  n = 8
  q = 2*jnp.pi*jnp.fft.fftfreq(n, d=.5)
  symbol = q[:, None]**2+q[None, :]**2
  field = jnp.exp(2j*jnp.pi*jnp.arange(n)[:, None]/n)*jnp.ones((1, n))
  result = pws.square_root_action(
    lambda u: jnp.fft.ifft2(symbol*jnp.fft.fft2(u)), field, krylov_dim=8,
  )
  assert result.converged
  np.testing.assert_allclose(result.solution, (2*np.pi/(n*.5))*field, atol=1e-12)


def test_default_precision_root_action():
  field = jnp.ones(8, dtype=jnp.complex64)
  values = jnp.linspace(1., 4., 8, dtype=jnp.float32)
  result = pws.square_root_action(lambda u: values*u, field, krylov_dim=8)
  assert result.converged
  np.testing.assert_allclose(result.solution, np.sqrt(values), rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize('kwargs', [dict(krylov_dim=0), dict(krylov_rtol=0)])
def test_invalid_krylov_configuration(kwargs):
  with pytest.raises(ValueError):
    pws.SolverConfig(method='spectral', stepper='lanczos',
                     propagator='helmholtz', **kwargs)
