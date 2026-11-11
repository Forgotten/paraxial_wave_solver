"""Independent references for implicit paraxial and one-way Helmholtz steps."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from paraxial_wave_solver import PMLConfig, SimulationConfig, SolverConfig
from paraxial_wave_solver.src.linear_solvers import solve_linear_system
from paraxial_wave_solver.src.rational import pade_coefficients, pade_multiplier
from paraxial_wave_solver.src.solvers import ParaxialWaveSolver

NO_PML = PMLConfig(0, 0, 0.0)


def _problem(nz=4):
  sim = SimulationConfig(6, 6, 1.0, 1.0, 0.4 / nz, nz, 1.0)
  q = 2 * np.pi * np.fft.fftfreq(sim.nx, sim.dx)
  symbol = -(q[:, None]**2 + q[None, :]**2)
  basis = np.eye(sim.nx * sim.ny).reshape(-1, sim.nx, sim.ny)
  lap = np.stack([
    np.fft.ifft2(symbol * np.fft.fft2(v)).ravel() for v in basis
  ], axis=1)
  x = np.arange(sim.nx)[:, None]
  index = np.broadcast_to(0.06 * np.cos(2 * np.pi * x / sim.nx), symbol.shape)
  psi = jnp.asarray((1 + 0.3 * np.cos(2 * np.pi * x / sim.nx))
                    * np.ones_like(symbol), dtype=complex)
  return sim, lap, index, psi


def test_linear_solve_residual_and_failure(x64):
  matrix = jnp.array([[3 + 1j, 0.2], [0.4j, 2 - 1j]])
  rhs = jnp.array([1 + 2j, -0.3j])
  result = solve_linear_system(lambda x: matrix @ x, rhs, rtol=1e-11)
  assert result.converged
  np.testing.assert_allclose(result.solution, np.linalg.solve(matrix, rhs), atol=1e-10)
  failed = solve_linear_system(lambda x: matrix @ x, rhs,
                               rtol=1e-12, restart=1, maxiter=1)
  assert not failed.converged
  zero = solve_linear_system(lambda x: matrix @ x, jnp.zeros_like(rhs))
  assert zero.converged
  assert jnp.all(zero.solution == 0)


def test_cn_second_order_against_dense_exponential(x64):
  sim, lap, index, psi = _problem()
  generator = 1j * lap / (2 * sim.k) + 1j * sim.k0 * np.diag(index.ravel())
  exact = expm(sim.lz * generator) @ np.asarray(psi).ravel()
  errors = []
  for nz in (4, 8, 16):
    cfg = dataclasses.replace(sim, nz=nz, dz=sim.lz / nz)
    solver = ParaxialWaveSolver(
      cfg, SolverConfig(method='spectral', stepper='crank_nicolson',
                        linear_rtol=1e-11), NO_PML, lambda z, m: m,
    )
    out, _ = solver.solve(psi, medium=jnp.asarray(index), return_history=False)
    errors.append(np.linalg.norm(np.asarray(out).ravel() - exact))
    assert float(jnp.linalg.norm(out)) == pytest.approx(float(jnp.linalg.norm(psi)),
                                                      rel=1e-10)
  assert np.all(np.array(errors[:-1]) / errors[1:] > 3.9)


@pytest.mark.parametrize('order', [2, 4, 6])
def test_cn_fd_plane_wave(order, x64):
  sim = SimulationConfig(16, 16, 0.4, 0.4, 0.2, 3, 1.0)
  theta = 2 * np.pi * 3 / sim.nx
  weights = {2: (-2, 1), 4: (-2.5, 4/3, -1/12),
             6: (-49/18, 1.5, -3/20, 1/90)}[order]
  eigenvalue = (weights[0] + 2 * sum(
    w * np.cos(j * theta) for j, w in enumerate(weights[1:], 1)
  )) / sim.dx**2
  psi = jnp.exp(1j * theta * jnp.arange(16)[:, None]) * jnp.ones((1, 16))
  solver = ParaxialWaveSolver(
    sim, SolverConfig(method='finite_difference', fd_order=order,
                      stepper='crank_nicolson', linear_rtol=1e-11), NO_PML,
  )
  out, _ = solver.solve(psi, return_history=False)
  a = 1j * sim.dz * eigenvalue / (4 * sim.k)
  np.testing.assert_allclose(out, psi * ((1+a)/(1-a))**sim.nz, atol=1e-10)


def test_pade_degree_convergence_against_dense_root(x64):
  sim, lap, index, psi = _problem()
  hamiltonian = lap + sim.k0**2 * np.diag((sim.n0 + index.ravel())**2)
  values, vectors = np.linalg.eigh(hamiltonian)
  exact = vectors @ (np.exp(1j * sim.lz * (np.sqrt(values + 0j) - sim.k))
                     * (vectors.conj().T @ np.asarray(psi).ravel()))
  errors = []
  for degree in (2, 4, 6):
    solver = ParaxialWaveSolver(
      sim, SolverConfig(method='spectral', stepper='pade', propagator='helmholtz',
                        pade_order=degree, linear_rtol=1e-11),
      NO_PML, lambda z, m: m,
    )
    out, _ = solver.solve(psi, medium=jnp.asarray(index), return_history=False)
    errors.append(np.linalg.norm(np.asarray(out).ravel() - exact)
                  / np.linalg.norm(exact))
  assert errors[1] < errors[0] / 20
  assert errors[2] < errors[1] / 20
  assert errors[2] < 1e-7


def test_pade_outgoing_branch_and_scalar_error(x64):
  eigenvalues = jnp.array([-1.0, -0.5, -0.1, 0.2, 0.5, 1.0, 1.5])
  num, den = pade_coefficients(0.1, order=6)
  actual = pade_multiplier(eigenvalues, num, den, np.pi / 6)
  exact = jnp.exp(0.1j * (jnp.sqrt(eigenvalues.astype(complex)) - 1))
  assert jnp.all(jnp.abs(actual[:3]) < 1)
  assert jnp.max(jnp.abs(actual[:3] - exact[:3])) < 0.01
  assert jnp.max(jnp.abs(actual[3:] - exact[3:])) < 1e-6


@pytest.mark.parametrize('stepper', ['crank_nicolson', 'pade'])
def test_implicit_gradients_history_and_batching(stepper, x64):
  sim, _, index, psi = _problem(nz=2)
  config = SolverConfig(method='spectral', stepper=stepper, linear_rtol=1e-11,
                        propagator='helmholtz' if stepper == 'pade' else 'paraxial')
  solver = ParaxialWaveSolver(sim, config, NO_PML, lambda z, m: m * (1 + z))
  index = jnp.asarray(index)

  def objective(a):
    operators = solver.build_operators({'dz': a * sim.dz, 'k0': a * sim.k0})
    out, history = solver.solve(psi, medium=a * index, operators=operators,
                               observable_fn=lambda f, z: z)
    return jnp.real(out[0, 0]) + jnp.sum(history)

  a, eps = 1.1, 1e-5
  numeric = (objective(a+eps) - objective(a-eps)) / (2*eps)
  assert float(jax.grad(objective)(a)) == pytest.approx(float(numeric), rel=1e-5)
  batched = jax.vmap(objective)(jnp.array([1.0, 1.1]))
  np.testing.assert_allclose(batched, [objective(1.0), objective(1.1)], atol=1e-10)


@pytest.mark.parametrize('stepper', ['crank_nicolson', 'pade'])
def test_implicit_rejects_kerr(stepper, x64):
  sim, _, _, _ = _problem()
  config = SolverConfig(method='spectral', stepper=stepper,
                        propagator='helmholtz' if stepper == 'pade' else 'paraxial')
  with pytest.raises(ValueError, match='n2=0'):
    ParaxialWaveSolver(dataclasses.replace(sim, n2=0.1), config, NO_PML)


def test_pade_restrictions(x64):
  sim, _, _, psi = _problem()
  config = SolverConfig(method='spectral', stepper='pade', propagator='helmholtz')
  with pytest.raises(ValueError, match='no PML'):
    ParaxialWaveSolver(sim, config, PMLConfig(1, 1, 1.0))
  solver = ParaxialWaveSolver(sim, config, NO_PML, lambda z, m: 0.01j)
  out, _ = solver.solve(psi, return_history=False)
  assert jnp.all(jnp.isnan(out))


def test_cn_absorber_matches_dense_cayley_step(x64):
  sim, lap, index, psi = _problem(nz=1)
  solver = ParaxialWaveSolver(
    sim, SolverConfig(method='spectral', stepper='crank_nicolson',
                      linear_rtol=1e-11), PMLConfig(1, 1, 1.5),
    lambda z, m: m + 0.02j,
  )
  generator = (1j*lap/(2*sim.k)
               + np.diag((1j*sim.k0*(index+0.02j)
                          - np.asarray(solver.pml_profile)).ravel()))
  identity = np.eye(psi.size)
  exact = np.linalg.solve(identity-sim.dz/2*generator,
                          (identity+sim.dz/2*generator) @ np.asarray(psi).ravel())
  out, _ = solver.solve(psi, medium=jnp.asarray(index), return_history=False)
  np.testing.assert_allclose(np.asarray(out).ravel(), exact, atol=1e-9)
  assert jnp.linalg.norm(out) < jnp.linalg.norm(psi)


def test_pade_fd_constant_mode_uses_full_index(x64):
  sim, _, _, psi = _problem()
  psi = jnp.ones_like(psi)
  solver = ParaxialWaveSolver(
    sim, SolverConfig(method='finite_difference', fd_order=4, stepper='pade',
                      propagator='helmholtz', pade_order=6, linear_rtol=1e-11),
    NO_PML, lambda z, m: 0.1,
  )
  out, _ = solver.solve(psi, return_history=False)
  np.testing.assert_allclose(out, psi*np.exp(1j*sim.k0*0.1*sim.lz), atol=1e-7)


@pytest.mark.parametrize('kwargs', [
  {'pade_order': 0}, {'pade_order': 7}, {'pade_rotation': 0.0},
  {'linear_rtol': 0.0}, {'linear_maxiter': 0}, {'linear_restart': 1.5},
  {'stepper': 'pade'}, {'propagator': 'helmholtz'},
])
def test_implicit_config_validation(kwargs):
  with pytest.raises(ValueError):
    SolverConfig(method='spectral', **kwargs)


def test_pade_requires_double_precision():
  previous = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", False)
  try:
    sim, _, _, _ = _problem()
    with pytest.raises(ValueError, match='enable_x64'):
      ParaxialWaveSolver(
        sim, SolverConfig(method='spectral', stepper='pade',
                          propagator='helmholtz'), NO_PML,
      )

  finally:
    jax.config.update("jax_enable_x64", previous)


def test_nonconverged_step_is_not_silently_accepted(x64):
  sim, _, index, psi = _problem(nz=1)
  solver = ParaxialWaveSolver(
    sim, SolverConfig(method='spectral', stepper='crank_nicolson',
                      linear_rtol=1e-12, linear_restart=1, linear_maxiter=1),
    NO_PML, lambda z, m: m,
  )
  out, _ = solver.solve(psi, medium=jnp.asarray(index), return_history=False)
  assert jnp.all(jnp.isnan(out))


def test_pade_propagates_an_evanescent_mode(x64):
  n, q = 8, np.sqrt(1.5)
  spacing = 2*np.pi/(n*q)
  sim = SimulationConfig(n, n, spacing, spacing, 0.1, 1, 2*np.pi)
  solver = ParaxialWaveSolver(
    sim, SolverConfig(method='spectral', stepper='pade', propagator='helmholtz',
                      pade_order=6), NO_PML,
  )
  psi = jnp.exp(2j*jnp.pi*jnp.arange(n)[:, None]/n)*jnp.ones((1, n))
  out, _ = solver.solve(psi, return_history=False)
  exact = psi*np.exp(0.1j*(np.sqrt(-0.5+0j)-1))
  # This negative eigenvalue is much farther from the expansion centre than
  # the positive-spectrum dense test. Degree six has about 0.5% error here.
  assert jnp.linalg.norm(out-exact)/jnp.linalg.norm(exact) < 0.006
  assert jnp.linalg.norm(out) < jnp.linalg.norm(psi)


def test_underflowed_pade_coefficients_are_rejected(x64):
  sim = SimulationConfig(4, 4, 1., 1., 10000., 1, 2*np.pi)
  solver = ParaxialWaveSolver(
    sim, SolverConfig(method='spectral', stepper='pade', propagator='helmholtz'),
    NO_PML,
  )
  out, _ = solver.solve(jnp.ones((4, 4), dtype=complex), return_history=False)
  assert jnp.all(jnp.isnan(out))
