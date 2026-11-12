"""Magnus order, matrix-free actions, and fixed-step gradients."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import paraxial_wave_solver as pws
from paraxial_wave_solver.src.magnus import exponential_action, exponential_plan


@pytest.mark.parametrize('scale', [.1, 3., 20.])
@pytest.mark.parametrize('backend', ['adaptive_taylor', 'fixed_taylor'])
def test_exponential_action_against_dense_nonnormal_matrix(scale, backend, x64):
  matrix = scale*np.array([[-1., 2j, 0.], [0., -.3, 1.], [0., 0., -2.]])
  field = jnp.array([1., 2j, -.3])
  actual = exponential_action(lambda u: jnp.asarray(matrix)@u, field,
                              np.linalg.norm(matrix, 2), backend=backend)
  np.testing.assert_allclose(actual, expm(matrix)@field, atol=2e-13, rtol=2e-12)


def test_exponential_capacity_and_zero(x64):
  field = jnp.ones(4, dtype=complex)
  assert np.all(np.isnan(exponential_action(
    lambda u: u, field, 3., 2, backend='fixed_taylor')))
  np.testing.assert_array_equal(exponential_action(lambda u: 0*u, field, 0.), field)
  np.testing.assert_array_equal(exponential_action(lambda u: u, 0*field, 1.), 0)


@pytest.mark.parametrize('method', ['spectral', 'finite_difference'])
@pytest.mark.parametrize('absorber', [False, True])
def test_magnus_fourth_order_noncommuting_medium(method, absorber, x64):
  n, length = 6, 1.2
  sim = pws.SimulationConfig(n, n, 1., 1., .1, 12, 1.)
  sc = pws.SolverConfig(method=method, stepper='magnus4')
  pml = pws.PMLConfig(1, 1, .15) if absorber else pws.PMLConfig(0, 0, 0.)
  x = np.arange(n)[:, None]
  medium = np.broadcast_to(.2*np.cos(2*np.pi*x/n), (n, n)).copy()
  initial = np.broadcast_to(1+.2*np.sin(2*np.pi*x/n), (n, n)).astype(complex)
  sigma = np.asarray(pws.ParaxialWaveSolver(sim, sc, pml).pml_profile)
  q = 2*np.pi*np.fft.fftfreq(n)
  symbol = -(q[:, None]**2+q[None, :]**2)

  def rhs(z, u):
    field = u.reshape(n, n)
    if method == 'spectral':
      lap = np.fft.ifft2(symbol*np.fft.fft2(field))
    else:
      lap = sum(np.roll(field, s, axis=a) for a in (0, 1) for s in (-1, 1))
      lap -= 4*field
    return (1j*lap/(2*sim.k)
            + (1j*sim.k0*(medium*np.cos(2*z)+.02j)-sigma)*field).ravel()

  ref = solve_ivp(rhs, (0, length), initial.ravel(), method='DOP853',
                  rtol=2e-13, atol=2e-14).y[:, -1]
  errors = []
  for nz in (4, 8, 16):
    obj = pws.ParaxialWaveSolver(dataclasses.replace(sim, dz=length/nz, nz=nz),
                                 sc, pml, lambda z, m: m*jnp.cos(2*z)+.02j)
    field, _ = obj.solve(jnp.asarray(initial), medium=jnp.asarray(medium),
                         return_history=False)
    errors.append(np.linalg.norm(np.asarray(field).ravel()-ref)/np.linalg.norm(ref))
  rates = np.log2(np.array(errors[:-1])/errors[1:])
  assert np.all((rates > 3.8) & (rates < 4.2)), (errors, rates)


def test_gradient_overrides_and_history(x64):
  sim = pws.SimulationConfig(6, 6, 1., 1., .1, 4, 1.)
  obj = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='magnus4'),
    pws.PMLConfig(0, 0, 0.), lambda z, m: m*jnp.cos(z),
  )
  field = jnp.ones((6, 6), dtype=complex)

  def loss(a):
    out, zs = obj.solve(field, z_0=.2, medium=a*.2,
                        operators=obj.build_operators({'dz': a*.1, 'k0': a*6.}),
                        observable_fn=lambda f, z: z)
    return jnp.real(out[0, 0])+jnp.sum(zs)

  eps = 1e-5
  expected = (loss(1+eps)-loss(1-eps))/(2*eps)
  assert float(jax.grad(loss)(1.)) == pytest.approx(float(expected), rel=1e-7)
  out, _ = obj.solve(field, medium=.2)
  exact = np.exp(1j*sim.k0*.2*np.sin(sim.lz))
  np.testing.assert_allclose(out, exact, atol=2e-8)


def test_magnus_rejects_nonlinear_and_stretched_models():
  sim = pws.SimulationConfig(6, 6, 1., 1., .1, 4, 1.)
  sc = pws.SolverConfig(method='spectral', stepper='magnus4')
  with pytest.raises(ValueError, match='n2=0'):
    pws.ParaxialWaveSolver(dataclasses.replace(sim, n2=.1), sc,
                           pws.PMLConfig(0, 0))
  with pytest.raises(ValueError, match='stretching'):
    pws.ParaxialWaveSolver(sim, sc, pws.PMLConfig(1, 1, use_complex_stretching=True))
  with pytest.raises(ValueError, match='exponential_max_substeps'):
    pws.SolverConfig(method='spectral', stepper='magnus4', exponential_max_substeps=0)


def test_selected_cost_and_remainder_budget(x64):
  import math

  for bound in (.01, .1, 1., 3., 20., 100.):
    plan = exponential_plan(bound)
    fixed = exponential_plan(bound, backend='fixed_taylor')
    assert plan.valid
    assert int(plan.operator_evaluations) < int(fixed.operator_evaluations)
    theta = bound/int(plan.substeps)
    tail = (math.exp(theta)*theta**(int(plan.degree)+1)
            / math.factorial(int(plan.degree)+1))
    assert tail <= 1e-12/256*(1+1e-12)
  assert not exponential_plan(1e10, max_substeps=1).valid
  assert exponential_plan(0.).operator_evaluations == 0


def test_action_tolerance_refinement(x64):
  matrix = jnp.array([[0., 1j], [1j, 0.]])
  field = jnp.array([1., .3j])
  exact = expm(np.asarray(matrix))@field
  errors = []
  for tolerance in (1e-2, 1e-6, 1e-12):
    result = exponential_action(lambda v: matrix@v, field, 1., 1, rtol=tolerance)
    errors.append(np.linalg.norm(np.asarray(result)-exact))
  assert np.all(np.diff(errors) < 0), errors
  assert errors[-1] < 1e-12 and errors[-1] < errors[0]/1000, errors


def test_action_selection_gradient_and_float32(x64):
  field = jnp.array([1., .3j])
  def objective(a):
    return jnp.real(exponential_action(lambda v: 1j*a*v, field, jnp.abs(a))[0])
  assert float(jax.grad(objective)(1.3)) == pytest.approx(-np.sin(1.3), abs=1e-12)
  field32 = jnp.ones(4, dtype=jnp.complex64)
  actual = exponential_action(lambda v: 2j*v, field32, jnp.float32(2), rtol=1e-6)
  np.testing.assert_allclose(actual, np.exp(2j), atol=2e-6)
  for kwargs in ({'exponential_rtol': 0}, {'exponential_backend': 'unknown'}):
    with pytest.raises(ValueError):
      pws.SolverConfig(method='spectral', stepper='magnus4', **kwargs)
