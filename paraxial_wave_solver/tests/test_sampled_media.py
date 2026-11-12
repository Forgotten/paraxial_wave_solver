"""Sample accuracy, regularity, sample gradients, and one-sided interfaces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

import paraxial_wave_solver as pws


def test_nonuniform_cubic_matches_independent_spline_and_c2(x64):
  z = np.array([0., .1, .4, .7, 1.])
  values = np.exp(z)*(1+.2j)
  medium = pws.make_sampled_medium(z, values)
  reference = CubicSpline(z, values)
  queries = np.linspace(0, 1, 31)
  actual = jax.jit(jax.vmap(pws.sample_medium, in_axes=(0, None)))(queries, medium)
  np.testing.assert_allclose(actual, reference(queries), atol=2e-14)
  coefficients = np.asarray(medium.coefficients)
  h = np.diff(z)[:-1]
  # Match value, slope, and second derivative at every interior knot.
  a, b, c, d = coefficients[:-1].T
  np.testing.assert_allclose(a+h*(b+h*(c+h*d)), coefficients[1:, 0], atol=1e-14)
  np.testing.assert_allclose(b+2*c*h+3*d*h*h, coefficients[1:, 1], atol=1e-14)
  np.testing.assert_allclose(2*c+6*d*h, 2*coefficients[1:, 2], atol=1e-14)


@pytest.mark.parametrize('kind,order', [('linear', 2), ('cubic', 4)])
def test_sample_grid_convergence(kind, order, x64):
  errors = []
  query = np.linspace(0, 1, 1001)
  for intervals in (8, 16, 32):
    z = np.linspace(0, 1, intervals+1)
    medium = pws.make_sampled_medium(z, np.exp(z), kind)
    actual = jax.vmap(pws.sample_medium, in_axes=(0, None))(query, medium)
    errors.append(np.max(np.abs(actual-np.exp(query))))
  rates = np.log2(np.array(errors[:-1])/errors[1:])
  assert np.all(np.abs(rates-order) < .25), (errors, rates)


def test_sample_gradients_and_invalid_inputs(x64):
  z = np.linspace(0, 1, 5)
  values = jnp.exp(jnp.asarray(z))
  def objective(v):
    return pws.sample_medium(.37, pws.make_sampled_medium(z, v))
  direction = jnp.arange(5.)
  eps = 1e-5
  expected = (objective(values+eps*direction)-objective(values-eps*direction))/(2*eps)
  assert float(jnp.dot(jax.grad(objective)(values), direction)) == pytest.approx(
    float(expected), rel=1e-9)
  medium = pws.make_sampled_medium(z, values)
  assert np.isnan(pws.sample_medium(-.01, medium))
  assert np.isnan(pws.sample_medium(1.01, medium))
  for knots in ([0, 0, 1, 2], [0, 1, np.nan, 2], [0, 1]):
    with pytest.raises(ValueError):
      pws.make_sampled_medium(knots, np.ones(len(knots)))
  with pytest.raises(ValueError, match='dimension'):
    pws.make_sampled_medium(z, np.ones(4))


@pytest.mark.parametrize('stepper', ['etdrk4', 'radau5'])
def test_layered_media_endpoint_sampling(stepper, x64):
  sim = pws.SimulationConfig(4, 4, 1., 1., .6, 2, 1.)
  solver = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper=stepper, linear_rtol=1e-12),
    pws.PMLConfig(0, 0), lambda z, medium: medium,
  )
  boundaries = [.2, .63, 1.4]
  indices = [.3+.01j, -.2+.02j]
  result = solver.solve_segments(
    jnp.ones((4, 4), dtype=complex), boundaries, indices,
    config=pws.AdaptiveConfig(rtol=1e-9, atol=1e-11),
  )
  phases = sim.k0*np.cumsum(np.array(indices)*np.diff(boundaries))
  np.testing.assert_allclose(result.history[1:, 0, 0], np.exp(1j*phases), atol=1e-7)
  np.testing.assert_array_equal(result.z, boundaries)
  assert sum(result.step_sizes) == pytest.approx(1.2)


def test_mandatory_stops_do_not_add_history(x64):
  sim = pws.SimulationConfig(4, 4, 1., 1., .5, 2, 1.)
  solver = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='etdrk4'), pws.PMLConfig(0, 0),
  )
  result = solver.solve_adaptive(jnp.ones((4, 4), dtype=complex),
                                 stop_z=[.13, .71])
  np.testing.assert_array_equal(result.z, [1.])
  for stop in (.13, .71, 1.):
    assert np.min(np.abs(np.cumsum(result.step_sizes)-stop)) < 1e-14
  with pytest.raises(ValueError, match='stop_z'):
    solver.solve_adaptive(jnp.ones((4, 4)), stop_z=[.5, .1])


def test_propagation_converges_with_sample_refinement(x64):
  sim = pws.SimulationConfig(4, 4, 1., 1., .2, 5, 1.)
  solver = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='magnus4'),
    pws.PMLConfig(0, 0), pws.sample_medium,
  )
  errors = []
  exact = np.exp(1j*sim.k0*.1*(np.e-1))
  for intervals in (4, 8, 16):
    z = np.linspace(0, 1, intervals+1)
    medium = pws.make_sampled_medium(z, .1*np.exp(z))
    result = solver.solve_adaptive(
      jnp.ones((4, 4), dtype=complex), medium=medium, stop_z=z,
      config=pws.AdaptiveConfig(rtol=1e-11, atol=1e-13),
    )
    errors.append(float(jnp.abs(result.field[0, 0]-exact)))
  assert errors[-1] < errors[0]/100, errors
  assert np.all(np.diff(errors) < 0), errors
