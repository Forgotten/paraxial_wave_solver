"""Adaptive local control, endpoint handling, and tolerance convergence."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import paraxial_wave_solver as pws


def solver(stepper='etdrk4', **kwargs):
  sim = pws.SimulationConfig(6, 6, 1., 1., .8, 2, 1., **kwargs)
  return pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper=stepper),
    pws.PMLConfig(0, 0, 0.), lambda z, m: m*(1+jnp.sin(z)),
  )


def test_rejections_outputs_and_nonzero_origin(x64):
  obj = solver()
  result = obj.solve_adaptive(
    jnp.ones((6, 6), dtype=complex), 2., z_0=.3, medium=.6,
    save_z=[.3, .7, 1.1], config=pws.AdaptiveConfig(rtol=1e-9, atol=1e-11),
    params={'k0': 3.},
  )
  exact = np.exp(3j*.6*(result.z-.3 + np.cos(.3)-np.cos(result.z)))
  np.testing.assert_allclose(result.history[:, 0, 0], exact, atol=2e-7)
  np.testing.assert_array_equal(result.z, [.3, .7, 1.1, 2.])
  assert result.rejected > 0
  assert result.accepted == len(result.step_sizes)
  assert sum(result.step_sizes) == pytest.approx(1.7)
  np.testing.assert_array_equal(result.field, result.history[-1])


@pytest.mark.parametrize('stepper', ['etdrk4', 'rk4', 'split_step',
                                    'crank_nicolson'])
def test_tolerance_convergence_against_independent_ode(stepper, x64):
  obj = solver(stepper)
  n = 6
  x = np.arange(n)[:, None]
  medium = np.broadcast_to(.2*np.cos(2*np.pi*x/n), (n, n))
  initial = np.broadcast_to(1+.2*np.sin(2*np.pi*x/n), (n, n)).astype(complex)
  q = 2*np.pi*np.fft.fftfreq(n)
  symbol = -(q[:, None]**2+q[None, :]**2)

  def rhs(z, u):
    field = u.reshape(n, n)
    return (1j*np.fft.ifft2(symbol*np.fft.fft2(field))/(4*np.pi)
            + 2j*np.pi*medium*(1+np.sin(z))*field).ravel()

  ref = solve_ivp(rhs, (0, 1.6), initial.ravel(), method='DOP853',
                  rtol=2e-13, atol=2e-14).y[:, -1]
  errors, counts = [], []
  for tol in (1e-3, 1e-5, 1e-7):
    result = obj.solve_adaptive(jnp.asarray(initial), medium=jnp.asarray(medium),
                               config=pws.AdaptiveConfig(rtol=tol, atol=tol*.01))
    errors.append(np.linalg.norm(np.asarray(result.field).ravel()-ref)
                  / np.linalg.norm(ref))
    counts.append(result.accepted)
  assert np.all(np.diff(errors) < 0), errors
  assert errors[-1] < errors[0]/30, errors
  assert counts[-1] > counts[0], counts


def test_zero_field_and_step_bound(x64):
  result = solver().solve_adaptive(jnp.zeros((6, 6), dtype=complex), .31,
                                   medium=.1, config=pws.AdaptiveConfig(max_step=.1))
  np.testing.assert_array_equal(result.field, 0)
  assert np.all(result.step_sizes <= .1)
  assert result.rejected == 0
  assert result.z[-1] == .31


@pytest.mark.parametrize('kwargs', [dict(rtol=0), dict(atol=float('nan')),
                                   dict(max_step=0), dict(max_steps=0),
                                   dict(safety=1)])
def test_invalid_controls(kwargs):
  with pytest.raises(ValueError):
    pws.AdaptiveConfig(**kwargs)


def test_failure_limits_and_invalid_planes(x64):
  obj = solver()
  psi = jnp.ones((6, 6), dtype=complex)
  with pytest.raises(RuntimeError, match='min_step'):
    obj.solve_adaptive(psi, medium=1.,
                       config=pws.AdaptiveConfig(min_step=.8, rtol=1e-12))
  with pytest.raises(RuntimeError, match='step count'):
    obj.solve_adaptive(psi, medium=1., config=pws.AdaptiveConfig(max_steps=1))
  for planes in ([.2, .1], [2.], [float('nan')]):
    with pytest.raises(ValueError, match='save_z'):
      obj.solve_adaptive(psi, medium=0., save_z=planes)
  with pytest.raises(ValueError, match='params'):
    obj.solve_adaptive(psi, params={'dz': .1})
