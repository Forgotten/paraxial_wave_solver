"""Implicit collocation: independent temporal order, damping, and gradients."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import paraxial_wave_solver as pws
from paraxial_wave_solver.src.collocation import collocation_tableau
from paraxial_wave_solver.src.solvers import _make_laplacian_fn


@pytest.mark.parametrize('stepper,order', [('gauss4', 4), ('radau5', 5)])
@pytest.mark.parametrize('boundary', ['spectral', 'fd_absorber', 'fd_stretch'])
def test_range_dependent_order(stepper, order, boundary, x64):
  n, length = 4, 1.2
  base = pws.SimulationConfig(n, n, 1., 1., .1, 12, 1.)
  sc = pws.SolverConfig(method='spectral' if boundary == 'spectral'
                        else 'finite_difference', stepper=stepper,
                        linear_rtol=2e-13, linear_restart=32)
  pml = (pws.PMLConfig(0, 0) if boundary == 'spectral' else
         pws.PMLConfig(1, 1, .2, use_complex_stretching=boundary == 'fd_stretch'))
  obj = pws.ParaxialWaveSolver(base, sc, pml)
  # Form the independently integrated semidiscrete ODE, not an RK reference.
  lap_fn = _make_laplacian_fn(sc, base, obj._use_stretch)
  basis = np.eye(n*n).reshape(-1, n, n)
  lap = np.stack([np.asarray(lap_fn(jnp.asarray(v), obj._operators)).ravel()
                  for v in basis], axis=1)
  sigma = 0 if obj._use_stretch else np.asarray(obj.pml_profile).ravel()
  index = np.broadcast_to(.2*np.cos(2*np.pi*np.arange(n)[:, None]/n), (n, n))
  initial = (1+.2*np.arange(n*n)/(n*n)).astype(complex)

  def rhs(z, field):
    return (1j*lap@field/(2*base.k)
            + (1j*base.k0*(index.ravel()*np.cos(2*z)+.02j)-sigma)*field)

  exact = solve_ivp(rhs, (0, length), initial, method='DOP853',
                    rtol=2e-13, atol=2e-14).y[:, -1]
  errors = []
  for nz in (4, 8, 16):
    obj = pws.ParaxialWaveSolver(dataclasses.replace(base, dz=length/nz, nz=nz),
                                 sc, pml, lambda z, m: m*jnp.cos(2*z)+.02j)
    out, _ = obj.solve(jnp.asarray(initial.reshape(n, n)), medium=jnp.asarray(index),
                       return_history=False)
    errors.append(np.linalg.norm(np.asarray(out).ravel()-exact)/np.linalg.norm(exact))
  rates = np.log2(np.array(errors[:-1])/errors[1:])
  assert np.all(np.abs(rates-order) < .35), (errors, rates)


def test_tableau_consistency(x64):
  for name in ('gauss4', 'radau5'):
    a, b, c = collocation_tableau(name)
    np.testing.assert_allclose(a.sum(axis=1), c, atol=1e-15)
    assert float(b.sum()) == pytest.approx(1.)
  with pytest.raises(ValueError):
    collocation_tableau('unknown')


def test_gauss_conservation_and_radau_stiff_decay(x64):
  sim = pws.SimulationConfig(4, 4, 1., 1., 1., 1, 1.)
  outputs = {}
  for stepper in ('gauss4', 'radau5'):
    sc = pws.SolverConfig(method='spectral', stepper=stepper, linear_rtol=1e-12)
    obj = pws.ParaxialWaveSolver(sim, sc, pws.PMLConfig(0, 0), lambda z, m: m)
    psi = jnp.ones((4, 4), dtype=complex)
    out, _ = obj.solve(psi, medium=1000j/sim.k0)
    outputs[stepper] = float(jnp.abs(out[0, 0]))
    if stepper == 'gauss4':
      out, _ = obj.solve(psi, medium=.3*jnp.arange(4)[:, None])
      assert float(jnp.linalg.norm(out)) == pytest.approx(4., rel=1e-10)
  assert outputs['radau5'] < .01
  assert outputs['gauss4'] > .9


@pytest.mark.parametrize('stepper', ['gauss4', 'radau5'])
def test_gradient_overrides_and_failure(stepper, x64):
  sim = pws.SimulationConfig(4, 4, 1., 1., .1, 3, 1.)
  sc = pws.SolverConfig(method='spectral', stepper=stepper, linear_rtol=1e-12)
  obj = pws.ParaxialWaveSolver(sim, sc, pws.PMLConfig(0, 0), lambda z, m: m*(1+z))
  psi = jnp.ones((4, 4), dtype=complex)

  def objective(a):
    out, zs = obj.solve(psi, medium=a*.2,
                        operators=obj.build_operators({'dz': .1*a, 'k0': 6.*a}),
                        observable_fn=lambda f, z: z)
    return jnp.real(out[0, 0])+jnp.sum(zs)

  eps = 1e-5
  exact = (objective(1+eps)-objective(1-eps))/(2*eps)
  assert float(jax.grad(objective)(1.)) == pytest.approx(float(exact), rel=1e-7)
  failed = pws.ParaxialWaveSolver(
    sim, dataclasses.replace(sc, linear_maxiter=1, linear_restart=1),
    pws.PMLConfig(0, 0), lambda z, m: m,
  )
  out, _ = failed.solve(psi, medium=jnp.arange(4)[:, None])
  assert np.all(np.isnan(out))
  with pytest.raises(ValueError, match='n2=0'):
    pws.ParaxialWaveSolver(dataclasses.replace(sim, n2=.1), sc, pws.PMLConfig(0, 0))
