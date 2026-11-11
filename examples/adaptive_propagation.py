"""Tolerance refinement against an analytic range-dependent medium."""

import jax.numpy as jnp
import numpy as np

import paraxial_wave_solver as pws


def main():
  pws.enable_x64()
  sim = pws.SimulationConfig(8, 8, 1., 1., .8, 2, 1.)
  solver = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='etdrk4'),
    pws.PMLConfig(0, 0, 0.), lambda z, m: m*(1+jnp.sin(z)),
  )
  exact = np.exp(1j*sim.k0*.2*(sim.lz+1-np.cos(sim.lz)))
  errors = []
  for tolerance in (1e-3, 1e-5, 1e-7):
    result = solver.solve_adaptive(
      jnp.ones((8, 8), dtype=complex), medium=.2,
      config=pws.AdaptiveConfig(rtol=tolerance, atol=tolerance*.01),
    )
    error = float(jnp.max(jnp.abs(result.field-exact)))
    errors.append(error)
    print(f'rtol={tolerance:g}: error={error:.3e}, '
          f'accepted={result.accepted}, rejected={result.rejected}')
  assert np.all(np.diff(errors) < 0), errors


if __name__ == '__main__':
  main()
