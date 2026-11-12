"""Separate sample-grid error from propagation error, then cross a layer jump."""

import jax.numpy as jnp
import numpy as np

import paraxial_wave_solver as pws


def main():
  pws.enable_x64()
  sim = pws.SimulationConfig(4, 4, 1., 1., .2, 5, 1.)
  solver = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='magnus4'),
    pws.PMLConfig(0, 0), pws.sample_medium,
  )
  psi = jnp.ones((4, 4), dtype=complex)
  exact = np.exp(1j*sim.k0*.1*(np.e-1))
  errors = []
  for intervals in (4, 8, 16):
    z = np.linspace(0, 1, intervals+1)
    medium = pws.make_sampled_medium(z, .1*np.exp(z))
    result = solver.solve_adaptive(
      psi, medium=medium, stop_z=z,
      config=pws.AdaptiveConfig(rtol=1e-11, atol=1e-13),
    )
    errors.append(float(jnp.abs(result.field[0, 0]-exact)))
    print(f'{intervals} sample intervals: field error={errors[-1]:.3e}')
  assert errors[-1] < errors[0]/100
  layers = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='etdrk4'),
    pws.PMLConfig(0, 0), lambda z, value: value,
  )
  result = layers.solve_segments(
    psi, [0., .37, 1.], [.2, -.1],
    config=pws.AdaptiveConfig(rtol=1e-9, atol=1e-11),
  )
  expected = np.exp(1j*sim.k0*(.37*.2-.63*.1))
  np.testing.assert_allclose(result.field, expected, atol=1e-7)
  print('Layered-medium result agrees with the analytic phase integral.')


if __name__ == '__main__':
  main()
