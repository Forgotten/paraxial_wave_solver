"""Refine Hermitian square-root actions against a dense eigenvalue reference."""

import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

import paraxial_wave_solver as pws


def main():
  pws.enable_x64()
  n = 6
  rng = np.random.default_rng(5)
  initial = rng.normal(size=(n, n))+1j*rng.normal(size=(n, n))
  q = 2*np.pi*np.fft.fftfreq(n)
  symbol = -(q[:, None]**2+q[None, :]**2)
  index = .1*np.cos(2*np.pi*np.arange(n)[:, None]/n)*np.ones((1, n))
  potential = (2*np.pi)**2*(1+index)**2
  basis = np.eye(n*n).reshape(-1, n, n)
  lap = np.stack([np.fft.ifft2(symbol*np.fft.fft2(v)).ravel()
                  for v in basis], axis=1)
  values, vectors = eigh(lap+np.diag(potential.ravel()))
  exact = (vectors*np.sqrt(values+0j))@vectors.conj().T@initial.ravel()

  def matvec(field):
    return jnp.fft.ifft2(jnp.asarray(symbol)*jnp.fft.fft2(field))+potential*field

  errors = []
  for dimension in (4, 8, 16, 36):
    result = pws.square_root_action(matvec, jnp.asarray(initial),
                                    krylov_dim=dimension, rtol=1e-10)
    error = (np.linalg.norm(np.asarray(result.solution).ravel()-exact)
             / np.linalg.norm(exact))
    errors.append(error)
    print(f'dimension={dimension}: error={error:.3e}, '
          f'indicator={float(result.relative_difference):.3e}, '
          f'accepted={bool(result.converged)}')
  assert errors[-1] < 1e-12 and errors[-1] < errors[0]/1000, errors
  sim = pws.SimulationConfig(n, n, 1., 1., .2, 1, 1.)
  solver = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='lanczos',
                          propagator='helmholtz', krylov_dim=n*n),
    pws.PMLConfig(0, 0), lambda z, m: m,
  )
  final, _ = solver.solve(jnp.asarray(initial), medium=jnp.asarray(index),
                           return_history=False)
  expected = (vectors*np.exp(1j*sim.dz*(np.sqrt(values+0j)-sim.k)))
  expected = expected@vectors.conj().T@initial.ravel()
  np.testing.assert_allclose(np.asarray(final).ravel(), expected, atol=1e-12)
  print('Full-index propagation agrees with the dense reference.')


if __name__ == '__main__':
  main()
