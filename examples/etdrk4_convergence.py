"""Compare ETDRK4 and Strang against a dense paraxial matrix exponential.

Run: python examples/etdrk4_convergence.py

The fixed periodic grid isolates longitudinal error. The transversely varying
index does not commute with diffraction. This is an accuracy study, not a
runtime benchmark or a spatial convergence study.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

import paraxial_wave_solver as pws


def benchmark():
  """Returns step counts and relative errors for the two steppers."""
  pws.enable_x64()
  n, length, k = 8, 2., 2*np.pi
  x = np.arange(n)[:, None]
  medium = np.broadcast_to(.12*np.cos(2*np.pi*x/n), (n, n)).copy()
  psi = ((1+.2*np.cos(2*np.pi*x/n))*np.ones((1, n))).astype(complex)
  q = 2*np.pi*np.fft.fftfreq(n)
  symbol = -(q[:, None]**2 + q[None, :]**2)
  basis = np.eye(n*n).reshape(-1, n, n)
  lap = np.stack([np.fft.ifft2(symbol*np.fft.fft2(v)).ravel()
                  for v in basis], axis=1)
  generator = 1j*lap/(2*k) + 1j*k*np.diag(medium.ravel())
  exact = expm(length*generator) @ psi.ravel()
  counts = [8, 16, 32, 64]
  errors = {}
  for stepper in ('split_step', 'etdrk4'):
    errors[stepper] = []
    for nz in counts:
      sim = pws.SimulationConfig(n, n, 1., 1., length/nz, nz, 1.)
      solver = pws.ParaxialWaveSolver(
        sim, pws.SolverConfig(method='spectral', stepper=stepper),
        pws.PMLConfig(0, 0, 0.), lambda z, m: m,
      )
      out, _ = solver.solve(jnp.asarray(psi), medium=jnp.asarray(medium),
                             return_history=False)
      errors[stepper].append(float(np.linalg.norm(np.asarray(out).ravel()-exact)
                                  / np.linalg.norm(exact)))
  return counts, errors


def main():
  counts, errors = benchmark()
  fig, ax = plt.subplots(figsize=(7, 5))
  for stepper, values in errors.items():
    rates = np.log2(np.array(values[:-1])/values[1:])
    print(f'{stepper}: errors={values}, measured orders={rates}')
    expected = 4 if stepper == 'etdrk4' else 2
    if not np.all(rates > expected-.2):
      raise RuntimeError(f'{stepper} failed its convergence check')
    ax.loglog(counts, values, 'o-', label=f'{stepper} (order {expected})')
  ax.set_xlabel('Propagation steps (fixed total range)')
  ax.set_ylabel('Relative field L2 error')
  ax.grid(True, which='both', alpha=.3)
  ax.legend()
  fig.tight_layout()
  fig.savefig('etdrk4_convergence.png', dpi=150)
  plt.close(fig)


if __name__ == '__main__':
  main()
