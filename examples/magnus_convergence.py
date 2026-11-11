"""Fourth-order Magnus refinement for a noncommuting range-dependent medium."""

import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

import paraxial_wave_solver as pws


def benchmark():
  pws.enable_x64()
  n, length = 6, 1.2
  x = np.arange(n)[:, None]
  medium = np.broadcast_to(.2*np.cos(2*np.pi*x/n), (n, n)).copy()
  initial = np.broadcast_to(1+.2*np.sin(2*np.pi*x/n), (n, n)).astype(complex)
  q = 2*np.pi*np.fft.fftfreq(n)
  symbol = -(q[:, None]**2+q[None, :]**2)

  def rhs(z, flat):
    field = flat.reshape(n, n)
    return (1j*np.fft.ifft2(symbol*np.fft.fft2(field))/(4*np.pi)
            + 2j*np.pi*medium*np.cos(2*z)*field).ravel()

  reference = solve_ivp(rhs, (0, length), initial.ravel(), method='DOP853',
                        rtol=2e-13, atol=2e-14)
  if not reference.success:
    raise RuntimeError(reference.message)
  exact = reference.y[:, -1]
  counts, errors = [4, 8, 16, 32], []
  for nz in counts:
    sim = pws.SimulationConfig(n, n, 1., 1., length/nz, nz, 1.)
    solver = pws.ParaxialWaveSolver(
      sim, pws.SolverConfig(method='spectral', stepper='magnus4'),
      pws.PMLConfig(0, 0, 0.), lambda z, m: m*jnp.cos(2*z),
    )
    field, _ = solver.solve(jnp.asarray(initial), medium=jnp.asarray(medium),
                             return_history=False)
    errors.append(np.linalg.norm(np.asarray(field).ravel()-exact)
                  / np.linalg.norm(exact))
  return counts, errors


def main():
  counts, errors = benchmark()
  rates = np.log2(np.array(errors[:-1])/errors[1:])
  print(f'steps={counts}\nerrors={errors}\nobserved orders={rates}')
  assert np.all(np.abs(rates-4) < .2), rates


if __name__ == '__main__':
  main()
