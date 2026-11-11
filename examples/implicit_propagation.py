"""Validate CN and the experimental full-index square-root propagator.

Run: python examples/implicit_propagation.py

A small periodic grid permits independent dense matrix references. CN is
compared with the paraxial matrix exponential; Padé is compared with the
outgoing square root of the full transverse Helmholtz matrix. These are
*different equations*. Refining dz controls CN error; raising Padé degree
controls rational approximation error. Neither study measures spatial error.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

import paraxial_wave_solver as pws


def benchmark():
  """Returns refinement coordinates and relative errors for both models."""
  pws.enable_x64()
  n, spacing, length = 8, 1.0, 0.4
  k = 2 * np.pi
  x = np.arange(n)[:, None]
  index = np.broadcast_to(0.06 * np.cos(2 * np.pi * x / n), (n, n))
  psi = (1 + 0.3 * np.cos(2 * np.pi * x / n)) * np.ones((n, n))
  frequencies = 2 * np.pi * np.fft.fftfreq(n, spacing)
  symbol = -(frequencies[:, None]**2 + frequencies[None, :]**2)
  basis = np.eye(n*n).reshape(-1, n, n)
  laplacian = np.stack([
    np.fft.ifft2(symbol * np.fft.fft2(v)).ravel() for v in basis
  ], axis=1)
  paraxial = 1j * laplacian / (2*k) + 1j*k*np.diag(index.ravel())
  exact_cn = expm(length * paraxial) @ psi.ravel()
  hamiltonian = laplacian + k**2 * np.diag((1 + index.ravel())**2)
  values, vectors = np.linalg.eigh(hamiltonian)
  exact_root = vectors @ (
    np.exp(1j * length * (np.sqrt(values + 0j) - k))
    * (vectors.conj().T @ psi.ravel())
  )

  def run(stepper, nz, degree=4):
    sim = pws.SimulationConfig(n, n, spacing, spacing, length/nz, nz, 1.0)
    config = pws.SolverConfig(
      method='spectral', stepper=stepper,
      propagator='helmholtz' if stepper == 'pade' else 'paraxial',
      pade_order=degree, linear_rtol=1e-11,
    )
    solver = pws.ParaxialWaveSolver(
      sim, config, pws.PMLConfig(0, 0, 0.0), lambda z, medium: medium,
    )
    out, _ = solver.solve(jnp.asarray(psi, dtype=complex),
                          medium=jnp.asarray(index), return_history=False)
    return np.asarray(out).ravel()

  counts, degrees = [4, 8, 16, 32], [2, 4, 6]
  cn_errors = [np.linalg.norm(run('crank_nicolson', nz) - exact_cn)
               / np.linalg.norm(exact_cn) for nz in counts]
  root_errors = [np.linalg.norm(run('pade', 4, m) - exact_root)
                 / np.linalg.norm(exact_root) for m in degrees]
  return counts, cn_errors, degrees, root_errors


def main():
  counts, cn_errors, degrees, root_errors = benchmark()
  for nz, error in zip(counts, cn_errors, strict=True):
    print(f'CN: nz={nz:2d}, relative error={error:.3e}')
  for degree, error in zip(degrees, root_errors, strict=True):
    print(f'Padé: degree={degree}, relative error={error:.3e}')
  if not (np.all(np.isfinite(cn_errors + root_errors))
          and cn_errors[-1] < cn_errors[0] / 50
          and root_errors[-1] < 1e-7):
    raise RuntimeError('Refinement checks failed; inspect solver convergence.')
  fig, axes = plt.subplots(1, 2, figsize=(10, 4))
  axes[0].loglog(counts, cn_errors, 'o-', label='CN vs matrix exponential')
  axes[0].loglog(counts, cn_errors[0] * (counts[0] / np.array(counts))**2,
                 'k--', label='second order')
  axes[0].set_xlabel('Propagation steps')
  axes[1].semilogy(degrees, root_errors, 'o-', label='Padé vs dense square root')
  axes[1].set_xlabel('Padé degree (fixed dz)')
  for ax in axes:
    ax.set_ylabel('Relative field L2 error')
    ax.grid(True, alpha=0.3)
    ax.legend()
  fig.tight_layout()
  fig.savefig('implicit_propagation.png', dpi=150)
  plt.close(fig)


if __name__ == '__main__':
  main()
