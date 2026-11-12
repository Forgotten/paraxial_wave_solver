"""Accuracy/cost studies with independent references and synchronized JAX timing.

Run with --output /tmp/solver_benchmarks.json. Results are hardware-specific;
compilation, warm execution, and compiler-reported buffers are kept separate.
"""

import argparse
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

import paraxial_wave_solver as pws

CASES = ('smooth', 'kerr', 'absorption', 'near_cutoff')


def problem(case, n=6):
  """Returns one fixed spatial problem and its independent final reference."""
  if case not in CASES:
    raise ValueError(f'Unknown case {case!r}.')
  length, dx = 1.2, .35 if case == 'near_cutoff' else 1.
  sim = pws.SimulationConfig(n, n, dx, dx, .1, 12, 1.,
                             n2=.05 if case == 'kerr' else 0.)
  x = np.arange(n)[:, None]
  medium = np.broadcast_to(.15*np.cos(2*np.pi*x/n), (n, n)).copy()
  initial = np.broadcast_to(1+.2*np.sin(2*np.pi*x/n), (n, n)).astype(complex)
  if case == 'absorption':
    medium = medium + .08j*(1+.4*np.cos(2*np.pi*x/n))
  q = 2*np.pi*np.fft.fftfreq(n, d=dx)
  symbol = -(q[:, None]**2+q[None, :]**2)
  if case == 'near_cutoff':
    initial = np.random.default_rng(9).normal(size=(n, n)).astype(complex)
    basis = np.eye(n*n).reshape(-1, n, n)
    lap = np.stack([np.fft.ifft2(symbol*np.fft.fft2(v)).ravel()
                    for v in basis], axis=1)
    values, vectors = eigh(lap+np.diag((sim.k0**2*(1+medium)**2).ravel()))
    exact = ((vectors*np.exp(1j*length*(np.sqrt(values+0j)-sim.k)))
             @ vectors.conj().T @ initial.ravel()).reshape(n, n)
  else:
    def rhs(z, flat):
      field = flat.reshape(n, n)
      lap = np.fft.ifft2(symbol*np.fft.fft2(field))
      return (1j*lap/(2*sim.k) + 1j*sim.k0*(medium*np.cos(z)
              + sim.n2*np.abs(field)**2)*field).ravel()
    ref = solve_ivp(rhs, (0., length), initial.ravel(), method='DOP853',
                    rtol=2e-13, atol=2e-14)
    if not ref.success:
      raise RuntimeError(ref.message)
    exact = ref.y[:, -1].reshape(n, n)
  return sim, initial, medium, exact


def run_case(case='smooth', counts=(8, 16, 32), steppers=None, repeats=3, n=6):
  """Collects refinement rows; failures remain visible rather than ranked."""
  if repeats < 1 or not counts or any(c < 1 or int(c) != c for c in counts):
    raise ValueError('Require positive repeats and integer step counts.')
  pws.enable_x64()
  base, initial, medium, exact = problem(case, n)
  allowed = (('pade', 'lanczos') if case == 'near_cutoff' else
             ('split_step', 'rk4', 'etdrk4') if case == 'kerr' else
             ('split_step', 'rk4', 'etdrk4', 'crank_nicolson', 'magnus4',
              'gauss4', 'radau5'))
  steppers = allowed if steppers is None else steppers
  if any(s not in allowed for s in steppers):
    raise ValueError('Selected stepper does not solve this benchmark model.')
  rows = []
  for stepper in steppers:
    for nz in counts:
      sim = pws.SimulationConfig(n, n, base.dx, base.dy, base.lz/nz, nz, 1.,
                                 n2=base.n2)
      sc = pws.SolverConfig(
        method='spectral', stepper=stepper, linear_rtol=1e-12,
        propagator='helmholtz' if case == 'near_cutoff' else 'paraxial',
        krylov_dim=n*n, pade_order=6,
      )
      def index(z, data):
        return data if case == 'near_cutoff' else data*jnp.cos(z)
      solver = pws.ParaxialWaveSolver(sim, sc, pws.PMLConfig(0, 0), index)
      def propagate(field, data, obj=solver):
        return obj.solve(field, medium=data, return_history=False,
                            checkpoint=False)[0]
      field, data = jnp.asarray(initial), jnp.asarray(medium)
      jax.block_until_ready((field, data))
      start = time.perf_counter()
      compiled = jax.jit(propagate).lower(field, data).compile()
      compilation = time.perf_counter()-start
      compiled(field, data).block_until_ready()
      durations = []
      for _ in range(repeats):
        start = time.perf_counter()
        final = compiled(field, data).block_until_ready()
        durations.append(time.perf_counter()-start)
      finite = bool(np.all(np.isfinite(final)))
      memory = compiled.memory_analysis()
      fft_per_step = {'split_step': 2, 'rk4': 8, 'etdrk4': 10}.get(stepper)
      rows.append(dict(
        case=case, stepper=stepper, nz=int(nz), n=n,
        relative_error=(float(np.linalg.norm(np.asarray(final)-exact)
                              / np.linalg.norm(exact)) if finite else None),
        finite=finite, compile_seconds=compilation,
        median_seconds=float(np.median(durations)),
        temp_bytes=None if memory is None else memory.temp_size_in_bytes,
        argument_bytes=None if memory is None else memory.argument_size_in_bytes,
        output_bytes=None if memory is None else memory.output_size_in_bytes,
        fft_calls=None if fft_per_step is None else fft_per_step*nz,
      ))
  return rows


def matched_accuracy(rows, tolerance):
  """Fastest measured run per case/stepper reaching the requested error."""
  if not np.isfinite(tolerance) or tolerance <= 0:
    raise ValueError('tolerance must be positive and finite.')
  selected = []
  for case, stepper in sorted({(r['case'], r['stepper']) for r in rows}):
    candidates = [r for r in rows if r['case'] == case and r['stepper'] == stepper
                  and r['finite'] and r['relative_error'] <= tolerance]
    selected.append(dict(case=case, stepper=stepper, reached=bool(candidates),
                         best=min(candidates, key=lambda r: r['median_seconds'])
                         if candidates else None))
  return selected


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', default='solver_benchmarks.json')
  parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
  parser.add_argument('--counts', nargs='+', type=int, default=[8, 16, 32])
  parser.add_argument('--tolerance', type=float, default=1e-5)
  args = parser.parse_args()
  rows = [row for case in args.cases for row in run_case(case, args.counts)]
  report = dict(jax_version=jax.__version__, platform=platform.platform(),
                devices=[str(d) for d in jax.devices()], rows=rows,
                matched_accuracy=matched_accuracy(rows, args.tolerance))
  with open(args.output, 'w') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
  for entry in report['matched_accuracy']:
    best = entry['best']
    print(entry['case'], entry['stepper'],
          'not reached' if best is None else
          f"nz={best['nz']}, error={best['relative_error']:.2e}, "
          f"time={best['median_seconds']:.3e}s")


if __name__ == '__main__':
  main()
