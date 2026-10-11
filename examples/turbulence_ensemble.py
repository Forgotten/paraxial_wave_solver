"""Monte-Carlo statistics of a beam propagating through turbulence.

A single propagation through a random medium is one realization and carries no
statistical meaning on its own: the speckle pattern it produces is a property
of that particular draw of the medium. The quantities of physical interest are
moments over the ensemble of realizations, and this example estimates them by
propagating many realizations simultaneously with `ensemble_solve`.

Two points are made quantitatively. First, averaging over the ensemble
converts a speckled instantaneous pattern into the smooth, broadened
long-exposure beam that a time-integrating detector records. Second, the
ensemble size required for a given precision depends strongly on which
statistic is being estimated: beam wander, a first-moment quantity, converges
roughly as the inverse square root of the ensemble size, whereas the
scintillation index is a ratio of fourth to second intensity moments and
requires substantially larger ensembles for comparable relative precision.

Run after installing the package (`pip install -e .`):

    python examples/turbulence_ensemble.py
"""

import jax
import jax.numpy as jnp
from _plotting import plot_ensemble_summary

import paraxial_wave_solver as pws

# A helium-neon beam through 2.5 m of water with a turbulent index
# perturbation, matching the geometry used by the other turbulence examples.
WAVELENGTH = 632.8e-9
REFRACTIVE_INDEX = 1.33
WAIST = 3e-3
CORRELATION_LENGTH = 2e-3

# Index-perturbation amplitude per screen. Both regimes are chosen so that the
# resulting speckle is resolved by the transverse grid: a saturated regime
# (strength 3e-6 here) produces a coherence radius of only one or two cells, at
# which point the scintillation index is a discretization artefact rather than
# a physical quantity. The ratio is reported below so that this can be checked.
REGIMES = (
  ('weak', 5e-7),
  ('moderate', 1.1e-6),
)

N_REALIZATIONS = 512
CHUNK_SIZE = 64
N_RESAMPLES = 200


def build_solver(
  strength: float,
) -> tuple[pws.ParaxialWaveSolver, pws.SimulationConfig]:
  """Returns a solver whose medium argument is a PRNG key.

  The index perturbation is generated from the key by `phase_screen`, with the
  propagation step folded in so that each z-slice draws an independent screen.
  Because the medium is never materialized as a volume, the memory cost of a
  realization is one transverse field rather than an (nz, nx, ny) array, which
  is what makes an ensemble of this size practical.

  Args:
    strength: Root-mean-square index perturbation of each screen.

  Returns:
    A tuple (solver, sim_config).
  """
  sim_config = pws.SimulationConfig(
    nx=192, ny=192, dx=1.333e-4, dy=1.333e-4, dz=2.5e-2, nz=100,
    wavelength=WAVELENGTH, n0=REFRACTIVE_INDEX,
  )
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  pml_config = pws.PMLConfig(width_x=24, width_y=24, strength=2.0)

  def delta_n_fn(z, key):
    step = jnp.floor(z / sim_config.dz).astype(int)
    return pws.phase_screen(
      jax.random.fold_in(key, step), sim_config, CORRELATION_LENGTH, strength
    )

  solver = pws.ParaxialWaveSolver(
    sim_config, solver_config, pml_config, delta_n_fn
  )
  return solver, sim_config


def run(strength: float, seed: int) -> tuple:
  """Propagates one ensemble and reduces it to statistics with error bars.

  Args:
    strength: Root-mean-square index perturbation of each screen.
    seed: Seed for the ensemble's realizations.

  Returns:
    A tuple (fields, statistics, errors, sim_config).
  """
  solver, sim_config = build_solver(strength)
  psi_0 = pws.gaussian_beam(sim_config, w0=WAIST)
  keys = jax.random.split(jax.random.PRNGKey(seed), N_REALIZATIONS)

  fields, _ = pws.ensemble_solve(
    solver, psi_0, keys, chunk_size=CHUNK_SIZE
  )
  statistics = pws.ensemble_statistics(fields, sim_config)
  errors = pws.ensemble_standard_error(
    fields,
    lambda ensemble: pws.ensemble_statistics(ensemble, sim_config),
    jax.random.PRNGKey(seed + 1),
    n_resamples=N_RESAMPLES,
  )
  return fields, statistics, errors, sim_config


def convergence(fields: jnp.ndarray, sim_config: pws.SimulationConfig) -> dict:
  """Returns the relative standard error of two statistics against ensemble size.

  Subsets of the ensemble already computed are reused rather than propagated
  again, so this costs only the bootstrap.

  Args:
    fields: Complex fields of shape (n_realizations, nx, ny).
    sim_config: Simulation configuration.

  Returns:
    A dict mapping ensemble size to a tuple of relative standard errors,
    (scintillation index, beam wander), each as a percentage.
  """
  def on_axis_scintillation(ensemble):
    intensity = jnp.abs(ensemble)**2
    mean_intensity = jnp.mean(intensity, axis=0)
    index = jnp.unravel_index(
      jnp.argmax(mean_intensity), mean_intensity.shape
    )
    on_axis = intensity[:, index[0], index[1]]
    return jnp.mean(on_axis**2) / jnp.mean(on_axis)**2 - 1.0

  def wander(ensemble):
    return pws.beam_wander(ensemble, sim_config)[0]

  table = {}
  for size in (16, 64, 256, fields.shape[0]):
    subset = fields[:size]
    key = jax.random.PRNGKey(size)
    results = []
    for statistic in (on_axis_scintillation, wander):
      value = float(statistic(subset))
      error = float(pws.ensemble_standard_error(
        subset, statistic, key, n_resamples=N_RESAMPLES
      ))
      results.append(100.0 * error / abs(value) if value else float('nan'))
    table[size] = tuple(results)
  return table


def main():
  pws.enable_x64()

  results = {}
  for label, strength in REGIMES:
    print(f"Propagating {N_REALIZATIONS} realizations of {label} turbulence "
          f"in chunks of {CHUNK_SIZE}...")
    results[label] = run(strength, seed=2024)

  for label, _ in REGIMES:
    fields, statistics, errors, sim_config = results[label]
    plot_ensemble_summary(
      jnp.abs(fields[0])**2,
      statistics['mean_intensity'],
      statistics['scintillation_index'],
      sim_config,
      f'turbulence_ensemble_{label}.png',
      f'{N_REALIZATIONS} realizations, {label} turbulence',
    )
    print(f"Saved figure to turbulence_ensemble_{label}.png")

  print(f"\nEnsemble statistics, value +/- bootstrap standard error "
        f"({N_RESAMPLES} resamples):")
  header = f"  {'quantity':<30}"
  for label, _ in REGIMES:
    header += f"{label:>26}"
  print(header)
  rows = (
    ('on-axis scintillation index', 'on_axis_scintillation_index', 1.0, ''),
    ('beam wander', 'wander_x', 1e6, ' um'),
    ('coherence radius', 'coherence_radius', 1e3, ' mm'),
    ('short-exposure width', 'short_exposure_width_x', 1e3, ' mm'),
    ('long-exposure width', 'long_exposure_width_x', 1e3, ' mm'),
  )
  for title, key, scale, unit in rows:
    cells = ''
    for label, _ in REGIMES:
      _, statistics, errors, _ = results[label]
      value = float(statistics[key]) * scale
      error = float(errors[key]) * scale
      cells += f'{value:14.3f} +/- {error:<7.3f}{unit:<4}'
    print(f"  {title:<30}{cells}")

  # The precondition for any of the above to be physical rather than numerical.
  cells = ''
  for label, _ in REGIMES:
    _, statistics, _, sim_config = results[label]
    cells += f'{float(statistics["coherence_radius"]) / sim_config.dx:26.2f}'
  print(f"  {'coherence radius / dx':<30}{cells}")
  print("  (below roughly three cells the speckle is unresolved and the")
  print("   scintillation index is a discretization artefact)")

  print("\nThe long-exposure width exceeds the short-exposure width because")
  print("beam wander displaces the instantaneous pattern between")
  print("realizations, which the ensemble average records as a broadening.")

  print("\nRelative standard error against ensemble size (%):")
  header = f"  {'N':>6}"
  for label, _ in REGIMES:
    header += f"{label + ': sigma_I^2':>22}{'wander':>10}"
  print(header)
  tables = {
    label: convergence(results[label][0], results[label][3])
    for label, _ in REGIMES
  }
  for size in sorted(next(iter(tables.values()))):
    cells = ''
    for label, _ in REGIMES:
      scintillation, wander = tables[label][size]
      cells += f'{scintillation:22.1f}{wander:10.1f}'
    print(f"  {size:>6}{cells}")

  print("\nBoth statistics improve as the inverse square root of the ensemble")
  print("size, but with markedly different prefactors: the scintillation index")
  print("is a ratio of fourth to second intensity moments and is the more")
  print("expensive of the two to estimate. It is also biased low at small")
  print("ensemble size, so a short run reports a confidently wrong value")
  print("rather than a visibly noisy one.")


if __name__ == "__main__":
  main()
