"""Beam quality through turbulence, measured inside the propagation loop.

Propagates a Laguerre-Gaussian beam through several strengths of Von Karman
turbulence and records diagnostics at every plane. The diagnostics are
computed by the solver as it goes, so what comes back is a handful of numbers
per plane rather than the field volume: for the grid used here that is about
30 kB instead of 210 MB.

Run after installing the package (`pip install -e .`):

    python examples/turbulence_diagnostics.py
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import paraxial_wave_solver as pws

# Underwater tank parameters, matching turbulence_propagation.py.
OUTER_SCALE = 2e-2      # L0, metres.
INNER_SCALE = 1e-3      # l0, metres.
POWER = 2e-3            # Total laser power, watts.
WAIST = 3.0e-3          # Beam waist, metres.


def build_config() -> pws.SimulationConfig:
  """Returns the grid shared by every run."""
  return pws.SimulationConfig(
    nx=256, ny=256,
    dx=2e-4, dy=2e-4, dz=2.5e-2,
    nz=100,
    wavelength=632.8e-9,
    n0=1.33,
  )


def run(cn2: float, seed: int = 0):
  """Propagates an LG_01 beam through one realization of turbulence.

  Args:
    cn2: Refractive index structure constant; 0.0 gives a clear medium.
    seed: PRNG seed selecting the realization.

  Returns:
    A tuple (history, psi_0, psi_final, sim_config), where history is a dict
    of per-plane diagnostics.
  """
  sim_config = build_config()
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  pml_config = pws.PMLConfig(width_x=32, width_y=32, strength=2.0)

  psi_0 = pws.laguerre_gaussian_beam(
    sim_config, WAIST, p=0, l=1, power=POWER, envelope_only=True
  )

  def delta_n_fn(z, medium):
    index = jnp.clip(
      jnp.round(z / sim_config.dz).astype(int), 0, sim_config.nz - 1
    )
    return medium[:, :, index]

  solver = pws.ParaxialWaveSolver(
    sim_config, solver_config, pml_config, delta_n_fn
  )

  if cn2 == 0.0:
    volume = jnp.zeros((sim_config.nx, sim_config.ny, sim_config.nz))
  else:
    volume = pws.random_medium_spectral(
      sim_config, cn2, OUTER_SCALE, INNER_SCALE, jax.random.PRNGKey(seed)
    )

  # The reference is the same mode in a clear medium, so overlap measures only
  # what turbulence did. The scintillation index is restricted to the
  # illuminated region: averaged over the whole plane it would mostly report
  # how much dark background surrounds the beam.
  reference = psi_0
  illuminated = jnp.abs(reference)**2 > 0.01 * jnp.max(jnp.abs(reference)**2)

  def observable(psi, z):
    metrics = pws.beam_diagnostics(psi, sim_config, mask=illuminated)
    metrics['overlap'] = pws.overlap(psi, reference)
    metrics['peak_ratio'] = pws.strehl_ratio(psi, reference)
    return metrics

  psi_final, history = solver.solve(
    psi_0, medium=volume, observable_fn=observable
  )
  return history, psi_0, psi_final, sim_config


def main():
  pws.enable_x64()
  strengths = [
    ('clear', 0.0, 'k'),
    ('weak (4.4e-13)', 4.4e-13, 'tab:blue'),
    ('moderate (4.4e-11)', 4.4e-11, 'tab:orange'),
    ('strong (4.4e-9)', 4.4e-9, 'tab:red'),
  ]

  results = {}
  for label, cn2, _ in strengths:
    print(f"Propagating through {label} turbulence...")
    results[label] = run(cn2)

  sim_config = results['clear'][3]
  z = jnp.arange(sim_config.nz) * sim_config.dz

  panels = [
    ('m2_x', 'Beam quality $M^2_x$', None),
    ('width_x', 'D4$\\sigma$ width $w_x$ (m)', None),
    ('peak_ratio', 'Peak intensity vs clear beam', None),
    ('overlap', 'Overlap with LG$_{01}$', None),
    ('scintillation_index', 'Spatial scintillation index', 'log'),
    ('power_retained', 'Power retained (PML losses)', None),
  ]

  # Absolute power is uninformative on a shared axis; the fraction retained
  # shows what the PML absorbed as turbulence pushed the beam outwards.
  for label, _, _ in strengths:
    history = results[label][0]
    history['power_retained'] = history['power'] / history['power'][0]

  fig, axes = plt.subplots(2, 3, figsize=(16, 9))
  for ax, (key, title, scale) in zip(axes.ravel(), panels, strict=True):
    for label, _, colour in strengths:
      history = results[label][0]
      ax.plot(z, history[key], color=colour, label=label, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel('z (m)')
    if scale == 'log':
      ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
  axes[0, 0].legend(fontsize=8)

  fig.suptitle(
    'LG$_{01}$ through Von Karman turbulence, measured inside the loop',
    fontsize=13,
  )
  fig.tight_layout()
  fig.savefig('turbulence_diagnostics.png')
  print("Saved figure to turbulence_diagnostics.png")

  print("\nFinal-plane summary:")
  print(f"  {'turbulence':<20}{'M2_x':>8}{'overlap':>10}{'sigma_I':>10}")
  print("  (sigma_I is spatial, over the illuminated region; the clear row is")
  print("   the baseline set by the mode's own intensity structure)")
  for label, _, _ in strengths:
    history = results[label][0]
    print(f"  {label:<20}{float(history['m2_x'][-1]):8.2f}"
          f"{float(history['overlap'][-1]):10.3f}"
          f"{float(history['scintillation_index'][-1]):10.3f}")

  fields_bytes = sim_config.nz * sim_config.nx * sim_config.ny * 16
  metrics_bytes = sum(
    v.size * v.dtype.itemsize for v in results['clear'][0].values()
  )
  print(f"\nRecorded {metrics_bytes / 1e3:.1f} kB of diagnostics instead of "
        f"{fields_bytes / 1e6:.0f} MB of fields.")


if __name__ == "__main__":
  main()
