"""Gaussian beam propagating through a random refractive index medium.

Run after installing the package (`pip install -e .`):

    python examples/random_media.py
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from _plotting import plot_xz_intensity

import paraxial_wave_solver as pws


def main():
  sim_config = pws.SimulationConfig(
    nx=400, ny=400,
    dx=1e-4, dy=1e-4, dz=1e-2,
    nz=250,
    wavelength=632.8e-9,
    n0=1.33,
  )
  pml_config = pws.PMLConfig(width_x=20, width_y=20, strength=5.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')

  psi_0 = pws.gaussian_beam(sim_config, w0=1e-3)

  delta_n = pws.random_medium(
    sim_config, correlation_length=1e-3, strength=4.4e-7,
    key=jax.random.PRNGKey(42),
  )

  # The medium is an argument rather than a closure, so one compiled solver
  # serves any number of realizations without retracing.
  def delta_n_fn(z, medium):
    index = jnp.clip(
      jnp.round(z / sim_config.dz).astype(int), 0, sim_config.nz - 1
    )
    return medium[:, :, index]

  print("Running simulation in random media...")
  solver = pws.ParaxialWaveSolver(
    sim_config, solver_config, pml_config, delta_n_fn
  )
  psi_final, psi_history = solver.solve(psi_0, medium=delta_n)
  print("Simulation complete.")

  centre_y = sim_config.ny // 2
  fig, axes = plt.subplots(2, 2, figsize=(15, 10))

  image = axes[0, 0].imshow(jnp.abs(psi_0).T, origin='lower', cmap='inferno')
  axes[0, 0].set_title('Initial Intensity (z=0)')
  axes[0, 0].axis('off')
  plt.colorbar(image, ax=axes[0, 0], fraction=0.046)

  image = axes[0, 1].imshow(jnp.abs(psi_final).T, origin='lower',
                            cmap='inferno')
  axes[0, 1].set_title(f'Numerical Intensity (z={sim_config.lz:.2e})')
  axes[0, 1].axis('off')
  plt.colorbar(image, ax=axes[0, 1], fraction=0.046)

  plot_xz_intensity(axes[1, 0], sim_config, psi_history,
                    'Propagation in Random Media (xz)')

  ax = axes[1, 1]
  extent = (0, sim_config.lz, 0, sim_config.ly)
  image = ax.imshow(delta_n[:, centre_y, :], extent=extent, origin='lower',
                    cmap='gray', aspect='auto')
  plt.colorbar(image, ax=ax, label='delta n', fraction=0.046)
  ax.set_xlabel('z')
  ax.set_ylabel('x')
  ax.set_title('Refractive Index Perturbation (xz)')

  fig.tight_layout()
  fig.savefig('random_media.png')
  print("Saved plot to random_media.png")


if __name__ == "__main__":
  main()
