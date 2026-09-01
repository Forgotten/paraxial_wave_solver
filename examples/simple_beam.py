"""Gaussian beam propagating in vacuum, checked against the analytical mode.

Run after installing the package (`pip install -e .`):

    python examples/simple_beam.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from _plotting import plot_xz_intensity, relative_l2_error

import paraxial_wave_solver as pws


def main():
  sim_config = pws.SimulationConfig(
    nx=512, ny=512,
    dx=0.25, dy=0.25, dz=1.0,
    nz=1000,
    wavelength=1.0,
    n0=1.0,
  )
  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')

  w0 = 10.0
  psi_0 = pws.gaussian_beam(sim_config, w0=w0, z=0.0)

  # Vacuum: delta_n = 0 is the solver default.
  print("Running simulation...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config)
  # Keep every 10th plane: the xz view does not need 1000 of them, and the
  # full history would be 1 GB.
  psi_final, psi_history = solver.solve(psi_0, save_every=10)
  print("Simulation complete.")

  z_final = sim_config.lz
  psi_analytical = pws.gaussian_beam(
    sim_config, w0=w0, z=z_final, envelope_only=True
  )
  print(f"Relative L2 Error at z={z_final:.2f}: "
        f"{relative_l2_error(psi_final, psi_analytical):.2e}")

  x = jnp.arange(sim_config.nx) * sim_config.dx
  centre_y = sim_config.ny // 2

  fig, axes = plt.subplots(2, 2, figsize=(15, 10))

  plot_xz_intensity(axes[0, 0], sim_config, psi_history,
                    'Numerical Propagation (xz)')

  ax = axes[0, 1]
  ax.plot(x, jnp.abs(psi_final[:, centre_y])**2, 'b-', label='Numerical',
          linewidth=2)
  ax.plot(x, jnp.abs(psi_analytical[:, centre_y])**2, 'r--',
          label='Analytical', linewidth=2)
  ax.set_title(f'Profile at z={z_final:.2f}')
  ax.set_xlabel('x')
  ax.set_ylabel('Intensity')
  ax.legend()
  ax.grid(True, alpha=0.3)

  ax = axes[1, 0]
  image = ax.imshow(jnp.abs(psi_final - psi_analytical).T, origin='lower',
                    cmap='viridis')
  plt.colorbar(image, ax=ax, label='|Error|', fraction=0.046)
  ax.set_title(f'Absolute Error Field at z={z_final:.2f}')
  ax.axis('off')

  ax = axes[1, 1]
  ax.plot(x, jnp.unwrap(jnp.angle(psi_final[:, centre_y])), 'b-',
          label='Numerical')
  ax.plot(x, jnp.unwrap(jnp.angle(psi_analytical[:, centre_y])), 'r--',
          label='Analytical')
  ax.set_title('Phase Comparison')
  ax.set_xlabel('x')
  ax.set_ylabel('Phase (rad)')
  ax.legend()
  ax.grid(True, alpha=0.3)

  fig.tight_layout()
  fig.savefig('gaussian_beam_comparison.png')
  print("Saved comparison plot to gaussian_beam_comparison.png")


if __name__ == "__main__":
  main()
