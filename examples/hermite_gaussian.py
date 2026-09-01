import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add project root to path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paraxial_wave_solver as pws


def main():
  # Setup Configuration.
  sim_config = pws.SimulationConfig(
    nx=512, ny=512,
    dx=0.2, dy=0.2, dz=1.0,
    nz=200,
    wavelength=1.0,
    n0=1.0
  )

  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')

  # Parameters for HG_11 mode.
  w0 = 8.0
  n_mode = 1
  m_mode = 1

  # Initial Condition (z=0).
  print(f"Initializing Hermite-Gaussian HG_{n_mode}{m_mode} beam...")
  psi_0 = pws.hermite_gaussian_beam(sim_config, w0, n=n_mode, m=m_mode, z=0.0)

  # Vacuum: delta_n = 0 is the solver default.
  # Run Simulation.
  print("Running simulation...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config)
  psi_final, psi_history = solver.solve(psi_0)
  print("Simulation complete.")

  # Analytical Solution at final z (envelope form to match solver).
  z_final = sim_config.lz
  psi_analytical = pws.hermite_gaussian_beam(
    sim_config, w0, n=n_mode, m=m_mode, z=z_final, envelope_only=True
  )

  # Compute Error.
  error_field = jnp.abs(psi_final - psi_analytical)
  l2_norm_ana = jnp.linalg.norm(psi_analytical)
  l2_error = jnp.linalg.norm(error_field) / l2_norm_ana
  print(f"Relative L2 Error at z={z_final:.2f}: {l2_error:.2e}")

  # Visualize.
  center_y = sim_config.ny // 2

  plt.figure(figsize=(15, 10))

  # 1. Intensity at z=0.
  plt.subplot(2, 3, 1)
  plt.imshow(jnp.abs(psi_0).T, origin='lower', cmap='inferno')
  plt.title(f'Initial Intensity (z=0)\nHG_{n_mode}{m_mode}')
  plt.axis('off')

  # 2. Numerical Intensity at z=final.
  plt.subplot(2, 3, 2)
  plt.imshow(jnp.abs(psi_final).T, origin='lower', cmap='inferno')
  plt.title(f'Numerical Intensity (z={z_final})')
  plt.axis('off')

  # 3. Analytical Intensity at z=final.
  plt.subplot(2, 3, 3)
  plt.imshow(jnp.abs(psi_analytical).T, origin='lower', cmap='inferno')
  plt.title(f'Analytical Intensity (z={z_final})')
  plt.axis('off')

  # 4. Error Field.
  plt.subplot(2, 3, 4)
  plt.imshow(error_field.T, origin='lower', cmap='viridis')
  plt.colorbar(label='|Error|')
  plt.title('Absolute Error Field')
  plt.axis('off')

  # 5. X-cut Comparison.
  plt.subplot(2, 3, 5)
  x = jnp.arange(sim_config.nx) * sim_config.dx
  plt.plot(x, jnp.abs(psi_final[:, center_y])**2, 'b-', label='Numerical')
  plt.plot(x, jnp.abs(psi_analytical[:, center_y])**2, 'r--', label='Analytical')
  plt.legend()
  plt.title('Intensity Profile (X-cut)')
  plt.xlabel('x')

  # 6. Phase Comparison (X-cut).
  plt.subplot(2, 3, 6)
  phase_num = jnp.unwrap(jnp.angle(psi_final[:, center_y]))
  phase_ana = jnp.unwrap(jnp.angle(psi_analytical[:, center_y]))
  plt.plot(x, phase_num, 'b-', label='Numerical')
  plt.plot(x, phase_ana, 'r--', label='Analytical')
  plt.legend()
  plt.title('Phase Profile (X-cut)')
  plt.xlabel('x')

  plt.tight_layout()
  plt.savefig('hermite_gaussian_benchmark.png')
  print("Saved benchmark plot to hermite_gaussian_benchmark.png")


if __name__ == "__main__":
  main()
