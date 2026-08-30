import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paraxial_wave_solver as pws

def main():
  # Setup Configuration.
  sim_config = pws.SimulationConfig(
    nx=400, ny=400, 
    dx=1e-4, dy=1e-4, dz=1e-2, 
    nz=250, 
    wavelength=632.8e-9,
    n0=1.33
  )
  
  pml_config = pws.PMLConfig(width_x=20, width_y=20, strength=5.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  
  # Initial Condition.
  w0 = 1e-3
  psi_0 = pws.gaussian_beam(sim_config, w0=w0)
  
  # Random Medium.
  key = jax.random.PRNGKey(42)
  delta_n = pws.random_medium(
    sim_config, correlation_length=1e-3, strength=4.4e-7, key=key
    )
  
  # Define refractive index function.
  def n_ref_fn(z):
    # Find index
    idx = jnp.clip(jnp.round(z / sim_config.dz).astype(int), 0, 
                   sim_config.nz - 1)
    return delta_n[:, :, idx]
    
  # Run Simulation.
  print("Running simulation in random media...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_fn)
  psi_final, psi_history = solver.solve(psi_0, z_0=0.0)
  print("Simulation complete.")
  
  # Defining the visualization.
  center_y = sim_config.ny // 2
  field_xz = psi_history[:, :, center_y].T
  intensity_xz = jnp.abs(field_xz)**2
  
  plt.figure(figsize=(12, 5))
  
  # Visualize
  plt.figure(figsize=(15, 10))
  
  # 1. Intensity at z=0
  plt.subplot(2, 3, 1)
  plt.imshow(jnp.abs(psi_0).T, origin='lower', cmap='inferno')
  plt.title(f'Initial Intensity (z=0)')#\nLG_{p_mode}{l_mode}')
  plt.axis('off')
  
  # 2. Numerical Intensity at z=final
  plt.subplot(2, 3, 2)
  plt.imshow(jnp.abs(psi_final).T, origin='lower', cmap='inferno')
  plt.title(f'Numerical Intensity (z={sim_config.nz*sim_config.dz:.2e})')
  plt.axis('off')

  plt.subplot(2, 3, 3)
  plt.imshow(intensity_xz, origin='lower', cmap='inferno', 
             aspect='auto')
  plt.colorbar(label='Intensity')
  plt.xlabel('z')
  plt.ylabel('x')
  plt.title('Propagation in Random Media (XZ)')
  
  plt.subplot(2, 3, 4)
  extent = (0, sim_config.nz*sim_config.dz, 0, sim_config.ny*sim_config.dy)
  # Plot refractive index slice
  plt.imshow(delta_n[:, center_y, :], extent=extent, origin='lower', 
             cmap='gray', aspect='auto')
  plt.colorbar(label='delta n')
  plt.xlabel('z')
  plt.ylabel('x')
  plt.title('Refractive Index Perturbation (XZ)')
  
  plt.tight_layout()
  plt.savefig('random_media.png')
  print("Saved plot to random_media.png")

if __name__ == "__main__":
  main()
