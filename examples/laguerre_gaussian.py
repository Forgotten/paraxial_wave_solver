import os
import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add project root to path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paraxial_wave_solver as pws


def run_simulation(p_mode, l_mode, output_filename, title_suffix=""):
  """Runs the simulation for a given LG mode and saves the plot."""
  
  # Setup Configuration.
  sim_config = pws.SimulationConfig(
    nx=512, ny=512, 
    dx=0.0075, dy=0.0075, dz=1.0, 
    nz=250, # propagation length (2.5 m).
    wavelength=632.8e-7 # Wavelength in centimeters (632.8 nm).
  )
  
  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  
  w0 = 3.0e-1 # Beam waist in centimeters (3 mm).

  # Initial Condition (z=0).
  print(f"\n--- Running Simulation for LG_{p_mode}{l_mode} ---")
  print(f"Initializing Laguerre-Gaussian LG_{p_mode}{l_mode} beam...")
  psi_0 = pws.laguerre_gaussian_beam(sim_config, w0, p=p_mode, l=l_mode, z=0.0)
  
  # Refractive Index (Vacuum).
  def n_ref_fn(z):
    return jnp.ones((sim_config.nx, sim_config.ny))
    
  # Run Simulation.
  print("Running simulation...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_fn)
  psi_final, psi_history = solver.solve(psi_0)
  print("Simulation complete.")
  
  # Analytical Solution at final z.
  z_final = sim_config.lz
  
  # Compute Error History and Phase Check.
  print("\n--- Error Analysis over Propagation ---")
  zs_hist = jnp.arange(sim_config.nz + 1) * sim_config.dz
  
  # Check error every 25 steps.
  for i in range(0, sim_config.nz + 1, 25):
    z_curr = zs_hist[i]
    psi_num = psi_history[i]
    
    # Determine analytical solution (Envelope directly).
    psi_ana_env = pws.laguerre_gaussian_beam(
      sim_config, w0, p=p_mode, l=l_mode, z=z_curr, envelope_only=True
    )
    
    err_f = jnp.abs(psi_num - psi_ana_env)
    norm_ana = jnp.linalg.norm(psi_ana_env)
    rel_err = jnp.linalg.norm(err_f) / norm_ana
    print(f"z={z_curr:6.2f}: Rel L2 Error = {rel_err:.2e}")

  # For plotting and final error report, use the final step.
  psi_analytical = pws.laguerre_gaussian_beam(
    sim_config, w0, p=p_mode, l=l_mode, z=z_final, envelope_only=True
  )
  
  # Compute Error.
  error_field = jnp.abs(psi_final - psi_analytical)
  l2_norm_ana = jnp.linalg.norm(psi_analytical)
  l2_error = jnp.linalg.norm(error_field) / l2_norm_ana
  print(f"\nFinal Relative L2 Error at z={z_final:.2f}: {l2_error:.2e}")
  
  # Visualize.
  plt.figure(figsize=(15, 10))
  
  # 1. Intensity at z=0.
  plt.subplot(2, 3, 1)
  plt.imshow(jnp.abs(psi_0).T, origin='lower', cmap='inferno')
  plt.title(f'Initial Intensity (z=0)\nLG_{p_mode}{l_mode}')
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
  
  # 5. Phase at z=final (Numerical).
  plt.subplot(2, 3, 5)
  plt.imshow(jnp.angle(psi_final).T, origin='lower', cmap='hsv')
  plt.title('Numerical Phase')
  plt.axis('off')
  
  # 6. Phase at z=final (Analytical).
  plt.subplot(2, 3, 6)
  plt.imshow(jnp.angle(psi_analytical).T, origin='lower', cmap='hsv')
  plt.title('Analytical Phase')
  plt.axis('off')
  
  plt.tight_layout()
  plt.savefig(output_filename)
  print(f"Saved benchmark plot to {output_filename}")


def main():
  # Run original requested case: LG_01 (Donut).
  run_simulation(p_mode=0, l_mode=1, output_filename='laguerre_gaussian_01.png')

  # Run higher order cases: LG_14, LG_0-6, LG_1-9 (More complex structure).
  run_simulation(p_mode=1, l_mode=4, output_filename='laguerre_gaussian_benchmark_14.png')
  run_simulation(p_mode=0, l_mode=-6, output_filename='laguerre_gaussian_benchmark_0-6.png')
  run_simulation(p_mode=1, l_mode=9, output_filename='laguerre_gaussian_benchmark_19.png')


if __name__ == "__main__":
  main()
