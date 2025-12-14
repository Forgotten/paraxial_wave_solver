import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paraxial_wave_solver import (
    SimulationConfig, 
    SolverConfig, 
    PMLConfig, 
    ParaxialWaveSolver, 
    gaussian_beam
)

def main():
  # Setup Configuration
  sim_config = SimulationConfig(
    nx=256, ny=256, 
    dx=0.5, dy=0.5, dz=1.0, 
    nz=100, 
    wavelength=1.0
  )
  
  # Vacuum propagation.
  pml_config = PMLConfig(width_x=20, width_y=20, strength=2.0)
  solver_config = SolverConfig(method='spectral', stepper='split_step')
  
  # Initial Condition.
  w0 = 10.0
  psi_0 = gaussian_beam(sim_config, w0=w0)
  
  # Refractive Index (Vacuum).
  def n_ref_fn(z):
    return jnp.ones((sim_config.nx, sim_config.ny))
    
  # Run Simulation.
  print("Running simulation...")
  solver = ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_fn)
  psi_final, psi_history = solver.solve(psi_0)
  print("Simulation complete.")
  
  # Visualize.
  # Plot XZ cut at center Y
  center_y = sim_config.ny // 2
  field_xz = psi_history[:, :, center_y].T # (nz, nx) -> (nx, nz)
  intensity_xz = jnp.abs(field_xz)**2
  
  plt.figure(figsize=(10, 6))
  extent = [0, sim_config.lz, 0, sim_config.lx]
  plt.imshow(intensity_xz, extent=extent, origin='lower', cmap='inferno', 
             aspect='auto')
  plt.colorbar(label='Intensity')
  plt.xlabel('z')
  plt.ylabel('x')
  plt.title('Gaussian Beam Propagation (Vacuum)')
  plt.savefig('gaussian_beam_xz.png')
  print("Saved plot to gaussian_beam_xz.png")

if __name__ == "__main__":
  main()
