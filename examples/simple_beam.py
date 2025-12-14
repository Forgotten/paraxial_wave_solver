import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paraxial_wave_solver as pws
Field = pws.Field

def get_analytical_beam(sim_config: pws.SimulationConfig, w0: float, z: float) -> Field:
  """Computes the analytical Gaussian beam solution at distance z."""
  k0 = sim_config.k0
  z_R = k0 * w0**2 / 2.0
  
  # Defining the grid.
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')
  
  # Center coordinates.
  x0 = sim_config.lx / 2.0
  y0 = sim_config.ly / 2.0
  r2 = (X - x0)**2 + (Y - y0)**2
  
  if z == 0:
    return jnp.exp(-r2 / w0**2)
    
  # Beam parameters at z.
  w_z = w0 * jnp.sqrt(1 + (z/z_R)**2)
  R_z = z * (1 + (z_R/z)**2)
  zeta_z = jnp.arctan(z/z_R)
  
  # Field amplitude and phase derived from propagator exp(i k r^2 / 2z).
  psi = (w0 / w_z) * jnp.exp(-r2 / w_z**2) * \
        jnp.exp(1j * k0 * r2 / (2 * R_z)) * \
        jnp.exp(-1j * zeta_z)
        
  return psi

def main():
  # Setup Configuration
  sim_config = pws.SimulationConfig(
    nx=512,
    ny=512,
    dx=0.25,
    dy=0.25,
    dz=1.0,
    nz=200,
    wavelength=1.0
  )
  
  # Vacuum propagation.
  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  
  # Initial condition.
  w0 = 10.0
  psi_0 = pws.gaussian_beam(sim_config, w0=w0)
  
  # Refractive index (vacuum).
  def n_ref_fn(z):
    return jnp.ones((sim_config.nx, sim_config.ny))
    
  # Run Simulation.
  print("Running simulation...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_fn)
  psi_final, psi_history = solver.solve(psi_0)
  print("Simulation complete.")
  
  # Analytical solution at final z.
  z_final = sim_config.lz
  psi_analytical = get_analytical_beam(sim_config, w0, z_final)
  
  # Compute error.
  # Normalize both to remove any global phase factor ambiguity if necessary,
  # though they should match.
  error_field = jnp.abs(psi_final - psi_analytical)
  l2_error = jnp.linalg.norm(error_field) / jnp.linalg.norm(psi_analytical)
  print(f"Relative L2 Error at z={z_final:.2f}: {l2_error:.2e}")
  
  # Visualize
  center_y = sim_config.ny // 2
  
  # XZ Intensity Plot
  field_xz = psi_history[:, :, center_y].T # (nz, nx) -> (nx, nz)
  intensity_xz = jnp.abs(field_xz)**2
  
  # Final slice comparison
  x = jnp.arange(sim_config.nx) * sim_config.dx
  intensity_final_num = jnp.abs(psi_final[:, center_y])**2
  intensity_final_ana = jnp.abs(psi_analytical[:, center_y])**2
  
  plt.figure(figsize=(15, 10))
  
  # 1. Propagation (XZ)
  plt.subplot(2, 2, 1)
  extent = [0, sim_config.lz, 0, sim_config.lx]
  plt.imshow(intensity_xz, extent=extent, origin='lower', cmap='inferno', 
             aspect='auto')
  plt.colorbar(label='Intensity')
  plt.xlabel('z')
  plt.ylabel('x')
  plt.title('Numerical Propagation (XZ)')
  
  # 2. Final Profile Comparison (1D cut)
  plt.subplot(2, 2, 2)
  plt.plot(x, intensity_final_num, 'b-', label='Numerical', linewidth=2)
  plt.plot(x, intensity_final_ana, 'r--', label='Analytical', linewidth=2)
  plt.xlabel('x')
  plt.ylabel('Intensity')
  plt.title(f'Profile at z={z_final:.2f}')
  plt.legend()
  plt.grid(True, alpha=0.3)
  
  # 3. Error Image (XY at final z)
  plt.subplot(2, 2, 3)
  plt.imshow(jnp.abs(psi_final - psi_analytical).T, origin='lower', cmap='viridis')
  plt.colorbar(label='|Error|')
  plt.title(f'Absolute Error Field at z={z_final:.2f}')
  
  # 4. Phase Comparison (1D cut)
  plt.subplot(2, 2, 4)
  phase_num = jnp.angle(psi_final[:, center_y])
  phase_ana = jnp.angle(psi_analytical[:, center_y])
  # Unwrap for better comparison
  phase_num = jnp.unwrap(phase_num)
  phase_ana = jnp.unwrap(phase_ana)
  
  plt.plot(x, phase_num, 'b-', label='Numerical')
  plt.plot(x, phase_ana, 'r--', label='Analytical')
  plt.xlabel('x')
  plt.ylabel('Phase (rad)')
  plt.title('Phase Comparison')
  plt.legend()
  plt.grid(True, alpha=0.3)
  
  plt.tight_layout()
  plt.savefig('gaussian_beam_comparison.png')
  print("Saved comparison plot to gaussian_beam_comparison.png")

if __name__ == "__main__":
  main()
