import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys
from scipy.special import genlaguerre
from math import factorial

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paraxial_wave_solver as pws
Field = pws.Field

def get_laguerre_gaussian_analytical(
    sim_config: pws.SimulationConfig, 
    w0: float, 
    p: int, 
    l: int, 
    z: float
) -> Field:
  """Computes the analytical Laguerre-Gaussian (LG_pl) beam solution."""
  k0 = sim_config.k0
  z_R = k0 * w0**2 / 2.0
  
  # Grid
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')
  
  # Center coordinates
  x0 = sim_config.lx / 2.0
  y0 = sim_config.ly / 2.0
  dx = X - x0
  dy = Y - y0
  r2 = dx**2 + dy**2
  r = jnp.sqrt(r2)
  phi = jnp.arctan2(dy, dx)
  
  # Beam parameters at z
  w_z = w0 * jnp.sqrt(1 + (z/z_R)**2)
  R_z = z * (1 + (z_R/z)**2) if z != 0 else jnp.inf
  zeta_z = jnp.arctan(z/z_R)
  
  # Generalized Laguerre polynomial L_p^|l|(x)
  # Argument is 2 * r^2 / w(z)^2
  arg = 2 * r2 / w_z**2
  Lpl = jnp.polyval(jnp.array(genlaguerre(p, abs(l)).coef), arg)
  
  # Amplitude terms.
  # term: (r * sqrt(2) / w(z))^|l|
  term_r = (r * jnp.sqrt(2) / w_z)**abs(l)
  
  # Normalization Constant: sqrt(2 * p! / (pi * (p + |l|)!))
  norm_factor = jnp.sqrt(2 * factorial(p) / (jnp.pi * factorial(p + abs(l))))
  
  amplitude = norm_factor * (w0 / w_z) * term_r * Lpl * jnp.exp(-r2 / w_z**2)
  
  # Phase terms
  # Gouy phase: (2p + |l| + 1) * zeta_z
  gouy_phase = (2 * p + abs(l) + 1) * zeta_z
  
  # Propagation phase
  propagation_phase = k0 * z

  # Curvature phase: k * r^2 / 2R
  curvature_phase = k0 * r2 / (2 * R_z) if z != 0 else 0.0
  
  # Azimuthal phase: l * phi
  azimuthal_phase = l * phi
  
  # Total field
  psi = amplitude * \
        jnp.exp(1j * curvature_phase) * \
        jnp.exp(-1j * gouy_phase) * \
        jnp.exp(1j * azimuthal_phase) * \
        jnp.exp(1j * propagation_phase)
  
  return psi

def run_simulation(p_mode, l_mode, output_filename, title_suffix=""):
  """Runs the simulation for a given LG mode and saves the plot."""
  
  # Setup Configuration
  sim_config = pws.SimulationConfig(
    nx=512, ny=512, 
    dx=0.2, dy=0.2, dz=1.0, 
    nz=200, 
    wavelength=1.0
  )
  
  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  
  w0 = 8.0

  # Initial Condition (z=0)
  print(f"\n--- Running Simulation for LG_{p_mode}{l_mode} ---")
  print(f"Initializing Laguerre-Gaussian LG_{p_mode}{l_mode} beam...")
  psi_0 = get_laguerre_gaussian_analytical(sim_config, w0, p_mode, l_mode, z=0.0)
  
  # Refractive Index (Vacuum)
  def n_ref_fn(z):
    return jnp.ones((sim_config.nx, sim_config.ny))
    
  # Run Simulation
  print("Running simulation...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_fn)
  psi_final, psi_history = solver.solve(psi_0)
  print("Simulation complete.")
  
  # Analytical Solution at final z
  z_final = sim_config.lz
  
  # Compute Error History and Phase Check
  print("\n--- Error Analysis over Propagation ---")
  zs_hist = jnp.arange(sim_config.nz + 1) * sim_config.dz
  
  # Check error every 25 steps
  for i in range(0, sim_config.nz + 1, 25):
      z_curr = zs_hist[i]
      psi_num = psi_history[i]
      
      # Determine analytical solution (Full Field)
      psi_ana_full = get_laguerre_gaussian_analytical(sim_config, w0, p_mode, l_mode, z_curr)
      
      # Demodulate to compare with Envelope from solver
      # Solver computes envelope psi, Analytical has psi * exp(i k z)
      psi_ana_env = psi_ana_full * jnp.exp(-1j * sim_config.k0 * z_curr)
      
      err_f = jnp.abs(psi_num - psi_ana_env)
      norm_ana = jnp.linalg.norm(psi_ana_env)
      rel_err = jnp.linalg.norm(err_f) / norm_ana
      print(f"z={z_curr:6.2f}: Rel L2 Error = {rel_err:.2e}")

  # For plotting and final error report, use the final step
  psi_analytical_full = get_laguerre_gaussian_analytical(sim_config, w0, p_mode, l_mode, z_final)
  psi_analytical = psi_analytical_full * jnp.exp(-1j * sim_config.k0 * z_final)
  
  # Compute Error
  error_field = jnp.abs(psi_final - psi_analytical)
  l2_norm_ana = jnp.linalg.norm(psi_analytical)
  l2_error = jnp.linalg.norm(error_field) / l2_norm_ana
  print(f"\nFinal Relative L2 Error at z={z_final:.2f}: {l2_error:.2e}")
  
  # Visualize
  plt.figure(figsize=(15, 10))
  
  # 1. Intensity at z=0
  plt.subplot(2, 3, 1)
  plt.imshow(jnp.abs(psi_0).T, origin='lower', cmap='inferno')
  plt.title(f'Initial Intensity (z=0)\nLG_{p_mode}{l_mode}')
  plt.axis('off')
  
  # 2. Numerical Intensity at z=final
  plt.subplot(2, 3, 2)
  plt.imshow(jnp.abs(psi_final).T, origin='lower', cmap='inferno')
  plt.title(f'Numerical Intensity (z={z_final})')
  plt.axis('off')
  
  # 3. Analytical Intensity at z=final
  plt.subplot(2, 3, 3)
  plt.imshow(jnp.abs(psi_analytical).T, origin='lower', cmap='inferno')
  plt.title(f'Analytical Intensity (z={z_final})')
  plt.axis('off')
  
  # 4. Error Field
  plt.subplot(2, 3, 4)
  plt.imshow(error_field.T, origin='lower', cmap='viridis')
  plt.colorbar(label='|Error|')
  plt.title('Absolute Error Field')
  plt.axis('off')
  
  # 5. Phase at z=final (Numerical)
  plt.subplot(2, 3, 5)
  plt.imshow(jnp.angle(psi_final).T, origin='lower', cmap='hsv')
  plt.title('Numerical Phase')
  plt.axis('off')
  
  # 6. Phase at z=final (Analytical)
  plt.subplot(2, 3, 6)
  plt.imshow(jnp.angle(psi_analytical).T, origin='lower', cmap='hsv')
  plt.title('Analytical Phase')
  plt.axis('off')
  
  plt.tight_layout()
  plt.savefig(output_filename)
  print(f"Saved benchmark plot to {output_filename}")

def main():
  # Run original requested case: LG_01 (Donut)
  run_simulation(p_mode=0, l_mode=1, output_filename='laguerre_gaussian_benchmark.png')
  
  # Run higher order case: LG_22 (More complex structure)
  # p=2 gives 2 radial nodes, l=2 gives azimuthal variation
  run_simulation(p_mode=2, l_mode=2, output_filename='laguerre_gaussian_benchmark_high_order.png')

if __name__ == "__main__":
  main()
