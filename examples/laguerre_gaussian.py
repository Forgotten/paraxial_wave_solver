import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys
from scipy.special import genlaguerre

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
  
  # Amplitude terms
  # term: (r * sqrt(2) / w(z))^|l|
  term_r = (r * jnp.sqrt(2) / w_z)**abs(l)
  
  amplitude = (w0 / w_z) * term_r * Lpl * jnp.exp(-r2 / w_z**2)
  
  # Phase terms
  # Gouy phase: (2p + |l| + 1) * zeta_z
  gouy_phase = (2 * p + abs(l) + 1) * zeta_z
  
  # Curvature phase: k * r^2 / 2R
  curvature_phase = k0 * r2 / (2 * R_z) if z != 0 else 0.0
  
  # Azimuthal phase: l * phi
  azimuthal_phase = l * phi
  
  # Total field
  psi = amplitude * \
        jnp.exp(1j * curvature_phase) * \
        jnp.exp(-1j * gouy_phase) * \
        jnp.exp(1j * azimuthal_phase)
  
  return psi

def main():
  # Setup Configuration
  sim_config = pws.SimulationConfig(
    nx=512, ny=512, 
    dx=0.2, dy=0.2, dz=1.0, 
    nz=200, 
    wavelength=1.0
  )
  
  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  
  # Parameters for LG_01 mode (Donut mode)
  w0 = 8.0
  p_mode = 0
  l_mode = 1
  
  # Initial Condition (z=0)
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
  psi_analytical = get_laguerre_gaussian_analytical(sim_config, w0, p_mode, l_mode, z_final)
  
  # Compute Error
  error_field = jnp.abs(psi_final - psi_analytical)
  l2_norm_ana = jnp.linalg.norm(psi_analytical)
  l2_error = jnp.linalg.norm(error_field) / l2_norm_ana
  print(f"Relative L2 Error at z={z_final:.2f}: {l2_error:.2e}")
  
  # Visualize
  center_y = sim_config.ny // 2
  
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
  plt.savefig('laguerre_gaussian_benchmark.png')
  print("Saved benchmark plot to laguerre_gaussian_benchmark.png")

if __name__ == "__main__":
  main()
