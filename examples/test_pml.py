
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paraxial_wave_solver as pws

def get_tilted_gaussian_analytical(sim_config, w0, theta, z):
  """Computes analytical tilted Gaussian beam."""
  k0 = sim_config.k0
  z_R = k0 * w0**2 / 2.0
  
  # Grid.
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')
  
  # Initial Center (same as in test setup).
  x0 = 35.0
  y0 = sim_config.ly / 2.0
  
  # Center shift due to tilt.
  # x_c(z) = x0 + z * tan(theta) ~ x0 + z * theta.
  xc = x0 + z * jnp.tan(theta)
  yc = y0
  
  # Beam parameters at z.
  w_z = w0 * jnp.sqrt(1 + (z/z_R)**2)
  R_z = z * (1 + (z_R/z)**2) if z != 0 else jnp.inf
  zeta_z = jnp.arctan(z/z_R) # Gouy phase.
  
  # Transverse coordinates relative to beam center.
  dx = X - xc
  dy = Y - yc
  r2 = dx**2 + dy**2
  
  # Amplitude.
  amplitude = (w0 / w_z) * jnp.exp(-r2 / w_z**2)
  
  # Phases.
  # Tilt vector k_t = (k sin(theta), 0).
  # Phase is exp(i k_t . r - i (k_t^2 / 2k) z).
  # k_t ~ k theta.
  qx = k0 * jnp.sin(theta)
  tilt_phase = qx * X - (qx**2 / (2 * k0)) * z
  
  curvature_phase = k0 * r2 / (2 * R_z) if z != 0 else 0.0
  gouy_phase = zeta_z
  
  psi = amplitude * jnp.exp(1j * (tilt_phase + curvature_phase - gouy_phase))
  return psi

def run_pml_test():
  """Runs a test comparing standard absorbing FD vs complex coordinate stretching FD."""
  
  # Configuration to hit the boundary.
  sim_config = pws.SimulationConfig(
    nx=256, ny=256, 
    dx=0.2, dy=0.2, dz=0.1, 
    nz=400, # Propagate further: 400 * 0.1 = 40.0 units.
    wavelength=1.0
  )
  
  # We use a Gaussian beam moving towards the boundary to test PML.
  w0 = 5.0
  k0 = sim_config.k0
  
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')
  
  # Initial setup (must match analytical).
  x0 = 35.0 
  y0 = sim_config.ly / 2.0
  
  # Initial condition: Gaussian.
  r2 = (X - x0)**2 + (Y - y0)**2
  psi_0_amp = jnp.exp(-r2 / w0**2)
  
  # Add phase to make it move towards X boundary (right).
  theta = 0.3 # radians.
  psi_0 = psi_0_amp * jnp.exp(1j * k0 * jnp.sin(theta) * X)

  # 1. Standard FD with absorbing layer.
  print("Running FD with Standard Absorbing Layer...")
  width_pixels = 20
  pml_config_std = pws.PMLConfig(
    width_x=width_pixels,
    width_y=width_pixels,
    strength=2.0,
    use_complex_stretching=False
  )
  solver_config = pws.SolverConfig(method='finite_difference', fd_order=4)
  solver_std = pws.ParaxialWaveSolver(
    sim_config,
    solver_config,
    pml_config_std,
    lambda z: jnp.ones((256,256))
  )
  psi_final_std, _ = solver_std.solve(psi_0)
  
  # 2. FD with complex coordinate stretching.
  print("Running FD with Complex Coordinate Stretching...")
  pml_config_ccs = pws.PMLConfig(
    width_x=width_pixels,
    width_y=width_pixels,
    strength=2.0,
    use_complex_stretching=True
  )
  solver_ccs = pws.ParaxialWaveSolver(
    sim_config, solver_config, pml_config_ccs, lambda z: jnp.ones((256,256))
  )
  psi_final_ccs, _ = solver_ccs.solve(psi_0)

  # 3. Analytical solution.
  print("Computing Analytical Solution...")
  psi_analytical = get_tilted_gaussian_analytical(
    sim_config, w0, theta, sim_config.lz
  )

  # Error Analysis (Inner Domain).
  # Exclude PML regions.
  mask = jnp.ones_like(X)
  pml_w = width_pixels * sim_config.dx
  mask = jnp.where((X < pml_w) | (X > sim_config.lx - pml_w), 0.0, mask)
  mask = jnp.where((Y < pml_w) | (Y > sim_config.ly - pml_w), 0.0, mask)
  
  err_std_field = jnp.abs(psi_final_std - psi_analytical) * mask
  err_ccs_field = jnp.abs(psi_final_ccs - psi_analytical) * mask
  
  l2_err_std = jnp.linalg.norm(err_std_field) / jnp.linalg.norm(psi_analytical * mask)
  l2_err_ccs = jnp.linalg.norm(err_ccs_field) / jnp.linalg.norm(psi_analytical * mask)
  
  print(f"Standard Absorbing L2 Error (Inner): {l2_err_std:.2e}")
  print(f"Complex Stretching L2 Error (Inner): {l2_err_ccs:.2e}")

  # Visualization.
  # 2 rows, 4 columns. Last column is for colorbars.
  # We use width_ratios to make the colorbar column thin.
  fig, axes = plt.subplots(
    2, 4, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]}
  )
  
  # --- Row 1: Intensities ---
  # Determine common scale.
  max_val = max(jnp.max(jnp.abs(psi_analytical)), 
                jnp.max(jnp.abs(psi_final_std)), 
                jnp.max(jnp.abs(psi_final_ccs)))
  
  # 1. Analytical.
  ax_ana = axes[0, 0]
  im_ana = ax_ana.imshow(
    jnp.abs(psi_analytical).T,
    origin='lower',
    cmap='inferno',
    vmin=0,
    vmax=max_val
  )
  ax_ana.set_title("Analytical |psi|")
  ax_ana.axis('off')
  
  # 2. Standard.
  ax_std = axes[0, 1]
  ax_std.imshow(
    jnp.abs(psi_final_std).T,
    origin='lower',
    cmap='inferno',
    vmin=0,
    vmax=max_val
  )
  ax_std.set_title("Standard Absorbing |psi|")
  ax_std.axis('off')
  
  # 3. CCS.
  ax_ccs = axes[0, 2]
  ax_ccs.imshow(
    jnp.abs(psi_final_ccs).T,
    origin='lower',
    cmap='inferno',
    vmin=0,
    vmax=max_val
  )
  ax_ccs.set_title("Complex Stretching |psi|")
  ax_ccs.axis('off')
  
  # 4. Colorbar for intensities.
  ax_cbar1 = axes[0, 3]
  plt.colorbar(im_ana, cax=ax_cbar1)
  ax_cbar1.set_title('|psi|')
  
  # --- Row 2: Errors ---
  # Determine common scale for errors.
  max_err = max(jnp.max(err_std_field), jnp.max(err_ccs_field))
  
  # 1. Blank.
  axes[1, 0].axis('off')
  
  # 2. Error Standard.
  ax_err_std = axes[1, 1]
  im_err = ax_err_std.imshow(
    err_std_field.T, origin='lower', cmap='viridis', vmin=0, vmax=max_err
  )
  ax_err_std.set_title(f"Error Standard (Inner)\nL2 = {l2_err_std:.2e}")
  ax_err_std.axis('off')
  
  # 3. Error CCS.
  ax_err_ccs = axes[1, 2]
  ax_err_ccs.imshow(
    err_ccs_field.T, origin='lower', cmap='viridis', vmin=0, vmax=max_err
  )
  ax_err_ccs.set_title(f"Error Stretching (Inner)\nL2 = {l2_err_ccs:.2e}")
  ax_err_ccs.axis('off')
  
  # 4. Colorbar for errors.
  ax_cbar2 = axes[1, 3]
  plt.colorbar(im_err, cax=ax_cbar2)
  ax_cbar2.set_title('|Error|')
  
  plt.tight_layout()
  plt.savefig('pml_comparison_test.png')
  print("Saved pml_comparison_test.png")

  # Check stability / NaN.
  if jnp.isnan(psi_final_ccs).any():
    print("ERROR: NaNs detected in Complex Coordinate Stretching solution.")
  else:
    print("Complex Coordinate Stretching solution is stable (no NaNs).")

if __name__ == "__main__":
  run_pml_test()
