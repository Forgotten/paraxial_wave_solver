import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys
from scipy.special import fresnel

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paraxial_wave_solver as pws
Field = pws.Field

def fresnel_integral_diffraction(
    u: jax.Array, 
    w: float, 
    z: float, 
    k: float
) -> jax.Array:
  """Computes the 1D Fresnel diffraction pattern for a slit of width w.
  
  Field U(x) = (exp(ikz) / sqrt(i)) * integral ...
  Using scipy.special.fresnel (S(x), C(x)).
  
  Args:
    u: Coordinate array (x or y).
    w: Slit width.
    z: Propagation distance.
    k: Wavenumber.
    
  Returns:
    Complex field amplitude (1D).
  """
  # Fresnel number coordinates
  # The integral limits are related to sqrt(2 / (lambda * z)) * (x +/- w/2)
  # lambda = 2*pi / k
  # alpha = sqrt(k / (pi * z))
  
  alpha = jnp.sqrt(k / (jnp.pi * z))
  
  xi1 = alpha * (u + w/2)
  xi2 = alpha * (u - w/2)
  
  # Fresnel integrals C(x) and S(x)
  # Note: scipy.special.fresnel returns (S, C)
  s1, c1 = fresnel(xi1)
  s2, c2 = fresnel(xi2)
  
  # C = c1 - c2, S = s1 - s2
  C = c1 - c2
  S = s1 - s2
  
  # U(x) = (exp(ikz) / sqrt(2i)) * ( (C + iS) ) ?
  # Standard result: 
  # U(x) = exp(ikz) * (1/sqrt(2i)) * [ (C(xi1)-C(xi2)) + i(S(xi1)-S(xi2)) ]
  # 1/sqrt(2i) = 1/sqrt(2 * exp(i pi/2)) = 1 / (sqrt(2) * exp(i pi/4))
  # = exp(-i pi/4) / sqrt(2) = (1 - i) / 2
  
  factor = (1 - 1j) / 2.0
  field = jnp.exp(1j * k * z) * factor * (C + 1j * S)
  
  return field

from jax.scipy.special import erf

def get_square_aperture_analytical(
    sim_config: pws.SimulationConfig, 
    width: float, 
    z: float,
    smooth_sigma: float = 0.0
) -> Field:
  """Computes analytical diffraction from a square aperture.
  
  If smooth_sigma > 0, the initial aperture is convolved with a Gaussian 
  of standard deviation smooth_sigma (mollified).
  """
  
  # Grid
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  
  # Center coordinates
  x0 = sim_config.lx / 2.0
  y0 = sim_config.ly / 2.0
  
  dx = x - x0
  dy = y - y0
  
  if z == 0:
    if smooth_sigma > 0:
      # Smoothed box using Erf: 0.5 * (erf((x+w/2)/sig) - erf((x-w/2)/sig)).
      
      def smoothed_box(u, w, sig):
        return 0.5 * (erf((u + w/2) / (jnp.sqrt(2) * sig)) - 
                      erf((u - w/2) / (jnp.sqrt(2) * sig)))
      
      mask_x = smoothed_box(dx, width, smooth_sigma)
      mask_y = smoothed_box(dy, width, smooth_sigma)
      return jnp.outer(mask_x, mask_y).astype(complex)
    else:
      # Top hat
      mask_x = jnp.abs(dx) <= width / 2.0
      mask_y = jnp.abs(dy) <= width / 2.0
      return jnp.outer(mask_x, mask_y).astype(complex)
  
  # 1D diffraction patterns (Hard Aperture)
  # Redefine helper to NOT include exp(ikz)
  def fresnel_envelope(u, w, z, k):
    alpha = jnp.sqrt(k / (jnp.pi * z))
    xi1 = alpha * (u + w/2)
    xi2 = alpha * (u - w/2)
    s1, c1 = fresnel(xi1)
    s2, c2 = fresnel(xi2)
    C = c1 - c2
    S = s1 - s2
    factor = (1 - 1j) / 2.0
    return factor * (C + 1j * S)

  Ux = fresnel_envelope(dx, width, z, sim_config.k0)
  Uy = fresnel_envelope(dy, width, z, sim_config.k0)
  
  if smooth_sigma > 0:
    # Convolve 1D results with Gaussian kernel using FFT (Circular Convolution)
    # to match the periodic boundary conditions of the spectral solver.
    
    # Construct kernel on the grid.
    x_kernel = jnp.arange(sim_config.nx) * sim_config.dx
    # Wrap coordinates: [0, 1, ..., L/2, -L/2, ..., -1]
    x_kernel = jnp.where(x_kernel > sim_config.lx/2, x_kernel - sim_config.lx, x_kernel)
    
    kernel = jnp.exp(-x_kernel**2 / (2 * smooth_sigma**2))
    kernel = kernel / jnp.sum(kernel) # Normalize
    
    # FFT Convolution
    Ux = jnp.fft.ifft(jnp.fft.fft(Ux) * jnp.fft.fft(kernel))
    Uy = jnp.fft.ifft(jnp.fft.fft(Uy) * jnp.fft.fft(kernel))
  
  return jnp.outer(Ux, Uy)

def main():
  # Setup configuration. Here we need a bigger domain to avoid PML artifacts.
  sim_config = pws.SimulationConfig(
    nx=1024,
    ny=1024,
    dx=0.1,
    dy=0.1,
    dz=0.5,
    nz=200,
    wavelength=1.0
  )
  
  # Need larger PML or domain because wide diffraction spread.
  pml_config = pws.PMLConfig(width_x=200, width_y=200, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  
  # Square aperture.
  width = 10.0
  smooth_sigma = 0.5 # Mollification width (5 grid points)
  
  # Initial condition (z=0).
  print(f"Initializing Square Aperture (width={width}, sigma={smooth_sigma})...")
  psi_0 = get_square_aperture_analytical(
    sim_config,
    width,
    z=0.0,
    smooth_sigma=smooth_sigma)
  
  # Refractive index (vacuum).
  def n_ref_fn(z):
    return jnp.ones((sim_config.nx, sim_config.ny))
    
  # Run Simulation
  print("Running simulation...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_fn)
  psi_final, psi_history = solver.solve(psi_0)
  print("Simulation complete.")
  
  # Analytical Solution at final z
  z_final = sim_config.lz
  psi_analytical = get_square_aperture_analytical(sim_config, width, z_final,
                                                  smooth_sigma=smooth_sigma)
  
  # Compute Error (excluding PML region)
  wx = pml_config.width_x
  wy = pml_config.width_y
  
  s_x = slice(wx, -wx) if wx > 0 else slice(None)
  s_y = slice(wy, -wy) if wy > 0 else slice(None)
  
  psi_final_inner = psi_final[s_x, s_y]
  psi_analytical_inner = psi_analytical[s_x, s_y]
  
  error_field_inner = jnp.abs(psi_final_inner - psi_analytical_inner)
  l2_norm_ana = jnp.linalg.norm(psi_analytical_inner)
  l2_error = jnp.linalg.norm(error_field_inner) / l2_norm_ana
  print(f"Relative L2 Error (inner domain) at z={z_final:.2f}: {l2_error:.2e}")
  
  # Full error field for visualization
  error_field = jnp.abs(psi_final - psi_analytical)
  
  # Visualize
  center_y = sim_config.ny // 2
  
  plt.figure(figsize=(15, 10))
  
  # 1. Intensity at z=0
  plt.subplot(2, 3, 1)
  plt.imshow(jnp.abs(psi_0).T, origin='lower', cmap='gray')
  plt.title(f'Initial Intensity (z=0)')
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
  
  # 5. X-cut Comparison
  plt.subplot(2, 3, 5)
  x = jnp.arange(sim_config.nx) * sim_config.dx
  plt.plot(x, jnp.abs(psi_final[:, center_y])**2, 'b-', label='Numerical')
  plt.plot(x, jnp.abs(psi_analytical[:, center_y])**2, 'r--', label='Analytical')
  plt.legend()
  plt.title('Intensity Profile (X-cut)')
  plt.xlabel('x')
  
  # 6. Phase Comparison (X-cut)
  plt.subplot(2, 3, 6)
  phase_num = jnp.unwrap(jnp.angle(psi_final[:, center_y]))
  phase_ana = jnp.unwrap(jnp.angle(psi_analytical[:, center_y]))
  plt.plot(x, phase_num, 'b-', label='Numerical')
  plt.plot(x, phase_ana, 'r--', label='Analytical')
  plt.legend()
  plt.title('Phase Profile (X-cut)')
  plt.xlabel('x')
  
  plt.tight_layout()
  plt.savefig('aperture_diffraction_benchmark.png')
  print("Saved benchmark plot to aperture_diffraction_benchmark.png")

if __name__ == "__main__":
  main()
