from typing import Callable
from numpy import power
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import os
import sys
from scipy.special import genlaguerre
from math import factorial
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paraxial_wave_solver as pws
from paraxial_wave_solver.src.utils import random_medium_spectral
#from paraxial_wave_solver.src.utils import get_thermal_turbulence_fn
Field = pws.Field



def get_laguerre_gaussian_analytical(
    sim_config: pws.SimulationConfig, 
    w0: float,
    power: float,
    p: int, 
    l: int,
    z: float
) -> Field:
  """Computes the analytical Laguerre-Gaussian (LG_pl) beam solution."""
  k0 = sim_config.k0 * sim_config.n0
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

  # Normalize to desired power  
  P_current = jnp.sum(jnp.abs(psi)**2) * sim_config.dx * sim_config.dy
  scale = jnp.sqrt(power / (P_current))
  psi *= scale
  return psi


def initialize_laguerre(a1, a2, a3, a4, sim_config, w0, power):
  """Initializes the LG beam."""
  def lg_beam(z):
    p_mode = 0
    l_mode = 1
    psi_0 = a1*get_laguerre_gaussian_analytical(sim_config, w0, power, p_mode, l_mode, z=z)
    p_mode = 1
    l_mode = 4
    psi_0 += a2*get_laguerre_gaussian_analytical(sim_config, w0, power, p_mode, l_mode, z=z)
    p_mode = 0
    l_mode = -6
    psi_0 += a3*get_laguerre_gaussian_analytical(sim_config, w0, power, p_mode, l_mode, z=z)
    p_mode = 1
    l_mode = 9
    psi_0 += a4*get_laguerre_gaussian_analytical(sim_config, w0, power, p_mode, l_mode, z=z)
    return psi_0
  return lg_beam


def get_turbulence_func(Cn2, L0, l0):
  """Returns the turbulence function."""
  def n_ref_fn_r(z):
    key = jr.PRNGKey(0)
    V = random_medium_spectral(
        sim_config, Cn2=Cn2, L0=L0, l0=l0, key=key
    )
    # Find index
    idx = jnp.clip(jnp.round(z / sim_config.dz).astype(int), 0, 
                    sim_config.nz - 1)
    return jnp.sqrt(sim_config.n0**2 + V[:, :, idx]) # water medium (n=1.33)
  return n_ref_fn_r


def run_simulation_total(a1, a2, a3, a4, Cn2, N_simulations=3):
  # Setup Configuration
  sim_config = pws.SimulationConfig(
    nx=500, ny=500, 
    dx=1e-4, dy=1e-4, dz=2.5e-2, 
    nz=100, # propagation length (2.5 m)
    wavelength=632.8e-9, # Wavelength in meters (632.8 nm)
    n0=1.33 # refractive index (underwater=1.33, vacuum=1.0)
  )
  # Experimental Parameters
  # Ra = 1.85e9 (Rayleigh number) -> Indicates strong thermal turbulence
  #Cn2=4.4e-10 # Measured Cn2 [m^-2/3]
  #Cn2=4.4e-8
  # Estimated scales for tank experiment
  L0=2e-2     # Outer scale (0.02 m) - Estimate
  l0=1e-3    # Inner scale (0.001 m) - Estimate for size of tank (5x5 cm^2)

  power = 2e-3  # Total laser power in Watts (2mW)
  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  w0 = 3.0e-3 # Beam waist in meters (3 mm)
  lg_beam = initialize_laguerre(a1, a2, a3, a4, sim_config, w0, power)
  
  # Initialize field
  psi_0 = lg_beam(0.0)
  
  # Total propagation distance = 2.5 meters
  # Chunk size = 0.5 meters (500 steps)
  # num_chunks = 5
  
  
  # Use N_simulations argument as number of chunks to extend
  num_chunks = N_simulations
  if num_chunks < 1: num_chunks = 1
  
  psi_current = psi_0
  psi_history_total = None # Will initialize after first chunk
  
  current_z = 0.0
  print(f"Starting simulation: {num_chunks} chunks of {sim_config.nz} steps each.")

  for i in range(num_chunks):
      print(f"Simulating chunk {i+1}/{num_chunks} (z={current_z:.2f}m)...")
      
      # IMPORTANT: Use a different seed for each chunk to get fresh random medium
      # Note: This breaks longitudinal correlation at the boundary, but L0 ~ 0.5m.
      seed = i 
      #n_ref_fn_r = get_thermal_turbulence_fn(sim_config, Cn2=Cn2, L0=L0, l0=l0, seed=seed)
      
      # We must wrap n_ref_fn_r to handle the absolute z passed by the solver
      # The generated volume is always for indices 0..nz-1
      key = jax.random.PRNGKey(seed)
      #delta_n = pws.random_medium(
      #  sim_config, correlation_length=l0, strength=strength, key=key
      #  )
      delta_n = random_medium_spectral(sim_config, Cn2, L0, l0, key)
      chunk_start_z = current_z
      def n_ref_wrapper(z):
          # Shift z to be relative to the start of this chunk
          z_rel = z - chunk_start_z 
          idx = jnp.clip(jnp.round(z_rel / sim_config.dz).astype(int), 0, 
                           sim_config.nz - 1)
          return delta_n[:, :, idx]
      
      solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_wrapper)
      
      # Solve for this chunk
      psi_final_chunk, psi_history_chunk = solver.solve(psi_current, current_z)#, save_interval=10)
      
      # Concatenate history
      if psi_history_total is None:
          psi_history_total = psi_history_chunk[:-1]
      else:
          psi_history_total = jnp.concatenate([psi_history_total, psi_history_chunk[:-1]], axis=0)
          
      # Update state for next chunk
      psi_current = psi_final_chunk
      current_z += sim_config.nz * sim_config.dz
      
  # Append final field
  psi_history_total = jnp.concatenate([psi_history_total, psi_current[None, ...]], axis=0)
  
  z_final = current_z
  print(f"Calculating analytical solution at z={z_final:.2f}m...")
  psi_analytical = lg_beam(z_final) * jnp.exp(-1j * sim_config.k0 * sim_config.n0 * z_final)
  
  return psi_history_total, psi_0, psi_analytical, z_final, sim_config

def main(N_simulations, Cn2, output_filename):
  a1 = 1.0
  a2 = 0.0
  a3 = 1.0
  a4 = 1.0

  psi_history_total, psi_0, psi_analytical, z_final, sim_config = run_simulation_total(a1, a2, a3, a4, Cn2, N_simulations)

  psi_final = psi_history_total[-1]

  # Compute Error
  error_field = jnp.abs(psi_final - psi_analytical)
  l2_norm_ana = jnp.linalg.norm(psi_analytical)
  l2_error = jnp.linalg.norm(error_field) / l2_norm_ana
  print(f"\nFinal Relative L2 Error at z={z_final:.2e}: {l2_error:.2e}")

  # Visualize
  plt.figure(figsize=(15, 10))
  
  # 1. Intensity at z=0
  plt.subplot(2, 2, 1)
  plt.imshow(jnp.abs(psi_0).T, origin='lower', cmap='inferno')
  plt.title(f'Initial Intensity (z=0)')#\nLG_{p_mode}{l_mode}')
  plt.axis('off')
  
  # 2. Numerical Intensity at z=final
  plt.subplot(2, 2, 2)
  plt.imshow(jnp.abs(psi_final).T, origin='lower', cmap='inferno')
  plt.title(f'Turbulent Intensity (z={z_final:.2e})')
  plt.axis('off')
  
  # 3. Analytical Intensity at z=final
#  plt.subplot(2, 3, 3)
#  plt.imshow(jnp.abs(psi_analytical).T, origin='lower', cmap='inferno')
#  plt.title(f'Analytical Intensity (z={z_final:.2e})')
#  plt.axis('off')
  
  # 4. Error Field
 # plt.subplot(2, 2, 3)
 # plt.imshow(error_field.T, origin='lower', cmap='viridis')
 # plt.colorbar(label='|Error|')
 # plt.title('Absolute Error Field')
 # plt.axis('off')
  
  # 5. XZ Slice of Intensity History
  #intensity_history = jnp.abs(psi_history_total)**2
  x_center = sim_config.nx // 2
  xz_slice = psi_history_total[:, x_center, :].T  # Shape (nz+1, ny)  
  intensity_xz = jnp.abs(xz_slice)**2
  plt.subplot(2, 2, 3)
  plt.imshow(intensity_xz, origin='lower', cmap='inferno', aspect='auto')
  plt.title('XZ Slice of Intensity History')
  plt.axis('off')
  plt.xlabel('Propagation Distance (z)')
  plt.ylabel('x')

  # 6. YZ Slice of Intensity History
  y_center = sim_config.ny // 2
  yz_slice = psi_history_total[:, :, y_center].T  # Shape (nz+1, nx)
  intensity_yz = jnp.abs(yz_slice)**2
  plt.subplot(2, 2, 4)
  plt.imshow(intensity_yz, origin='lower', cmap='inferno', aspect='auto')
  plt.title('YZ Slice of Intensity History')
  plt.axis('off')
  plt.xlabel('Propagation Distance (z)')
  plt.ylabel('x')
  
  plt.tight_layout()
  plt.savefig(output_filename)
  print(f"Saved benchmark plot to {output_filename}")

if __name__ == "__main__":
  main(N_simulations=1, Cn2=4.4e-10, output_filename="test_turbulence_easy.png")
#  main(N_simulations=1, Cn2=6e-10, output_filename="test_turbulence_e-m_1.png")
#  main(N_simulations=1, Cn2=8e-10, output_filename="test_turbulence_e-m_2.png")
#  main(N_simulations=1, Cn2=1e-9, output_filename="test_turbulence_e-m_3.png")
#  main(N_simulations=1, Cn2=3e-9, output_filename="test_turbulence_e-m_4.png")
  main(N_simulations=1, Cn2=4.4e-9, output_filename="test_turbulence_medium.png")
#  main(N_simulations=1, Cn2=8e-9, output_filename="test_turbulence_m-h_2.png")
#  main(N_simulations=1, Cn2=1e-8, output_filename="test_turbulence_m-h_3.png")
#  main(N_simulations=1, Cn2=3e-8, output_filename="test_turbulence_m-h_4.png")
  main(N_simulations=1, Cn2=4.4e-8, output_filename="test_turbulence_hard.png")
#  main(N_simulations=1, Cn2=1e-6, output_filename="test_turbulence_limit_1_high.png")
#  main(N_simulations=1, Cn2=1e-5, output_filename="test_turbulence_limit_2_high.png")
#  main(N_simulations=1, Cn2=1e-4, output_filename="test_turbulence_limit_3_high.png")
  
  
  

