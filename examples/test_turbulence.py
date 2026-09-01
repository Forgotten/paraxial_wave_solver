import os
import sys
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

# Add project root to path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paraxial_wave_solver as pws


def initialize_laguerre(
  coeffs: list[float] | jax.Array,
  sim_config: pws.SimulationConfig,
  w0: float,
  power: float,
  modes: None | list[tuple[int, int]] = None,
):
  """Initializes a superposition of LG beams from a vector of coefficients.

  Args:
    coeffs: Vector of superposition coefficients.
    sim_config: Simulation configuration.
    w0: Beam waist radius in meters.
    power: Total laser power in Watts.
    modes: Optional list of (p, l) mode tuples. Defaults to [(0, 1), (1, 4), (0, -6), (1, 9)].

  Returns:
    Callable lg_beam(z) returning the composite beam at distance z.
  """
  if modes is None:
    modes = [(0, 1), (1, 4), (0, -6), (1, 9)]

  if len(coeffs) != len(modes):
    raise ValueError(
      f"Length of coeffs ({len(coeffs)}) does not match length of modes ({len(modes)})."
    )

  def lg_beam(z):
    psi = jnp.zeros((sim_config.nx, sim_config.ny), dtype=jnp.complex64)
    for c, (p, l) in zip(coeffs, modes):
      if c != 0.0:
        psi = psi + c * pws.laguerre_gaussian_beam(
          sim_config, w0=w0, p=p, l=l, z=z, power=power
        )
    return psi

  return lg_beam


def run_simulation_total(coeffs, Cn2, N_simulations=3):
  """Runs beam propagation simulation over multiple chunks of turbulent medium.

  Args:
    coeffs: Vector of superposition coefficients for the initial LG beam.
    Cn2: Refractive index structure constant for turbulence strength.
    N_simulations: Number of chunks to propagate through (default: 3).

  Returns:
    Tuple of (psi_history_total, psi_0, psi_analytical, z_final, sim_config).
  """
  # Setup Configuration.
  sim_config = pws.SimulationConfig(
    nx=500, ny=500, 
    dx=1e-4, dy=1e-4, dz=2.5e-2, 
    nz=100, # propagation length (2.5 m).
    wavelength=632.8e-9, # Wavelength in meters (632.8 nm).
    n0=1.33 # refractive index (underwater=1.33, vacuum=1.0).
  )

  # Estimated scales for tank experiment.
  L0 = 2e-2  # Outer scale (0.02 m) - Estimate.
  l0 = 1e-3  # Inner scale (0.001 m) - Estimate for size of tank (5x5 cm^2).

  power = 2e-3  # Total laser power in Watts (2mW).
  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  w0 = 3.0e-3 # Beam waist in meters (3 mm).
  lg_beam = initialize_laguerre(coeffs, sim_config, w0, power)
  
  # Initialize field.
  psi_0 = lg_beam(0.0)
  
  # Use N_simulations argument as number of chunks to extend.
  num_chunks = max(1, N_simulations)
  
  psi_current = psi_0
  psi_history_total = None # Will initialize after first chunk.
  
  current_z = 0.0
  print(f"Starting simulation: {num_chunks} chunks of {sim_config.nz} steps each.")

  for i in range(num_chunks):
    print(f"Simulating chunk {i+1}/{num_chunks} (z={current_z:.2f}m)...")
    
    # IMPORTANT: Use a different seed for each chunk to get fresh random medium.
    # Note: This breaks longitudinal correlation at the boundary, but L0 ~ 0.5m.
    seed = i 
    
    # We must wrap delta_n to handle the absolute z passed by the solver.
    # The generated volume is always for indices 0..nz-1.
    key = jax.random.PRNGKey(seed)
    delta_n = pws.random_medium_spectral(sim_config, Cn2, L0, l0, key)
    chunk_start_z = current_z

    def n_ref_wrapper(z):
      # Shift z to be relative to the start of this chunk.
      z_rel = z - chunk_start_z 
      idx = jnp.clip(jnp.round(z_rel / sim_config.dz).astype(int), 0, 
                     sim_config.nz - 1)
      return delta_n[:, :, idx]
    
    solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_wrapper)
    
    # Solve for this chunk.
    psi_final_chunk, psi_history_chunk = solver.solve(psi_current, current_z)
    
    # Concatenate history.
    if psi_history_total is None:
      psi_history_total = psi_history_chunk[:-1]
    else:
      psi_history_total = jnp.concatenate([psi_history_total, psi_history_chunk[:-1]], axis=0)
        
    # Update state for next chunk.
    psi_current = psi_final_chunk
    current_z += sim_config.nz * sim_config.dz
      
  # Append final field.
  psi_history_total = jnp.concatenate([psi_history_total, psi_current[None, ...]], axis=0)
  
  z_final = current_z
  print(f"Calculating analytical solution at z={z_final:.2f}m...")
  psi_analytical = lg_beam(z_final) * jnp.exp(-1j * sim_config.k0 * sim_config.n0 * z_final)
  
  return psi_history_total, psi_0, psi_analytical, z_final, sim_config


def main(N_simulations, Cn2, output_filename, coeffs=None):
  """Runs a turbulent beam propagation simulation and saves visualization plots.

  Args:
    N_simulations: Number of sequential propagation chunks to simulate.
    Cn2: Refractive index structure constant for turbulence strength.
    output_filename: Output image filename for benchmark plots.
    coeffs: Optional vector of superposition coefficients. Defaults to [1.0, 0.0, 1.0, 1.0].
  """
  if coeffs is None:
    coeffs = jnp.array([1.0, 0.0, 1.0, 1.0])

  psi_history_total, psi_0, psi_analytical, z_final, sim_config = run_simulation_total(
    coeffs, Cn2, N_simulations
  )

  psi_final = psi_history_total[-1]

  # Compute Error.
  error_field = jnp.abs(psi_final - psi_analytical)
  l2_norm_ana = jnp.linalg.norm(psi_analytical)
  l2_error = jnp.linalg.norm(error_field) / l2_norm_ana
  print(f"\nFinal Relative L2 Error at z={z_final:.2e}: {l2_error:.2e}")

  # Visualize.
  plt.figure(figsize=(15, 10))
  
  # 1. Intensity at z=0.
  plt.subplot(2, 2, 1)
  plt.imshow(jnp.abs(psi_0).T, origin='lower', cmap='inferno')
  plt.title(f'Initial Intensity (z=0)')
  plt.axis('off')
  
  # 2. Numerical Intensity at z=final.
  plt.subplot(2, 2, 2)
  plt.imshow(jnp.abs(psi_final).T, origin='lower', cmap='inferno')
  plt.title(f'Turbulent Intensity (z={z_final:.2e})')
  plt.axis('off')
  
  # 3. XZ Slice of Intensity History.
  x_center = sim_config.nx // 2
  xz_slice = psi_history_total[:, x_center, :].T  # Shape (nz+1, ny).
  intensity_xz = jnp.abs(xz_slice)**2
  plt.subplot(2, 2, 3)
  plt.imshow(intensity_xz, origin='lower', cmap='inferno', aspect='auto')
  plt.title('XZ Slice of Intensity History')
  plt.axis('off')
  plt.xlabel('Propagation Distance (z)')
  plt.ylabel('x')

  # 4. YZ Slice of Intensity History.
  y_center = sim_config.ny // 2
  yz_slice = psi_history_total[:, :, y_center].T  # Shape (nz+1, nx).
  intensity_yz = jnp.abs(yz_slice)**2
  plt.subplot(2, 2, 4)
  plt.imshow(intensity_yz, origin='lower', cmap='inferno', aspect='auto')
  plt.title('YZ Slice of Intensity History')
  plt.axis('off')
  plt.xlabel('Propagation Distance (z)')
  plt.ylabel('y')
  
  plt.tight_layout()
  plt.savefig(output_filename)
  print(f"Saved benchmark plot to {output_filename}")


if __name__ == "__main__":
  main(N_simulations=1, Cn2=4.4e-10, output_filename="test_turbulence_easy.png")
  main(N_simulations=1, Cn2=4.4e-9, output_filename="test_turbulence_medium.png")
  main(N_simulations=1, Cn2=4.4e-8, output_filename="test_turbulence_hard.png")
