"""Superposed Laguerre-Gaussian modes through chunks of Von Karman turbulence.

Each chunk draws a fresh medium; because the medium is passed to solve() as an
argument rather than closed over, all chunks share one compiled solver.

Run after installing the package (`pip install -e .`):

    python examples/turbulence_propagation.py
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
    modes: Optional list of (p, l) mode tuples. Defaults to
           [(0, 1), (1, 4), (0, -6), (1, 9)].

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
    """Returns the composite envelope at distance z."""
    terms = [
      c * pws.laguerre_gaussian_beam(
        sim_config, w0=w0, p=p, l=l, z=z, power=power, envelope_only=True
      )
      for c, (p, l) in zip(coeffs, modes, strict=True) if c != 0.0
    ]
    return sum(terms)

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

  # The turbulent volume is passed as an argument, together with the absolute
  # z at which it starts, so that one compiled solver serves every chunk.
  def delta_n_fn(z, medium):
    volume, z_start = medium
    idx = jnp.clip(jnp.round((z - z_start) / sim_config.dz).astype(int), 0,
                   sim_config.nz - 1)
    return volume[:, :, idx]

  solver = pws.ParaxialWaveSolver(
    sim_config, solver_config, pml_config, delta_n_fn
  )

  psi_current = psi_0
  histories = []
  current_z = 0.0
  print(f"Starting simulation: {num_chunks} chunks of {sim_config.nz} steps each.")

  for i in range(num_chunks):
    print(f"Simulating chunk {i+1}/{num_chunks} (z={current_z:.2f}m)...")

    # A different seed per chunk gives a fresh medium. This breaks the
    # longitudinal correlation at the chunk boundary, acceptable while the
    # chunk length stays well above the outer scale L0.
    key = jax.random.PRNGKey(i)
    delta_n = pws.random_medium_spectral(sim_config, Cn2, L0, l0, key)

    psi_current, history = solver.solve(
      psi_current, current_z, medium=(delta_n, current_z)
    )
    # history[j] is the field at current_z + j * dz, so chunks concatenate
    # without any overlap to trim.
    histories.append(history)
    current_z += sim_config.lz

  psi_history_total = jnp.concatenate(
    histories + [psi_current[None, ...]], axis=0
  )

  z_final = current_z
  print(f"Calculating analytical solution at z={z_final:.2f}m...")
  psi_analytical = lg_beam(z_final)

  return psi_history_total, psi_0, psi_analytical, z_final, sim_config


def main(N_simulations, Cn2, output_filename, coeffs=None):
  """Runs a turbulent beam propagation simulation and saves visualization plots.

  Args:
    N_simulations: Number of sequential propagation chunks to simulate.
    Cn2: Refractive index structure constant for turbulence strength.
    output_filename: Output image filename for benchmark plots.
    coeffs: Optional vector of superposition coefficients. Defaults to
            [1.0, 0.0, 1.0, 1.0].
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
  plt.title('Initial Intensity (z=0)')
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
