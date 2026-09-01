"""Laguerre-Gaussian modes propagating in vacuum at a HeNe wavelength.

Unlike the other vacuum benchmarks this one uses a physical wavelength, so
k0 * lz is not a multiple of 2*pi. That makes it the example that exposes any
error in the carrier phase convention rather than hiding it.

Run after installing the package (`pip install -e .`):

    python examples/laguerre_gaussian.py
"""

from _plotting import plot_benchmark, relative_l2_error

import paraxial_wave_solver as pws


def run_simulation(p_mode, l_mode, output_filename, use_x64=True):
  """Propagates a single LG mode and writes its benchmark figure.

  Args:
    p_mode: Radial mode index.
    l_mode: Azimuthal mode index.
    output_filename: File to write the benchmark figure to.
    use_x64: Run in float64. The envelope itself is well within float32
             range, but float64 takes the error from ~2e-5 to ~3e-13.
  """
  if use_x64:
    pws.enable_x64()

  sim_config = pws.SimulationConfig(
    nx=512, ny=512,
    dx=0.0075, dy=0.0075, dz=1.0,
    nz=250,                  # Propagation length: 2.5 m.
    wavelength=632.8e-7,     # 632.8 nm, expressed in centimetres.
  )
  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')

  w0 = 3.0e-1  # Beam waist: 3 mm.

  print(f"\n--- Running Simulation for LG_{p_mode}{l_mode} ---")
  psi_0 = pws.laguerre_gaussian_beam(
    sim_config, w0, p=p_mode, l=l_mode, z=0.0
  )

  # Vacuum: delta_n = 0 is the solver default.
  print("Running simulation...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config)
  psi_final, psi_history = solver.solve(psi_0, save_every=25)
  print("Simulation complete.")

  # Error against the analytical envelope at every saved plane.
  print("\n--- Error Analysis over Propagation ---")
  for index in range(psi_history.shape[0]):
    z_current = index * 25 * sim_config.dz
    psi_reference = pws.laguerre_gaussian_beam(
      sim_config, w0, p=p_mode, l=l_mode, z=z_current, envelope_only=True
    )
    error = relative_l2_error(psi_history[index], psi_reference)
    print(f"z={z_current:6.2f}: Rel L2 Error = {error:.2e}")

  psi_analytical = pws.laguerre_gaussian_beam(
    sim_config, w0, p=p_mode, l=l_mode, z=sim_config.lz, envelope_only=True
  )
  l2_error = plot_benchmark(
    sim_config, psi_0, psi_final, psi_analytical, output_filename,
    initial_title=f'Initial Intensity (z=0)\nLG_{p_mode}{l_mode}',
    compare='phase',
  )
  print(f"\nFinal Relative L2 Error at z={sim_config.lz:.2f}: {l2_error:.2e}")
  return l2_error


def main():
  run_simulation(0, 1, 'laguerre_gaussian_01.png')
  run_simulation(1, 4, 'laguerre_gaussian_benchmark_14.png')
  run_simulation(0, -6, 'laguerre_gaussian_benchmark_0-6.png')
  run_simulation(1, 9, 'laguerre_gaussian_benchmark_19.png')


if __name__ == "__main__":
  main()
