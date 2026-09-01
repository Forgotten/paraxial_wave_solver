"""Hermite-Gaussian HG_11 mode propagating in vacuum.

Run after installing the package (`pip install -e .`):

    python examples/hermite_gaussian.py
"""

from _plotting import plot_benchmark

import paraxial_wave_solver as pws


def main():
  sim_config = pws.SimulationConfig(
    nx=512, ny=512,
    dx=0.2, dy=0.2, dz=1.0,
    nz=200,
    wavelength=1.0,
    n0=1.0,
  )
  pml_config = pws.PMLConfig(width_x=40, width_y=40, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')

  w0 = 8.0
  n_mode, m_mode = 1, 1

  print(f"Initializing Hermite-Gaussian HG_{n_mode}{m_mode} beam...")
  psi_0 = pws.hermite_gaussian_beam(sim_config, w0, n=n_mode, m=m_mode, z=0.0)

  # Vacuum: delta_n = 0 is the solver default.
  print("Running simulation...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config)
  psi_final, _ = solver.solve(psi_0, return_history=False)
  print("Simulation complete.")

  psi_analytical = pws.hermite_gaussian_beam(
    sim_config, w0, n=n_mode, m=m_mode, z=sim_config.lz, envelope_only=True
  )

  l2_error = plot_benchmark(
    sim_config, psi_0, psi_final, psi_analytical,
    'hermite_gaussian_benchmark.png',
    initial_title=f'Initial Intensity (z=0)\nHG_{n_mode}{m_mode}',
  )
  print(f"Relative L2 Error at z={sim_config.lz:.2f}: {l2_error:.2e}")


if __name__ == "__main__":
  main()
