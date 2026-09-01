"""Kerr self-focusing, and the numerical options that make it tractable.

Three demonstrations in one figure:

1. A Gaussian beam in a Kerr medium, at powers below, near and above the
   critical power for self-focusing. Below it diffraction wins and the beam
   spreads; above it the nonlinear lens wins and the beam collapses.
2. The 2/3 dealiasing rule. A collapsing beam pushes energy to high transverse
   wavenumbers, which fold back onto the grid as spurious structure unless the
   upper third of each axis is removed.
3. The Yoshida composition against Strang splitting, as a step-size refinement
   study through the same medium.

Run after installing the package (`pip install -e .`):

    python examples/kerr_self_focusing.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from _plotting import plot_xz_intensity

import paraxial_wave_solver as pws


def beam_width(sim_config: pws.SimulationConfig, psi: pws.Field) -> float:
  """Returns the intensity-weighted RMS radius of a transverse field."""
  x = jnp.arange(sim_config.nx) * sim_config.dx - sim_config.lx / 2
  y = jnp.arange(sim_config.ny) * sim_config.dy - sim_config.ly / 2
  intensity = jnp.abs(psi)**2
  r2 = x[:, None]**2 + y[None, :]**2
  return float(jnp.sqrt(jnp.sum(r2 * intensity) / jnp.sum(intensity)))


def build_config(n2: float) -> pws.SimulationConfig:
  """Grid used for every Kerr run; only the Kerr coefficient changes."""
  return pws.SimulationConfig(
    nx=256, ny=256,
    dx=0.05, dy=0.05, dz=0.002,
    nz=500,
    wavelength=1.0,
    n2=n2,
  )


def run_kerr(
  amplitude: float, n2: float, dealias: bool = True
) -> tuple[pws.SimulationConfig, pws.Field, pws.Field, pws.Field]:
  """Propagates a Gaussian of the given peak amplitude through a Kerr medium.

  Args:
    amplitude: Peak field amplitude at z=0, which sets the optical power and
      therefore the strength of the nonlinear lens.
    n2: Kerr coefficient.
    dealias: Whether to apply the 2/3 rule.

  Returns:
    A tuple (sim_config, psi_0, psi_final, psi_history).
  """
  sim_config = build_config(n2)
  solver_config = pws.SolverConfig(
    method='spectral', stepper='split_step',
    splitting_order=4,      # The nonlinear term rewards the higher order.
    dealias=dealias,
  )
  pml_config = pws.PMLConfig(width_x=32, width_y=32, strength=3.0)

  psi_0 = amplitude * pws.gaussian_beam(sim_config, w0=1.0)
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config)
  psi_final, history = solver.solve(psi_0, save_every=10)
  return sim_config, psi_0, psi_final, history


def splitting_study() -> tuple[list[int], list[float], list[float]]:
  """Refines dz for both splitting orders through a fixed Kerr medium.

  Returns:
    A tuple (step_counts, strang_errors, yoshida_errors).
  """
  nx = ny = 64
  dx = dy = 0.2
  lz = 20.0

  def delta_n_fn(z, medium):
    x = jnp.arange(nx) * dx
    y = jnp.arange(ny) * dy
    return 0.5 * jnp.cos(x[:, None] * 0.8) * jnp.cos(y[None, :] * 0.8)

  def make(nz):
    return pws.SimulationConfig(nx=nx, ny=ny, dx=dx, dy=dy, dz=lz / nz, nz=nz,
                                wavelength=1.0)

  x = jnp.arange(nx) * dx
  y = jnp.arange(ny) * dy
  r2 = (x[:, None] - nx * dx / 2)**2 + (y[None, :] - ny * dy / 2)**2
  psi_0 = jnp.exp(-r2 / 4.0).astype(complex)

  def run(nz, order):
    config = pws.SolverConfig(method='spectral', stepper='split_step',
                              splitting_order=order)
    solver = pws.ParaxialWaveSolver(
      make(nz), config, pws.PMLConfig(0, 0, 0.0), delta_n_fn
    )
    return solver.solve(psi_0, return_history=False)[0]

  reference = run(20480, 2)
  norm = jnp.linalg.norm(reference)
  counts = [160, 320, 640, 1280]
  strang, yoshida = [], []
  for nz in counts:
    strang.append(float(jnp.linalg.norm(run(nz, 2) - reference) / norm))
    yoshida.append(float(jnp.linalg.norm(run(nz, 4) - reference) / norm))
  return counts, strang, yoshida


def main():
  pws.enable_x64()

  n2 = 0.02
  print("Propagating through a Kerr medium at three powers...")
  runs = {}
  for label, amplitude in (('below', 1.0), ('near', 2.4), ('above', 3.2)):
    sim_config, psi_0, psi_final, history = run_kerr(amplitude, n2)
    runs[label] = (sim_config, psi_0, psi_final, history)
    print(f"  amplitude={amplitude:4.1f}  RMS radius "
          f"{beam_width(sim_config, psi_0):.3f} -> "
          f"{beam_width(sim_config, psi_final):.3f}")

  print("\nSame run with dealiasing disabled, for comparison...")
  aliased = run_kerr(3.2, n2, dealias=False)

  print("\nRefining dz for both splitting orders...")
  counts, strang, yoshida = splitting_study()
  for nz, e2, e4 in zip(counts, strang, yoshida, strict=True):
    print(f"  nz={nz:5d}  Strang {e2:.3e}   Yoshida {e4:.3e}")

  fig, axes = plt.subplots(2, 3, figsize=(16, 9))

  for ax, label, title in zip(
    axes[0], ('below', 'near', 'above'),
    ('Below critical power', 'Near critical power', 'Above: self-focusing'),
    strict=True,
  ):
    sim_config, _, _, history = runs[label]
    plot_xz_intensity(ax, sim_config, history, title)

  ax = axes[1, 0]
  sim_config = runs['above'][0]
  x = jnp.arange(sim_config.nx) * sim_config.dx
  for label, style in (('below', 'b-'), ('near', 'g-'), ('above', 'r-')):
    cfg, _, psi_final, _ = runs[label]
    ax.plot(x, jnp.abs(psi_final[:, sim_config.ny // 2])**2, style,
            label=label)
  ax.set_title(f'Final intensity profile (n2={n2})')
  ax.set_xlabel('x')
  ax.set_ylabel('Intensity')
  ax.legend()
  ax.grid(True, alpha=0.3)

  # The 2/3 rule acts in wavenumber space, so that is where to look at it. In
  # real space the two runs are almost indistinguishable at this power; the
  # rule is insurance for stronger nonlinearity, not a visible correction here.
  ax = axes[1, 1]
  clean = runs['above'][2]
  kx = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nx, d=sim_config.dx)
  order = jnp.argsort(kx)

  def spectrum(psi):
    # Index 0 of the second axis is ky = 0, not the middle of the array.
    line = jnp.abs(jnp.fft.fft2(psi)[:, 0])**2
    return line[order]

  clean_spectrum = spectrum(clean)
  peak = float(clean_spectrum.max())

  ax.semilogy(kx[order], spectrum(aliased[2]), 'r-', label='no dealiasing')
  ax.semilogy(kx[order], clean_spectrum, 'b-', label='2/3 rule')
  cutoff = (2 / 3) * jnp.pi / sim_config.dx
  for sign in (-1, 1):
    ax.axvline(sign * cutoff, color='k', ls=':', alpha=0.6)
  ax.set_title('Transverse spectrum, with the 2/3 cutoff')
  ax.set_xlabel('$k_x$')
  ax.set_ylabel('Spectral power')
  ax.set_ylim(peak * 1e-24, peak * 10)
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)

  ax = axes[1, 2]
  ax.loglog(counts, strang, 'o-', label='Strang (order 2)')
  ax.loglog(counts, yoshida, 's-', label='Yoshida (order 4)')
  reference_slope = [strang[0] * (counts[0] / n)**2 for n in counts]
  fourth_slope = [yoshida[0] * (counts[0] / n)**4 for n in counts]
  ax.loglog(counts, reference_slope, 'k:', alpha=0.5, label='dz$^2$')
  ax.loglog(counts, fourth_slope, 'k--', alpha=0.5, label='dz$^4$')
  ax.set_title('Convergence in z')
  ax.set_xlabel('steps')
  ax.set_ylabel('relative L2 error')
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3, which='both')

  fig.tight_layout()
  fig.savefig('kerr_self_focusing.png')
  print("\nSaved figure to kerr_self_focusing.png")


if __name__ == "__main__":
  main()
