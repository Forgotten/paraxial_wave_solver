"""Square aperture diffraction, checked against the Fresnel integrals.

Run after installing the package (`pip install -e .`):

    python examples/aperture_diffraction.py
"""

import jax
import jax.numpy as jnp
from _plotting import plot_benchmark
from jax.scipy.special import erf
from scipy.special import fresnel

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
  # A wide domain, so that the diffracted field does not reach the PML.
  sim_config = pws.SimulationConfig(
    nx=1024, ny=1024,
    dx=0.1, dy=0.1, dz=0.5,
    nz=200,
    wavelength=1.0,
  )
  # Diffraction spreads widely, so the absorbing layer is correspondingly deep.
  pml_config = pws.PMLConfig(width_x=200, width_y=200, strength=2.0)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')

  width = 10.0
  smooth_sigma = 0.5  # Mollification width, five grid points.

  print(f"Initializing square aperture (width={width}, sigma={smooth_sigma})...")
  psi_0 = get_square_aperture_analytical(
    sim_config, width, z=0.0, smooth_sigma=smooth_sigma
  )

  # Vacuum: delta_n = 0 is the solver default.
  print("Running simulation...")
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config)
  psi_final, _ = solver.solve(psi_0, return_history=False)
  print("Simulation complete.")

  z_final = sim_config.lz
  psi_analytical = get_square_aperture_analytical(
    sim_config, width, z_final, smooth_sigma=smooth_sigma
  )

  # Report the interior separately: the PML absorbs the outermost fringes by
  # design, so the whole-grid figure is dominated by that, not by the scheme.
  interior_x = slice(pml_config.width_x, -pml_config.width_x)
  interior_y = slice(pml_config.width_y, -pml_config.width_y)
  interior_error = (
    jnp.linalg.norm(psi_final[interior_x, interior_y]
                    - psi_analytical[interior_x, interior_y])
    / jnp.linalg.norm(psi_analytical[interior_x, interior_y])
  )
  print(f"Relative L2 Error (inner domain) at z={z_final:.2f}: "
        f"{interior_error:.2e}")

  plot_benchmark(
    sim_config, psi_0, psi_final, psi_analytical,
    'aperture_diffraction_benchmark.png',
    initial_title=f'Initial Aperture (width={width}, sigma={smooth_sigma})',
  )


if __name__ == "__main__":
  main()
