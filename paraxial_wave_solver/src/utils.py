import math
import jax
import jax.numpy as jnp
from scipy.special import genlaguerre, hermite

from .config import SimulationConfig, Field


def gaussian_beam(
  sim_config: SimulationConfig,
  w0: float,
  z: float = 0.0,
  power: None | float = None,
  x0: None | float = None,
  y0: None | float = None,
  kx0: float = 0.0,
  ky0: float = 0.0,
  envelope_only: bool = False,
) -> Field:
  """Generates an analytical fundamental Gaussian (TEM_00) beam profile.

  Args:
    sim_config: Simulation configuration.
    w0: Beam waist radius (1/e^2 intensity radius) at z=0.
    z: Propagation distance along z-axis (default: 0.0).
    power: Optional total optical power to normalize the beam to.
    x0: Center x-position (default: lx / 2).
    y0: Center y-position (default: ly / 2).
    kx0: Transverse wavenumber in x (tilt).
    ky0: Transverse wavenumber in y (tilt).
    envelope_only: If True, returns envelope without carrier phase exp(i*k*z).

  Returns:
    psi: Complex field array of shape (nx, ny) representing the Gaussian beam.
  """
  if x0 is None:
    x0 = sim_config.lx / 2.0
  if y0 is None:
    y0 = sim_config.ly / 2.0

  k = sim_config.k
  z_R = k * w0**2 / 2.0

  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')

  dx = X - x0
  dy = Y - y0
  r2 = dx**2 + dy**2
  phase_tilt = kx0 * X + ky0 * Y

  if z == 0.0:
    psi = jnp.exp(-r2 / (w0**2)) * jnp.exp(1j * phase_tilt)
  else:
    w_z = w0 * jnp.sqrt(1 + (z / z_R)**2)
    R_z = z * (1 + (z_R / z)**2)
    zeta_z = jnp.arctan(z / z_R)

    psi = (
      (w0 / w_z)
      * jnp.exp(-r2 / (w_z**2))
      * jnp.exp(1j * k * r2 / (2 * R_z))
      * jnp.exp(-1j * zeta_z)
      * jnp.exp(1j * phase_tilt)
    )

    if not envelope_only:
      psi = psi * jnp.exp(1j * k * z)

  if power is not None:
    p_current = jnp.sum(jnp.abs(psi)**2) * sim_config.dx * sim_config.dy
    psi = psi * jnp.sqrt(power / p_current)

  return psi


def laguerre_gaussian_beam(
  sim_config: SimulationConfig,
  w0: float,
  p: int = 0,
  l: int = 0,
  z: float = 0.0,
  power: None | float = None,
  x0: None | float = None,
  y0: None | float = None,
  envelope_only: bool = False,
) -> Field:
  """Computes the analytical Laguerre-Gaussian (LG_pl) beam.

  Args:
    sim_config: Simulation configuration.
    w0: Beam waist radius (1/e^2 intensity radius) at z=0.
    p: Radial mode index (p >= 0).
    l: Azimuthal mode index (topological charge / OAM).
    z: Propagation distance along z-axis (default: 0.0).
    power: Optional target total optical power to normalize the beam to.
    x0: Center x-position (default: lx / 2).
    y0: Center y-position (default: ly / 2).
    envelope_only: If True, returns envelope without carrier phase exp(i*k*z).

  Returns:
    psi: Complex field array of shape (nx, ny) representing the LG beam.
  """
  if x0 is None:
    x0 = sim_config.lx / 2.0
  if y0 is None:
    y0 = sim_config.ly / 2.0

  k = sim_config.k
  z_R = k * w0**2 / 2.0

  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')

  dx = X - x0
  dy = Y - y0
  r2 = dx**2 + dy**2
  r = jnp.sqrt(r2)
  phi = jnp.arctan2(dy, dx)

  w_z = w0 * jnp.sqrt(1 + (z / z_R)**2)
  R_z = z * (1 + (z_R / z)**2) if z != 0 else jnp.inf
  zeta_z = jnp.arctan(z / z_R)

  arg = 2 * r2 / w_z**2
  Lpl = jnp.polyval(jnp.array(genlaguerre(p, abs(l)).coef), arg)

  term_r = (r * jnp.sqrt(2) / w_z)**abs(l)
  norm_factor = jnp.sqrt(2 * math.factorial(p) / (jnp.pi * math.factorial(p + abs(l))))

  amplitude = norm_factor * (w0 / w_z) * term_r * Lpl * jnp.exp(-r2 / w_z**2)

  gouy_phase = (2 * p + abs(l) + 1) * zeta_z
  curvature_phase = k * r2 / (2 * R_z) if z != 0 else 0.0
  azimuthal_phase = l * phi

  psi = (
    amplitude
    * jnp.exp(1j * curvature_phase)
    * jnp.exp(-1j * gouy_phase)
    * jnp.exp(1j * azimuthal_phase)
  )

  if not envelope_only and z != 0.0:
    psi = psi * jnp.exp(1j * k * z)

  if power is not None:
    p_current = jnp.sum(jnp.abs(psi)**2) * sim_config.dx * sim_config.dy
    psi = psi * jnp.sqrt(power / p_current)

  return psi


def hermite_gaussian_beam(
  sim_config: SimulationConfig,
  w0: float,
  n: int = 0,
  m: int = 0,
  z: float = 0.0,
  power: None | float = None,
  x0: None | float = None,
  y0: None | float = None,
  envelope_only: bool = False,
) -> Field:
  """Computes the analytical Hermite-Gaussian (HG_nm) beam.

  Args:
    sim_config: Simulation configuration.
    w0: Beam waist radius at z=0.
    n: Mode index in x-direction (n >= 0).
    m: Mode index in y-direction (m >= 0).
    z: Propagation distance along z-axis (default: 0.0).
    power: Optional target total optical power to normalize the beam to.
    x0: Center x-position (default: lx / 2).
    y0: Center y-position (default: ly / 2).
    envelope_only: If True, returns envelope without carrier phase exp(i*k*z).

  Returns:
    psi: Complex field array of shape (nx, ny) representing the HG beam.
  """
  if x0 is None:
    x0 = sim_config.lx / 2.0
  if y0 is None:
    y0 = sim_config.ly / 2.0

  k = sim_config.k
  z_R = k * w0**2 / 2.0

  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')

  dx = X - x0
  dy = Y - y0
  r2 = dx**2 + dy**2

  w_z = w0 * jnp.sqrt(1 + (z / z_R)**2)
  R_z = z * (1 + (z_R / z)**2) if z != 0 else jnp.inf
  zeta_z = jnp.arctan(z / z_R)

  Hn = jnp.polyval(jnp.array(hermite(n).coef), jnp.sqrt(2) * dx / w_z)
  Hm = jnp.polyval(jnp.array(hermite(m).coef), jnp.sqrt(2) * dy / w_z)

  gouy_phase = (n + m + 1) * zeta_z
  curvature_phase = k * r2 / (2 * R_z) if z != 0 else 0.0

  norm_factor = jnp.sqrt(2.0 / (jnp.pi * 2**(n + m) * math.factorial(n) * math.factorial(m)))
  amplitude = norm_factor * (w0 / w_z) * Hn * Hm * jnp.exp(-r2 / w_z**2)

  psi = amplitude * jnp.exp(1j * curvature_phase) * jnp.exp(-1j * gouy_phase)

  if not envelope_only and z != 0.0:
    psi = psi * jnp.exp(1j * k * z)

  if power is not None:
    p_current = jnp.sum(jnp.abs(psi)**2) * sim_config.dx * sim_config.dy
    psi = psi * jnp.sqrt(power / p_current)

  return psi


# Aliases for backward compatibility.
get_laguerre_gaussian_analytical = laguerre_gaussian_beam
get_hermite_gaussian_analytical = hermite_gaussian_beam
get_analytical_beam = gaussian_beam


def random_medium(
  sim_config: SimulationConfig,
  correlation_length: float,
  strength: float,
  key: jax.Array,
) -> Field:
  """Generates a random refractive index perturbation.

  Uses Fourier filtering of white noise to generate a Gaussian random field with a
  specified correlation length.

  Args:
    sim_config: Simulation configuration.
    correlation_length: Correlation length of the random medium (physical units).
    strength: Standard deviation of the refractive index fluctuation (delta_n).
    key: JAX random key for reproducibility.

  Returns:
    delta_n: 3D array of shape (nx, ny, nz) containing refractive index perturbations.
  """
  # Generate white noise.
  noise = jax.random.normal(key, (sim_config.nx, sim_config.ny, sim_config.nz))

  # Filter in Fourier domain to impose correlation length.
  kx = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nx, d=sim_config.dx)
  ky = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.ny, d=sim_config.dy)
  kz = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nz, d=sim_config.dz)

  KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
  K2 = KX**2 + KY**2 + KZ**2

  # Gaussian correlation function -> Gaussian power spectrum.
  power_spectrum = jnp.exp(-K2 * correlation_length**2 / 4.0)

  noise_k = jnp.fft.fftn(noise)
  filtered_noise_k = noise_k * jnp.sqrt(power_spectrum)
  delta_n = jnp.real(jnp.fft.ifftn(filtered_noise_k))

  # Normalize to desired strength.
  current_std = jnp.std(delta_n)
  delta_n = delta_n * (strength / current_std)

  return delta_n


def random_medium_spectral(
  sim_config: SimulationConfig,
  Cn2: float,
  L0: float,
  l0: float,
  key: jax.Array,
) -> Field:
  """Generates a random refractive index perturbation with Von Karman spectrum.

  Args:
    sim_config: Simulation configuration.
    Cn2: Structure constant for refractive index fluctuations.
    L0: Outer scale of turbulence.
    l0: Inner scale of turbulence.
    key: JAX random key for reproducibility.

  Returns:
    V: 3D array of shape (nx, ny, nz) containing refractive index fluctuations.
  """
  kx = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nx, d=sim_config.dx)
  ky = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.ny, d=sim_config.dy)
  kz = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nz, d=sim_config.dz)

  KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
  kappa = jnp.sqrt(KX**2 + KY**2 + KZ**2)

  dkx = kx[1] - kx[0]
  dky = ky[1] - ky[0]
  dkz = kz[1] - kz[0]

  dVk = dkx * dky * dkz

  kappa0 = 2 * jnp.pi / L0
  kappam = 5.92 / l0

  # Von Karman Spectrum.
  Phi_n = 0.033 * Cn2 * jnp.exp(-kappa**2 / kappam**2) / (kappa**2 + kappa0**2)**(11 / 6)

  sigma_k2 = Phi_n * dVk
  sigma_k2 = jnp.where(kappa == 0, 0.0, sigma_k2)

  # Sample V_hat.
  key_r, key_i = jax.random.split(key)
  xi_r = jax.random.normal(key_r, sigma_k2.shape)
  xi_i = jax.random.normal(key_i, sigma_k2.shape)

  # Scale by N_total for IFFT.
  n_total = sim_config.nx * sim_config.ny * sim_config.nz

  V_hat = n_total * jnp.sqrt(sigma_k2 / 2) * (xi_r + 1j * xi_i)

  V = jnp.real(jnp.fft.ifftn(V_hat)) * jnp.sqrt(2)

  return V
