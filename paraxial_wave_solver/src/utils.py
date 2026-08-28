import jax
import jax.numpy as jnp
from typing import Optional
from .config import SimulationConfig, Field

def gaussian_beam(
  sim_config: SimulationConfig,
  w0: float,
  x0: Optional[float] = None,
  y0: Optional[float] = None,
  kx0: float = 0.0,
  ky0: float = 0.0
) -> Field:
  """Generates a Gaussian beam profile as an initial condition.
  
  Args:
    sim_config: Simulation configuration.
    w0: Beam waist radius (1/e^2 intensity radius).
    x0: Center x-position (default: center of domain).
    y0: Center y-position (default: center of domain).
    kx0: Transverse wavenumber in x (tilt).
    ky0: Transverse wavenumber in y (tilt).
    
  Returns:
    psi: Complex field array of shape (nx, ny) representing the Gaussian beam.
  """
  if x0 is None:
    x0 = sim_config.lx / 2
  if y0 is None:
    y0 = sim_config.ly / 2
    
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')
  
  r2 = (X - x0)**2 + (Y - y0)**2
  phase = kx0 * X + ky0 * Y
  
  # Simple Gaussian profile (at waist)
  psi = jnp.exp(-r2 / (w0**2)) * jnp.exp(1j * phase)
  return psi

def random_medium(
  sim_config: SimulationConfig,
  correlation_length: float,
  strength: float,
  key: jax.Array
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
    delta_n: 3D array of shape (nx, ny, nz) containing the refractive index 
             perturbations.
  """
  # Generate white noise.
  noise = jax.random.normal(key, (sim_config.nx, sim_config.ny, sim_config.nz))

  # Filter in Fourier domain to impose correlation length
  kx = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nx, d=sim_config.dx)
  ky = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.ny, d=sim_config.dy)
  kz = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nz, d=sim_config.dz)

  KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
  K2 = KX**2 + KY**2 + KZ**2
  
  # Gaussian correlation function -> Gaussian power spectrum
  # C(r) ~ exp(-r^2/L^2) <-> P(k) ~ exp(-k^2 L^2 / 4)
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
  key: jax.Array
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
    delta_n: 3D array of shape (nx, ny, nz) containing the refractive index 
             perturbations.
  """
  
  # Filter in Fourier domain to impose correlation length
  kx = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nx, d=sim_config.dx)
  ky = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.ny, d=sim_config.dy)
  kz = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.nz, d=sim_config.dz)
  
  KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')
  kappa = jnp.sqrt(KX**2 + KY**2 + KZ**2)
  
  dkx = kx[1] - kx[0]
  dky = ky[1] - ky[0]
  dkz = kz[1] - kz[0]

  dVk = dkx * dky * dkz

  kappa0 = 2*jnp.pi / L0
  kappam = 5.92 / l0

  # Von Karman Spectrum
  Phi_n = 0.033 * Cn2 * jnp.exp(-kappa**2 / kappam**2) / (kappa**2 + kappa0**2)**(11/6)
  #Phi_n = 0.033 * Cn2 * kappa**(-11/3) * jnp.exp(-kappa**2 / kappam**2)

  sigma_k2 = Phi_n * dVk
  sigma_k2 = jnp.where(kappa == 0, 0.0, sigma_k2)

  # sample V_hat
  key_r, key_i = jax.random.split(key)
  xi_r = jax.random.normal(key_r, sigma_k2.shape)
  xi_i = jax.random.normal(key_i, sigma_k2.shape)

  # Since IFFT divides by N_total, we must scale inputs by N_total.
  n_total = sim_config.nx * sim_config.ny * sim_config.nz

  V_hat = n_total * jnp.sqrt(sigma_k2 / 2) * (xi_r + 1j * xi_i)

  V = jnp.real(jnp.fft.ifftn(V_hat)) * jnp.sqrt(2)

  return V


def phase_screen_von_karman(
    sim_config,
    Cn2: float,
    L: float,
    L0: float,
    l0: float,
    key: jax.Array,
):
    """
    Generate a single von Kármán phase screen for underwater turbulence.
    """
    nx, ny = sim_config.nx, sim_config.ny
    dx, dy = sim_config.dx, sim_config.dy
    wavelength = sim_config.wavelength

    k = 2 * jnp.pi / wavelength

    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, dy)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    kappa = jnp.sqrt(KX**2 + KY**2)

    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    dA = dkx * dky

    kappa0 = 2 * jnp.pi / L0
    kappam = 5.92 / l0

    Phi_phi = (
        0.023
        * k**2
        * Cn2
        * L
        * jnp.exp(-(kappa / kappam) ** 2)
        / (kappa**2 + kappa0**2) ** (11 / 6)
    )

    Phi_phi = jnp.where(kappa == 0, 0.0, Phi_phi)

    sigma2 = Phi_phi #* dA

    key_r, key_i = jax.random.split(key)
    noise = (
        jax.random.normal(key_r, (nx, ny))
        + 1j * jax.random.normal(key_i, (nx, ny))
    )

    phi_hat = jnp.sqrt(sigma2 / 2) * noise
    phi = jnp.real(jnp.fft.ifft2(phi_hat)) #* (2 * jnp.pi)**2

    # --- ENFORCE CORRECT PHASE VARIANCE ---
    phi = phi - jnp.mean(phi)

    phi_rms_target = jnp.sqrt(
        1.03 * k**2 * Cn2 * (L)**(5 / 3)
    )

    phi *= phi_rms_target / (jnp.std(phi))
    return phi

def get_thermal_turbulence_fn(
  sim_config: SimulationConfig,
  Cn2: float,
  L0: float,
  l0: float,
  seed: int = 0
) -> Callable[[float], Field]:
  """Factory for thermal turbulence refractive index function.
  
  Args:
      sim_config: Simulation configuration.
      Cn2: Refractive index structure constant [m^-2/3].
      L0: Outer scale [m].
      l0: Inner scale [m].
      seed: Random seed.
      
  Returns:
      A function n_ref_fn(z) -> Field (2D slice at z).
  """
  key = jax.random.PRNGKey(seed)
  # Generate the full 3D volume of refractive index fluctuations
  V_vol = random_medium_spectral(sim_config, Cn2=Cn2, L0=L0, l0=l0, key=key)
  
  # Return a function that slices this volume
  def n_ref_fn(z):
      # Map continuous z to index
      # We clip to ensure we don't go out of bounds
      idx = jnp.clip(jnp.round(z / sim_config.dz).astype(int), 0, sim_config.nz - 1)
      return V_vol[:, :, idx]
      
  return n_ref_fn
