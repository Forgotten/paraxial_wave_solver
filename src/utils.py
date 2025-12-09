import jax
import jax.numpy as jnp
from typing import Optional
from .config import SimulationConfig, Field

def gaussian_beam(sim_config: SimulationConfig, w0: float, x0: Optional[float] = None, y0: Optional[float] = None, kx0: float = 0.0, ky0: float = 0.0) -> Field:
  """
  Generates a Gaussian beam profile as an initial condition.
  
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

def random_medium(sim_config: SimulationConfig, correlation_length: float, strength: float, key: jax.Array) -> Field:
  """
  Generates a random refractive index perturbation with a specified correlation length.
  
  Uses Fourier filtering of white noise to generate a Gaussian random field.
  
  Args:
    sim_config: Simulation configuration.
    correlation_length: Correlation length of the random medium (physical units).
    strength: Standard deviation of the refractive index fluctuation (delta_n).
    key: JAX random key for reproducibility.
  
  Returns:
    delta_n: 3D array of shape (nx, ny, nz) containing the refractive index perturbations.
  """
  # Generate white noise
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
  
  # Normalize to desired strength
  current_std = jnp.std(delta_n)
  delta_n = delta_n * (strength / current_std)
  
  return delta_n
