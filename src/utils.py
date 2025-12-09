import jax
import jax.numpy as jnp
from .config import SimulationConfig

def gaussian_beam(sim_config: SimulationConfig, w0: float, x0: float = None, y0: float = None, kx0: float = 0.0, ky0: float = 0.0):
    """
    Generates a Gaussian beam profile.
    
    Args:
        sim_config: Simulation configuration.
        w0: Beam waist radius.
        x0, y0: Center position (default: center of domain).
        kx0, ky0: Transverse wavenumbers (tilt).
    """
    if x0 is None:
        x0 = sim_config.Lx / 2
    if y0 is None:
        y0 = sim_config.Ly / 2
        
    x = jnp.arange(sim_config.Nx) * sim_config.dx
    y = jnp.arange(sim_config.Ny) * sim_config.dy
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    r2 = (X - x0)**2 + (Y - y0)**2
    phase = kx0 * X + ky0 * Y
    
    # Simple Gaussian profile (at waist)
    psi = jnp.exp(-r2 / (w0**2)) * jnp.exp(1j * phase)
    return psi

def random_medium(sim_config: SimulationConfig, correlation_length: float, strength: float, key: jax.Array):
    """
    Generates a random refractive index perturbation.
    
    Args:
        sim_config: Simulation configuration.
        correlation_length: Correlation length of the random medium.
        strength: Standard deviation of refractive index fluctuation (dn).
        key: JAX random key.
    
    Returns:
        delta_n: 3D array (Nx, Ny, Nz) of refractive index perturbations.
    """
    # Generate white noise
    noise = jax.random.normal(key, (sim_config.Nx, sim_config.Ny, sim_config.Nz))
    
    # Filter in Fourier domain to impose correlation length
    kx = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.Nx, d=sim_config.dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.Ny, d=sim_config.dy)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(sim_config.Nz, d=sim_config.dz)
    
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
