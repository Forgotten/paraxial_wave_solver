import jax.numpy as jnp
from .config import SimulationConfig, PMLConfig, Field

from typing import Dict, Union, Tuple

def generate_pml_profile(sim_config: SimulationConfig, 
                         pml_config: PMLConfig) -> Union[Field, Dict[str, Field]]:
  """Generates the complex Perfectly Matched Layer (PML) absorption profile.

  The profile is zero in the physical domain and increases polynomially in the 
  PML regions at the boundaries.

  Args:
    sim_config: Simulation configuration containing grid details.
    pml_config: PML configuration containing width, strength, and order.

  Returns:
    sigma: Complex array of shape (nx, ny) representing the PML absorption 
           profile.
  """
  
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  
  lx = sim_config.lx
  ly = sim_config.ly
  
  # Define PML regions
  x_start_pml = pml_config.width_x * sim_config.dx
  x_end_pml = lx - pml_config.width_x * sim_config.dx
  
  y_start_pml = pml_config.width_y * sim_config.dy
  y_end_pml = ly - pml_config.width_y * sim_config.dy
  
  def sigma_1d(
    coord: Field,
    start: float,
    end: float,
    width_idx: int,
    d_step: float
  ) -> Tuple[Field, Field]:
    # Distance into PML.
    d_left = jnp.maximum(0.0, start - coord)
    d_right = jnp.maximum(0.0, coord - end)
    d = jnp.maximum(d_left, d_right)

    # Normalize by PML width.
    l_pml = width_idx * d_step
    # Avoid division by zero if width is 0.
    l_pml = jnp.where(l_pml > 0, l_pml, 1.0)
    d_norm = d / l_pml
    
    # Polynomial profile sigma(x)
    sigma = jnp.where(width_idx > 0, 
                     pml_config.strength * (d_norm ** pml_config.order), 
                     0.0)

    # Derivative sigma'(x)
    # d(d)/dx is -1 for left, +1 for right?
    # d_left = start - x (if < start). deriv is -1.
    # d_right = x - end (if > end). deriv is +1.
    grad_d = jnp.where(coord < start, -1.0, jnp.where(coord > end, 1.0, 0.0))
    
    # sigma = S * (d/L)^n
    # sigma' = S * n * (d/L)^(n-1) * (1/L) * grad_d
    # If using order 0, deriv is 0. 
    # Use careful power for n-1 to avoid div by zero if d=0
    # when d=0, sigma'=0 provided n > 1.
    
    if pml_config.order > 1:
        d_norm_pow = jnp.where(d_norm > 0, d_norm ** (pml_config.order - 1), 0.0)
        sigma_prime = pml_config.strength * pml_config.order * d_norm_pow * (1.0 / l_pml) * grad_d
        sigma_prime = jnp.where(width_idx > 0, sigma_prime, 0.0)
    elif pml_config.order == 1:
        sigma_prime = pml_config.strength * (1.0 / l_pml) * grad_d
        sigma_prime = jnp.where(width_idx > 0, sigma_prime, 0.0)
    else:
        sigma_prime = jnp.zeros_like(sigma)
        
    return sigma, sigma_prime

  sigma_x, sigma_x_prime = sigma_1d(x, x_start_pml, x_end_pml, pml_config.width_x, sim_config.dx)
  sigma_y, sigma_y_prime = sigma_1d(y, y_start_pml, y_end_pml, pml_config.width_y, sim_config.dy)
  
  # Combine x and y profiles.
  sigma_grid_x, sigma_grid_y = jnp.meshgrid(sigma_x, sigma_y, indexing='ij')
  
  # Complex stretch factors: s_x = 1 + 1j * sigma_x
  # We might want to scale sigma by k0? For now keep 1.0 factor or assume strength is appropriate.
  # Let's match the "absorbing layer" relative magnitude. 
  # In absorbing term: -sigma * psi.
  # If we have s_x = 1 + 1j * sigma, then 1/s_x^2 d^2/dx^2 ...
  # Let's use sx = 1 + 1j * sigma_x.
  sx = 1.0 + 1j * sigma_x
  sy = 1.0 + 1j * sigma_y
  
  # Derivative of s_x w.r.t x is 1j * sigma_x_prime
  sx_prime = 1j * sigma_x_prime
  sy_prime = 1j * sigma_y_prime
  
  # Meshgrid for full 2D arrays (needed for operators that expect 2D fields)
  sx_2d, sy_2d = jnp.meshgrid(sx, sy, indexing='ij')
  sx_prime_2d, sy_prime_2d = jnp.meshgrid(sx_prime, sy_prime, indexing='ij')

  if pml_config.use_complex_stretching:
      return {
          'sigma_sum': sigma_grid_x + sigma_grid_y, # Keep for backward compat/hybrid
          'sx': sx_2d,
          'sy': sy_2d,
          'sx_prime': sx_prime_2d,
          'sy_prime': sy_prime_2d
      }
  else:
      return sigma_grid_x + sigma_grid_y
