import jax.numpy as jnp
from .config import SimulationConfig, PMLConfig, Field

def generate_pml_profile(sim_config: SimulationConfig, pml_config: PMLConfig) -> Field:
    """
    Generates the complex Perfectly Matched Layer (PML) absorption profile sigma(x, y).
    
    The profile is zero in the physical domain and increases polynomially in the PML regions
    at the boundaries.
    
    Args:
        sim_config: Simulation configuration containing grid details.
        pml_config: PML configuration containing width, strength, and order.
        
    Returns:
        sigma: Complex array of shape (nx, ny) representing the PML absorption profile.
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
    
    def sigma_1d(coord: Field, start: float, end: float, width_idx: int, d_step: float) -> Field:
        # Distance into PML
        d_left = jnp.maximum(0.0, start - coord)
        d_right = jnp.maximum(0.0, coord - end)
        d = jnp.maximum(d_left, d_right)
        
        # Normalize by PML width
        l_pml = width_idx * d_step
        # Avoid division by zero if width is 0
        l_pml = jnp.where(l_pml > 0, l_pml, 1.0)
        d_norm = d / l_pml
        
        # Polynomial profile
        # If width is 0, d is 0, so result is 0 anyway, but we need to avoid NaN from 0/0
        return jnp.where(width_idx > 0, pml_config.strength * (d_norm ** pml_config.order), 0.0)

    sigma_x = sigma_1d(x, x_start_pml, x_end_pml, pml_config.width_x, sim_config.dx)
    sigma_y = sigma_1d(y, y_start_pml, y_end_pml, pml_config.width_y, sim_config.dy)
    
    # Combine x and y profiles
    sigma_grid_x, sigma_grid_y = jnp.meshgrid(sigma_x, sigma_y, indexing='ij')
    
    # Return the combined damping profile
    return sigma_grid_x + sigma_grid_y
