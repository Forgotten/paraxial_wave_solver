import jax.numpy as jnp
from .config import SimulationConfig, PMLConfig

def generate_pml_profile(sim_config: SimulationConfig, pml_config: PMLConfig):
    """
    Generates the complex PML profile sigma(x, y).
    
    Args:
        sim_config: Simulation configuration.
        pml_config: PML configuration.
        
    Returns:
        sigma: Complex array of shape (Nx, Ny) representing the PML absorption.
    """
    
    x = jnp.arange(sim_config.Nx) * sim_config.dx
    y = jnp.arange(sim_config.Ny) * sim_config.dy
    
    Lx = sim_config.Lx
    Ly = sim_config.Ly
    
    # Define PML regions
    x_start_pml = pml_config.width_x * sim_config.dx
    x_end_pml = Lx - pml_config.width_x * sim_config.dx
    
    y_start_pml = pml_config.width_y * sim_config.dy
    y_end_pml = Ly - pml_config.width_y * sim_config.dy
    
    def sigma_1d(coord, start, end, width_idx, d_step):
        # Distance into PML
        d_left = jnp.maximum(0.0, start - coord)
        d_right = jnp.maximum(0.0, coord - end)
        d = jnp.maximum(d_left, d_right)
        
        # Normalize by PML width
        L_pml = width_idx * d_step
        # Avoid division by zero if width is 0
        L_pml = jnp.where(L_pml > 0, L_pml, 1.0)
        d_norm = d / L_pml
        
        # Polynomial profile
        # If width is 0, d is 0, so result is 0 anyway, but we need to avoid NaN from 0/0
        return jnp.where(width_idx > 0, pml_config.strength * (d_norm ** pml_config.order), 0.0)

    sigma_x = sigma_1d(x, x_start_pml, x_end_pml, pml_config.width_x, sim_config.dx)
    sigma_y = sigma_1d(y, y_start_pml, y_end_pml, pml_config.width_y, sim_config.dy)
    
    # Combine x and y profiles (simple addition for corner regions is standard for split-field, 
    # but for complex coordinate stretching in paraxial, we often just add them to the potential)
    # For paraxial wave equation: 2ik dpsi/dz + Laplacian_perp psi + k^2(n^2 - 1) psi = 0
    # PML is usually implemented as a complex potential V_pml = -i * sigma
    
    sigma_grid_x, sigma_grid_y = jnp.meshgrid(sigma_x, sigma_y, indexing='ij')
    
    # Return the combined damping profile
    return sigma_grid_x + sigma_grid_y
