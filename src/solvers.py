import jax.numpy as jnp
from jax import lax, jit
from functools import partial
from typing import Callable, Tuple, Any

from .config import SimulationConfig, SolverConfig, PMLConfig
from .operators import (
    laplacian_fd_2nd, laplacian_fd_4th, laplacian_fd_6th,
    laplacian_spectral, get_spectral_k_grids
)
from .pml import generate_pml_profile

def get_laplacian_fn(config: SolverConfig, sim_config: SimulationConfig) -> Callable:
    """Returns the appropriate Laplacian function based on config."""
    if config.method == 'finite_difference':
        if config.fd_order == 2:
            return partial(laplacian_fd_2nd, dx=sim_config.dx, dy=sim_config.dy)
        elif config.fd_order == 4:
            return partial(laplacian_fd_4th, dx=sim_config.dx, dy=sim_config.dy)
        elif config.fd_order == 6:
            return partial(laplacian_fd_6th, dx=sim_config.dx, dy=sim_config.dy)
        else:
            raise ValueError(f"Unsupported FD order: {config.fd_order}")
    elif config.method == 'spectral':
        kx, ky = get_spectral_k_grids(sim_config.Nx, sim_config.Ny, sim_config.dx, sim_config.dy)
        return partial(laplacian_spectral, kx_grid=kx, ky_grid=ky)
    else:
        raise ValueError(f"Unsupported method: {config.method}")

def rhs_paraxial(psi, z, laplacian_fn, k0, n_ref_fn, pml_profile):
    """
    RHS for Paraxial Wave Equation: 2ik0 dpsi/dz = -Laplacian_perp psi - k0^2(n^2 - 1) psi - 2ik0 sigma psi
    dpsi/dz = (i/2k0) Laplacian_perp psi + (ik0/2)(n^2 - 1) psi - sigma psi
    
    Args:
        psi: Field at current z.
        z: Current z position.
        laplacian_fn: Function to compute transverse Laplacian.
        k0: Wavenumber.
        n_ref_fn: Function n(x, y, z) returning refractive index field.
        pml_profile: PML absorption profile sigma(x, y).
    """
    lap = laplacian_fn(psi)
    
    # We assume n_ref_fn returns the refractive index grid at z
    # If n_ref_fn is just a static array, we can wrap it. 
    # But for general inhomogeneous media, it might depend on z.
    # Here we assume n_ref_fn(z) returns the grid.
    n = n_ref_fn(z)
    chi = n**2 - 1.0
    
    term1 = (1j / (2 * k0)) * lap
    term2 = (1j * k0 / 2) * chi * psi
    term3 = -pml_profile * psi
    
    return term1 + term2 + term3

def step_rk4(psi, z, dz, rhs_fn):
    """Runge-Kutta 4 stepper."""
    k1 = rhs_fn(psi, z)
    k2 = rhs_fn(psi + 0.5 * dz * k1, z + 0.5 * dz)
    k3 = rhs_fn(psi + 0.5 * dz * k2, z + 0.5 * dz)
    k4 = rhs_fn(psi + dz * k3, z + dz)
    
    return psi + (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def step_split_step(psi, z, dz, k0, kx, ky, n_ref_fn, pml_profile):
    """
    Split-step Fourier stepper.
    
    Linear step (diffraction): exp(-i * dz * (kx^2 + ky^2) / (2k0))
    Nonlinear/Potential step (refraction + PML): exp(i * dz * k0 * (n^2 - 1) / 2 - dz * sigma)
    
    Note: This assumes n is constant over dz (or we evaluate at midpoint).
    """
    # 1. Half-step refraction + PML
    n = n_ref_fn(z + 0.5 * dz) # Midpoint evaluation
    chi = n**2 - 1.0
    # Potential operator: exp(i * dz/2 * (k0 * chi / 2) - dz/2 * sigma) ? 
    # Wait, usually split step is: Linear(dz/2) -> Nonlinear(dz) -> Linear(dz/2) or similar.
    # Let's do: Nonlinear(dz/2) -> Linear(dz) -> Nonlinear(dz/2)
    
    nonlinear_op_half = jnp.exp( (1j * k0 * chi / 2 - pml_profile) * (dz / 2) )
    psi = psi * nonlinear_op_half
    
    # 2. Full-step diffraction (Linear)
    psi_k = jnp.fft.fft2(psi)
    k_sq = kx**2 + ky**2
    linear_op = jnp.exp( -1j * dz * k_sq / (2 * k0) )
    psi = jnp.fft.ifft2(psi_k * linear_op)
    
    # 3. Half-step refraction + PML
    # If n depends on z, strictly we should re-evaluate, but for small dz, midpoint is fine.
    # Or we can do Linear(dz) -> Nonlinear(dz) (1st order)
    # Let's stick to the symmetric Strang splitting for 2nd order.
    psi = psi * nonlinear_op_half
    
    return psi

def propagate(psi_0, sim_config: SimulationConfig, solver_config: SolverConfig, pml_config: PMLConfig, n_ref_fn: Callable):
    """
    Main propagation loop.
    
    Args:
        psi_0: Initial field at z=0.
        sim_config: Simulation configuration.
        solver_config: Solver configuration.
        pml_config: PML configuration.
        n_ref_fn: Function taking z and returning refractive index grid n(x, y).
    
    Returns:
        psi_final: Field at z=Lz.
        psi_saved: Field saved at specified intervals (if we implement saving).
                   For now, let's return the full trajectory if it fits in memory, or just final.
                   Let's return the full trajectory for visualization.
    """
    
    pml_profile = generate_pml_profile(sim_config, pml_config)
    
    # Setup step function
    if solver_config.method == 'spectral' and solver_config.stepper == 'split_step':
        kx, ky = get_spectral_k_grids(sim_config.Nx, sim_config.Ny, sim_config.dx, sim_config.dy)
        step_fn = partial(step_split_step, k0=sim_config.k0, kx=kx, ky=ky, n_ref_fn=n_ref_fn, pml_profile=pml_profile)
    else:
        laplacian_fn = get_laplacian_fn(solver_config, sim_config)
        rhs = partial(rhs_paraxial, laplacian_fn=laplacian_fn, k0=sim_config.k0, n_ref_fn=n_ref_fn, pml_profile=pml_profile)
        step_fn = partial(step_rk4, rhs_fn=rhs)

    # Scan loop
    zs = jnp.linspace(0, sim_config.Lz, sim_config.Nz)
    dz = sim_config.dz
    
    def scan_body(carrier, z):
        psi = carrier
        psi_next = step_fn(psi, z, dz)
        return psi_next, psi_next

    psi_final, psi_history = lax.scan(scan_body, psi_0, zs)
    
    # Prepend initial condition to history
    psi_history = jnp.concatenate([psi_0[None, ...], psi_history], axis=0)
    
    return psi_final, psi_history
