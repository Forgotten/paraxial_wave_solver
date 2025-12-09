import jax.numpy as jnp
from jax import lax, jit
from jax.tree_util import Partial
from functools import partial
from typing import Callable, Tuple, Any

from .config import SimulationConfig, SolverConfig, PMLConfig, Field
from .operators import (
  laplacian_fd_2nd, laplacian_fd_4th, laplacian_fd_6th,
  laplacian_fd_9point,
  laplacian_spectral, get_spectral_k_grids
)
from .pml import generate_pml_profile

def get_laplacian_fn(config: SolverConfig, 
                     sim_config: SimulationConfig) -> Callable[[Field], Field]:
  """
  Returns the appropriate Laplacian function based on the solver configuration.
  
  Args:
    config: Solver configuration specifying the method and order.
    sim_config: Simulation configuration specifying grid parameters.
    
  Returns:
    A callable that takes a Field and returns its Laplacian as a Field.
  """
  if config.method == 'finite_difference':
    if config.compact:
      # Use the 9-point isotropic stencil
      return Partial(laplacian_fd_9point, dx=sim_config.dx, dy=sim_config.dy)

    if config.fd_order == 2:
      return Partial(laplacian_fd_2nd, dx=sim_config.dx, dy=sim_config.dy)
    elif config.fd_order == 4:
      return Partial(laplacian_fd_4th, dx=sim_config.dx, dy=sim_config.dy)
    elif config.fd_order == 6:
      return Partial(laplacian_fd_6th, dx=sim_config.dx, dy=sim_config.dy)
    else:
      raise ValueError(f"Unsupported FD order: {config.fd_order}")
  elif config.method == 'spectral':
    kx, ky = get_spectral_k_grids(sim_config.nx, sim_config.ny, 
                                  sim_config.dx, sim_config.dy)
    return Partial(laplacian_spectral, kx_grid=kx, ky_grid=ky)
  else:
    raise ValueError(f"Unsupported method: {config.method}")

def rhs_paraxial(psi: Field, z: float, laplacian_fn: Callable[[Field], Field], 
                 k0: float, n_ref_fn: Callable[[float], Field], 
                 pml_profile: Field) -> Field:
  """
  Computes the Right-Hand Side (RHS) of the Paraxial Wave Equation.
  
  The equation is: 
  2ik0 dpsi/dz = -Laplacian_perp psi - k0^2(n^2 - 1) psi - 2ik0 sigma psi
  
  Rearranging for dpsi/dz:
  dpsi/dz = (i/2k0) Laplacian_perp psi + (ik0/2)(n^2 - 1) psi - sigma psi
  
  Args:
    psi: Complex field amplitude at the current z-step.
    z: Current propagation distance.
    laplacian_fn: Function to compute the transverse Laplacian.
    k0: Vacuum wavenumber.
    n_ref_fn: Function n(z) returning the refractive index grid n(x, y) at z.
    pml_profile: PML absorption profile sigma(x, y).
    
  Returns:
    dpsi_dz: The derivative of the field with respect to z.
  """
  lap = laplacian_fn(psi)
  
  # We assume n_ref_fn returns the refractive index grid at z
  n = n_ref_fn(z)
  chi = n**2 - 1.0
  
  term1 = (1j / (2 * k0)) * lap
  term2 = (1j * k0 / 2) * chi * psi
  term3 = -pml_profile * psi
  
  return term1 + term2 + term3

def step_rk4(psi: Field, z: float, dz: float, 
             rhs_fn: Callable[[Field, float], Field]) -> Field:
  """
  Performs a single z-step using the 4th-order Runge-Kutta method.
  
  Args:
    psi: Field at current z.
    z: Current z position.
    dz: Step size.
    rhs_fn: Function computing the RHS dpsi/dz = f(psi, z).
    
  Returns:
    psi_next: Field at z + dz.
  """
  k1 = rhs_fn(psi, z)
  k2 = rhs_fn(psi + 0.5 * dz * k1, z + 0.5 * dz)
  k3 = rhs_fn(psi + 0.5 * dz * k2, z + 0.5 * dz)
  k4 = rhs_fn(psi + dz * k3, z + dz)
  
  return psi + (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def step_split_step(psi: Field, z: float, dz: float, k0: float, kx: Field, 
                    ky: Field, n_ref_fn: Callable[[float], Field], 
                    pml_profile: Field) -> Field:
  """
  Performs a single z-step using the Split-Step Fourier method.
  
  Uses symmetric Strang splitting (2nd order accuracy in z):
  1. Half-step of nonlinear/potential operator (refraction + PML).
  2. Full-step of linear diffraction operator (in Fourier space).
  3. Half-step of nonlinear/potential operator.
  
  Args:
    psi: Field at current z.
    z: Current z position.
    dz: Step size.
    k0: Vacuum wavenumber.
    kx, ky: Transverse wavenumber grids.
    n_ref_fn: Function n(z) returning refractive index grid.
    pml_profile: PML absorption profile.
    
  Returns:
    psi_next: Field at z + dz.
  """
  # 1. Half-step refraction + PML
  n = n_ref_fn(z + 0.5 * dz) # Midpoint evaluation
  chi = n**2 - 1.0
  
  nonlinear_op_half = jnp.exp( (1j * k0 * chi / 2 - pml_profile) * (dz / 2) )
  psi = psi * nonlinear_op_half
  
  # 2. Full-step diffraction (Linear)
  psi_k = jnp.fft.fft2(psi)
  k_sq = kx**2 + ky**2
  linear_op = jnp.exp( -1j * dz * k_sq / (2 * k0) )
  psi = jnp.fft.ifft2(psi_k * linear_op)
  
  # 3. Half-step refraction + PML
  psi = psi * nonlinear_op_half
  
  return psi

@jit
def _solve_scan(psi_0: Field, zs: Field, dz: float, 
                step_fn: Callable[[Field, float, float], Field]
                ) -> Tuple[Field, Field]:
  """ JIT-compiled scan loop for efficient propagation.
  
  Args:
    psi_0: Initial field.
    zs: Array of z positions.
    dz: Step size.
    step_fn: Stepper function.
    
  Returns:
    A tuple containing:
    - psi_final: Field at the end of propagation.
    - psi_history: History of the field at each step.
  """
  def scan_body(carrier: Field, z: float) -> Tuple[Field, Field]:
    psi = carrier
    psi_next = step_fn(psi, z, dz)
    return psi_next, psi_next

  psi_final, psi_history = lax.scan(scan_body, psi_0, zs)
  return psi_final, psi_history

class ParaxialWaveSolver:
  """
  Solver for the Paraxial Wave Equation.
  
  Encapsulates the simulation configuration, solver method, PML settings, and 
  refractive index profile. Provides a method to propagate an initial field 
  through the medium.
  """
  
  def __init__(self, sim_config: SimulationConfig, solver_config: SolverConfig, 
               pml_config: PMLConfig, n_ref_fn: Callable[[float], Field]):
    """
    Initialize the solver.
    
    Args:
      sim_config: Simulation configuration.
      solver_config: Solver configuration.
      pml_config: PML configuration.
      n_ref_fn: Function taking z and returning refractive index grid n(x, y).
    """
    self.sim_config = sim_config
    self.solver_config = solver_config
    self.pml_config = pml_config
    self.n_ref_fn = n_ref_fn
    
    # Pre-compute PML profile
    self.pml_profile = generate_pml_profile(sim_config, pml_config)
    
    # Wrap n_ref_fn in Partial if it's a plain function to ensure it's a valid 
    # PyTree node (though Partial treats the func as static, which is what we 
    # want for a pure function).
    # Note: If n_ref_fn is already a Partial or JAX-compatible callable, 
    # wrapping it again is harmless. For safety with JIT, we wrap it.
    self.n_ref_fn_partial = Partial(n_ref_fn)
    
    # Setup step function using Partial to be JIT-friendly
    if (solver_config.method == 'spectral' and 
        solver_config.stepper == 'split_step'):
      kx, ky = get_spectral_k_grids(sim_config.nx, sim_config.ny, 
                                    sim_config.dx, sim_config.dy)
      self.step_fn = Partial(step_split_step, k0=sim_config.k0, kx=kx, ky=ky, 
                             n_ref_fn=self.n_ref_fn_partial, 
                             pml_profile=self.pml_profile)
    else:
      laplacian_fn = get_laplacian_fn(solver_config, sim_config)
      rhs = Partial(rhs_paraxial, laplacian_fn=laplacian_fn, k0=sim_config.k0, 
                    n_ref_fn=self.n_ref_fn_partial, 
                    pml_profile=self.pml_profile)
      self.step_fn = Partial(step_rk4, rhs_fn=rhs)

  def solve(self, psi_0: Field) -> Tuple[Field, Field]:
    """
    Propagates the initial field psi_0 through the medium.
    
    Args:
      psi_0: Initial complex field amplitude at z=0.
      
    Returns:
      psi_final: Field at z=lz.
      psi_history: Field history at all z steps (including z=0).
    """
    zs = jnp.linspace(0, self.sim_config.lz, self.sim_config.nz)
    dz = self.sim_config.dz
    
    # Call the JIT-compiled scan loop
    psi_final, psi_history = _solve_scan(psi_0, zs, dz, self.step_fn)
    
    # Prepend initial condition to history
    psi_history = jnp.concatenate([psi_0[None, ...], psi_history], axis=0)
    
    return psi_final, psi_history

def propagate(psi_0: Field, sim_config: SimulationConfig, 
              solver_config: SolverConfig, pml_config: PMLConfig, 
              n_ref_fn: Callable[[float], Field]) -> Tuple[Field, Field]:
  """
  Main propagation loop (Legacy wrapper).
  
  Args:
    psi_0: Initial field at z=0.
    sim_config: Simulation configuration.
    solver_config: Solver configuration.
    pml_config: PML configuration.
    n_ref_fn: Function taking z and returning refractive index grid n(x, y).
  
  Returns:
    psi_final: Field at z=lz.
    psi_history: Field history at all z steps.
  """
  solver = ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_fn)
  return solver.solve(psi_0)
