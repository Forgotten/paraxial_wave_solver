import jax.numpy as jnp
import pytest
from src.config import SimulationConfig, SolverConfig, PMLConfig
from src.solvers import propagate

def test_energy_conservation_vacuum():
  """Test that energy is conserved in vacuum (no PML)."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0
  )
  # No PML
  pml_config = PMLConfig(width_x=0, width_y=0, strength=0.0)
  
  # Spectral solver + Split step is unitary for vacuum
  solver_config = SolverConfig(method='spectral', stepper='split_step')
  
  # Initial condition: Gaussian
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')
  psi_0 = jnp.exp(-((X-3.2)**2 + (Y-3.2)**2))
  
  # Vacuum refractive index
  def n_ref_fn(z):
    return jnp.ones((sim_config.nx, sim_config.ny))
    
  psi_final, _ = propagate(psi_0, sim_config, solver_config, pml_config, n_ref_fn)
  
  E_in = jnp.sum(jnp.abs(psi_0)**2)
  E_out = jnp.sum(jnp.abs(psi_final)**2)
  
  assert jnp.abs(E_in - E_out) / E_in < 1e-6

def test_pml_absorption():
  """Test that PML absorbs outgoing waves."""
  sim_config = SimulationConfig(
    nx=100, ny=100, dx=0.1, dy=0.1, dz=0.1, nz=50, wavelength=1.0
  )
  # Strong PML
  pml_config = PMLConfig(width_x=20, width_y=20, strength=10.0)
  
  solver_config = SolverConfig(method='spectral', stepper='split_step')
  
  # Initial condition: Gaussian near edge moving towards PML? 
  # Or just a spreading Gaussian in center that eventually hits PML.
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')
  psi_0 = jnp.exp(-((X-5.0)**2 + (Y-5.0)**2))
  
  def n_ref_fn(z):
    return jnp.ones((sim_config.nx, sim_config.ny))
    
  psi_final, _ = propagate(psi_0, sim_config, solver_config, pml_config, n_ref_fn)
  
  # Energy should decrease
  E_in = jnp.sum(jnp.abs(psi_0)**2)
  E_out = jnp.sum(jnp.abs(psi_final)**2)
  
  assert E_out < E_in
