import jax
import jax.numpy as jnp
import pytest
from src.operators import laplacian_fd_2nd, laplacian_fd_4th, laplacian_fd_6th, laplacian_fd_9point, laplacian_spectral, get_spectral_k_grids

def test_laplacian_spectral_gaussian():
  """Test spectral Laplacian against analytical derivative of a Gaussian."""
  jax.config.update("jax_enable_x64", True)
  nx, ny = 128, 128
  dx, dy = 0.1, 0.1
  x = jnp.arange(nx) * dx
  y = jnp.arange(ny) * dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')
  
  # Gaussian center
  x0, y0 = nx*dx/2, ny*dy/2
  sigma = 0.5  # Smaller sigma to avoid boundary effects
  
  psi = jnp.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * sigma**2))
  
  # Analytical Laplacian
  # d2/dx2 (exp(-x^2/2s^2)) = (x^2/s^4 - 1/s^2) exp(...)
  lap_analytical = ((X-x0)**2 / sigma**4 - 1/sigma**2) * psi + \
                   ((Y-y0)**2 / sigma**4 - 1/sigma**2) * psi
                   
  kx, ky = get_spectral_k_grids(nx, ny, dx, dy)
  lap_num = laplacian_spectral(psi, kx, ky)
  
  # Error should be small (spectral accuracy)
  # Note: Periodic boundaries assumed, Gaussian should decay to 0 at boundaries
  err = jnp.linalg.norm(lap_num - lap_analytical) / jnp.linalg.norm(lap_analytical)
  assert err < 1e-5

def test_laplacian_fd_order():
  """Test convergence order of FD schemes."""
  # We check if error decreases as expected when refining grid
  # 9-point stencil is 2nd order accurate (but isotropic)
  orders = [2, 4, 6, 2] 
  funcs = [laplacian_fd_2nd, laplacian_fd_4th, laplacian_fd_6th, laplacian_fd_9point]
  names = ["2nd", "4th", "6th", "9-point"]
  
  for order, func, name in zip(orders, funcs, names):
    errors = []
    dxs = [0.2, 0.1, 0.05]
    
    for dx in dxs:
      nx = int(10 / dx)
      x = jnp.arange(nx) * dx
      # 1D test for simplicity, broadcast to 2D
      psi_1d = jnp.sin(2 * jnp.pi * x / 10) # Periodic
      psi = jnp.outer(psi_1d, psi_1d)
      
      # Analytical laplacian of sin(kx)sin(ky) = -2k^2 sin(kx)sin(ky)
      k = 2 * jnp.pi / 10
      lap_analytical = -2 * k**2 * psi
      
      lap_num = func(psi, dx, dx)
      
      err = jnp.linalg.norm(lap_num - lap_analytical) / jnp.linalg.norm(lap_analytical)
      errors.append(err)
      
    # Check convergence rate: log(err1/err2) / log(dx1/dx2) ~ order
    rate = jnp.log(errors[0]/errors[1]) / jnp.log(dxs[0]/dxs[1])
    print(f"Scheme {name} (Expected Order {order}) measured rate: {rate}, Errors: {errors}")
    # With x64, this should pass. With float32, high orders hit machine precision fast.
    # We'll assert a bit loosely.
    assert rate > order - 1.0
