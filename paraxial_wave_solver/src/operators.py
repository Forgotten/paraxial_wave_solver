import jax.numpy as jnp
from jax import lax
from typing import Tuple
from .config import Field

def laplacian_fd_2nd(field: Field, dx: float, dy: float) -> Field:
  """Computes the 2D Laplacian using a 2nd-order finite difference scheme.
  
  Args:
    field: Input 2D field array of shape (nx, ny).
    dx: Grid spacing in the x-direction.
    dy: Grid spacing in the y-direction.
    
  Returns:
    The Laplacian of the input field, same shape as input.
  """
  def d2_2nd(u: Field, h: float, axis: int) -> Field:
    return ( jnp.roll(u, -1, axis=axis) + 
             -2 * u + 
             jnp.roll(u, 1, axis=axis)) / (h**2)
  
  return d2_2nd(field, dx, 0) + d2_2nd(field, dy, 1)

def laplacian_fd_4th(field: Field, dx: float, dy: float) -> Field:
  """Computes the 2D Laplacian using a 4th-order finite difference scheme.
  
  Args:
    field: Input 2D field array of shape (nx, ny).
    dx: Grid spacing in the x-direction.
    dy: Grid spacing in the y-direction.
    
  Returns:
    The Laplacian of the input field, same shape as input.
  """
  # Coefficients for 4th order central difference: 
  # [-1/12, 4/3, -5/2, 4/3, -1/12]
  
  def d2_4th(u: Field, h: float, axis: int) -> Field:
    return (-1/12 * jnp.roll(u, -2, axis=axis) + 
             4/3  * jnp.roll(u, -1, axis=axis) - 
             5/2  * u + 
             4/3  * jnp.roll(u, 1, axis=axis) - 
             1/12 * jnp.roll(u, 2, axis=axis)) / (h**2)

  return d2_4th(field, dx, 0) + d2_4th(field, dy, 1)

def laplacian_fd_6th(field: Field, dx: float, dy: float) -> Field:
  """Computes the 2D Laplacian using a 6th-order finite difference scheme.
  
  Args:
    field: Input 2D field array of shape (nx, ny).
    dx: Grid spacing in the x-direction.
    dy: Grid spacing in the y-direction.
    
  Returns:
    The Laplacian of the input field, same shape as input.
  """
  # Coefficients: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
  
  def d2_6th(u: Field, h: float, axis: int) -> Field:
    return ( 1/90 * jnp.roll(u, -3, axis=axis) - 
             3/20 * jnp.roll(u, -2, axis=axis) + 
             3/2  * jnp.roll(u, -1, axis=axis) - 
             49/18 * u + 
             3/2  * jnp.roll(u, 1, axis=axis) - 
             3/20 * jnp.roll(u, 2, axis=axis) + 
             1/90 * jnp.roll(u, 3, axis=axis)) / (h**2)

  return d2_6th(field, dx, 0) + d2_6th(field, dy, 1)

def laplacian_fd_9point(field: Field, dx: float, dy: float) -> Field:
  """Computes the 2D Laplacian using an isotropic 9-point stencil (compact 3x3).

  This stencil includes cross-terms to improve isotropy compared to the 
  5-point stencil.

  Args:
    field: Input 2D field (nx, ny).
    dx: Grid spacing in x.
    dy: Grid spacing in y.

  Returns:
    The Laplacian of the field.
  """
  # Standard 9-point isotropic stencil for dx=dy=h:
  # L = Dxx + Dyy + (h^2/6) DxxDyy
  if dx != dy:
    raise ValueError("dx and dy must be equal for 9-point stencil.")
  
  Dxx_u = (jnp.roll(field, -1, axis=0) - 2 * field + 
           jnp.roll(field, 1, axis=0)) / (dx**2)
  Dyy_u = (jnp.roll(field, -1, axis=1) - 2 * field + 
           jnp.roll(field, 1, axis=1)) / (dy**2)
  
  # We apply Dxx to Dyy_u
  DxxDyy_u = (jnp.roll(Dyy_u, -1, axis=0) - 2 * Dyy_u + 
              jnp.roll(Dyy_u, 1, axis=0)) / (dx**2)
  
  # L = Dxx + Dyy + (dx**2/6) * DxxDyy (assuming dx=dy).
  return Dxx_u + Dyy_u + (dx**2 / 6.0) * DxxDyy_u

def get_spectral_k_grids(nx: int, ny: int, dx: float, dy: float
                        ) -> Tuple[Field, Field]:
  """Generates the wavenumber grids (kx, ky) for spectral methods.
  
  Args:
    nx: Number of grid points in the x-direction.
    ny: Number of grid points in the y-direction.
    dx: Grid spacing in the x-direction.
    dy: Grid spacing in the y-direction.
    
  Returns:
    A tuple (kx_grid, ky_grid) of 2D arrays containing the wavenumbers.
  """
  kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
  ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
  return jnp.meshgrid(kx, ky, indexing='ij')

def laplacian_spectral(field: Field, kx_grid: Field, ky_grid: Field) -> Field:
  """Computes the 2D Laplacian using the pseudo-spectral method (FFT).
  
  Args:
    field: Input 2D field array of shape (nx, ny).
    kx_grid: 2D array of x-wavenumbers.
    ky_grid: 2D array of y-wavenumbers.
    
  Returns:
    The Laplacian of the input field, same shape as input.
  """
  field_k = jnp.fft.fft2(field)
  lap_k = -(kx_grid**2 + ky_grid**2) * field_k
  return jnp.fft.ifft2(lap_k)
