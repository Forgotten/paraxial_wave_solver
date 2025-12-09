import jax.numpy as jnp
from jax import lax

def laplacian_fd_2nd(field, dx, dy):
    """2nd order finite difference Laplacian."""
    # d2/dx2
    d2x = (jnp.roll(field, -1, axis=0) - 2 * field + jnp.roll(field, 1, axis=0)) / (dx**2)
    # d2/dy2
    d2y = (jnp.roll(field, -1, axis=1) - 2 * field + jnp.roll(field, 1, axis=1)) / (dy**2)
    return d2x + d2y

def laplacian_fd_4th(field, dx, dy):
    """4th order finite difference Laplacian."""
    # Coefficients for 4th order central difference: [-1/12, 4/3, -5/2, 4/3, -1/12]
    
    def d2_4th(u, h, axis):
        return (-1/12 * jnp.roll(u, -2, axis=axis) + 
                 4/3  * jnp.roll(u, -1, axis=axis) - 
                 5/2  * u + 
                 4/3  * jnp.roll(u, 1, axis=axis) - 
                 1/12 * jnp.roll(u, 2, axis=axis)) / (h**2)

    return d2_4th(field, dx, 0) + d2_4th(field, dy, 1)

def laplacian_fd_6th(field, dx, dy):
    """6th order finite difference Laplacian."""
    # Coefficients: [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
    
    def d2_6th(u, h, axis):
        return ( 1/90 * jnp.roll(u, -3, axis=axis) - 
                 3/20 * jnp.roll(u, -2, axis=axis) + 
                 3/2  * jnp.roll(u, -1, axis=axis) - 
                 49/18 * u + 
                 3/2  * jnp.roll(u, 1, axis=axis) - 
                 3/20 * jnp.roll(u, 2, axis=axis) + 
                 1/90 * jnp.roll(u, 3, axis=axis)) / (h**2)

    return d2_6th(field, dx, 0) + d2_6th(field, dy, 1)

def get_spectral_k_grids(Nx, Ny, dx, dy):
    """Returns kx and ky grids for spectral method."""
    kx = 2 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(Ny, d=dy)
    return jnp.meshgrid(kx, ky, indexing='ij')

def laplacian_spectral(field, kx_grid, ky_grid):
    """Spectral Laplacian using FFT."""
    field_k = jnp.fft.fft2(field)
    lap_k = -(kx_grid**2 + ky_grid**2) * field_k
    return jnp.fft.ifft2(lap_k)
