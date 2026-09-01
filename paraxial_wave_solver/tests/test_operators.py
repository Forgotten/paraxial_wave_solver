import jax.numpy as jnp
import pytest

from paraxial_wave_solver.src.operators import (
  get_spectral_k_grids,
  laplacian_fd,
  laplacian_fd_9point,
  laplacian_spectral,
)


def test_laplacian_spectral_gaussian(x64):
  """Test spectral Laplacian against the analytical derivative of a Gaussian."""
  nx, ny = 128, 128
  dx, dy = 0.1, 0.1
  x = jnp.arange(nx) * dx
  y = jnp.arange(ny) * dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')

  x0, y0 = nx * dx / 2, ny * dy / 2
  sigma = 0.5  # Small enough that the Gaussian decays before the boundary.

  psi = jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))

  # d2/dx2 exp(-x^2 / 2s^2) = (x^2 / s^4 - 1 / s^2) exp(...).
  lap_analytical = (
    ((X - x0)**2 / sigma**4 - 1 / sigma**2) * psi
    + ((Y - y0)**2 / sigma**4 - 1 / sigma**2) * psi
  )

  kx, ky = get_spectral_k_grids(nx, ny, dx, dy)
  lap_num = laplacian_spectral(psi, kx, ky)

  err = (jnp.linalg.norm(lap_num - lap_analytical)
         / jnp.linalg.norm(lap_analytical))
  assert err < 1e-5


@pytest.mark.parametrize("order", [2, 4, 6])
def test_laplacian_fd_convergence_order(order, x64):
  """Test that each FD stencil converges at its nominal order."""
  errors = []
  dxs = [0.2, 0.1, 0.05]

  for dx in dxs:
    nx = int(10 / dx)
    x = jnp.arange(nx) * dx
    psi_1d = jnp.sin(2 * jnp.pi * x / 10)  # Periodic on the domain.
    psi = jnp.outer(psi_1d, psi_1d)

    k = 2 * jnp.pi / 10
    lap_analytical = -2 * k**2 * psi
    lap_num = laplacian_fd(psi, dx, dx, order)

    errors.append(
      jnp.linalg.norm(lap_num - lap_analytical)
      / jnp.linalg.norm(lap_analytical)
    )

  rate = jnp.log(errors[0] / errors[1]) / jnp.log(dxs[0] / dxs[1])
  assert rate > order - 1.0, f"order {order}: measured rate {rate}"


def test_laplacian_9point_convergence_order(x64):
  """The compact 9-point stencil is 2nd order accurate but more isotropic."""
  errors = []
  dxs = [0.2, 0.1, 0.05]

  for dx in dxs:
    nx = int(10 / dx)
    x = jnp.arange(nx) * dx
    psi_1d = jnp.sin(2 * jnp.pi * x / 10)
    psi = jnp.outer(psi_1d, psi_1d)

    k = 2 * jnp.pi / 10
    lap_analytical = -2 * k**2 * psi
    lap_num = laplacian_fd_9point(psi, dx, dx)

    errors.append(
      jnp.linalg.norm(lap_num - lap_analytical)
      / jnp.linalg.norm(lap_analytical)
    )

  rate = jnp.log(errors[0] / errors[1]) / jnp.log(dxs[0] / dxs[1])
  assert rate > 1.0


def test_laplacian_fd_rejects_unknown_order():
  """An unsupported stencil order is reported rather than silently ignored."""
  field = jnp.zeros((8, 8))
  with pytest.raises(ValueError, match="Unsupported FD order"):
    laplacian_fd(field, 0.1, 0.1, order=3)


def test_9point_requires_equal_spacing():
  """The compact stencil is only isotropic for dx == dy, and says so."""
  field = jnp.zeros((8, 8))
  with pytest.raises(ValueError, match="dx == dy"):
    laplacian_fd_9point(field, 0.1, 0.2)


def test_9point_rejects_stretching():
  """Coordinate stretching is unimplemented for the compact stencil."""
  field = jnp.zeros((8, 8))
  params = {'sx': jnp.ones((8, 1)), 'sy': jnp.ones((1, 8)),
            'sx_prime': jnp.zeros((8, 1)), 'sy_prime': jnp.zeros((1, 8))}
  with pytest.raises(NotImplementedError):
    laplacian_fd_9point(field, 0.1, 0.1, params)
