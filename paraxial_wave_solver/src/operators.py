"""Transverse Laplacian operators: finite difference stencils and spectral."""

from collections.abc import Callable

import jax.numpy as jnp

from .config import FD_ORDERS, Field

StencilFn = Callable[[Field, float, int], Field]


def d1_2nd(u: Field, h: float, axis: int) -> Field:
  return (jnp.roll(u, -1, axis=axis) - jnp.roll(u, 1, axis=axis)) / (2 * h)


def d1_4th(u: Field, h: float, axis: int) -> Field:
  return (
    -jnp.roll(u, -2, axis=axis)
    + 8 * jnp.roll(u, -1, axis=axis)
    - 8 * jnp.roll(u, 1, axis=axis)
    + jnp.roll(u, 2, axis=axis)
  ) / (12 * h)


def d1_6th(u: Field, h: float, axis: int) -> Field:
  return (
    1 / 60 * jnp.roll(u, -3, axis=axis)
    - 3 / 20 * jnp.roll(u, -2, axis=axis)
    + 3 / 4 * jnp.roll(u, -1, axis=axis)
    - 3 / 4 * jnp.roll(u, 1, axis=axis)
    + 3 / 20 * jnp.roll(u, 2, axis=axis)
    - 1 / 60 * jnp.roll(u, 3, axis=axis)
  ) / h


def d2_2nd(u: Field, h: float, axis: int) -> Field:
  return (
    jnp.roll(u, -1, axis=axis)
    - 2 * u
    + jnp.roll(u, 1, axis=axis)
  ) / (h**2)


def d2_4th(u: Field, h: float, axis: int) -> Field:
  return (
    -1 / 12 * jnp.roll(u, -2, axis=axis)
    + 4 / 3 * jnp.roll(u, -1, axis=axis)
    - 5 / 2 * u
    + 4 / 3 * jnp.roll(u, 1, axis=axis)
    - 1 / 12 * jnp.roll(u, 2, axis=axis)
  ) / (h**2)


def d2_6th(u: Field, h: float, axis: int) -> Field:
  return (
    1 / 90 * jnp.roll(u, -3, axis=axis)
    - 3 / 20 * jnp.roll(u, -2, axis=axis)
    + 3 / 2 * jnp.roll(u, -1, axis=axis)
    - 49 / 18 * u
    + 3 / 2 * jnp.roll(u, 1, axis=axis)
    - 3 / 20 * jnp.roll(u, 2, axis=axis)
    + 1 / 90 * jnp.roll(u, 3, axis=axis)
  ) / (h**2)


# Second- and first-derivative stencil pairs, keyed by order of accuracy. The
# first derivative is only needed when complex coordinate stretching is active.
STENCILS: dict[int, tuple[StencilFn, StencilFn]] = {
  2: (d2_2nd, d1_2nd),
  4: (d2_4th, d1_4th),
  6: (d2_6th, d1_6th),
}


def apply_stretched_op(
  u: Field,
  d2_fn: StencilFn,
  d1_fn: StencilFn,
  h: float,
  axis: int,
  s: Field,
  s_prime: Field,
) -> Field:
  """Applies the stretched derivative operator: (1/s) d/dx ((1/s) d/dx u).

  Expands to: (1/s^2) d^2u/dx^2 - (s'/s^3) du/dx.

  Args:
    u: Input field.
    d2_fn: Second-derivative stencil.
    d1_fn: First-derivative stencil.
    h: Grid spacing along the axis.
    axis: Axis to differentiate along.
    s: Complex stretch factor, broadcastable to u's shape.
    s_prime: Derivative of the stretch factor, broadcastable to u's shape.

  Returns:
    The stretched second derivative of u along the given axis.
  """
  d2_u = d2_fn(u, h, axis)
  d1_u = d1_fn(u, h, axis)
  return (1.0 / s**2) * d2_u - (s_prime / s**3) * d1_u


def laplacian_fd(
  field: Field,
  dx: float,
  dy: float,
  order: int = 2,
  pml_params: dict[str, Field] | None = None,
) -> Field:
  """Computes the 2D Laplacian with a central finite difference stencil.

  Args:
    field: Input 2D field array of shape (nx, ny).
    dx: Grid spacing in the x-direction.
    dy: Grid spacing in the y-direction.
    order: Order of accuracy; one of FD_ORDERS.
    pml_params: Optional dict with 'sx', 'sy', 'sx_prime' and 'sy_prime'
                complex coordinate-stretching fields.

  Returns:
    The Laplacian of the input field, same shape as input.
  """
  if order not in STENCILS:
    raise ValueError(f"Unsupported FD order {order!r}; expected one of {FD_ORDERS}.")
  d2_fn, d1_fn = STENCILS[order]

  if pml_params is None:
    return d2_fn(field, dx, 0) + d2_fn(field, dy, 1)

  lap_x = apply_stretched_op(
    field, d2_fn, d1_fn, dx, 0, pml_params['sx'], pml_params['sx_prime']
  )
  lap_y = apply_stretched_op(
    field, d2_fn, d1_fn, dy, 1, pml_params['sy'], pml_params['sy_prime']
  )
  return lap_x + lap_y


def laplacian_fd_9point(
  field: Field,
  dx: float,
  dy: float,
  pml_params: dict[str, Field] | None = None,
) -> Field:
  """Computes the 2D Laplacian using an isotropic 9-point stencil (compact 3x3).

  This stencil includes cross-terms to improve isotropy compared to the
  5-point stencil: L = Dxx + Dyy + (h^2/6) Dxx Dyy, valid for dx == dy.

  Args:
    field: Input 2D field of shape (nx, ny).
    dx: Grid spacing in x.
    dy: Grid spacing in y; must equal dx.
    pml_params: Unsupported; must be None.

  Returns:
    The Laplacian of the field.
  """
  if pml_params is not None:
    raise NotImplementedError(
      "Complex coordinate stretching is not implemented for the 9-point "
      "stencil. Use use_complex_stretching=False, or an fd_order in "
      f"{FD_ORDERS}."
    )
  if dx != dy:
    raise ValueError(
      f"The 9-point stencil requires dx == dy, got dx={dx!r}, dy={dy!r}."
    )

  dxx_u = d2_2nd(field, dx, 0)
  dyy_u = d2_2nd(field, dy, 1)
  dxx_dyy_u = d2_2nd(dyy_u, dx, 0)
  return dxx_u + dyy_u + (dx**2 / 6.0) * dxx_dyy_u


def get_spectral_k_grids(
  nx: int,
  ny: int,
  dx: float,
  dy: float,
) -> tuple[Field, Field]:
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
