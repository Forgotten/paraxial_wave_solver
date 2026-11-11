"""Fourth-order exponential Runge–Kutta propagation for spectral diffraction.

Cox–Matthews ETDRK4 integrates u' = L*u + N(z, u). L is diagonal in Fourier
space; refraction, Kerr, and the absorbing potential belong to N. Coefficients
use Taylor series near zero to avoid cancellation in the phi functions.
"""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from .config import Field


def phi_functions(value: Field) -> tuple[Field, Field, Field]:
  """Evaluates phi_1, phi_2, phi_3 with their analytic limits at zero.

  phi_j(x) = sum(x**m / (m+j)!, m=0..infinity). A 20-term series is used
  for |x| <= 1; elsewhere recurrence from expm1 avoids large polynomials.
  Both branches stay finite at zero for reverse-mode differentiation.
  """
  small = jnp.abs(value) <= 1
  denominator = jnp.where(small, 1, value)
  first = jnp.expm1(value) / denominator
  second = (first - 1) / denominator
  third = (second - 0.5) / denominator
  x = jnp.where(small, value, 0)
  series = []
  for degree, initial in ((1, 1.), (2, 0.5), (3, 1/6)):
    term = jnp.ones_like(value) * initial
    total = term
    for power in range(1, 20):
      term = term * x / (power + degree)
      total = total + term
    series.append(total)
  return tuple(jnp.where(small, s, r)
               for s, r in zip(series, (first, second, third), strict=True))


def etd_coefficients(symbol: Field, step: Any) -> dict[str, Field]:
  """Builds Fourier multipliers for a single ETDRK4 step.

  Args:
    symbol: Fourier symbol of the linear generator L, not the Laplacian.
    step: Propagation step, possibly traced.

  Returns:
    Multipliers including step factors; at symbol=0 these recover RK4.
  """
  x = step * symbol
  p1, p2, p3 = phi_functions(x)
  half_phi = phi_functions(x / 2)[0]
  return {
    'e': jnp.exp(x), 'e_half': jnp.exp(x / 2),
    'q': step / 2 * half_phi,
    'b1': step * (p1 - 3*p2 + 4*p3),
    'b2': step * (2*p2 - 4*p3),
    'b3': step * (-p2 + 4*p3),
  }


def make_etdrk4_kernel(
  n2: float,
  delta_n_fn: Callable[[Any, Any], Any] | None,
  has_pml: bool,
) -> Callable[[Field, Any, dict[str, Any], Any], Field]:
  """Builds ETDRK4 with medium samples at z, z+h/2, and z+h.

  The local terms are explicit: their stability restrictions remain even
  though diffraction is integrated exactly. No outer absorber splitting is
  used. Smooth range dependence is required for fourth-order convergence.
  """
  def step(psi: Field, z: Any, operators: dict[str, Any], medium: Any) -> Field:
    h, k0 = operators['dz'], operators['k0']
    coefficients = operators['etd']

    def nonlinear(spectrum: Field, position: Any) -> Field:
      field = jnp.fft.ifft2(spectrum)
      index = 0.0 if delta_n_fn is None else delta_n_fn(position, medium)
      if n2 != 0:
        index = index + n2 * jnp.abs(field)**2
      rhs = 1j * k0 * index * field
      if has_pml:
        rhs = rhs - operators['sigma'] * field
      return jnp.fft.fft2(rhs)

    u = jnp.fft.fft2(psi)
    e, half, q = coefficients['e'], coefficients['e_half'], coefficients['q']
    nu = nonlinear(u, z)
    a = half*u + q*nu
    na = nonlinear(a, z + h/2)
    b = half*u + q*na
    nb = nonlinear(b, z + h/2)
    c = half*a + q*(2*nb - nu)
    nc = nonlinear(c, z + h)
    return jnp.fft.ifft2(
      e*u + coefficients['b1']*nu + coefficients['b2']*(na+nb)
      + coefficients['b3']*nc
    )

  return step
