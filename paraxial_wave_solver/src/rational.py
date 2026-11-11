"""Rotated Padé approximation of the one-way Helmholtz propagation function.

For X = H/k**2 - I, approximate exp(i*h*k*(sqrt(I+X)-I)). Rotation moves the
expansion centre off the real axis so negative eigenvalues use the outgoing
square-root branch. This is an approximation, not a universal stability
guarantee: degree, range step, rotation, and spectral extent must be checked.

Coefficients are computed in JAX, so h and k remain differentiable. Applying
the numerator and denominator as operator polynomials avoids differentiating
polynomial roots or introducing nested iterative square-root evaluations.
"""

from collections.abc import Callable

import jax.numpy as jnp

from .config import Field


def pade_coefficients(
  phase: Field,
  order: int = 4,
  rotation: float = 0.5235987755982988,
) -> tuple[Field, Field]:
  """Returns ascending-power numerator/denominator coefficients in Y.

  Y = exp(-i*rotation) * H/k**2 - I. The approximated scalar function is
  exp(i*phase*(exp(i*rotation/2)*sqrt(1+Y)-1)). Its Taylor series is formed by
  the exponential recurrence and matched through degree 2*order. Only the
  constant exponential is factored out; this reduces unnecessary scaling in
  the small coefficient solve.

  Args:
    phase: h*k, dimensionless and possibly traced. Must be nonzero.
    order: Numerator and denominator degree, between one and six.
    rotation: Branch rotation in radians, strictly between zero and pi.

  Returns:
    Two complex arrays of length order+1. Use float64 inputs for accuracy.
  """
  if not isinstance(order, int) or not 1 <= order <= 6:
    raise ValueError("order must be an integer between one and six.")
  if not 0 < rotation < jnp.pi:
    raise ValueError("rotation must lie strictly between zero and pi.")
  root_rotation = jnp.exp(0.5j * rotation)
  root = [1.0]
  for n in range(1, 2 * order + 1):
    root.append(root[-1] * (1.5 - n) / n)
  log_series = [1j * phase * root_rotation * b for b in root]
  series = [jnp.ones_like(log_series[0])]
  for n in range(1, 2 * order + 1):
    series.append(sum(
      j * log_series[j] * series[n - j] for j in range(1, n + 1)
    ) / n)
  matrix = jnp.stack([
    jnp.stack([series[n - j] for j in range(1, order + 1)])
    for n in range(order + 1, 2 * order + 1)
  ])
  tail = jnp.linalg.solve(matrix, -jnp.stack(series[order + 1:]))
  denominator = jnp.concatenate([jnp.ones_like(tail[:1]), tail])
  numerator = jnp.stack([
    sum(denominator[j] * series[n - j] for j in range(n + 1))
    for n in range(order + 1)
  ])
  numerator = numerator * jnp.exp(
    1j * phase * (root_rotation - 1)
  )
  return numerator, denominator


def polynomial_action(
  coefficients: Field,
  matvec: Callable[[Field], Field],
  field: Field,
) -> Field:
  """Applies an ascending-power operator polynomial using Horner's rule."""
  result = coefficients[-1] * field
  for index in range(coefficients.shape[0] - 2, -1, -1):
    result = matvec(result) + coefficients[index] * field
  return result


def pade_multiplier(
  normalized_eigenvalue: Field,
  numerator: Field,
  denominator: Field,
  rotation: float,
) -> Field:
  """Evaluates the scalar propagator for eigenvalues of H/k**2.

  Useful for inspecting phase error and evanescent attenuation before a run.
  This scalar diagnostic does not establish stability for nonnormal operators.
  """
  y = jnp.exp(-1j * rotation) * normalized_eigenvalue - 1
  return jnp.polyval(numerator[::-1], y) / jnp.polyval(denominator[::-1], y)
