"""Fourth-order commutator-free Magnus propagation for linear paraxial media.

Two Gauss nodes and two exponential actions retain the range ordering of the
operator. Matrix-free scaled Taylor actions avoid forming a dense propagator.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from .config import Field


def exponential_action(
  matvec: Callable[[Field], Field], field: Field, norm_bound: Any,
  max_substeps: int = 256,
) -> Field:
  """Applies exp(B) with 24-term Taylor actions at ||B/s|| <= 1.

  norm_bound must bound the induced 2-norm of B. Splitting into s repeated
  actions controls truncation error without dense matrices or eigensystems.
  The fixed loop bound preserves reverse-mode differentiation; the selected
  integer scaling is held constant by autodiff. Returns NaNs if the supplied
  bound is invalid or needs more than max_substeps. Roundoff still limits the
  result, and large nonnormal amplification can degrade relative accuracy.
  """
  if not isinstance(max_substeps, int) or max_substeps < 1:
    raise ValueError('max_substeps must be a positive integer.')
  valid = (jnp.isfinite(norm_bound) & (norm_bound >= 0)
           & (norm_bound <= max_substeps))
  safe_bound = jnp.where(valid, norm_bound, 1.)
  count = jnp.maximum(1, jnp.ceil(safe_bound).astype(jnp.int32))

  def advance(value):
    def term(index, state):
      increment, total = state
      increment = matvec(increment) / (count*index)
      return increment, total+increment
    return jax.lax.fori_loop(1, 25, term, (value, value))[1]

  def substep(index, value):
    return jax.lax.cond(index < count, advance, lambda x: x, value)

  result = jax.lax.fori_loop(0, max_substeps, substep, field)
  return jnp.where(valid, result, jnp.nan+1j*jnp.nan)


def make_magnus_kernel(
  laplacian: Callable[[Field, dict[str, Any]], Field],
  delta_n_fn: Callable[[Any, Any], Any] | None,
  apply_sigma: bool, max_substeps: int,
) -> Callable[[Field, Any, dict[str, Any], Any], Field]:
  """CF4 with Gauss nodes 1/2 ± sqrt(3)/6; no frozen-midpoint reduction."""
  root = 3**.5
  a, b = (3-2*root)/12, (3+2*root)/12

  def step(psi, z, operators, medium):
    h, k0, n0 = operators['dz'], operators['k0'], operators['n0']
    first = (0. if delta_n_fn is None
             else delta_n_fn(z+h*(.5-root/6), medium))
    second = (0. if delta_n_fn is None
              else delta_n_fn(z+h*(.5+root/6), medium))
    sigma = operators['sigma'] if apply_sigma else 0.

    def apply(field, index):
      def generator(value):
        return h*(1j*laplacian(value, operators)/(4*k0*n0)
                  + (1j*k0*index-sigma/2)*value)
      bound = jnp.abs(h)*(operators['laplacian_bound']/(4*jnp.abs(k0*n0))
                          + jnp.max(jnp.abs(1j*k0*index-sigma/2)))
      return exponential_action(generator, field, bound, max_substeps)

    psi = apply(psi, b*first+a*second)
    return apply(psi, a*first+b*second)

  return step
