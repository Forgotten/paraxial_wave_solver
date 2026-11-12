"""Fourth-order commutator-free Magnus propagation for linear paraxial media.

Two Gauss nodes and two exponential actions retain the range ordering of the
operator. Matrix-free scaled Taylor actions avoid forming a dense propagator.
"""

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .config import Field


class TaylorPlan(NamedTuple):
  """Chosen degree, repeated actions, matvec count, and capacity validity."""
  degree: Field
  substeps: Field
  operator_evaluations: Field
  valid: Field


_DEGREES = (8, 12, 16, 20, 24)


def _taylor_threshold(degree, log_budget):
  # exp(theta)*theta**(m+1)/(m+1)! bounds the absolute Taylor tail
  # for an operator of induced norm at most theta. Use logarithms.
  low, high = 0., 24.
  for _ in range(60):
    mid = (low+high)/2
    remainder = mid+(degree+1)*math.log(mid)-math.lgamma(degree+2)
    if remainder <= log_budget:
      low = mid
    else:
      high = mid
  return low


def exponential_plan(norm_bound, max_substeps=256, *, rtol=1e-12,
                     backend='adaptive_taylor') -> TaylorPlan:
  """Selects the least matvec cost among degrees 8, 12, 16, 20, and 24.

  Each scaled action receives an absolute operator-tail budget of
  rtol/max_substeps. This controls truncation locally, not the global relative
  error of a nonnormal exponential. The fixed backend retains 24 terms and
  norm <= 1. Integer choices are held constant under differentiation.
  """
  if not isinstance(max_substeps, int) or max_substeps < 1:
    raise ValueError('max_substeps must be a positive integer.')
  if not 0 < rtol < 1:
    raise ValueError('rtol must lie between zero and one.')
  if backend not in ('adaptive_taylor', 'fixed_taylor'):
    raise ValueError('Unknown exponential backend.')
  valid = jnp.isfinite(norm_bound) & (norm_bound >= 0)
  safe = jnp.where(valid, norm_bound, 1.)
  if backend == 'fixed_taylor':
    degree = jnp.asarray(24)
    count = jnp.maximum(1., jnp.ceil(safe))
  else:
    thresholds = jnp.asarray([
      _taylor_threshold(m, math.log(rtol)-math.log(max_substeps)) for m in _DEGREES
    ])
    counts = jnp.maximum(1., jnp.ceil(safe/thresholds))
    costs = jnp.where(counts <= max_substeps, counts*jnp.array(_DEGREES), jnp.inf)
    selected = jnp.argmin(costs)
    degree = jnp.array(_DEGREES)[selected]
    count = counts[selected]
  valid = valid & (count <= max_substeps)
  count = jnp.where(valid, count, 1.).astype(jnp.int32)
  count = jnp.where(norm_bound == 0, 0, count)
  return TaylorPlan(degree, count, degree*count, valid)


def exponential_action(
  matvec: Callable[[Field], Field], field: Field, norm_bound: Any,
  max_substeps: int = 256, *, rtol: float = 1e-12,
  backend: str = 'adaptive_taylor',
) -> Field:
  """Applies exp(B) by repeated scaled polynomial actions without dense matrices.

  norm_bound must bound the induced 2-norm of B. The selected degree and
  scaling minimize matvec count among the available candidates. A fixed loop
  capacity preserves reverse-mode differentiation. Invalid or over-capacity
  plans return NaNs. Roundoff and nonnormal amplification remain limitations;
  rtol controls the per-action Taylor budget, not guaranteed final error.
  """
  plan = exponential_plan(norm_bound, max_substeps, rtol=rtol, backend=backend)

  def branch(degree):
    def advance(value):
      def term(index, state):
        increment, total = state
        increment = matvec(increment)/(jnp.maximum(1, plan.substeps)*index)
        return increment, total+increment
      return jax.lax.fori_loop(1, degree+1, term, (value, value))[1]
    return advance

  branches = tuple(branch(m) for m in _DEGREES)
  selected = (plan.degree-8)//4

  def substep(index, value):
    return jax.lax.cond(index < plan.substeps,
                        lambda u: jax.lax.switch(selected, branches, u),
                        lambda u: u, value)

  result = jax.lax.fori_loop(0, max_substeps, substep, field)
  return jnp.where(plan.valid, result, jnp.nan+1j*jnp.nan)


def make_magnus_kernel(
  laplacian: Callable[[Field, dict[str, Any]], Field],
  delta_n_fn: Callable[[Any, Any], Any] | None,
  apply_sigma: bool, max_substeps: int, *, rtol: float = 1e-12,
  backend: str = 'adaptive_taylor',
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
      return exponential_action(generator, field, bound, max_substeps,
                                 rtol=rtol, backend=backend)

    psi = apply(psi, b*first+a*second)
    return apply(psi, a*first+b*second)

  return step
