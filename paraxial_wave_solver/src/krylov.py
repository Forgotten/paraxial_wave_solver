"""Hermitian matrix-function actions with fully reorthogonalized Lanczos bases.

Only operator-vector products and small projected eigendecompositions are
needed. Nested subspace differences provide a refinement indicator, not a
certified bound, particularly for spectra near the square-root branch point.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import Field


class KrylovResult(NamedTuple):
  """Candidate action, relative nested-space difference, and acceptance flag.

  An invariant Krylov subspace is accepted up to roundoff. Otherwise the
  indicator compares dimensions m and max(1, m//2). A false flag does not
  discard the candidate, allowing dimension-refinement diagnostics.
  """
  solution: Field
  relative_difference: Field
  converged: Field


def hermitian_function_action(
  matvec: Callable[[Field], Field], field: Field,
  function: Callable[[Field], Field], *, krylov_dim: int = 32,
  rtol: float = 1e-6,
) -> KrylovResult:
  """Approximates f(H) field for Hermitian H; the caller owns that assumption.

  Two orthogonalization passes suppress loss of orthogonality. Storage is
  O(krylov_dim * field.size). The projected eigenproblem costs O(krylov_dim**3).
  The function and operator must be JAX-compatible. JIT and vmap are supported;
  derivatives at breakdown, repeated Ritz values, and zero eigenvalues are
  not supported. Use the differentiable Padé solver for inverse problems.
  """
  if not isinstance(krylov_dim, int) or krylov_dim < 1:
    raise ValueError('krylov_dim must be a positive integer.')
  if not 0 < rtol < 1:
    raise ValueError('rtol must lie between zero and one.')
  field = jnp.asarray(field).astype(jnp.result_type(field, 1j))
  if not field.size:
    raise ValueError('field must be nonempty.')
  m = min(krylov_dim, field.size)
  beta = jnp.linalg.norm(field)
  dtype = field.real.dtype
  eps, tiny = jnp.finfo(dtype).eps, jnp.finfo(dtype).tiny
  basis = jnp.zeros((m+1, field.size), dtype=field.dtype)
  basis = basis.at[0].set(field.ravel()/jnp.where(beta > 0, beta, 1.))
  hessenberg = jnp.zeros((m+1, m), dtype=field.dtype)

  def column(j, state):
    q, h = state
    value = matvec(q[j].reshape(field.shape)).ravel()
    original_norm = jnp.linalg.norm(value)
    coefficients = q.conj() @ value
    value = value - coefficients @ q
    correction = q.conj() @ value
    value = value - correction @ q
    residual = jnp.linalg.norm(value)
    active = residual > 20*eps*jnp.maximum(original_norm, tiny)
    h = h.at[:, j].set(coefficients+correction)
    h = h.at[j+1, j].set(jnp.where(active, residual, 0.))
    q = q.at[j+1].set(jnp.where(active, value/jnp.where(active, residual, 1.), 0.))
    return q, h

  basis, hessenberg = jax.lax.fori_loop(0, m, column, (basis, hessenberg))
  projected = hessenberg[:m]
  hermitian = (jnp.linalg.norm(projected-projected.conj().T)
               <= 100*eps*jnp.maximum(jnp.linalg.norm(projected), tiny))
  projected = (projected+projected.conj().T)/2

  def evaluate(size):
    values, vectors = jnp.linalg.eigh(projected[:size, :size])
    coefficients = vectors @ (function(values)*vectors[0].conj())
    return beta*(coefficients @ basis[:size]).reshape(field.shape)

  candidate = evaluate(m)
  coarse = evaluate(max(1, m//2))
  difference = jnp.linalg.norm(candidate-coarse)/jnp.maximum(
    jnp.linalg.norm(candidate), tiny)
  invariant = hessenberg[m, m-1] == 0
  # A one-vector comparison carries no refinement information unless invariant.
  converged = (invariant | ((m > 1) & (difference <= rtol))) & hermitian
  finite = jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(difference)
  return KrylovResult(candidate, difference, converged & finite)


def square_root_action(
  matvec: Callable[[Field], Field], field: Field, *, krylov_dim: int = 32,
  rtol: float = 1e-6,
) -> KrylovResult:
  """Principal sqrt(H) field; negative eigenvalues take the positive-imaginary root.

  For sqrt(-Laplacian), supply the negative discrete Laplacian as matvec.
  This action is distinct from the carrier-removed Helmholtz propagator.
  """
  return hermitian_function_action(matvec, field, lambda x: jnp.sqrt(x+0j),
                                   krylov_dim=krylov_dim, rtol=rtol)


def make_lanczos_kernel(config, laplacian, delta_n_fn):
  """Midpoint-frozen full-index one-way root, with outgoing evanescent decay."""
  def step(psi, z, operators, medium):
    h, k0, n0 = operators['dz'], operators['k0'], operators['n0']
    index = 0. if delta_n_fn is None else delta_n_fn(z+h/2, medium)

    def helmholtz(field):
      return laplacian(field, operators)+k0**2*(n0+index)**2*field

    def multiplier(values):
      return jnp.exp(1j*h*(jnp.sqrt(values+0j)-k0*n0))

    result = hermitian_function_action(
      helmholtz, psi, multiplier, krylov_dim=config.krylov_dim,
      rtol=config.krylov_rtol,
    )
    valid = result.converged & jnp.all(jnp.imag(index) == 0)
    return jnp.where(valid, result.solution, jnp.nan+1j*jnp.nan)

  return step
