"""Matrix-free complex linear solves shared by implicit propagation kernels."""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from .config import Field


class LinearSolveResult(NamedTuple):
  """Solution, relative residual, and convergence flag (all JAX arrays)."""

  solution: Field
  relative_residual: Field
  converged: Field


def solve_linear_system(
  matvec: Callable[[Field], Field],
  rhs: Field,
  preconditioner: Callable[[Field], Field] | None = None,
  *,
  rtol: float = 1e-6,
  maxiter: int = 50,
  restart: int = 20,
) -> LinearSolveResult:
  """Solves a complex system and checks its unpreconditioned residual.

  JAX's GMRES differentiates implicitly, using a transpose solve in reverse
  mode. Both solves must converge for accurate derivatives. The returned
  residual measures the forward solve; directional derivative tests remain
  necessary when changing a model or its tolerances.

  Args:
    matvec: Linear operator, preserving the shape of rhs.
    rhs: Right-hand side, usually a transverse field.
    preconditioner: Approximate inverse, or None.
    rtol: Relative residual tolerance, positive.
    maxiter: Maximum number of restart cycles, positive.
    restart: Krylov dimension per cycle, positive.

  Returns:
    A LinearSolveResult. On failure the candidate is returned for inspection;
    propagation kernels replace failed steps with NaNs rather than accepting
    an unconverged field. This behavior also works under jit and vmap.
  """
  if not 0 < rtol < 1:
    raise ValueError("rtol must lie strictly between zero and one.")
  for name, value in (('maxiter', maxiter), ('restart', restart)):
    if not isinstance(value, int) or value < 1:
      raise ValueError(f"{name} must be a positive integer.")

  solution, _ = gmres(
    matvec, rhs, M=preconditioner, tol=rtol, atol=0.0,
    maxiter=maxiter, restart=min(restart, rhs.size), solve_method='incremental',
  )
  norm = jnp.linalg.norm(rhs)
  residual = jnp.linalg.norm(matvec(solution) - rhs)
  relative = residual / jnp.where(norm > 0, norm, 1.0)
  return LinearSolveResult(
    solution, relative, jnp.isfinite(relative) & (relative <= rtol),
  )
