"""Implicit linear propagation kernels using the shared operator pytree."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from .config import Field, SolverConfig
from .linear_solvers import solve_linear_system
from .rational import polynomial_action


def make_implicit_kernel(
  config: SolverConfig,
  laplacian: Callable[[Field, dict[str, Any]], Field],
  delta_n_fn: Callable[[Any, Any], Any] | None,
  apply_sigma: bool,
) -> Callable[[Field, Any, dict[str, Any], Any], Field]:
  """Builds CN for the paraxial equation or Padé for the full-index root.

  The medium is frozen at the step midpoint. Both kernels are linear in the
  field and reject Kerr at solver construction. The Padé kernel is initially
  restricted to periodic, real-index media without an absorber. Its rational
  error is independent of the second-order midpoint error for z variation.
  """
  def step(psi: Field, z: Any, operators: dict[str, Any], medium: Any) -> Field:
    h, k0, n0 = operators['dz'], operators['k0'], operators['n0']
    k = k0 * n0
    index = (0.0 if delta_n_fn is None
             else delta_n_fn(z + h / 2, medium))

    if config.stepper == 'crank_nicolson':
      def generator(field: Field) -> Field:
        out = (1j / (2 * k)) * laplacian(field, operators)
        out = out + 1j * k0 * index * field
        if apply_sigma:
          out = out - operators['sigma'] * field
        return out

      def lhs(field: Field) -> Field:
        return field - h / 2 * generator(field)

      rhs = psi + h / 2 * generator(psi)
    else:
      rotation = jnp.exp(-1j * config.pade_rotation)

      def y_operator(field: Field) -> Field:
        return rotation * (
          laplacian(field, operators) / k**2 + (1 + index / n0)**2 * field
        ) - field

      def lhs(field: Field) -> Field:
        return polynomial_action(operators['pade_den'], y_operator, field)

      rhs = polynomial_action(operators['pade_num'], y_operator, psi)

    def preconditioner(field: Field) -> Field:
      return jnp.fft.ifft2(
        jnp.fft.fft2(field) * operators['inverse_reference']
      )

    result = solve_linear_system(
      lhs, rhs, preconditioner,
      rtol=config.linear_rtol, maxiter=config.linear_maxiter,
      restart=config.linear_restart,
    )
    valid = result.converged
    if config.stepper == 'pade':
      # Complex-index extensions need separate branch/stability validation.
      # A traced medium cannot be rejected with a Python boolean under jit.
      numerator, denominator = operators['pade_num'], operators['pade_den']
      valid = (valid & jnp.all(jnp.imag(index) == 0)
               & jnp.all(jnp.isfinite(numerator))
               & jnp.all(jnp.isfinite(denominator))
               & jnp.any(numerator != 0))
    return jnp.where(valid, result.solution, jnp.nan + 1j * jnp.nan)

  return step
