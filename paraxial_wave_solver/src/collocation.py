"""Linear Gauss–Legendre and Radau IIA implicit Runge–Kutta kernels."""

import jax
import jax.numpy as jnp

from .linear_solvers import solve_linear_system


def collocation_tableau(stepper):
  """Returns A, b, c for Gauss order 4 or stiffly accurate Radau IIA order 5."""
  if stepper == 'gauss4':
    q = 3**.5/6
    a = [[.25, .25-q], [.25+q, .25]]
    b, c = [.5, .5], [.5-q, .5+q]
  elif stepper == 'radau5':
    q = 6**.5
    a = [[(88-7*q)/360, (296-169*q)/1800, (-2+3*q)/225],
         [(296+169*q)/1800, (88+7*q)/360, (-2-3*q)/225],
         [(16-q)/36, (16+q)/36, 1/9]]
    b, c = a[-1], [(4-q)/10, (4+q)/10, 1.]
  else:
    raise ValueError("Expected stepper='gauss4' or 'radau5'.")
  return jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)


def make_collocation_kernel(config, laplacian, delta_n_fn, apply_sigma):
  """Solves all stage fields together; samples the medium at each stage node.

  Only linear field equations are supported. The shared residual-checked GMRES
  supplies implicit differentiation, including its transpose solve.
  """
  def step(psi, z, operators, medium):
    a, b, c = collocation_tableau(config.stepper)
    h, k0, n0 = operators['dz'], operators['k0'], operators['n0']
    samples = jnp.stack([
      jnp.broadcast_to(0. if delta_n_fn is None else delta_n_fn(z+h*t, medium),
                       psi.shape) for t in c
    ])

    def generator(field, index):
      value = 1j*laplacian(field, operators)/(2*k0*n0)+1j*k0*index*field
      if apply_sigma:
        value = value-operators['sigma']*field
      return value

    def apply(stages):
      return jax.vmap(generator)(stages, samples)

    def lhs(stages):
      return stages-h*jnp.einsum('ij,jxy->ixy', a, apply(stages))

    def preconditioner(stages):
      spectrum = jnp.moveaxis(jnp.fft.fft2(stages), 0, -1)
      filtered = jnp.einsum('xyij,xyj->xyi', operators['irk_inverse'], spectrum)
      return jnp.fft.ifft2(jnp.moveaxis(filtered, -1, 0))

    result = solve_linear_system(
      lhs, jnp.broadcast_to(psi, (len(c),)+psi.shape), preconditioner,
      rtol=config.linear_rtol, maxiter=config.linear_maxiter,
      restart=config.linear_restart,
    )
    if config.stepper == 'radau5':
      final = result.solution[-1]
    else:
      final = psi+h*jnp.einsum('i,ixy->xy', b, apply(result.solution))
    return jnp.where(result.converged, final, jnp.nan+1j*jnp.nan)

  return step
