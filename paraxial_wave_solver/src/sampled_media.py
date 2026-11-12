"""Differentiable sampled-medium interpolation and explicit interface marches."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .adaptive import AdaptiveResult
from .config import Field


class SampledMedium(NamedTuple):
  """JAX pytree of physical coordinates and ascending local polynomial powers."""
  z: Field
  coefficients: Field


def make_sampled_medium(z, values, kind='cubic') -> SampledMedium:
  """Builds linear or not-a-knot cubic interpolation along the leading axis.

  Coordinates must be concrete, finite, and strictly increasing. Values have
  shape (len(z), ...), including scalar, transverse, or complex index samples.
  Construction and evaluation differentiate with respect to values, not the
  knot schedule. Cubic needs at least four knots and is C2; linear is C0.
  No positivity or monotonicity constraint is imposed on cubic interpolation.
  """
  if kind not in ('linear', 'cubic'):
    raise ValueError("kind must be 'linear' or 'cubic'.")
  coordinates = np.asarray(z)
  minimum = 4 if kind == 'cubic' else 2
  if (coordinates.ndim != 1 or len(coordinates) < minimum
      or np.iscomplexobj(coordinates) or not np.all(np.isfinite(coordinates))
      or np.any(np.diff(coordinates) <= 0)):
    raise ValueError(f'z requires at least {minimum} increasing finite real knots.')
  knots = jnp.asarray(coordinates, dtype=jnp.result_type(coordinates, 1.))
  values = jnp.asarray(values)
  if values.ndim < 1 or values.shape[0] != len(coordinates):
    raise ValueError('The leading values dimension must equal the knot count.')
  widths = jnp.diff(knots)
  h = widths.reshape((-1,)+(1,)*(values.ndim-1))
  secants = jnp.diff(values, axis=0)/h
  if kind == 'linear':
    second = jnp.zeros_like(values)
  else:
    n = len(coordinates)
    system = jnp.zeros((n, n), dtype=knots.dtype)
    i = jnp.arange(1, n-1)
    system = system.at[i, i-1].set(widths[:-1])
    system = system.at[i, i].set(2*(widths[:-1]+widths[1:]))
    system = system.at[i, i+1].set(widths[1:])
    system = system.at[0, :3].set(jnp.array(
      [-widths[1], widths[0]+widths[1], -widths[0]]))
    system = system.at[-1, -3:].set(jnp.array(
      [widths[-1], -widths[-2]-widths[-1], widths[-2]]))
    rhs = jnp.zeros_like(values, dtype=jnp.result_type(values, knots))
    rhs = rhs.at[1:-1].set(6*jnp.diff(secants, axis=0))
    second = jnp.linalg.solve(system, rhs.reshape(n, -1)).reshape(values.shape)
  coefficients = jnp.stack([
    values[:-1], secants-h*(2*second[:-1]+second[1:])/6,
    second[:-1]/2, (second[1:]-second[:-1])/(6*h),
  ], axis=1)
  return SampledMedium(knots, coefficients)


def sample_medium(z, medium: SampledMedium):
  """Evaluates at one scalar coordinate; use vmap for multiple coordinates.

  Domain endpoints are included. Out-of-domain samples return NaNs instead of
  silently extrapolating. Use as the solver's delta_n_fn directly.
  """
  i = jnp.clip(jnp.searchsorted(medium.z, z, side='right')-1,
               0, medium.z.size-2)
  t = z-medium.z[i]
  a, b, c, d = medium.coefficients[i]
  value = a+t*(b+t*(c+t*d))
  return jnp.where((z >= medium.z[0]) & (z <= medium.z[-1]), value, jnp.nan)


def solve_segments(solver, psi_0, boundaries, media, *, config=None, params=None):
  """Adaptive march with a separate medium on each closed range interval.

  The callback receives media[i] throughout [boundaries[i], boundaries[i+1]],
  including endpoint stages. This avoids ambiguous one-sided values at jumps.
  The field is continuous across interfaces: no backward reflection, Fresnel
  coefficient, or change of reference carrier is modeled. Returns all boundary
  fields, including the initial field, and combined controller statistics.
  """
  points = np.asarray(boundaries, dtype=float)
  if (points.ndim != 1 or points.size < 2 or not np.all(np.isfinite(points))
      or np.any(np.diff(points) <= 0) or len(media) != len(points)-1):
    raise ValueError('Require increasing boundaries and one medium per interval.')
  if solver.solver_config.splitting_order == 4:
    raise ValueError('Segmented marching excludes negative Yoshida substeps.')
  field, fields, steps, accepted, rejected = psi_0, [jnp.asarray(psi_0)], [], 0, 0
  for left, right, medium in zip(points[:-1], points[1:], media, strict=True):
    result = solver.solve_adaptive(field, float(right), z_0=float(left),
                                   medium=medium, config=config, params=params)
    field = result.field
    fields.append(field)
    steps.append(result.step_sizes)
    accepted += result.accepted
    rejected += result.rejected
  return AdaptiveResult(field, points, jnp.stack(fields), np.concatenate(steps),
                        accepted, rejected)
