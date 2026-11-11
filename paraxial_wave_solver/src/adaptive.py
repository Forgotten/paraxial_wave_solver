"""Host-controlled step doubling with JIT-compiled propagation trials.

The controller is intentionally outside JAX transformations. Fixed-step solve
remains the differentiable API; accepted steps are returned for inspection.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .config import Field


@dataclass(frozen=True, slots=True)
class AdaptiveConfig:
  """Local weighted-RMS error tolerances and positive step-size bounds.

  max_steps counts attempted steps, including rejected trials. A requested
  output plane can require a last step smaller than min_step. Tolerances
  control the local estimator, not a guaranteed global error bound.
  """
  rtol: float = 1e-5
  atol: float = 1e-8
  min_step: float = 1e-10
  max_step: float = math.inf
  max_steps: int = 100000
  safety: float = .9

  def __post_init__(self):
    for name in ('rtol', 'atol', 'min_step'):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f'{name} must be positive and finite.')
    if math.isnan(self.max_step) or self.max_step < self.min_step:
      raise ValueError('max_step must be at least min_step.')
    if not isinstance(self.max_steps, int) or self.max_steps < 1:
      raise ValueError('max_steps must be a positive integer.')
    if not 0 < self.safety < 1:
      raise ValueError('safety must lie between zero and one.')


@dataclass(frozen=True)
class AdaptiveResult:
  """Final field, requested planes (including final), and controller statistics.

  history is shaped (len(z), nx, ny); step_sizes records accepted macrosteps,
  each computed as two half steps. Rejected trials never enter the history.
  """
  field: Field
  z: np.ndarray
  history: Field
  step_sizes: np.ndarray
  accepted: int
  rejected: int


@partial(jax.jit, static_argnames=('solver',))
def _trial(solver, field, z, h, medium, params):
  coarse_ops = solver.build_operators({**params, 'dz': h})
  half_ops = solver.build_operators({**params, 'dz': h/2})
  coarse = solver._step_fn(field, z, coarse_ops, medium)
  half = solver._step_fn(field, z, half_ops, medium)
  fine = solver._step_fn(half, z+h/2, half_ops, medium)
  return coarse, fine


def adaptive_solve(
  solver, psi_0: Field, z_end: float | None = None, *, z_0: float = 0.,
  medium: Any = None, config: AdaptiveConfig | None = None,
  save_z: Any = None, params: dict[str, Any] | None = None,
) -> AdaptiveResult:
  """Propagates to z_end using local step-doubling error control.

  Supports RK4, ETDRK4, CN, and unfiltered split-step propagation. Output
  planes are hit exactly (no interpolation); the final plane is always saved.
  The initial trial size is SimulationConfig.dz, clipped to configured bounds.
  params accepts k0/n0 overrides; dz is owned by the controller. Neither this
  host loop nor its accept/reject decisions support jit, vmap, or gradients.
  """
  cfg = config or AdaptiveConfig()
  sc = solver.solver_config
  if sc.stepper not in ('rk4', 'etdrk4', 'crank_nicolson', 'split_step'):
    raise ValueError('Adaptive stepping requires RK4, ETDRK4, CN, or split-step.')
  if sc.dealias:
    raise ValueError('Adaptive step doubling does not support dealias filtering.')
  order = 4 if sc.stepper in ('rk4', 'etdrk4') else 2
  if sc.stepper == 'split_step' and not solver._has_pml:
    order = sc.splitting_order
  params = dict(params or {})
  if set(params) - {'k0', 'n0'}:
    raise ValueError('Adaptive params accepts only k0 and n0.')
  z = float(z_0)
  end = z + solver.sim_config.lz if z_end is None else float(z_end)
  if not math.isfinite(z) or not math.isfinite(end) or end <= z:
    raise ValueError('Require finite z_end > z_0.')
  planes = np.asarray([] if save_z is None else save_z, dtype=float)
  if (planes.ndim != 1 or not np.all(np.isfinite(planes))
      or np.any(np.diff(planes) <= 0) or np.any(planes < z)
      or np.any(planes > end)):
    raise ValueError('save_z must be increasing finite planes within the interval.')
  if not len(planes) or planes[-1] != end:
    planes = np.append(planes, end)
  field = jnp.asarray(psi_0)
  if field.shape != (solver.sim_config.nx, solver.sim_config.ny):
    raise ValueError('psi_0 must match the transverse grid.')
  if not bool(jnp.all(jnp.isfinite(field))):
    raise ValueError('psi_0 must be finite.')
  field = field.astype(jnp.result_type(field, 1j))
  h = min(cfg.max_step, max(cfg.min_step, solver.sim_config.dz))
  steps, saved = [], []
  rejected = 0
  for target in planes:
    while z < target:
      if len(steps) + rejected >= cfg.max_steps:
        raise RuntimeError('Adaptive maximum attempted step count exceeded.')
      remaining = float(target-z)
      h = min(h, remaining, cfg.max_step)
      if z+h == z:
        raise RuntimeError('Adaptive step cannot advance the propagation coordinate.')
      coarse, fine = _trial(solver, field, z, h, medium, params)
      scale = cfg.atol + cfg.rtol*jnp.maximum(jnp.abs(field), jnp.abs(fine))
      error = float(jnp.sqrt(jnp.mean(
        (jnp.abs(fine-coarse) / ((2**order-1)*scale))**2)))
      accepted = math.isfinite(error) and error <= 1
      if accepted:
        field = fine
        z = float(target) if h == remaining else z+h
        steps.append(h)
      else:
        rejected += 1
        if h <= cfg.min_step:
          raise RuntimeError('Adaptive tolerance cannot be met at min_step.')
      factor = (5. if error == 0 else
                max(.2, min(5., cfg.safety*error**(-1/(order+1))))
                if math.isfinite(error) else .2)
      if not accepted:
        factor = min(1., factor)
      h = min(cfg.max_step, max(cfg.min_step, h*factor))
    saved.append(field)
  return AdaptiveResult(field, planes, jnp.stack(saved), np.asarray(steps),
                        len(steps), rejected)
