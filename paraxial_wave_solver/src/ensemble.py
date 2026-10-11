"""Monte-Carlo propagation over ensembles of random media.

One realization of a random medium is one PRNG key: `delta_n_fn` receives the
key as its `medium` argument and derives the index perturbation from it, so a
batch of keys is a batch of independent media. Mapping the solver over that
batch is therefore the whole of ensemble propagation, and `jax.vmap` already
expresses it.

What `vmap` alone does not do is bound memory. A vectorized map over n
realizations materializes n transverse fields together with n copies of every
temporary inside the step, so an ensemble of 1000 on a 512 x 512 complex64
grid requires roughly 2.1 GB of live field before FFT scratch is counted.
`ensemble_solve` composes the two JAX mapping primitives instead: `vmap`
inside, over the realizations of one chunk, and `lax.map` outside, over the
chunks. Since `lax.map` is a scan, peak memory is set by the chunk rather
than by the ensemble, while the vectorization within a chunk is retained.

`ensemble_standard_error` answers the question that follows: how large the
ensemble has to be. It bootstraps the realization axis through the same
chunked map, so the convergence check costs no more memory than the
propagation did.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .config import Field
from .solvers import ParaxialWaveSolver


def ensemble_solve(
  solver: ParaxialWaveSolver,
  psi_0: Field,
  keys: jax.Array,
  *,
  chunk_size: int | None = None,
  return_history: bool = False,
  **solve_kwargs: Any,
) -> tuple[Field, Any]:
  """Propagates one initial field through an ensemble of random media.

  Each entry along the leading axis of `keys` indexes one realization, and is
  forwarded to `solver.delta_n_fn` as its `medium` argument. The initial field
  is shared across realizations; an ensemble of distinct inputs is a plain
  `jax.vmap` over `psi_0` and needs no driver.

  `chunk_size` is a memory control and not a physical parameter: the returned
  values are independent of it up to floating-point reassociation, which the
  implementation avoids by keeping each realization's propagation entirely
  within one chunk. Lowering it reduces peak memory and some of the
  vectorization benefit; leaving it unset vectorizes over the whole ensemble
  at once.

  When the ensemble size is not a multiple of `chunk_size` the key array is
  padded with repeats of its first entry and the corresponding results are
  discarded. This wastes at most `chunk_size - 1` propagations, and is
  preferred over a ragged final chunk, which would compile a second
  specialization of the propagation.

  Args:
    solver: A constructed solver. Its `delta_n_fn` must accept a PRNG key as
            its second argument; see `paraxial_wave_solver.src.utils.phase_screen`.
    psi_0: Initial complex field of shape (nx, ny), shared by all realizations.
    keys: PRNG keys of shape (n_realizations, ...), as returned by
          `jax.random.split`. Both raw uint32 and typed keys are accepted.
    chunk_size: Number of realizations propagated simultaneously. Defaults to
                the whole ensemble.
    return_history: Forwarded to `solve`. Defaults to False here, in contrast
                    to `solve` itself, because an ensemble of histories has
                    shape (n_realizations, nz // save_every, nx, ny) and is
                    rarely what is wanted. Pass an `observable_fn` to record
                    diagnostics over z instead of fields.
    **solve_kwargs: Forwarded verbatim to `solver.solve`, including
                    `observable_fn`, `save_every`, `z_0`, `checkpoint` and
                    `operators`.

  Returns:
    A tuple (psi_final, history) matching `solve`, with a leading axis of
    length n_realizations prepended to every array. history is None when
    return_history is False and no observable_fn is given.

  Raises:
    ValueError: If the ensemble is empty, if chunk_size is not positive, or
                if `donate_psi_0` is requested. Donation is incompatible with
                an ensemble: `psi_0` is shared by every realization, so
                handing its buffer to the first chunk would destroy the input
                the remaining chunks still need.
  """
  if keys.shape[0] == 0:
    raise ValueError("keys must hold at least one realization.")
  if solve_kwargs.get("donate_psi_0"):
    raise ValueError(
      "donate_psi_0 is incompatible with ensemble_solve: psi_0 is shared "
      "across realizations and would be destroyed by the first chunk."
    )

  if chunk_size is not None and int(chunk_size) < 1:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

  def one(key: jax.Array) -> tuple[Field, Any]:
    return solver.solve(
      psi_0, medium=key, return_history=return_history, **solve_kwargs
    )

  return _chunked_map(one, keys, chunk_size)


def ensemble_standard_error(
  samples: jax.Array,
  statistic_fn: Callable[[jax.Array], Any],
  key: jax.Array,
  *,
  n_resamples: int = 200,
  chunk_size: int | None = 1,
) -> Any:
  """Returns the bootstrap standard error of an ensemble statistic.

  Resamples the realization axis with replacement, re-evaluates
  `statistic_fn` on each resample, and returns the standard deviation of the
  results. This is the quantity that says whether an ensemble is large
  enough, and it differs sharply between statistics: on the same ensemble,
  beam wander and the scintillation index can disagree by more than an order
  of magnitude in relative precision, because the latter is a ratio of fourth
  to second intensity moments and inherits the heavy tail of a near-exponential
  intensity distribution.

  The estimate is itself noisy, at a relative precision of about
  1 / sqrt(2 * n_resamples), so the default of 200 resamples resolves the
  standard error to roughly 5%. It also inherits any bias in the statistic:
  a quantity biased at small ensemble size has a small standard error and a
  wrong central value, which the standard error cannot reveal.

  Args:
    samples: Array whose leading axis is the realization axis. Usually the
      fields returned by `ensemble_solve`, but any per-realization array
      works.
    statistic_fn: Callable mapping a resampled `samples` array to a scalar or
      to a pytree of arrays, for example
      `lambda f: ensemble_statistics(f, sim_config)`.
    key: PRNG key for the resampling.
    n_resamples: Number of bootstrap resamples.
    chunk_size: Resamples evaluated simultaneously. Defaults to 1, because a
      resample gathers the whole ensemble and vectorizing over resamples
      multiplies that: 200 resamples of 512 realizations on a 96 x 96 grid is
      7.5 GB vectorized against 38 MB one at a time. Raise it to trade memory
      for speed.

  Returns:
    The standard error, with the same structure as `statistic_fn`'s output.
    Array-valued entries get their standard error elementwise.
  """
  n_realizations = samples.shape[0]

  def one(resample_key: jax.Array) -> Any:
    indices = jax.random.randint(
      resample_key, (n_realizations,), 0, n_realizations
    )
    return statistic_fn(samples[indices])

  resampled = _chunked_map(
    one, jax.random.split(key, n_resamples), chunk_size
  )
  return jax.tree_util.tree_map(lambda a: jnp.std(a, axis=0), resampled)


def _chunked_map(
  fn: Callable[[jax.Array], Any],
  xs: jax.Array,
  chunk_size: int | None,
) -> Any:
  """Maps `fn` over the leading axis of `xs`, vectorized within chunks.

  `vmap` inside a `lax.map`: the vectorization is retained within a chunk
  while peak memory is set by the chunk rather than by the full axis. The
  axis is padded with repeats of its first entry so that every chunk has the
  same shape and `fn` compiles once, and the padded results are discarded.
  """
  n = xs.shape[0]
  chunk = min(n if chunk_size is None else int(chunk_size), n)

  padding = (-n) % chunk
  if padding:
    xs = jnp.concatenate([xs, xs[jnp.zeros(padding, dtype=int)]])
  chunked = xs.reshape(-1, chunk, *xs.shape[1:])

  stacked = lax.map(jax.vmap(fn), chunked)

  def flatten(leaf: Field) -> Field:
    # (n_chunks, chunk, ...) -> (n_chunks * chunk, ...), then drop the padding.
    return leaf.reshape(-1, *leaf.shape[2:])[:n]

  return jax.tree_util.tree_map(flatten, stacked)
