"""Tests for the chunked ensemble driver.

The pair that carries the most weight is `test_ensemble_matches_serial_loop`
and `test_chunk_size_does_not_change_results`: between them they pin that the
driver is a pure reorganization of work, so that chunking cannot be mistaken
for a physical parameter.
"""

import jax
import jax.numpy as jnp
import pytest

from paraxial_wave_solver.src.config import (
  PMLConfig,
  SimulationConfig,
  SolverConfig,
)
from paraxial_wave_solver.src.ensemble import ensemble_solve
from paraxial_wave_solver.src.solvers import ParaxialWaveSolver
from paraxial_wave_solver.src.utils import gaussian_beam, phase_screen

CORRELATION_LENGTH = 2e-3
STRENGTH = 2e-6


def _turbulence_solver(nx=32, nz=8):
  """Returns a small solver whose medium argument is a PRNG key."""
  sim_config = SimulationConfig(
    nx=nx, ny=nx, dx=4e-4, dy=4e-4, dz=2.5e-2, nz=nz,
    wavelength=632.8e-9, n0=1.33,
  )
  solver_config = SolverConfig(method='spectral', stepper='split_step')
  pml_config = PMLConfig(width_x=4, width_y=4, strength=2.0)

  def delta_n_fn(z, key):
    step = jnp.floor(z / sim_config.dz).astype(int)
    return phase_screen(
      jax.random.fold_in(key, step), sim_config, CORRELATION_LENGTH, STRENGTH
    )

  solver = ParaxialWaveSolver(sim_config, solver_config, pml_config, delta_n_fn)
  return solver, sim_config, gaussian_beam(sim_config, w0=2e-3)


# --------------------------------------------------------------------------
# The driver is a pure reorganization of work
# --------------------------------------------------------------------------

def test_ensemble_matches_serial_loop(x64):
  """The driver reproduces a Python loop over the same keys, realization by
  realization."""
  solver, _, psi_0 = _turbulence_solver()
  keys = jax.random.split(jax.random.PRNGKey(0), 6)

  batched, _ = ensemble_solve(solver, psi_0, keys, chunk_size=3)
  serial = jnp.stack(
    [solver.solve(psi_0, medium=k, return_history=False)[0] for k in keys]
  )

  assert batched.shape == serial.shape
  assert jnp.allclose(batched, serial, atol=1e-12)


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 4, 8, 16])
def test_chunk_size_does_not_change_results(chunk_size, x64):
  """chunk_size is a memory control, so every value gives the same fields."""
  solver, _, psi_0 = _turbulence_solver()
  keys = jax.random.split(jax.random.PRNGKey(1), 8)

  reference, _ = ensemble_solve(solver, psi_0, keys)
  chunked, _ = ensemble_solve(solver, psi_0, keys, chunk_size=chunk_size)

  assert chunked.shape == (8, 32, 32)
  assert jnp.allclose(reference, chunked, atol=1e-12)


@pytest.mark.parametrize(
  'n_realizations,chunk_size', [(10, 4), (7, 3), (5, 8), (1, 1), (9, 2)]
)
def test_ragged_ensemble_is_trimmed(n_realizations, chunk_size, x64):
  """A ragged final chunk is padded internally, never returned."""
  solver, _, psi_0 = _turbulence_solver()
  keys = jax.random.split(jax.random.PRNGKey(2), n_realizations)

  fields, _ = ensemble_solve(solver, psi_0, keys, chunk_size=chunk_size)
  serial = jnp.stack(
    [solver.solve(psi_0, medium=k, return_history=False)[0] for k in keys]
  )

  assert fields.shape[0] == n_realizations
  assert jnp.allclose(fields, serial, atol=1e-12)


def test_distinct_keys_give_distinct_fields():
  """Each realization sees its own medium, not a shared one."""
  solver, _, psi_0 = _turbulence_solver()
  keys = jax.random.split(jax.random.PRNGKey(3), 4)

  fields, _ = ensemble_solve(solver, psi_0, keys, chunk_size=2)

  for i in range(4):
    for j in range(i + 1, 4):
      assert not jnp.allclose(fields[i], fields[j], atol=1e-8)


def test_typed_prng_keys_are_accepted():
  """Both raw uint32 and typed PRNG keys index realizations."""
  solver, _, psi_0 = _turbulence_solver()
  typed = jax.random.split(jax.random.key(4), 4)

  fields, _ = ensemble_solve(solver, psi_0, typed, chunk_size=2)

  assert fields.shape == (4, 32, 32)
  assert bool(jnp.all(jnp.isfinite(fields)))


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

def test_ensemble_peak_memory_is_flat_in_n():
  """At fixed chunk_size, compiled scratch does not grow with the ensemble.

  This is the property that distinguishes `lax.map` over chunks from a plain
  `vmap` over the whole ensemble. Measured through the compiled module's own
  accounting rather than process RSS, which is a high-water mark and reports
  a growing loop as flat.
  """
  solver, _, psi_0 = _turbulence_solver(nx=64, nz=8)

  def temp_bytes(n_realizations):
    keys = jax.random.split(jax.random.PRNGKey(5), n_realizations)
    fn = jax.jit(
      lambda k: ensemble_solve(solver, psi_0, k, chunk_size=2)[0]
    )
    compiled = fn.lower(keys).compile()
    return compiled.memory_analysis().temp_size_in_bytes

  small = temp_bytes(4)
  large = temp_bytes(32)

  # An 8x ensemble at the same chunk size. A plain vmap would grow with it.
  assert large < 2.0 * small, f"{small} -> {large} bytes over an 8x ensemble"


# --------------------------------------------------------------------------
# Composition with the rest of the package
# --------------------------------------------------------------------------

def test_ensemble_is_jittable_and_differentiable(x64):
  """grad flows through the driver, with a value fixed by linearity.

  The solver is linear in psi_0 and the objective is quadratic in it, so
  scaling the input by `a` scales the objective by a^2 exactly. The gradient
  at a is therefore 2 * a * objective(1), independently of the medium.
  """
  solver, _, psi_0 = _turbulence_solver()
  keys = jax.random.split(jax.random.PRNGKey(6), 4)

  @jax.jit
  def objective(amplitude):
    fields, _ = ensemble_solve(
      solver, amplitude * psi_0, keys, chunk_size=2
    )
    return jnp.mean(jnp.abs(fields)**2)

  gradient = jax.grad(objective)(2.0)
  expected = 2 * 2.0 * objective(1.0)

  assert jnp.isfinite(gradient)
  assert jnp.allclose(gradient, expected, rtol=1e-8)


def test_ensemble_with_observable_fn(x64):
  """An observable collapses each realization to numbers, not fields."""
  solver, sim_config, psi_0 = _turbulence_solver(nz=8)
  keys = jax.random.split(jax.random.PRNGKey(7), 6)

  fields, history = ensemble_solve(
    solver, psi_0, keys,
    chunk_size=3,
    return_history=True,
    save_every=2,
    observable_fn=lambda psi, z: jnp.sum(jnp.abs(psi)**2),
  )

  assert fields.shape == (6, 32, 32)
  # (n_realizations, nz // save_every), not a field volume.
  assert history.shape == (6, 4)

  serial = jnp.stack([
    solver.solve(
      psi_0, medium=k, save_every=2,
      observable_fn=lambda psi, z: jnp.sum(jnp.abs(psi)**2),
    )[1]
    for k in keys
  ])
  assert jnp.allclose(history, serial, atol=1e-12)


def test_ensemble_history_of_fields_has_a_realization_axis(x64):
  """With return_history and no observable, histories stack per realization."""
  solver, _, psi_0 = _turbulence_solver(nz=8)
  keys = jax.random.split(jax.random.PRNGKey(8), 3)

  _, history = ensemble_solve(
    solver, psi_0, keys, chunk_size=3, return_history=True, save_every=4
  )

  assert history.shape == (3, 2, 32, 32)


# --------------------------------------------------------------------------
# Rejected inputs
# --------------------------------------------------------------------------

def test_empty_ensemble_is_rejected():
  solver, _, psi_0 = _turbulence_solver()
  keys = jax.random.split(jax.random.PRNGKey(9), 1)[:0]

  with pytest.raises(ValueError, match='at least one realization'):
    ensemble_solve(solver, psi_0, keys)


@pytest.mark.parametrize('chunk_size', [0, -1])
def test_non_positive_chunk_size_is_rejected(chunk_size):
  solver, _, psi_0 = _turbulence_solver()
  keys = jax.random.split(jax.random.PRNGKey(10), 4)

  with pytest.raises(ValueError, match='chunk_size must be positive'):
    ensemble_solve(solver, psi_0, keys, chunk_size=chunk_size)


def test_donation_is_rejected():
  """psi_0 is shared across realizations, so donating it is a correctness bug
  rather than an optimization."""
  solver, _, psi_0 = _turbulence_solver()
  keys = jax.random.split(jax.random.PRNGKey(11), 4)

  with pytest.raises(ValueError, match='donate_psi_0 is incompatible'):
    ensemble_solve(solver, psi_0, keys, donate_psi_0=True)
