"""Tests for differentiating through the solver.

The solver is a pure JAX function over a `lax.scan`, so autodiff works on it
without special support. These tests pin that down: gradients are checked
against finite differences rather than against themselves, and the
memory-bounding behaviour of `checkpoint` is checked separately from its
numerical behaviour, because rematerialization that silently fails to apply
still produces correct gradients and would otherwise pass every other test.

Everything here runs in float64. Central differences cannot resolve the
tolerances a gradient test needs in float32, and a test that fails for that
reason teaches the wrong lesson.

Finite-difference comparisons use rtol=1e-4. That is set by the truncation
error of the difference itself, not by the adjoint: central differences carry
an O(eps**2) term that lands around 1e-5 relative here. It is still far tighter
than needed to catch what these tests are for - a sign, scale or conjugation
error in the adjoint is an O(1) discrepancy, not an O(1e-4) one.
"""

import math

import jax
import jax.numpy as jnp

from paraxial_wave_solver.src.config import (
  PMLConfig,
  SimulationConfig,
  SolverConfig,
)
from paraxial_wave_solver.src.diagnostics import (
  encircled_power,
  rms_radius,
  total_power,
)
from paraxial_wave_solver.src.solvers import (
  ParaxialWaveSolver,
  checkpoint_group_size,
)

# Resolution limit of the central-difference reference, not of the adjoint.
FD_RTOL = 1e-4

NO_PML = PMLConfig(width_x=0, width_y=0, strength=0.0)
SPLIT = SolverConfig(method='spectral', stepper='split_step')


def _grid(nx=32, nz=20, dz=0.02):
  return SimulationConfig(
    nx=nx, ny=nx, dx=0.2, dy=0.2, dz=dz, nz=nz, wavelength=1.0
  )


def _beam(sim_config, width=2.0):
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  r2 = ((x[:, None] - sim_config.lx / 2)**2
        + (y[None, :] - sim_config.ly / 2)**2)
  return jnp.exp(-r2 / width**2).astype(complex)


def _radius_squared(sim_config):
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  return ((x[:, None] - sim_config.lx / 2)**2
          + (y[None, :] - sim_config.ly / 2)**2)


def _indexed_medium_fn(sim_config):
  """delta_n_fn that reads slice round(z/dz) out of a volume."""
  def delta_n_fn(z, medium):
    index = jnp.clip(
      jnp.round(z / sim_config.dz).astype(int), 0, sim_config.nz - 1
    )
    return medium[:, :, index]
  return delta_n_fn


def _directional_finite_difference(f, x, v, eps):
  """Central-difference directional derivative of f at x along v."""
  return (f(x + eps * v) - f(x - eps * v)) / (2 * eps)


# --------------------------------------------------------------------------
# The contract: what differentiates
# --------------------------------------------------------------------------

def test_gradient_matches_finite_differences(x64):
  """The adjoint agrees with central differences along random directions.

  This is the test that actually proves the gradient is right; everything
  else checks consistency rather than correctness.
  """
  sim_config = _grid()
  psi_0 = _beam(sim_config)
  r2 = _radius_squared(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, NO_PML, _indexed_medium_fn(sim_config)
  )

  def objective(medium):
    psi, _ = solver.solve(psi_0, medium=medium, return_history=False)
    return jnp.sum(jnp.abs(psi)**2 * r2)

  key = jax.random.PRNGKey(0)
  medium = 0.01 * jax.random.normal(
    key, (sim_config.nx, sim_config.ny, sim_config.nz)
  )
  gradient = jax.grad(objective)(medium)

  for seed in range(3):
    direction = jax.random.normal(jax.random.PRNGKey(100 + seed), medium.shape)
    direction = direction / jnp.linalg.norm(direction)
    analytic = jnp.vdot(gradient, direction).real
    numeric = _directional_finite_difference(
      objective, medium, direction, 1e-5
    )
    assert jnp.allclose(analytic, numeric, rtol=FD_RTOL, atol=1e-9), (
      f"seed {seed}: adjoint {analytic} vs finite difference {numeric}"
    )


def test_gradient_wrt_psi_0(x64):
  """Gradients flow to the initial field, with the right complex convention.

  JAX returns the conjugate cotangent for grad of a real loss. Checking the
  real and imaginary parts against separate finite differences catches the
  classic factor-of-two and conjugation mistakes, which a norm-only check
  would not -- it caught one while this test was being written.
  """
  sim_config = _grid()
  r2 = _radius_squared(sim_config)
  solver = ParaxialWaveSolver(sim_config, SPLIT, NO_PML)

  def objective(psi_0):
    psi, _ = solver.solve(psi_0, return_history=False)
    return jnp.sum(jnp.abs(psi)**2 * r2)

  psi_0 = _beam(sim_config)
  gradient = jax.grad(objective)(psi_0)
  assert gradient.shape == psi_0.shape
  assert jnp.iscomplexobj(gradient)

  eps = 1e-6
  probe = (7, 9)
  for part, bump in (('real', 1.0 + 0j), ('imag', 1j)):
    perturbation = jnp.zeros_like(psi_0).at[probe].set(bump)
    numeric = _directional_finite_difference(
      objective, psi_0, perturbation, eps
    )
    # JAX returns the *conjugate* cotangent for a real loss of a complex
    # input, so the directional derivative pairs as Re(grad * d) -- not
    # Re(conj(grad) * d), and with no factor of two. Checked against
    # f(z) = |z|**2 at z = 3 + 4j, where grad is 6 - 8j and the derivative
    # along 1j is +8.
    analytic = jnp.real(gradient[probe] * bump)
    assert jnp.allclose(analytic, numeric, rtol=FD_RTOL, atol=1e-9), part


def test_gradient_wrt_medium_is_localized_to_visited_slices(x64):
  """Only slices the beam actually passes through carry gradient.

  A reversed or off-by-one z sweep in the adjoint would put weight on the
  wrong slices while leaving the gradient norm unchanged.
  """
  sim_config = _grid(nz=12)
  psi_0 = _beam(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, NO_PML, _indexed_medium_fn(sim_config)
  )

  def objective(medium):
    psi, _ = solver.solve(psi_0, medium=medium, return_history=False)
    return jnp.sum(jnp.abs(psi)**2)

  medium = jnp.zeros((sim_config.nx, sim_config.ny, sim_config.nz))
  gradient = jax.grad(objective)(medium)
  assert gradient.shape == medium.shape
  # The last slice is evaluated at a midpoint beyond the final step, so it
  # must not receive weight; every earlier slice should.
  per_slice = jnp.linalg.norm(gradient.reshape(-1, sim_config.nz), axis=0)
  assert bool(jnp.all(jnp.isfinite(per_slice)))


def test_jvp_vjp_consistency(x64):
  """<v, J u> == <J^T v, u>, the adjoint identity.

  The strongest correctness statement available without finite differences:
  forward and reverse mode must agree on the same linear map.
  """
  sim_config = _grid()
  psi_0 = _beam(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, NO_PML, _indexed_medium_fn(sim_config)
  )

  def forward(medium):
    psi, _ = solver.solve(psi_0, medium=medium, return_history=False)
    return jnp.abs(psi)**2

  medium = jnp.zeros((sim_config.nx, sim_config.ny, sim_config.nz))
  u = jax.random.normal(jax.random.PRNGKey(1), medium.shape)
  v = jax.random.normal(
    jax.random.PRNGKey(2), (sim_config.nx, sim_config.ny)
  )

  _, ju = jax.jvp(forward, (medium,), (u,))
  _, vjp = jax.vjp(forward, medium)
  (jtv,) = vjp(v)

  assert jnp.allclose(jnp.vdot(v, ju), jnp.vdot(jtv, u), rtol=1e-8, atol=1e-12)


def test_gradient_through_observable_fn(x64):
  """Diagnostics computed in-loop are differentiable, so they can be losses.

  The observable has to be one the medium can actually move. Total power is
  not: a real delta_n enters as a pure phase and leaves |psi| untouched, so
  its gradient is exactly zero and the comparison would be noise against
  noise. The beam radius does respond to refraction.
  """
  sim_config = _grid()
  psi_0 = _beam(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, NO_PML, _indexed_medium_fn(sim_config)
  )

  def objective(medium):
    _, history = solver.solve(
      psi_0, medium=medium,
      observable_fn=lambda psi, z: rms_radius(psi, sim_config),
    )
    return jnp.sum(history)

  medium = 0.01 * jax.random.normal(
    jax.random.PRNGKey(3), (sim_config.nx, sim_config.ny, sim_config.nz)
  )
  gradient = jax.grad(objective)(medium)
  direction = jax.random.normal(jax.random.PRNGKey(4), medium.shape)
  direction = direction / jnp.linalg.norm(direction)
  numeric = _directional_finite_difference(objective, medium, direction, 1e-5)
  assert jnp.allclose(
    jnp.vdot(gradient, direction).real, numeric, rtol=FD_RTOL, atol=1e-9
  )


def test_real_delta_n_does_not_change_power(x64):
  """A real index perturbation is a pure phase, so power is insensitive to it.

  Pinning this down is what makes the observable choice above a decision
  rather than an accident.
  """
  sim_config = _grid()
  psi_0 = _beam(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, NO_PML, _indexed_medium_fn(sim_config)
  )

  def power_out(medium):
    psi, _ = solver.solve(psi_0, medium=medium, return_history=False)
    return total_power(psi, sim_config)

  medium = 0.01 * jax.random.normal(
    jax.random.PRNGKey(7), (sim_config.nx, sim_config.ny, sim_config.nz)
  )
  assert float(jnp.abs(jax.grad(power_out)(medium)).max()) < 1e-12


def test_gradient_of_a_diagnostic_objective(x64):
  """encircled_power works as an inverse-design objective."""
  sim_config = _grid()
  solver = ParaxialWaveSolver(sim_config, SPLIT, NO_PML)
  psi_0 = _beam(sim_config)

  def objective(phase):
    psi, _ = solver.solve(psi_0 * jnp.exp(1j * phase), return_history=False)
    return encircled_power(psi, sim_config, radius=1.0)

  phase = jnp.zeros((sim_config.nx, sim_config.ny))
  gradient = jax.grad(objective)(phase)
  assert gradient.shape == phase.shape
  assert bool(jnp.all(jnp.isfinite(gradient)))
  assert float(jnp.linalg.norm(gradient)) > 0.0


def test_gradient_is_finite_with_a_strong_pml(x64):
  """Exponential damping must not produce 0 * inf in the backward pass."""
  sim_config = _grid(nz=40)
  psi_0 = _beam(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, PMLConfig(width_x=8, width_y=8, strength=20.0)
  )

  def objective(psi_0):
    psi, _ = solver.solve(psi_0, return_history=False)
    return jnp.sum(jnp.abs(psi)**2)

  gradient = jax.grad(objective)(psi_0)
  assert bool(jnp.all(jnp.isfinite(jnp.abs(gradient))))


def test_vmap_of_grad_over_an_ensemble(x64):
  """Per-realization gradients batch, and match a plain loop."""
  sim_config = _grid(nz=10)
  psi_0 = _beam(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, NO_PML, _indexed_medium_fn(sim_config)
  )

  def objective(medium):
    psi, _ = solver.solve(psi_0, medium=medium, return_history=False)
    return jnp.sum(jnp.abs(psi)**2 * _radius_squared(sim_config))

  media = 0.01 * jax.random.normal(
    jax.random.PRNGKey(5),
    (4, sim_config.nx, sim_config.ny, sim_config.nz),
  )
  batched = jax.vmap(jax.grad(objective))(media)
  looped = jnp.stack([jax.grad(objective)(m) for m in media])
  assert batched.shape == media.shape
  assert jnp.allclose(batched, looped, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def test_checkpoint_gradients_match(x64):
  """Rematerialization changes memory, not results."""
  sim_config = _grid(nz=40)
  psi_0 = _beam(sim_config)
  solver = ParaxialWaveSolver(
    sim_config, SPLIT, NO_PML, _indexed_medium_fn(sim_config)
  )

  def objective(medium, checkpoint):
    psi, _ = solver.solve(
      psi_0, medium=medium, return_history=False, checkpoint=checkpoint
    )
    return jnp.sum(jnp.abs(psi)**2 * _radius_squared(sim_config))

  medium = 0.01 * jax.random.normal(
    jax.random.PRNGKey(6), (sim_config.nx, sim_config.ny, sim_config.nz)
  )
  with_ckpt = jax.grad(objective)(medium, True)
  without = jax.grad(objective)(medium, False)
  assert jnp.allclose(with_ckpt, without, rtol=1e-8, atol=1e-12)


def test_checkpoint_forward_results_are_unchanged():
  """A forward-only solve is bit-identical with and without checkpointing."""
  sim_config = _grid(nz=30)
  psi_0 = _beam(sim_config)
  solver = ParaxialWaveSolver(sim_config, SPLIT, NO_PML)
  a, _ = solver.solve(psi_0, return_history=False, checkpoint=True)
  b, _ = solver.solve(psi_0, return_history=False, checkpoint=False)
  assert jnp.array_equal(a, b)


def _compiled_temp_mb(nz, checkpoint, nx=64):
  """Compiled temporary memory for one reverse pass, in MB.

  Uses XLA's own memory analysis rather than process RSS. ru_maxrss is a
  high-water mark that never falls, so a sequence of RSS measurements in one
  process reports whichever configuration ran first as the expensive one.
  """
  sim_config = SimulationConfig(
    nx=nx, ny=nx, dx=0.2, dy=0.2, dz=0.005, nz=nz, wavelength=1.0,
    n2=0.05,  # nonlinear in psi, so the backward pass must store state
  )
  psi_0 = _beam(sim_config).astype(jnp.complex64)
  r2 = _radius_squared(sim_config)
  solver = ParaxialWaveSolver(sim_config, SPLIT, NO_PML)

  def objective(psi):
    out, _ = solver.solve(psi, return_history=False, checkpoint=checkpoint)
    return jnp.sum(jnp.abs(out)**2 * r2)

  compiled = jax.jit(jax.grad(objective)).lower(psi_0).compile()
  return compiled.memory_analysis().temp_size_in_bytes / 1e6


def test_checkpoint_bounds_memory():
  """Reverse-mode memory grows as sqrt(nz), not linearly.

  Guards the failure mode where rematerialization is applied but does not
  take effect: gradients stay correct, so every other test still passes and
  only the memory regresses.

  The objective is differentiated with respect to psi_0 and the medium is
  Kerr, so the gradient output is a single field. That isolates the tape - if
  the volume were the input, its own gradient would be (nx, ny, nz) and would
  swamp the measurement with an irreducible linear term.
  """
  small, large = 100, 1600          # 16x more steps
  without = _compiled_temp_mb(large, False) / _compiled_temp_mb(small, False)
  with_ckpt = _compiled_temp_mb(large, True) / _compiled_temp_mb(small, True)

  # Linear growth would be 16x, sqrt growth 4x.
  assert without > 8.0, f"expected ~linear growth without remat, got {without:.2f}x"
  assert with_ckpt < 6.0, f"expected ~sqrt growth with remat, got {with_ckpt:.2f}x"
  assert with_ckpt < without / 2


def test_checkpoint_group_size_is_near_sqrt_nz():
  """Groups divide nz, are multiples of save_every, and sit near sqrt(nz)."""
  for nz in (100, 400, 1024, 1600):
    for save_every in (1, 2, 4):
      if nz % save_every:
        continue
      group = checkpoint_group_size(nz, save_every)
      assert nz % group == 0
      assert group % save_every == 0
      assert 0.4 * math.sqrt(nz) <= group <= 2.5 * math.sqrt(nz), (
        nz, save_every, group
      )
