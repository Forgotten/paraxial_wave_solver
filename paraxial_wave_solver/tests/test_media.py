import jax
import jax.numpy as jnp
import pytest

from paraxial_wave_solver.src.config import (
  PMLConfig,
  SimulationConfig,
  SolverConfig,
)
from paraxial_wave_solver.src.solvers import ParaxialWaveSolver
from paraxial_wave_solver.src.utils import (
  phase_screen,
  random_medium,
  random_medium_spectral,
)


def _config(nx=32, ny=32, nz=32):
  return SimulationConfig(
    nx=nx, ny=ny, dx=0.1, dy=0.1, dz=0.1, nz=nz, wavelength=1.0
  )


def test_random_medium_shape_and_strength():
  sim_config = _config()
  delta_n = random_medium(
    sim_config, correlation_length=0.5, strength=1e-3,
    key=jax.random.PRNGKey(0),
  )
  assert delta_n.shape == (sim_config.nx, sim_config.ny, sim_config.nz)
  assert jnp.isrealobj(delta_n)
  assert jnp.isclose(jnp.std(delta_n), 1e-3, rtol=1e-5)


def test_random_medium_is_reproducible():
  sim_config = _config()
  kwargs = dict(correlation_length=0.5, strength=1e-3)
  first = random_medium(sim_config, key=jax.random.PRNGKey(7), **kwargs)
  second = random_medium(sim_config, key=jax.random.PRNGKey(7), **kwargs)
  other = random_medium(sim_config, key=jax.random.PRNGKey(8), **kwargs)
  assert jnp.allclose(first, second)
  assert not jnp.allclose(first, other)


@pytest.mark.parametrize("correlation_length", [0.3, 1.0])
def test_random_medium_correlation_length(correlation_length, x64):
  """A longer correlation length gives a smoother field.

  Measured through the lag-one autocorrelation along x, which increases
  towards 1 as the correlation length grows relative to the grid spacing.
  """
  sim_config = _config(nx=64, ny=64, nz=64)
  delta_n = random_medium(
    sim_config, correlation_length=correlation_length, strength=1.0,
    key=jax.random.PRNGKey(3),
  )
  shifted = jnp.roll(delta_n, 1, axis=0)
  lag_one = float(jnp.mean(delta_n * shifted) / jnp.mean(delta_n**2))
  expected = float(jnp.exp(-sim_config.dx**2 / correlation_length**2))
  assert lag_one == pytest.approx(expected, abs=0.05)


def test_random_medium_spectral_shape_and_realness():
  sim_config = _config()
  volume = random_medium_spectral(
    sim_config, Cn2=1e-13, L0=2e-2, l0=1e-3, key=jax.random.PRNGKey(1)
  )
  assert volume.shape == (sim_config.nx, sim_config.ny, sim_config.nz)
  assert jnp.isrealobj(volume)
  assert bool(jnp.all(jnp.isfinite(volume)))


def test_random_medium_spectral_scales_with_cn2():
  """Fluctuation amplitude scales as sqrt(Cn2)."""
  sim_config = _config()
  kwargs = dict(L0=2e-2, l0=1e-3, key=jax.random.PRNGKey(1))
  weak = random_medium_spectral(sim_config, Cn2=1e-13, **kwargs)
  strong = random_medium_spectral(sim_config, Cn2=1e-11, **kwargs)
  ratio = float(jnp.std(strong) / jnp.std(weak))
  assert ratio == pytest.approx(10.0, rel=1e-3)


# --------------------------------------------------------------------------
# Streaming phase screens
# --------------------------------------------------------------------------

def test_phase_screen_shape_and_strength(x64):
  sim_config = _config(nx=64, ny=64)
  screen = phase_screen(
    jax.random.PRNGKey(0), sim_config, correlation_length=0.5, strength=1e-3
  )
  assert screen.shape == (sim_config.nx, sim_config.ny)
  assert jnp.isrealobj(screen)
  assert jnp.isclose(jnp.std(screen), 1e-3, rtol=1e-5)


def test_phase_screen_is_reproducible_and_key_dependent(x64):
  sim_config = _config(nx=64, ny=64)
  kwargs = dict(sim_config=sim_config, correlation_length=0.5, strength=1e-3)
  a = phase_screen(jax.random.PRNGKey(3), **kwargs)
  b = phase_screen(jax.random.PRNGKey(3), **kwargs)
  c = phase_screen(jax.random.PRNGKey(4), **kwargs)
  assert jnp.allclose(a, b)
  assert not jnp.allclose(a, c)
  # fold_in is the intended way to get a screen per propagation step.
  base = jax.random.PRNGKey(5)
  assert not jnp.allclose(
    phase_screen(jax.random.fold_in(base, 0), **kwargs),
    phase_screen(jax.random.fold_in(base, 1), **kwargs),
  )


def test_phase_screen_matches_volume_transverse_statistics(x64):
  """Screens reproduce a slice of random_medium across the transverse plane.

  This is what makes the streaming form a substitute rather than a different
  medium: same variance, same transverse correlation.
  """
  sim_config = _config(nx=128, ny=128, nz=64)
  correlation_length, strength = 0.8, 1e-3

  volume = random_medium(
    sim_config, correlation_length=correlation_length, strength=strength,
    key=jax.random.PRNGKey(0),
  )
  base = jax.random.PRNGKey(1)
  screens = jnp.stack(
    [
      phase_screen(
        jax.random.fold_in(base, i), sim_config, correlation_length, strength
      )
      for i in range(sim_config.nz)
    ],
    axis=-1,
  )

  def lag_one(field, axis):
    return float(
      jnp.mean(field * jnp.roll(field, 1, axis=axis)) / jnp.mean(field**2)
    )

  assert jnp.isclose(jnp.std(screens), jnp.std(volume), rtol=0.05)
  assert lag_one(screens, 0) == pytest.approx(lag_one(volume, 0), abs=0.02)
  assert lag_one(screens, 1) == pytest.approx(lag_one(volume, 1), abs=0.02)


def test_phase_screens_are_uncorrelated_along_z(x64):
  """The documented difference: independent screens carry no z-correlation.

  random_medium filters isotropically in three dimensions, so its slices are
  correlated along z. Screens drawn from independent keys are the thin-screen
  limit instead. Asserting it keeps the approximation explicit rather than
  letting it be discovered later.
  """
  sim_config = _config(nx=64, ny=64, nz=64)
  correlation_length, strength = 0.8, 1e-3

  volume = random_medium(
    sim_config, correlation_length=correlation_length, strength=strength,
    key=jax.random.PRNGKey(0),
  )
  base = jax.random.PRNGKey(1)
  screens = jnp.stack(
    [
      phase_screen(
        jax.random.fold_in(base, i), sim_config, correlation_length, strength
      )
      for i in range(sim_config.nz)
    ],
    axis=-1,
  )

  def longitudinal_lag_one(field):
    return float(
      jnp.mean(field * jnp.roll(field, 1, axis=2)) / jnp.mean(field**2)
    )

  assert longitudinal_lag_one(volume) > 0.5
  assert abs(longitudinal_lag_one(screens)) < 0.05


def test_streaming_screens_propagate_and_differentiate(x64):
  """A run whose medium is a key, not a volume, still solves and differentiates."""
  sim_config = _config(nx=64, ny=64, nz=20)
  solver_config = SolverConfig(method='spectral', stepper='split_step')

  def delta_n_fn(z, medium):
    key, strength = medium
    step = jnp.floor(z / sim_config.dz).astype(int)
    return phase_screen(
      jax.random.fold_in(key, step), sim_config,
      correlation_length=0.5, strength=strength,
    )

  solver = ParaxialWaveSolver(
    sim_config, solver_config, PMLConfig(0, 0, 0.0), delta_n_fn
  )
  x = jnp.arange(sim_config.nx) * sim_config.dx
  r2 = ((x[:, None] - sim_config.lx / 2)**2
        + (x[None, :] - sim_config.ly / 2)**2)
  psi_0 = jnp.exp(-r2 / 4.0).astype(complex)

  def objective(strength):
    psi, _ = solver.solve(
      psi_0, medium=(jax.random.PRNGKey(0), strength), return_history=False
    )
    return jnp.sum(jnp.abs(psi)**2 * r2)

  value = objective(1e-3)
  gradient = jax.grad(objective)(1e-3)
  assert jnp.isfinite(value)
  assert jnp.isfinite(gradient)
