import jax
import jax.numpy as jnp
import pytest

from paraxial_wave_solver.src.config import SimulationConfig
from paraxial_wave_solver.src.utils import random_medium, random_medium_spectral


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
