"""Shared pytest fixtures.

`jax_enable_x64` is process-global state, so a test that flips it changes the
precision of every test that runs afterwards. Setting it through a fixture
that restores the previous value keeps each test's precision explicit and
independent of collection order.
"""

import jax
import pytest


@pytest.fixture
def x64():
  """Runs the test in float64, restoring the previous setting afterwards."""
  previous = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", True)
  try:
    yield
  finally:
    jax.config.update("jax_enable_x64", previous)
