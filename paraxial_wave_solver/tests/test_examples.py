"""Check the quantitative claims in the new and corrected examples."""

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from paraxial_wave_solver import SimulationConfig

EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


def _load(name):
  spec = importlib.util.spec_from_file_location(name, EXAMPLES / f'{name}.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_implicit_example_converges(x64):
  _, cn, _, root = _load('implicit_propagation').benchmark()
  assert np.all(np.array(cn[:-1])/cn[1:] > 3.9)
  assert root[-1] < 1e-7
  assert np.all(np.diff(root) < 0)


def test_clear_diagnostics_compare_the_same_plane(x64, monkeypatch):
  example = _load('turbulence_diagnostics')
  sim = SimulationConfig(128, 128, 5e-4, 5e-4, 0.25, 4, 632.8e-9, n0=1.33)
  monkeypatch.setattr(example, 'build_config', lambda: sim)
  history, _, _, _ = example.run(0.)
  assert history['overlap'].shape == (sim.nz+1,)
  assert jnp.allclose(history['overlap'], 1., atol=1e-7)
  assert jnp.allclose(history['peak_ratio'], 1., atol=1e-6)


def test_tilted_reference_has_paraxial_group_velocity(x64):
  example = _load('pml_comparison')
  sim = SimulationConfig(256, 128, 0.3, 0.3, 0.1, 10, 1., n0=1.2)
  from paraxial_wave_solver import centroid
  theta, z = 0.3, 10.
  field = example.get_tilted_gaussian_analytical(sim, 3., theta, z)
  x, _ = centroid(field, sim)
  assert float(x) == pytest.approx(35 + z*np.sin(theta), abs=1e-7)


def test_etdrk4_example_converges(x64):
  _, errors = _load('etdrk4_convergence').benchmark()
  for name, order in (('etdrk4', 4), ('split_step', 2)):
    values = errors[name]
    rates = np.log2(np.array(values[:-1])/values[1:])
    assert np.all(rates > order-.2), (name, values, rates)


def test_etdrk4_notebook_executes(x64, monkeypatch):
  """Run notebook cells in order, including their convergence assertions."""
  import matplotlib.pyplot as plt

  monkeypatch.chdir(EXAMPLES)
  monkeypatch.syspath_prepend(str(EXAMPLES))
  monkeypatch.setattr(plt, 'show', lambda: None)
  notebook = json.loads((EXAMPLES / 'etdrk4_convergence.ipynb').read_text())
  namespace = {'__name__': '__main__'}
  try:
    for cell in notebook['cells']:
      if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), 'etdrk4_notebook', 'exec'), namespace)
    assert np.all(np.abs(namespace['rates']['etdrk4'] - 4) < .2)
  finally:
    plt.close('all')
