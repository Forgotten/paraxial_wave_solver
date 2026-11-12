"""Validate benchmark references and selection, not hardware-dependent timings."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
  'solver_benchmarks', Path(__file__).resolve().parents[2]
  / 'examples' / 'solver_benchmarks.py',
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_refinement_and_measurements(x64):
  rows = benchmark.run_case(counts=(4, 8, 16), steppers=('etdrk4',), repeats=1)
  errors = [r['relative_error'] for r in rows]
  assert np.all(np.log2(np.array(errors[:-1])/errors[1:]) > 3.7), errors
  assert all(r['median_seconds'] > 0 and r['compile_seconds'] > 0 for r in rows)
  assert [r['fft_calls'] for r in rows] == [40, 80, 160]
  assert all(r['temp_bytes'] is None or r['temp_bytes'] >= 0 for r in rows)
  assert benchmark.matched_accuracy(rows, 1e-3)[0]['reached']
  assert not benchmark.matched_accuracy(rows, 1e-20)[0]['reached']


@pytest.mark.parametrize('case', benchmark.CASES)
def test_reference_is_finite_and_absorption_passive(case, x64):
  _, initial, _, exact = benchmark.problem(case)
  assert np.all(np.isfinite(exact))
  if case in ('absorption', 'near_cutoff'):
    assert np.linalg.norm(exact) < np.linalg.norm(initial)
  else:
    assert np.linalg.norm(exact) == pytest.approx(np.linalg.norm(initial), rel=1e-11)


def test_incompatible_physics_rejected(x64):
  with pytest.raises(ValueError, match='model'):
    benchmark.run_case('kerr', steppers=('pade',))


def test_magnus_action_counts_and_accuracy(x64):
  fixed = benchmark.run_case(counts=(8,), steppers=('magnus4',), repeats=1,
                             exponential_backend='fixed_taylor')[0]
  adaptive = benchmark.run_case(counts=(8,), steppers=('magnus4',), repeats=1)[0]
  assert fixed['fft_calls'] == 8*2*24*2
  assert adaptive['fft_calls'] < fixed['fft_calls']
  assert abs(adaptive['relative_error']-fixed['relative_error']) < 1e-12
