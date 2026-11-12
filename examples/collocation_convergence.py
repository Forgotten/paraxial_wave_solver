"""Gauss/Radau temporal order using the independent benchmark ODE reference."""

import numpy as np
from solver_benchmarks import run_case


def main():
  for name, order in (('gauss4', 4), ('radau5', 5)):
    rows = run_case('smooth', counts=(4, 8, 16), steppers=(name,), repeats=1)
    errors = np.array([r['relative_error'] for r in rows])
    rates = np.log2(errors[:-1]/errors[1:])
    print(f'{name}: errors={errors}, orders={rates}')
    assert np.all(np.abs(rates-order) < .4), rates


if __name__ == '__main__':
  main()
