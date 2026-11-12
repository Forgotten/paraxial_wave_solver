"""Compare selected and fixed Taylor actions at the same propagation accuracy."""

from solver_benchmarks import run_case


def main():
  records = {}
  for backend in ('fixed_taylor', 'adaptive_taylor'):
    records[backend] = run_case(
      counts=(8, 16, 32), steppers=('magnus4',), repeats=5,
      exponential_backend=backend,
    )
  for fixed, adaptive in zip(records['fixed_taylor'], records['adaptive_taylor'],
                              strict=True):
    assert adaptive['fft_calls'] < fixed['fft_calls']
    assert abs(adaptive['relative_error']-fixed['relative_error']) < 1e-12
    print(f"nz={fixed['nz']}: FFTs {fixed['fft_calls']} -> {adaptive['fft_calls']}, "
          f"error={adaptive['relative_error']:.3e}, "
          f"warm runtime ratio fixed/selected="
          f"{fixed['median_seconds']/adaptive['median_seconds']:.2f}")
  print('Runtime ratios are measurements, not portable performance guarantees.')


if __name__ == '__main__':
  main()
