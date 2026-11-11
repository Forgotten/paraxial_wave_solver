# Adaptive propagation

`solve_adaptive` uses one full step and two half steps to estimate the local
error. It accepts the two-half-step result when the weighted RMS estimate is
at most one, otherwise retries from the unchanged input. The estimator divides
the difference by `2**p - 1`, with the actual order of the configured method.
The controller uses exponent `1/(p+1)` and bounded step-size changes.

```python
import jax.numpy as jnp
import paraxial_wave_solver as pws

pws.enable_x64()
sim = pws.SimulationConfig(16, 16, 1., 1., .2, 10, 1.)
solver = pws.ParaxialWaveSolver(
  sim, pws.SolverConfig(method='spectral', stepper='etdrk4'),
  pws.PMLConfig(0, 0, 0.), lambda z, m: m*jnp.cos(z),
)
result = solver.solve_adaptive(
  jnp.ones((16, 16), dtype=complex), medium=.2,
  config=pws.AdaptiveConfig(rtol=1e-7, atol=1e-9, max_step=.3),
  save_z=[0., .5, 1., 1.5, 2.],
)
assert result.z[-1] == 2.
print(result.accepted, result.rejected)
```

Supported methods are ETDRK4, RK4, Magnus-4, Crank–Nicolson, and split-step without
dealias filtering. Kerr and absorbers follow the chosen method's restrictions;
Yoshida with an outer absorber uses a second-order estimator. Local error
control does not guarantee global error or stability. Smooth medium callbacks
are required: split calls at known discontinuities, with a callback appropriate
to each interval; merely requesting an output at a jump is insufficient for
methods sampling that endpoint.

Output planes are reached exactly by shortening steps, without interpolation.
Only requested planes and the final field are saved; the final plane is always
appended. `z_end` defaults to `z_0 + sim.lz`. `params` can override `k0` and `n0`;
the controller owns `dz`. An output-plane step can be shorter than `min_step`.
Failed tolerance checks at that minimum, exhausted attempts, or inability to
advance the coordinate raise `RuntimeError`. Nonfinite trial errors are rejected.

The controller runs on the host and synchronizes one scalar error per attempt.
It is **not** compatible with JIT, vmap, or differentiation. Use the existing
fixed-step `solve` API for inverse problems and batched ensembles. Accepted
macrostep sizes and accepted/rejected counts are returned for diagnostics.
Float64 is recommended for tight tolerances. Tests compare tolerance refinement
against an independent SciPy DOP853 reference, in addition to the existing
fixed-step convergence tests.
