# Fourth-order exponential propagation

Select `SolverConfig(method='spectral', stepper='etdrk4')` to use Cox–Matthews
ETDRK4 for the paraxial equation. It integrates the homogeneous spectral
Laplacian exactly and treats refraction, Kerr, and the absorbing potential
with four explicit stages. It uses the existing `solve` interface, including
observables, history, checkpointing, batching, and parameter overrides.

```python
import jax.numpy as jnp
import paraxial_wave_solver as pws

pws.enable_x64()
sim = pws.SimulationConfig(32, 32, 0.25, 0.25, 0.02, 20, 1.0, n2=0.01)
solver = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='etdrk4'),
    pws.PMLConfig(4, 4, 0.5), lambda z, medium: medium*jnp.cos(z),
)
initial = pws.gaussian_beam(sim, w0=1.0, envelope_only=True)
final, history = solver.solve(initial, medium=0.02, save_every=5)
assert bool(jnp.all(jnp.isfinite(final)))
assert history.shape == (4, 32, 32)
```

## Supported physics and restrictions

- Spectral, periodic, paraxial diffraction only. Finite differences and the
  `wide_angle`/`helmholtz` propagators are not supported by this initial version.
- Real or complex index, Kerr, and an absorbing potential are supported.
  Complex coordinate stretching is rejected.
- `splitting_order` stays at its default value of 2: it does not control ETDRK4.
  ETDRK4 itself is fourth order. Nondefault splitting order and `dealias=True`
  are rejected, rather than silently ignored.
- Diffraction has no RK4 stability restriction, but local index, Kerr, and
  absorption terms remain explicit. Large local phase or damping increments
  can destabilize the method; refine `dz` and check finite fields and errors.
- Norm is not preserved exactly. Check power drift and transverse resolution
  for conservative nonlinear runs.

## Order and medium sampling

At fixed spatial discretization, the method has fourth-order convergence in z
for sufficiently smooth solutions and medium callbacks. The four evaluations
use `z`, `z+dz/2`, `z+dz/2`, and `z+dz`. Endpoint sampling differs from Strang's
midpoint-only sampling. A callback indexing a finite volume must handle the
last endpoint and genuinely discontinuous interfaces deliberately. Piecewise
constant slice lookup does not generally give fourth-order accuracy.

The absorber participates in the stage evaluations; it is not an outer
second-order split. This removes that source of order reduction but does not
remove the absorber's explicit stability restriction. Fourth-order convergence
at fixed grid is not a uniform stiff-order guarantee under arbitrary spatial
refinement or rough potentials.

## Coefficients and derivatives

`src.exponential.etd_coefficients` builds the exponential and phi-function
multipliers once per operator cache. `phi_functions` uses 20-term Taylor
series for arguments of magnitude at most one and recurrence elsewhere. This
avoids division by zero and small-argument cancellation, including in gradients.
At zero diffraction the stages reduce to classical RK4. The implementation
uses the Cox–Matthews formula, with series evaluation rather than the contour
quadrature discussed by [Kassam and Trefethen](https://people.maths.ox.ac.uk/trefethen/fourth-order.pdf).

`build_operators` keeps `k0`, `n0`, and `dz` differentiable. Changing `dz` changes
the callback positions and final distance. Medium and initial-field derivatives
flow through the same JAX scan used by the other steppers.

## Validation and comparison

Run `python examples/etdrk4_convergence.py` for an independent dense matrix
exponential comparison. It checks fourth-order ETDRK4 and second-order Strang
at fixed grid, writing `etdrk4_convergence.png`. The comparison is about error
per step count, not error per runtime: ETDRK4 evaluates four local stages.

Tests additionally cover smooth range variation, Kerr with absorption and PML,
phi functions against augmented matrix exponentials, vacuum propagation beyond
the RK4 diffraction limit, gradients, batching, and callback coordinates.
Adaptive step doubling is available through [solve_adaptive](adaptive_stepping.md).
Magnus integrators remain future work; ETDRK4
uses fixed steps.
