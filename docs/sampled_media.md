# Sampled media and range interfaces

Build a medium with `make_sampled_medium(z, values, kind='cubic')` and pass
`sample_medium` as the solver's callback. Coordinates are physical, strictly
increasing range values. Samples have leading dimension `len(z)` and any
trailing shape broadcastable against the field, including complex index data.

`kind='linear'` is piecewise linear and C0, with order-two interpolation error
for smooth data. `kind='cubic'` is a not-a-knot cubic spline, C2, with order-four
interpolation error for sufficiently smooth data and regular sample meshes.
Cubic requires at least four knots. It can overshoot; it does not preserve
positivity of absorption or monotonicity. Values outside the sampled interval
return NaNs rather than silently extrapolating. Endpoints are included.

The constructor solves a dense system along the knot axis once; the stored
piecewise coefficients make evaluation inexpensive. This setup is intended for
moderate knot counts, not millions of samples. Coefficients form a JAX pytree.
Both setup and evaluation differentiate with respect to sample values; knot
validation is on the host, and differentiation of knot schedules is unsupported.
Use `jax.vmap(sample_medium, in_axes=(0, None))` for multiple query positions.

```python
import jax.numpy as jnp
import numpy as np
import paraxial_wave_solver as pws

pws.enable_x64()
z = np.linspace(0, 1, 9)
medium = pws.make_sampled_medium(z, .1*np.exp(z))
sim = pws.SimulationConfig(4, 4, 1., 1., .2, 5, 1.)
solver = pws.ParaxialWaveSolver(
  sim, pws.SolverConfig(method='spectral', stepper='magnus4'),
  pws.PMLConfig(0, 0), pws.sample_medium,
)
result = solver.solve_adaptive(jnp.ones((4, 4), dtype=complex), medium=medium,
                               stop_z=z, save_z=[0., 1.])
assert result.history.shape == (2, 4, 4)
```

`stop_z` introduces mandatory step boundaries without extra saved fields.
Stopping at spline knots avoids crossing jumps in higher derivatives within a
step. Interpolation regularity alone does not justify arbitrarily high temporal
order. Refine the medium sample grid separately from dz; tighter integration
tolerances cannot remove interpolation error.

For discontinuous media, merely stopping is insufficient when a method samples
the interface itself. Use `solver.solve_segments(psi, boundaries, media)`:
the callback receives a **different** `media[i]` throughout each closed interval,
so endpoint stages use the appropriate one-sided extension. The result includes
all boundary fields, including the initial field. Each interval runs the adaptive
controller; its limitations still apply. Negative Yoshida substeps are excluded.
A segment may itself use a sampled medium covering that interval.

The envelope is continuous across these range interfaces. This is a one-way
march, not an interface scattering model: no reflected wave, Fresnel matching,
or carrier change is added. Run `python examples/medium_interpolation.py` for
sample refinement and an analytic layered-medium phase check. Tests additionally
compare the spline against SciPy, check C2 continuity, gradients, mandatory stops,
and ETDRK4/Radau endpoint behavior.
