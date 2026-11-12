# Higher-order implicit propagation

Select `stepper='gauss4'` for two-stage Gauss–Legendre order four, or
`stepper='radau5'` for three-stage Radau IIA order five. Both integrate the
**linear paraxial equation**, sample the medium at their distinct stage nodes,
and support spectral or finite-difference discretization. Complex index,
absorbing potentials, and supported FD complex stretching are included in the
stage equations. Kerr is rejected; no nonlinear stage solver is implied.

The coupled system is
`Y_i - h sum_j a_ij A(z+c_j*h) Y_j = psi`.
A single matrix-free GMRES solve handles all stage fields together. The
preconditioner inverts the small stage block for each Fourier mode of the
periodic homogeneous reference, including the selected FD stencil symbol.
The actual stage residual is checked; failed solves return NaNs.

Gauss methods preserve quadratic invariants for conservative linear generators
up to solve and roundoff error. They do not strongly damp infinitely stiff
modes. Radau IIA is L-stable and uses its final stage as the output, avoiding
a cancellation-prone update for strong absorption. Neither stability property
removes phase-accuracy constraints. The order claims assume smooth range
variation at a fixed spatial grid; stiff order reduction remains possible.
The [NASA collocation reference, Table 3.2](https://ntrs.nasa.gov/api/citations/20220008600/downloads/Huynh_ICCFD_2022_2.pdf)
provides the Radau IIA tableau.

```python
import jax.numpy as jnp
import paraxial_wave_solver as pws

pws.enable_x64()
sim = pws.SimulationConfig(8, 8, 1., 1., .1, 10, 1.)
solver = pws.ParaxialWaveSolver(
  sim, pws.SolverConfig(method='finite_difference', fd_order=4,
                        stepper='radau5', linear_rtol=1e-11),
  pws.PMLConfig(1, 1, .2, use_complex_stretching=True),
  lambda z, medium: medium*jnp.cos(z),
)
field, _ = solver.solve(jnp.ones((8, 8), dtype=complex), medium=.1+.01j,
                         return_history=False)
assert jnp.all(jnp.isfinite(field))
```

Fixed-step propagation supports gradients via the shared implicit linear-solve
rule. Both forward and transpose solves must converge. Tighten `linear_rtol`
below the desired propagation accuracy and verify directional derivatives.
`solve_adaptive` supports these methods using orders four and five; its host
controller remains nondifferentiable and its tolerances do not replace the
linear residual tolerance.

Run `python examples/collocation_convergence.py` for measured order against an
independent DOP853 reference. Tests cover smooth range variation with spectral,
FD absorption, and FD stretching, as well as conservation, stiff damping,
failed solves, parameter overrides, and gradients.
