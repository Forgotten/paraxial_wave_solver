# Hermitian square-root actions

Use `stepper='lanczos', propagator='helmholtz'` to approximate the **action**
of `exp(i*h*(sqrt(H)-k))`, where `H = L_perp + k0**2*(n0+delta_n)**2`.
This is an alternative to the rotated Padé whole-step approximation, not a
higher-order z integrator. The medium is frozen at the midpoint, so smooth
range dependence has order two. A constant medium has only Krylov and
roundoff errors; decreasing dz does not replace Krylov-dimension refinement.

```python
import jax.numpy as jnp
import paraxial_wave_solver as pws

pws.enable_x64()
sim = pws.SimulationConfig(8, 8, 1., 1., .1, 10, 1.)
solver = pws.ParaxialWaveSolver(
  sim, pws.SolverConfig(method='spectral', stepper='lanczos',
                        propagator='helmholtz', krylov_dim=32, krylov_rtol=1e-8),
  pws.PMLConfig(0, 0),
)
final, _ = solver.solve(jnp.ones((8, 8), dtype=complex), return_history=False)
assert jnp.all(jnp.isfinite(final))
```

Spectral and periodic FD discretizations are supported. The index must be real;
Kerr and all PML/stretching are rejected. A complex index passed through a
traced callback produces NaNs. Negative eigenvalues use the positive-imaginary
square root, giving outgoing evanescent **decay** for a positive range step.

## Standalone square-root action

`pws.square_root_action(matvec, field, krylov_dim=32, rtol=1e-6)` returns a
`KrylovResult(solution, relative_difference, converged)`. Supplying
`matvec=lambda u: -laplacian(u)` computes `sqrt(-Laplacian) @ field`.
This fractional-Laplacian action differs from the Helmholtz envelope propagator.
For a periodic spectral Laplacian alone, an exact Fourier multiplier is cheaper;
the Krylov interface is useful for non-diagonal Hermitian operators.

The candidate is `||field|| Q f(T) e1`, with a fully reorthogonalized Hermitian
Krylov projection. Two orthogonalization passes control basis loss; a small
Hermitian eigensystem applies the outgoing branch. Storage scales as O(mN)
and projected eigendecomposition as O(m³), with m capped at N. No dense N×N
transverse matrix is built unless the user deliberately chooses m=N.

The acceptance check compares dimensions m and max(1,m//2). This is a
**refinement indicator, not a certified error bound**; agreement can be
misleading, especially near zero eigenvalues/cutoff. An invariant subspace
is accepted to roundoff without requiring coarse-space agreement. A single
non-invariant basis vector cannot pass. The solver returns NaNs when the
indicator fails; the standalone action retains its candidate for diagnostics.
Increase `krylov_dim` and compare independent results. Float64 is recommended.
The caller must supply a Hermitian matvec; a projected symmetry check cannot
certify symmetry on unexplored directions.

JIT and batching are supported for forward evaluation. Derivatives through
basis breakdown, repeated Ritz eigenvalues, or the square-root branch point
are not supported; use Padé for differentiable full-index work. Adaptive
propagation currently excludes root steppers because temporal error control
alone cannot control their separate approximation errors.

Run `python examples/square_root_actions.py` for dimension refinement and a
dense propagation reference. Unit tests additionally check mixed positive and
negative spectra, invariant and zero fields, float32, parameter overrides,
failure reporting, and second-order midpoint convergence against an independent
range-dependent dense-root ODE reference.

For background, see [Saad's Krylov matrix-function analysis](https://epubs.siam.org/doi/10.1137/0729014)
and the [2024 Lanczos matrix-function approximation paper](https://proceedings.nips.cc/paper_files/paper/2024/hash/fcc3dc27672a12510babe448d665e152-Abstract-Conference.html).
These analyses motivate the approach; their guarantees are not a certification
of the nested-space indicator used here, particularly across the branch point.
