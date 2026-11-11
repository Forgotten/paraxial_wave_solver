# Implicit propagation and square-root models

This guide covers `crank_nicolson` and the experimental `pade` stepper. Both
reuse `ParaxialWaveSolver.solve`, including history, observables, checkpointing,
operator overrides, and JAX differentiation. They currently require `n2=0`.
See the [README](../README.md) for the shared interface and
[enhancement plan](../ENHANCEMENT_PLAN.md) for research directions.

## Choose the equation before the stepper

Write `k = k0*n0` and supply `delta_n = n - n0` through the medium callback.
All outputs are envelopes, with the carrier `exp(1j*k*z)` removed.

| Configuration | Generator of propagation | Scope |
|---|---|---|
| `crank_nicolson`, `paraxial` | `i*L/(2*k) + i*k0*delta_n - sigma` | Linear paraxial equation; weak index contrast |
| `split_step`, `wide_angle` | Homogeneous `i*(sqrt(k*k + L)-k)` split from `i*k0*delta_n` | Exact homogeneous diffraction symbol; approximate heterogeneous model |
| `pade`, `helmholtz` | `i*(sqrt(L + k0*k0*(n0+delta_n)**2)-k)` | Experimental full-index one-way model |

Here `L` is the selected discrete transverse Laplacian. The full-index root
includes the noncommuting diffraction and index operators together. For
range-dependent media it is frozen at each step midpoint; this is a one-way
approximation, not a solver for reflections in the full Helmholtz equation.
Neither square-root option is the fractional Laplacian `sqrt(-L)`, whose
Fourier symbol is `|q|`. A standalone fractional-Laplacian API is not implemented.

## Supported combinations

| Capability | Crank–Nicolson | Padé |
|---|---|---|
| Spectral Laplacian | Yes | Yes |
| FD2/4/6 | Yes | Yes |
| Isotropic nine-point stencil | Yes; square cells required | Yes; square cells required |
| Real index | Yes | Yes |
| Complex index | Yes | No |
| Kerr term | No | No |
| Absorbing potential | Yes | No |
| FD coordinate stretching | Yes, except the nine-point stencil | No |
| Precision | float32 or float64; tolerance must be attainable | Enable float64 before construction and array creation |

All transverse grids are periodic underneath the absorber: FFTs and rolled FD
stencils wrap at the edges. There are no Dirichlet or Neumann boundary options.
Padé requires no active absorber and rejects `use_complex_stretching=True`.
`dealias` and nondefault `splitting_order` are not supported by either method.

## Runnable propagation example

After `pip install -e .`, run the following from Python. It uses a small
periodic index profile and enables double precision for both methods.

```python
import jax
import jax.numpy as jnp
import paraxial_wave_solver as pws

pws.enable_x64()
sim = pws.SimulationConfig(
    nx=8, ny=8, dx=1.0, dy=1.0, dz=0.1, nz=4, wavelength=1.0,
)
no_pml = pws.PMLConfig(0, 0, 0.0)
x = jnp.arange(sim.nx)[:, None]
index = jnp.broadcast_to(0.06*jnp.cos(2*jnp.pi*x/sim.nx), (sim.nx, sim.ny))
psi_0 = (1 + 0.3*jnp.cos(2*jnp.pi*x/sim.nx))*jnp.ones((1, sim.ny))
psi_0 = psi_0.astype(complex)

def delta_n_fn(z, medium):
    return medium

cn = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='crank_nicolson',
                          linear_rtol=1e-10), no_pml, delta_n_fn,
)
root = pws.ParaxialWaveSolver(
    sim, pws.SolverConfig(method='spectral', stepper='pade',
                          propagator='helmholtz', pade_order=4,
                          linear_rtol=1e-10), no_pml, delta_n_fn,
)
cn_final, cn_history = cn.solve(psi_0, medium=index, save_every=2)
root_final, _ = root.solve(psi_0, medium=index, return_history=False)
assert bool(jnp.all(jnp.isfinite(cn_final)))
assert bool(jnp.all(jnp.isfinite(root_final)))
assert cn_history.shape == (2, 8, 8)
```

The history contains planes at `z=0` and `z=0.2`; final fields are at `z=0.4`.
The difference between the two final fields includes a **model difference**,
so it is not an estimate of either method's numerical error.

## Accuracy and convergence

Crank–Nicolson solves the midpoint-frozen Cayley step
`(I-h*A/2) psi_next = (I+h*A/2) psi`. For smooth range dependence it is second
order in `h`. With a periodic skew-Hermitian generator it preserves norm in
exact arithmetic; iterative error perturbs this property. It has no explicit
RK4 diffraction stability limit, but coarse steps still have phase error. It
is not L-stable: very stiff absorption can be poorly resolved.

Padé approximates the whole function `exp(i*h*k*(sqrt(H/k**2)-1))`. The
rotation selects an approximation to the outgoing branch; negative real
eigenvalues should decay. A finite rational approximation need not be unitary
on propagating modes or contractive over every evanescent spectrum.

Check four errors separately:

1. Refine the transverse grid at fixed physical domain and medium.
2. Refine `dz` at fixed total range, maintaining the same physical medium.
3. For Padé, compare degrees and inspect the relevant spectral interval.
4. Tighten the linear tolerance to confirm that inner solves are sufficiently
   accurate; check directional derivatives separately for inverse problems.

Reducing `dz` alone does not eliminate the finite-degree root error. Near the
cutoff `H=0`, far into the evanescent spectrum, or for large `k*dz`, the Padé
approximation can be inaccurate or ill-conditioned. Large phase can also
overflow coefficients or underflow their exponential prefactor.

Run `python examples/implicit_propagation.py` for a small-grid comparison with
independent dense matrix references. It writes `implicit_propagation.png`.
CN is checked against a matrix exponential; Padé against an eigendecomposition
of the full discrete Helmholtz matrix. This isolates propagation error from
spatial discretization error. Its positive-spectrum accuracy does not establish
accuracy near cutoff or for arbitrary evanescent modes.

### Inspect the rational multiplier

This standalone diagnostic evaluates normalized eigenvalues `lambda(H)/k**2`.
Use an interval appropriate to your operator, not merely the values below.

```python
import jax.numpy as jnp
import paraxial_wave_solver as pws
from paraxial_wave_solver.src.rational import pade_coefficients, pade_multiplier

pws.enable_x64()
phase, rotation = 0.1, float(jnp.pi/6)
eigenvalues = jnp.array([-1.0, -0.5, -0.1, 0.2, 0.5, 1.0, 1.5])
num, den = pade_coefficients(phase, order=6, rotation=rotation)
approximate = pade_multiplier(eigenvalues, num, den, rotation)
exact = jnp.exp(1j*phase*(jnp.sqrt(eigenvalues.astype(complex))-1))
print('Maximum multiplier error:', float(jnp.max(jnp.abs(approximate-exact))))
print('Evanescent magnitudes:', jnp.abs(approximate[:3]))
```

Coefficients have ascending powers of `Y = exp(-i*rotation)*H/k**2-I`.
`phase` must be nonzero; standalone coefficient construction does not validate
a traced phase or enforce double precision. A scalar spectral diagnostic does
not establish stability for a nonnormal operator.

## Linear solves and failure behavior

Both methods use GMRES with an FFT inverse of a periodic homogeneous reference
as preconditioner. For FD this uses the symbol of the selected stencil.

| Setting | Default | Meaning |
|---|---|---|
| `linear_rtol` | `1e-6` | Required unpreconditioned relative residual; strictly between 0 and 1 |
| `linear_restart` | `20` | Positive Krylov dimension, capped internally by the field size |
| `linear_maxiter` | `50` | Positive maximum number of restart cycles, not individual iterations |
| `pade_order` | `4` | Numerator and denominator degree; integer 1–6 |
| `pade_rotation` | `pi/6` | Rotation in radians; strictly between 0 and pi |

`solve` still returns only `(final_field, history)`. A step that fails its
residual check returns NaNs. Padé also returns NaNs for complex callback values,
nonfinite coefficients, or a completely underflowed numerator. Unsupported
static configurations raise at construction instead. Finite output is
necessary but does not certify rational or physical accuracy.

The lower-level utility exposes the candidate and residual for a supplied
linear system. It does not expose per-step diagnostics from `solve`:

```python
import jax.numpy as jnp
from paraxial_wave_solver.src.linear_solvers import solve_linear_system

rhs = jnp.array([1+1j, 2-1j])
diagonal = jnp.array([2+0.2j, 3-0.1j])
result = solve_linear_system(lambda field: diagonal*field, rhs)
print(result.relative_residual, result.converged)
assert bool(result.converged)
```

For nonzero `rhs`, the checked residual is `norm(A*x-rhs)/norm(rhs)`; for zero
`rhs`, it is the absolute residual. Failure returns the candidate in this
utility, whereas propagation replaces the failed field with NaNs.

## Differentiation and parameter overrides

JAX differentiates the linear system implicitly. Both the forward and transpose
solves must converge for reliable derivatives; the reported residual checks
only the forward solve. Check gradients against central directional differences
and repeat with tighter tolerances. Padé additionally differentiates its
coefficient construction, so ill-conditioned coefficients can impair gradients.

Use `solver.build_operators({'k0': ..., 'n0': ..., 'dz': ...})` and pass the
result as `operators=` to differentiate these constants. Overrides affect both
the kernels and propagation coordinates. They must remain physically valid;
the builder checks key names but does not revalidate traced values. Overriding
`dz` changes the final distance, and callbacks that index sampled media must
use the sampling coordinates of that data. Kerr's `n2` is not an operator
override. For the shared API and complex-gradient convention, see the
[README differentiation section](../README.md#differentiation).
