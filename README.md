# Paraxial Wave Solver

[![CI](https://github.com/Forgotten/paraxial_wave_solver/actions/workflows/ci.yml/badge.svg)](https://github.com/Forgotten/paraxial_wave_solver/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Forgotten/paraxial_wave_solver/blob/main/examples/demo.ipynb)

A JAX-based numerical solver for the paraxial wave equation in 3D, with examples
for optical beam propagation in vacuum and in inhomogeneous media such as
atmospheric or underwater turbulence.

## Features

- **Modular design.** Simulation, solver and PML settings are frozen dataclasses,
  validated on construction.
- **Fast.** The propagation loop is a JIT-compiled `jax.lax.scan`, and every
  z-independent operator is built once rather than per step.
- **Multiple solvers.** Pseudo-spectral (split-step Fourier or RK4) and finite
  difference at 2nd, 4th and 6th order, plus a compact isotropic 9-point stencil.
- **Absorbing boundaries.** Perfectly matched layers, as an absorbing potential
  or via complex coordinate stretching.
- **Analytical beams.** Gaussian, Laguerre-Gaussian and Hermite-Gaussian modes,
  branchless in `z`, so they can be `jit`-ed and `vmap`-ed over propagation
  distance.
- **Beyond the linear paraxial problem.** Optional Kerr nonlinearity, complex
  refractive index for absorption and gain, a wide-angle propagator, 4th-order
  splitting and 2/3-rule dealiasing — all opt-in, with defaults unchanged.
- **Diagnostics.** Power, centroid, D4-sigma widths, M², Strehl, mode overlap,
  scintillation and encircled power — computable inside the propagation loop.
- **Monte-Carlo ensembles.** `ensemble_solve` propagates many realizations of a
  random medium at once, vectorized within chunks so that peak memory is set by
  the chunk rather than by the ensemble, with ensemble statistics and bootstrap
  error bars alongside.
- **Differentiable.** `jax.grad` runs through `solve` end to end, with
  square-root checkpointing, so that reverse mode remains tractable at large
  `nz`.
- **Typed and linted.** Type hints throughout; `ruff` clean.

## Installation

```bash
git clone https://github.com/Forgotten/paraxial_wave_solver.git
cd paraxial_wave_solver
pip install -e .
```

The examples assume an editable install. For the test and lint tooling:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import paraxial_wave_solver as pws

sim_config = pws.SimulationConfig(
    nx=256, ny=256, dx=0.5, dy=0.5, dz=1.0, nz=100, wavelength=1.0
)
solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
pml_config = pws.PMLConfig(width_x=20, width_y=20, strength=2.0)

# Vacuum: omit delta_n_fn entirely.
solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config)

psi_0 = pws.gaussian_beam(sim_config, w0=10.0)
psi_final, psi_history = solver.solve(psi_0)
```

`psi_final` is the envelope at `z_0 + nz * dz`. `psi_history[j]` is the envelope
at `z_0 + j * save_every * dz`, so index `0` is `psi_0` itself.

## The equation being solved

Every method in this package integrates the same quantity: the slowly varying
**envelope** $\psi$, not the physical field. This section states exactly what
$\psi$ is and exactly what equation it obeys, because both conventions below
are frequently misapplied and neither can be inferred from the API.

### Derivation

Start from the scalar Helmholtz equation for a monochromatic field $E$, with
$k_0 = 2\pi/\lambda$ the vacuum wavenumber and $n(x,y,z)$ the refractive index:

```math
\nabla^2 E + k_0^2\, n^2(x,y,z)\, E = 0
```

Factor out the fast carrier along the propagation axis,

```math
E(x,y,z) = \psi(x,y,z)\, e^{i k z}, \qquad k = k_0 n_0
```

Substituting and cancelling the carrier gives an equation that is **still
exact**:

```math
\frac{\partial^2 \psi}{\partial z^2}
  + 2 i k \frac{\partial \psi}{\partial z}
  + \nabla_\perp^2 \psi
  + k_0^2 \left( n^2 - n_0^2 \right) \psi = 0
```

Two approximations turn this into what the solver integrates.

**1. The paraxial (slowly varying envelope) approximation** drops the second
$z$-derivative, on the grounds that the envelope changes little over a
wavelength:

```math
\left| \frac{\partial^2 \psi}{\partial z^2} \right|
  \ll \left| 2 k \frac{\partial \psi}{\partial z} \right|
```

This is the step that makes the problem an initial-value problem in $z$: one
first-order equation marching forward, rather than a boundary-value problem.
It also discards the backward-propagating wave, so there are no reflections
from index structure.

**2. Weak index contrast.** Writing $n = n_0 + \delta n$ with
$\delta n \ll n_0$,

```math
n^2 - n_0^2 = 2 n_0\, \delta n + \delta n^2 \;\approx\; 2 n_0\, \delta n
```

What remains, solved for the $z$-derivative, is the equation this package
integrates:

```math
\boxed{\;\frac{\partial \psi}{\partial z}
  = \frac{i}{2 k_0 n_0} \nabla_\perp^2 \psi
  + i k_0\, \delta n\, \psi \;}
```

### The full equation, with every optional term

```math
\frac{\partial \psi}{\partial z}
  = \underbrace{\frac{i}{2 k_0 n_0} \mathcal{L}_\perp \psi}_{\text{diffraction}}
  + \underbrace{i k_0 \left( \delta n(x,y,z) + n_2 \lvert \psi \rvert^2 \right) \psi}_{\text{refraction and Kerr}}
  - \underbrace{\sigma(x,y)\, \psi}_{\text{PML}}
```

| Term | Meaning | Controlled by |
|---|---|---|
| $\frac{i}{2 k_0 n_0} \mathcal{L}_\perp \psi$ | Diffraction | `method`, `fd_order`, `propagator` |
| $i k_0\thinspace \delta n\thinspace \psi$ | Refraction; complex $\delta n$ gives absorption or gain | `delta_n_fn` |
| $i k_0 n_2 \lvert \psi \rvert^2 \psi$ | Kerr self-phase modulation | `n2` |
| $-\sigma\thinspace \psi$ | PML absorption, non-physical, zero in the interior | `PMLConfig` |

$\mathcal{L}_\perp$ is the transverse Laplacian
$\partial^2/\partial x^2 + \partial^2/\partial y^2$, discretized by the chosen
`method`. Under complex coordinate stretching it becomes

```math
\mathcal{L}_\perp
  = \frac{1}{s_x} \frac{\partial}{\partial x}
    \left( \frac{1}{s_x} \frac{\partial}{\partial x} \right)
  + \frac{1}{s_y} \frac{\partial}{\partial y}
    \left( \frac{1}{s_y} \frac{\partial}{\partial y} \right),
  \qquad s = 1 + i\sigma
```

which is where the absorption lives in that mode, and why $\sigma$ is then not
also applied as a potential.

With no PML and real $\delta n$, the equation is norm-conserving: the total
power $\sum \lvert \psi \rvert^2$ is invariant. That is what
`test_energy_conservation_vacuum` checks.

### Wide-angle: the approximation that is not made

`propagator='wide_angle'` skips approximation 1. Rather than dropping the
second $z$-derivative, it factors Helmholtz into forward- and
backward-travelling parts and keeps the forward one:

```math
\frac{\partial \psi}{\partial z}
  = i \left( \sqrt{k^2 + \nabla_\perp^2} - k \right) \psi
```

The square root of an operator is awkward in general, which is why the
literature reaches for Padé approximants. In Fourier space it is diagonal, so
no approximation is needed:

```math
\widehat{D}_{\text{paraxial}}(h) = \exp\left( -\frac{i h k_\perp^2}{2k} \right),
  \qquad
  \widehat{D}_{\text{wide}}(h) = \exp\left( i h \left( \sqrt{k^2 - k_\perp^2} - k \right) \right)
```

Expanding the root for $k_\perp \ll k$ gives $-k_\perp^2 / 2k$, so the paraxial
operator is the leading term of the wide-angle one. Past the light line
($k_\perp \gt k$) the root turns imaginary and the multiplier decays, which is
the correct treatment of evanescent components.

### What the steppers do with it

**`split_step`** alternates the two halves of the equation, each solved exactly
in its own domain — diffraction as a multiplier in Fourier space, everything
else pointwise in real space. One Strang step is

```math
\psi(z + \Delta z) = \mathcal{N}\!\left( \tfrac{\Delta z}{2} \right)
  \, \mathcal{D}(\Delta z) \,
  \mathcal{N}\!\left( \tfrac{\Delta z}{2} \right) \psi(z)
```

where $\mathcal{D}$ applies $\widehat{D}$ above and

```math
\mathcal{N}(h) = \exp\left[ i k_0 \left( \delta n + n_2 \lvert \psi \rvert^2 \right) h \right]
```

with the PML applied as $e^{-\sigma \Delta z / 2}$ at each end of the full
step. `splitting_order=4` composes three such steps with Yoshida weights
$w_1 = \left( 2 - 2^{1/3} \right)^{-1}$ and $w_0 = 1 - 2 w_1$, the middle one
running backwards.

**`rk4`** applies the classic four-stage Runge–Kutta scheme directly to the
right-hand side above, with $\mathcal{L}_\perp$ a finite-difference stencil or
the spectral Laplacian.

### Two conventions

**The solver propagates an envelope, not the field.** $E = \psi e^{ikz}$ with
$k = 2\pi n_0/\lambda$. Analytical beams return the full field by default; pass
`envelope_only=True` when comparing them against solver output.

**Refractive index is supplied as a perturbation.** `delta_n_fn(z, medium)`
must return $\delta n = n - n_0$, so **vacuum is zero, not one**. Omit
`delta_n_fn` entirely and the solver treats the domain as vacuum and drops the
refraction term altogether. Returning $1$ for vacuum injects a spurious
$e^{i k_0 L_z}$, which remains undetected whenever $k_0 L_z$ is a multiple of
$2\pi$, as it is for `wavelength=1.0` with an integer propagation distance.

## Choosing a solver

Measured on a 256×256 grid, `dz=0.005`, `nz=200`, float32, against the analytical
Gaussian in vacuum:

| `method` | `stepper` | Relative L2 error | Time | Notes |
|---|---|---|---|---|
| `spectral` | `split_step` | 1.7e-05 | **35 ms** | 2nd order in z, unconditionally stable |
| `spectral` | `rk4` | **5.9e-07** | 150 ms | 4th order in z, but z-step limited |
| `finite_difference` | `rk4`, `fd_order=2` | 2.0e-04 | 198 ms | |
| `finite_difference` | `rk4`, `fd_order=4` | 1.9e-06 | 750 ms | |
| `finite_difference` | `rk4`, `fd_order=6` | 1.9e-06 | 2000 ms | |
| `finite_difference` | `rk4`, `compact=True` | 2.6e-04 | 535 ms | 2nd order, more isotropic |

**`spectral` with `split_step` is the recommended default.** It is roughly four
times faster than the next alternative and carries no step-size restriction.
Use `rk4` where the additional order in `z` outweighs the increased runtime,
and `finite_difference` where a local operator is required, for example to
employ complex coordinate stretching in the PML, which the spectral method
does not admit.

Two combinations are rejected at construction rather than silently substituted:
`split_step` requires `method='spectral'`, and `compact=True` requires
`method='finite_difference'` with `dx == dy`.

### Step-size stability

The paraxial diffraction operator is purely imaginary, so **RK4 is stable only
while `|lambda| * dz` stays within about 2.83**, where `lambda` is the largest
eigenvalue of the discrete Laplacian divided by `2 * k0 * n0`. Exceeding it causes
the solution to diverge. The solver checks this on construction and warns with the
maximum stable `dz`:

```
RuntimeWarning: RK4 step size is above the stability limit: |lambda| * dz = 4.24
exceeds 2.83. The propagation will diverge. Reduce dz below 0.0333, coarsen the
transverse grid, or use method='spectral' with stepper='split_step', which has no
step size restriction.
```

The limit tightens as the grid is refined — it scales as `dx**2` — so a run that
is stable at one resolution may not be at twice the resolution.
`split_step` has no such restriction.

## Optional numerical schemes

Every option below defaults to off, so existing code is unaffected.

### Nonlinear media

Set `n2` on the simulation config to add an intensity-dependent index
`n2 * |psi|**2`, turning the propagation into a nonlinear Schroedinger
equation. Both steppers implement it; the split-step form is a pure phase, so
it conserves power exactly.

```python
sim_config = pws.SimulationConfig(..., n2=0.02)
```

Above the critical power the nonlinear lens overcomes diffraction and the beam
self-focuses. `examples/kerr_self_focusing.py` sweeps through that threshold.

### Absorption and gain

`delta_n_fn` may return complex values. A positive imaginary part absorbs, at
the rate `exp(-2 * k0 * Im(delta_n) * z)` in power; a negative one amplifies.

```python
solver = pws.ParaxialWaveSolver(
    sim_config, solver_config, pml_config,
    lambda z, medium: 0.01j,      # uniform absorption
)
```

### Higher-order splitting

`splitting_order=4` uses a Yoshida composition of three Strang sub-steps, the
middle one running backwards. It costs three times as much per step and
therefore becomes advantageous once the step size is small enough to lie in the
asymptotic regime, beyond which it is substantially more accurate at equal
cost. Measured through a strongly refracting medium over `lz=20`:

| steps | Strang (order 2) | Yoshida (order 4) |
|---|---|---|
| 160 | 3.0e-02 | 1.5e-02 |
| 320 | 7.1e-03 | 2.0e-03 |
| 640 | 1.8e-03 | 1.5e-04 |
| 1280 | 4.4e-04 | 1.2e-05 |

The PML attenuation is applied once per full step rather than inside each
sub-step: a backwards sub-step through a damping term would amplify instead of
absorb. For `splitting_order=2` the two placements coincide exactly.

### Wide-angle propagation

`propagator='wide_angle'` replaces the small-angle operator with the exact
square root, `exp(1j * dz * (sqrt(k**2 - k_perp**2) - k))`. In Fourier space
this is a diagonal multiplier, so the exact root costs the same as the paraxial
form and no Pade approximation is needed. Components beyond the light line get
an imaginary root and decay, which is the correct treatment of evanescent
waves rather than mis-propagating them.

The two agree to `O(theta**4)`, where `theta` is the divergence angle, so the
difference only matters for tightly focused beams:

| `w0` | `theta` | difference |
|---|---|---|
| 0.6 | 0.53 | 1.2e-01 |
| 0.8 | 0.40 | 3.1e-02 |
| 1.2 | 0.27 | 5.3e-03 |
| 1.6 | 0.20 | 1.6e-03 |

### Dealiasing

`dealias=True` applies the 2/3 rule, zeroing the upper third of each transverse
wavenumber axis. A cubic nonlinearity spreads energy to three times its input
wavenumber, and anything past Nyquist folds back onto the grid as spurious
low-frequency structure. The mask folds into the precomputed propagator, so it
costs nothing per step. Enabling it is advisable whenever `n2` is non-zero.

## Usage

### Inhomogeneous media

Pass the medium to `solve` rather than closing over it. The medium then travels
as a traced argument, so one compiled solver serves any number of realizations
instead of recompiling for each:

```python
import jax
import jax.numpy as jnp

delta_n = pws.random_medium(
    sim_config, correlation_length=2.0, strength=0.01,
    key=jax.random.PRNGKey(0),
)

def delta_n_fn(z, medium):
    # floor, not round: the split-step samples at sub-step midpoints, and
    # round-half-to-even on i + 0.5 would visit half the slices twice and
    # the other half never.
    index = jnp.clip(jnp.floor(z / sim_config.dz).astype(int),
                     0, sim_config.nz - 1)
    return medium[:, :, index]

solver = pws.ParaxialWaveSolver(
    sim_config, solver_config, pml_config, delta_n_fn
)
psi_final, _ = solver.solve(psi_0, medium=delta_n, return_history=False)
```

`medium` is an arbitrary pytree, so it can carry more than one array — a volume
together with the absolute `z` at which it starts, for example, which is how
`examples/turbulence_propagation.py` chains chunks through a single solver.

Two volume generators are provided: `random_medium` for a Gaussian correlation
function, and `random_medium_spectral` for a Von Karman turbulence spectrum
parameterised by `Cn2`, and the outer and inner scales `L0` and `l0`.

### Streaming phase screens

A volume costs `nx * ny * nz` — 105 MB at 256×256×400, a gigabyte at
production sizes — and exists only so `delta_n_fn` can index it. `phase_screen`
generates one transverse screen at a time instead, so the medium becomes a
PRNG key:

```python
def delta_n_fn(z, medium):
    step = jnp.floor(z / sim_config.dz).astype(int)
    return pws.phase_screen(
        jax.random.fold_in(medium, step), sim_config,
        correlation_length=1e-3, strength=4.4e-7,
    )

solver.solve(psi_0, medium=jax.random.PRNGKey(0))
```

`fold_in` yields a reproducible screen per step without splitting `nz` keys in
advance, and it is traceable, so the step index may itself be a tracer.

This reproduces a slice of `random_medium` in the transverse plane — same
variance, same correlation length — because integrating the 3D spectrum over
`kz` leaves the same functional form. **It does not reproduce correlation along
z.** `random_medium` filters isotropically in three dimensions, so its
consecutive slices are correlated; independent screens are the thin-screen
limit. Measured lag-one correlation:

| | volume | screens |
|---|---|---|
| transverse | 0.986 | 0.984 |
| longitudinal | 0.983 | **0.000** |

Screens are appropriate when the step already exceeds the longitudinal
correlation length, which is the typical case; generate the volume otherwise.

### Turbulence ensembles

A single propagation through a random medium is one realization and carries no
statistical meaning in isolation: the speckle pattern it produces is a property
of that particular draw of the medium. The quantities of physical interest are
moments over the ensemble, and `ensemble_solve` estimates them by propagating
many realizations together:

```python
keys = jax.random.split(jax.random.PRNGKey(0), 512)
fields, _ = pws.ensemble_solve(solver, psi_0, keys, chunk_size=64)
statistics = pws.ensemble_statistics(fields, sim_config)
```

`fields` has shape `(n_realizations, nx, ny)`. `chunk_size` is a memory control
rather than a physical parameter: the driver applies `vmap` within a chunk and
`lax.map` across chunks, so peak scratch memory is set by the chunk. Measured
as compiled temporary memory on a 64×64 grid at `chunk_size=2`:

| realizations | chunked | plain `vmap` |
|---|---|---|
| 4 | 232 kB | 297 kB |
| 8 | 232 kB | 559 kB |
| 16 | 232 kB | 1083 kB |
| 32 | **232 kB** | **2132 kB** |

The chunked driver is flat in the ensemble size where a plain `vmap` grows
linearly with it, which is what makes ensembles of a realistic size feasible.

`ensemble_statistics` returns the long-exposure intensity, the pointwise and
on-axis scintillation indices, beam wander, the coherence radius, and both the
long- and short-exposure widths. The two widths quantify the central effect of
an ensemble average: wander displaces the instantaneous pattern between
realizations, which the average records as a broadening.

#### How many realizations are required

`ensemble_standard_error` bootstraps the realization axis, which is the only
reliable way to answer this, because the answer depends strongly on the
statistic. Measured relative standard error at 512 realizations:

| statistic | weak | moderate |
|---|---|---|
| beam wander | 3.1% | 3.1% |
| on-axis $\sigma_I^2$ | 6.2% | 8.7% |

Both converge as the inverse square root of the ensemble size, but with
different prefactors: the scintillation index is a ratio of fourth to second
intensity moments and inherits the heavy tail of a near-exponential intensity
distribution. Since the error falls as $N^{-1/2}$, closing that factor of two
to three costs four to nine times as many realizations. The scintillation index
is additionally biased low at small ensemble size, so a short run reports a
value that is wrong rather than one that is visibly noisy.

#### Spatial resolution is a precondition

Scintillation statistics are meaningless when the speckle they describe is not
resolved by the transverse grid. The relevant diagnostic is the measured
coherence radius in units of `dx`; below roughly three cells the scintillation
index is a discretization artefact. Measured at a fixed physical domain, with a
perturbation strong enough to saturate:

| `nx` | `dx` | coherence radius / `dx` | $\sigma_I^2$ |
|---|---|---|---|
| 128 | 0.200 mm | 1.41 | 7.76 |
| 192 | 0.133 mm | 2.00 | 6.30 |
| 256 | 0.100 mm | 2.83 | 5.29 |

The index falls by 32% across that refinement and has not converged at any of
the three resolutions. Verify the coherence radius against `dx` before
reporting a scintillation index; `examples/turbulence_ensemble.py` prints the
ratio for this reason.

### Controlling memory

A full history is `(nz, nx, ny)` complex values, which reaches gigabytes for long
runs — 839 MB for a 512×512 grid with `nz=400`. Store fewer planes, or none:

```python
psi_final, history = solver.solve(psi_0, save_every=10)   # nz // 10 planes
psi_final, _ = solver.solve(psi_0, return_history=False)  # nothing allocated
```

`save_every` must divide `nz`. It reduces what is allocated rather than
allocating everything and slicing. Omitting the history entirely is also the
fastest path, approximately 1.4 times faster than retaining it in full.

### Absorbing boundaries

`PMLConfig(width_x, width_y, strength)` sets the layer depth in grid points and
the peak absorption. The strength requires tuning: too weak and the boundary
transmits, too strong and the layer itself reflects. Interior residual for a beam driven
into the boundary, against a reference domain twice as wide:

| `strength` | 0.0 | 0.5 | 2.0 | **4.0** | 8.0 |
|---|---|---|---|---|---|
| residual | 0.49 | 0.28 | 0.094 | **0.011** | 0.093 |

Set `use_complex_stretching=True` to absorb inside the differential operator
instead of through a damping potential. This applies to the finite difference
solvers only; the spectral method falls back to the absorbing potential.

### Diagnostics

`paraxial_wave_solver.src.diagnostics` provides the standard beam measures, all
pure functions of a field. Moments follow the ISO 11146 definitions: widths are
D4-sigma, and `m_squared` uses the full space-frequency covariance including the
position–wavenumber cross term, so it is a propagation invariant rather than
growing away from the waist.

| Function | Returns |
|---|---|
| `total_power`, `peak_intensity` | Scalars |
| `centroid`, `beam_width`, `m_squared` | `(x, y)` pairs |
| `second_moments` | `(sigma_xx, sigma_yy, sigma_xy)` |
| `rms_radius`, `scintillation_index` | Scalars |
| `strehl_ratio`, `overlap` | Comparison against a reference field |
| `encircled_power` | Power fraction inside a radius |
| `ensemble_scintillation_index` | Pointwise index across a stack of runs |
| `long_exposure_intensity` | Ensemble-averaged intensity |
| `beam_wander` | Spread of the centroid across realizations |
| `mutual_coherence`, `coherence_radius` | Field correlation about the peak |
| `beam_diagnostics` | The common ones, as a dict |
| `ensemble_statistics` | The ensemble ones, as a dict |

Pass any of them to `solve` as `observable_fn` and they are evaluated inside the
loop, so the history holds numbers rather than fields:

```python
def observable(psi, z):
    return pws.beam_diagnostics(psi, sim_config)

psi_final, history = solver.solve(psi_0, observable_fn=observable)
history['m2_x']        # shape (nz // save_every,)
```

`examples/turbulence_diagnostics.py` tracks beam quality through four
turbulence strengths this way, recording 10 kB of metrics in place of 105 MB of
fields.

Two caveats apply. `strehl_ratio` is a ratio of peak values and may therefore
exceed unity for a speckled field in which a random hot spot exceeds the
unaberrated peak; `overlap` is the more robust measure of the fraction of a
mode that survives. The spatial `scintillation_index` is dominated by the dark
background unless the beam fills the grid, so a `mask` selecting the
illuminated region should be supplied.

### Differentiation

`solve` is a pure JAX function, so `jax.grad`, `jax.jvp` and `jax.vmap` work
through it directly — including through the in-loop diagnostics, which makes
any of them usable as an objective:

```python
def objective(phase):
    psi, _ = solver.solve(psi_0 * jnp.exp(1j * phase), return_history=False)
    return -pws.encircled_power(psi, sim_config, radius=1.0)

gradient = jax.grad(objective)(phase)
```

What carries a gradient:

| quantity | how |
|---|---|
| `psi_0` | directly |
| `medium`, and anything inside it | directly — it is an arbitrary pytree |
| parameters closed over by `delta_n_fn` | directly |
| `n0`, `k0`, `dz` | through `solver.build_operators({'n0': ...})`, passed to `solve(operators=...)` |

The diffraction operators are built once when the solver is constructed, from
concrete configuration values. To differentiate with respect to one of those
constants, rebuild the operators from a traced value:

```python
def objective(n0):
    operators = solver.build_operators({'n0': n0})
    psi, _ = solver.solve(psi_0, operators=operators, return_history=False)
    return beam_spread(psi)
```

Constructing an entirely new solver inside the traced function is also valid
and yields the same gradient; `build_operators` is less expensive, since it
reuses the PML profile and omits revalidation at every evaluation.

Two conventions apply. JAX returns the **conjugate** cotangent for a real loss
of a complex input, so the directional derivative pairs as `Re(grad * delta)`.
A *real* `delta_n` enters as a pure phase, so the total power is exactly
insensitive to it; a vanishing gradient is therefore correct rather than an
error. Differentiate instead a quantity on which the medium acts, such as the
beam radius or the encircled power.

### Checkpointing

Reverse mode must retain sufficient state to replay the propagation backwards,
a requirement that is linear in `nz` in the absence of mitigation. `solve` groups steps into
blocks of `sqrt(nz)` and rematerializes each group, so the cost grows as
`sqrt(nz)` instead. Measured on a Kerr run at 64×64, as compiled temporary
memory for one reverse pass:

| `nz` | `checkpoint=False` | `checkpoint=True` | |
|---|---|---|---|
| 100 | 23.0 MB | 2.7 MB | 8.5× |
| 400 | 91.8 MB | 5.3 MB | 17.2× |
| 1600 | 367.1 MB | 10.6 MB | **34.7×** |

The saving grows with `nz`, which is the signature of `sqrt(n)` against `n`:
over that 16× range the unchecked run grows 15.9× and the checkpointed one
3.9×, against the 4.0× that `sqrt` predicts. It costs one additional forward
pass per group during a backward pass and nothing when no gradient is taken,
and is therefore enabled by default. Pass `checkpoint=False` to benchmark
against it.

Per-step rematerialization, the more immediate choice, also reduces the
requirement, by a constant factor of approximately seven, but the scan still
stores the state entering every step, so the growth remains linear. The
grouping is what changes the scaling.

### Buffer donation

`solve(..., donate_psi_0=True)` lets XLA reuse the input buffer for the output.
It is disabled by default because **it destroys `psi_0`**: after the call the
array is deleted and reading it raises. This is consequential in practice,
since the input beam is generally also required for plotting or as the
reference in an `overlap`.

The saving is one field, approximately 2 MB on a 512×512 complex64 grid. It is
not a route to reclaiming the medium: XLA can reuse an input buffer only when
some output has the same shape and dtype, and `solve` returns a field, so an
`(nx, ny, nz)` volume has no counterpart. Donation of such an array is refused
with a warning and has no effect.

For memory reduction, `checkpoint` and `phase_screen` should be preferred; both
save orders of magnitude more.

### Precision

JAX defaults to float32, which limits the achievable accuracy. Switch before
creating any arrays:

```python
pws.enable_x64()
```

For the Laguerre-Gaussian benchmark at 632.8 nm this takes the error from 2.2e-05
to 3.5e-13.

## API reference

### `SimulationConfig`

| Field | Meaning |
|---|---|
| `nx`, `ny` | Transverse grid points |
| `dx`, `dy` | Transverse grid spacing |
| `dz`, `nz` | Propagation step size and number of steps |
| `wavelength` | Vacuum wavelength |
| `n0` | Background refractive index (default `1.0`) |
| `n2` | Kerr coefficient (default `0.0`, the linear problem) |

Derived properties: `k0` (vacuum wavenumber), `k` (`2*pi*n0/wavelength`), and
`lx`, `ly`, `lz` (domain extents).

### `SolverConfig`

| Field | Values | Meaning |
|---|---|---|
| `method` | `'spectral'`, `'finite_difference'` | Spatial discretization |
| `stepper` | `'rk4'`, `'split_step'` | z-propagation scheme |
| `fd_order` | `2`, `4`, `6` | Finite difference accuracy |
| `compact` | `bool` | Isotropic 9-point stencil; requires `dx == dy` |
| `splitting_order` | `2`, `4` | Strang, or a Yoshida composition |
| `propagator` | `'paraxial'`, `'wide_angle'` | Diffraction operator |
| `dealias` | `bool` | Apply the 2/3 rule |

The last three act on the split-step propagator and are rejected with any other
stepper.

### `PMLConfig`

| Field | Meaning |
|---|---|
| `width_x`, `width_y` | Layer depth in grid points; `0` disables |
| `strength` | Peak absorption |
| `order` | Polynomial profile order (default `2`) |
| `use_complex_stretching` | Absorb in the operator rather than as a potential |

### `ParaxialWaveSolver.solve`

```python
solve(psi_0, z_0=0.0, medium=None, save_every=1, return_history=True,
      observable_fn=None, checkpoint=True, operators=None,
      donate_psi_0=False)
```

Returns `(psi_final, psi_history)`. `psi_history` is `None` when
`return_history=False`. The solver takes `nz` steps of size `dz` from `z_0`, so
it finishes at `z_0 + nz * dz`; chaining runs by passing the previous end as the
next `z_0` reproduces one long run exactly.

### `ensemble_solve`

```python
ensemble_solve(solver, psi_0, keys, *, chunk_size=None, return_history=False,
               **solve_kwargs)
```

Returns `(psi_final, history)` as `solve` does, with a leading realization axis
prepended to every array. `return_history` defaults to `False` here, in
contrast to `solve`, because an ensemble of histories is rarely wanted; pass an
`observable_fn` to record diagnostics over `z` instead. `donate_psi_0` is
rejected, since `psi_0` is shared across realizations.

### Beams

`gaussian_beam`, `laguerre_gaussian_beam(p, l)` and
`hermite_gaussian_beam(n, m)` share the arguments `w0`, `z`, `power`, `x0`, `y0`
and `envelope_only`. `gaussian_beam` is peak-normalized to 1; the LG and HG
modes are normalized to unit total power. Passing `power` rescales any of them.

## Examples

Run these after `pip install -e .`:

| Script | What it does |
|---|---|
| `simple_beam.py` | Gaussian beam in vacuum against the analytical mode |
| `hermite_gaussian.py` | HG₁₁ mode |
| `laguerre_gaussian.py` | Four LG modes at 632.8 nm, in float64 |
| `aperture_diffraction.py` | Square aperture against the Fresnel integrals |
| `random_media.py` | Propagation through a random medium |
| `pml_comparison.py` | Absorbing layer versus coordinate stretching |
| `turbulence_propagation.py` | LG superposition through Von Karman turbulence |
| `kerr_self_focusing.py` | Self-focusing, dealiasing, and 2nd vs 4th order splitting |
| `turbulence_diagnostics.py` | Beam quality vs turbulence strength, measured in-loop |
| `adjoint_sensitivity.py` | Sensitivity maps and phase-mask inverse design, via `jax.grad` |
| `turbulence_ensemble.py` | Monte-Carlo ensemble statistics with bootstrap error bars |
| `demo.ipynb` | Notebook walkthrough of both solver families |
| `turbulence_ensemble.ipynb` | Derivation and demonstration of ensemble statistics |

```bash
python examples/simple_beam.py
```

## Project structure

```
paraxial_wave_solver/
├── paraxial_wave_solver/
│   ├── __init__.py         # Public API
│   ├── src/
│   │   ├── config.py       # Configuration dataclasses and conventions
│   │   ├── operators.py    # Laplacian operators
│   │   ├── pml.py          # PML profile generation
│   │   ├── diagnostics.py  # Beam quality and ensemble measures
│   │   ├── ensemble.py     # Chunked Monte-Carlo driver and bootstrap
│   │   ├── solvers.py      # Propagation kernels and the solver
│   │   └── utils.py        # Analytical beams and random media
│   └── tests/
│       ├── conftest.py
│       ├── test_beams.py
│       ├── test_diagnostics.py
│       ├── test_ensemble.py
│       ├── test_gradients.py
│       ├── test_media.py
│       ├── test_numerics.py
│       ├── test_operators.py
│       └── test_solvers.py
├── examples/
│   ├── _plotting.py        # Shared figure helpers
│   └── ...                 # See the table above
├── pyproject.toml          # Project metadata, pytest and ruff config
├── requirements.txt        # Dependency list, mirrors pyproject
└── LICENSE
```

## Testing

```bash
pytest
```

The suite is order-independent and may therefore be parallelised, as CI does:

```bash
pytest -n auto
```

Linting is configured in `pyproject.toml`:

```bash
ruff check .
```

### Continuous integration

`.github/workflows/ci.yml` runs `ruff` and then the suite on Python 3.10
through 3.13. The matrix is deliberately wide: the dependency pins have no
upper bound, so each interpreter resolves to a different JAX version, 3.10 to
0.6.2 and 3.12 and 3.13 to 0.11.1, and the workflow logs `pip list` so that
the resolved version is always recorded.

## License

MIT — see [LICENSE](LICENSE).
