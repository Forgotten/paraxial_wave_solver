# Paraxial Wave Solver

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
- **Typed and linted.** Type hints throughout; `ruff` clean.

## Installation

```bash
git clone https://github.com/Forgotten/paraxial_wave_solver.git
cd paraxial_wave_solver
pip install -e .
```

An editable install is what the examples expect. For the test and lint tooling:

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
are easy to get wrong and neither is guessable from the API.

### Derivation

Start from the scalar Helmholtz equation for a monochromatic field $E$, with
$k_0 = 2\pi/\lambda$ the vacuum wavenumber and $n(x,y,z)$ the refractive index:

$$
\nabla^2 E + k_0^2\, n^2(x,y,z)\, E = 0
$$

Factor out the fast carrier along the propagation axis,

$$
E(x,y,z) = \psi(x,y,z)\, e^{i k z}, \qquad k = k_0 n_0
$$

Substituting and cancelling the carrier gives an equation that is **still
exact**:

$$
\frac{\partial^2 \psi}{\partial z^2}
  + 2 i k \frac{\partial \psi}{\partial z}
  + \nabla_\perp^2 \psi
  + k_0^2 \left( n^2 - n_0^2 \right) \psi = 0
$$

Two approximations turn this into what the solver integrates.

**1. The paraxial (slowly varying envelope) approximation** drops the second
$z$-derivative, on the grounds that the envelope changes little over a
wavelength:

$$
\left| \frac{\partial^2 \psi}{\partial z^2} \right|
  \ll \left| 2 k \frac{\partial \psi}{\partial z} \right|
$$

This is the step that makes the problem an initial-value problem in $z$: one
first-order equation marching forward, rather than a boundary-value problem.
It also discards the backward-propagating wave, so there are no reflections
from index structure.

**2. Weak index contrast.** Writing $n = n_0 + \delta n$ with
$\delta n \ll n_0$,

$$
n^2 - n_0^2 = 2 n_0\, \delta n + \delta n^2 \;\approx\; 2 n_0\, \delta n
$$

What remains, solved for the $z$-derivative, is the equation this package
integrates:

$$
\boxed{\;\frac{\partial \psi}{\partial z}
  = \frac{i}{2 k_0 n_0} \nabla_\perp^2 \psi
  + i k_0\, \delta n\, \psi \;}
$$

### The full equation, with every optional term

$$
\frac{\partial \psi}{\partial z}
  = \underbrace{\frac{i}{2 k_0 n_0} \mathcal{L}_\perp \psi}_{\text{diffraction}}
  + \underbrace{i k_0 \left( \delta n(x,y,z) + n_2 \lvert \psi \rvert^2 \right) \psi}_{\text{refraction and Kerr}}
  - \underbrace{\sigma(x,y)\, \psi}_{\text{PML}}
$$

| Term | Meaning | Controlled by |
|---|---|---|
| $\frac{i}{2 k_0 n_0} \mathcal{L}_\perp \psi$ | Diffraction | `method`, `fd_order`, `propagator` |
| $i k_0\, \delta n\, \psi$ | Refraction; complex $\delta n$ gives absorption or gain | `delta_n_fn` |
| $i k_0 n_2 \lvert \psi \rvert^2 \psi$ | Kerr self-phase modulation | `n2` |
| $-\sigma\, \psi$ | PML absorption, non-physical, zero in the interior | `PMLConfig` |

$\mathcal{L}_\perp$ is the transverse Laplacian
$\partial^2/\partial x^2 + \partial^2/\partial y^2$, discretized by the chosen
`method`. Under complex coordinate stretching it becomes

$$
\mathcal{L}_\perp
  = \frac{1}{s_x} \frac{\partial}{\partial x}
    \left( \frac{1}{s_x} \frac{\partial}{\partial x} \right)
  + \frac{1}{s_y} \frac{\partial}{\partial y}
    \left( \frac{1}{s_y} \frac{\partial}{\partial y} \right),
  \qquad s = 1 + i\sigma
$$

which is where the absorption lives in that mode, and why $\sigma$ is then not
also applied as a potential.

With no PML and real $\delta n$, the equation is norm-conserving: the total
power $\sum \lvert \psi \rvert^2$ is invariant. That is what
`test_energy_conservation_vacuum` checks.

### Wide-angle: the approximation that is not made

`propagator='wide_angle'` skips approximation 1. Rather than dropping the
second $z$-derivative, it factors Helmholtz into forward- and
backward-travelling parts and keeps the forward one:

$$
\frac{\partial \psi}{\partial z}
  = i \left( \sqrt{k^2 + \nabla_\perp^2} - k \right) \psi
$$

The square root of an operator is awkward in general, which is why the
literature reaches for Padé approximants. In Fourier space it is diagonal, so
no approximation is needed:

$$
\widehat{D}_{\text{paraxial}}(h) = \exp\left( -\frac{i h k_\perp^2}{2k} \right),
  \qquad
  \widehat{D}_{\text{wide}}(h) = \exp\left( i h \left( \sqrt{k^2 - k_\perp^2} - k \right) \right)
$$

Expanding the root for $k_\perp \ll k$ gives $-k_\perp^2 / 2k$, so the paraxial
operator is the leading term of the wide-angle one. Past the light line
($k_\perp > k$) the root turns imaginary and the multiplier decays, which is
the correct treatment of evanescent components.

### What the steppers do with it

**`split_step`** alternates the two halves of the equation, each solved exactly
in its own domain — diffraction as a multiplier in Fourier space, everything
else pointwise in real space. One Strang step is

$$
\psi(z + \Delta z) = \mathcal{N}\!\left( \tfrac{\Delta z}{2} \right)
  \, \mathcal{D}(\Delta z) \,
  \mathcal{N}\!\left( \tfrac{\Delta z}{2} \right) \psi(z)
$$

where $\mathcal{D}$ applies $\widehat{D}$ above and

$$
\mathcal{N}(h) = \exp\left[ i k_0 \left( \delta n + n_2 \lvert \psi \rvert^2 \right) h \right]
$$

with the PML applied as $e^{-\sigma \Delta z / 2}$ at each end of the full
step. `splitting_order=4` composes three such steps with Yoshida weights
$w_1 = \left( 2 - 2^{1/3} \right)^{-1}$ and $w_0 = 1 - 2 w_1$, the middle one
running backwards.

**`rk4`** applies the classic four-stage Runge–Kutta scheme directly to the
right-hand side above, with $\mathcal{L}_\perp$ a finite-difference stencil or
the spectral Laplacian.

### Two conventions to keep straight

**The solver propagates an envelope, not the field.** $E = \psi e^{ikz}$ with
$k = 2\pi n_0/\lambda$. Analytical beams return the full field by default; pass
`envelope_only=True` when comparing them against solver output.

**Refractive index is supplied as a perturbation.** `delta_n_fn(z, medium)`
must return $\delta n = n - n_0$, so **vacuum is zero, not one**. Omit
`delta_n_fn` entirely and the solver treats the domain as vacuum and drops the
refraction term altogether. Returning $1$ for vacuum injects a spurious
$e^{i k_0 L_z}$, which stays invisible whenever $k_0 L_z$ happens to be a
multiple of $2\pi$ — as it is for `wavelength=1.0` with an integer propagation
distance.

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

**Use `spectral` + `split_step` unless you have a reason not to.** It is roughly
four times faster than the next option and has no step-size restriction. Reach
for `rk4` when the extra order in `z` matters more than the runtime, and for
`finite_difference` when you need a local operator — for instance to use complex
coordinate stretching in the PML, which the spectral method cannot.

Two combinations are rejected at construction rather than silently substituted:
`split_step` requires `method='spectral'`, and `compact=True` requires
`method='finite_difference'` with `dx == dy`.

### Step-size stability

The paraxial diffraction operator is purely imaginary, so **RK4 is stable only
while `|lambda| * dz` stays within about 2.83**, where `lambda` is the largest
eigenvalue of the discrete Laplacian divided by `2 * k0 * n0`. Exceeding it makes
the solution diverge. The solver checks this on construction and warns with the
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

Above the critical power the nonlinear lens beats diffraction and the beam
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
middle one running backwards. It costs three times as much per step, so it pays
off once the step size is small enough to be in the asymptotic regime — past
that point it wins comfortably at equal cost. Measured through a strongly
refracting medium over `lz=20`:

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
costs nothing per step. Worth enabling whenever `n2` is non-zero.

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
    index = jnp.clip(jnp.round(z / sim_config.dz).astype(int),
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

Two generators are provided: `random_medium` for a Gaussian correlation
function, and `random_medium_spectral` for a Von Karman turbulence spectrum
parameterised by `Cn2`, and the outer and inner scales `L0` and `l0`.

### Controlling memory

A full history is `(nz, nx, ny)` complex values, which reaches gigabytes for long
runs — 839 MB for a 512×512 grid with `nz=400`. Store fewer planes, or none:

```python
psi_final, history = solver.solve(psi_0, save_every=10)   # nz // 10 planes
psi_final, _ = solver.solve(psi_0, return_history=False)  # nothing allocated
```

`save_every` must divide `nz`. It reduces what is allocated rather than
allocating everything and slicing, and skipping the history entirely is also the
fastest path — about 1.4× quicker than keeping all of it.

### Absorbing boundaries

`PMLConfig(width_x, width_y, strength)` sets the layer depth in grid points and
the peak absorption. Strength wants tuning: too weak and the boundary transmits,
too strong and the layer itself reflects. Interior residual for a beam driven
into the boundary, against a reference domain twice as wide:

| `strength` | 0.0 | 0.5 | 2.0 | **4.0** | 8.0 |
|---|---|---|---|---|---|
| residual | 0.49 | 0.28 | 0.094 | **0.011** | 0.093 |

Set `use_complex_stretching=True` to absorb inside the differential operator
instead of through a damping potential. This applies to the finite difference
solvers only; the spectral method falls back to the absorbing potential.

### Diagnostics

`paraxial_wave_solver.src.diagnostics` provides the usual beam measures, all
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
| `beam_diagnostics` | The common ones, as a dict |

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

Two caveats worth knowing. `strehl_ratio` is a ratio of peaks, so it can exceed
1 for a speckled field where a random hot spot beats the unaberrated peak;
`overlap` is the robust choice for tracking how much of a mode survives. And
the spatial `scintillation_index` is dominated by dark background unless the
beam fills the grid — pass a `mask` selecting the illuminated region.

### Precision

JAX defaults to float32, which caps the achievable accuracy. Switch before
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
      observable_fn=None)
```

Returns `(psi_final, psi_history)`. `psi_history` is `None` when
`return_history=False`. The solver takes `nz` steps of size `dz` from `z_0`, so
it finishes at `z_0 + nz * dz`; chaining runs by passing the previous end as the
next `z_0` reproduces one long run exactly.

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
| `demo.ipynb` | Notebook walkthrough of both solver families |

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
│   │   ├── diagnostics.py  # Beam quality measures
│   │   ├── solvers.py      # Propagation kernels and the solver
│   │   └── utils.py        # Analytical beams and random media
│   └── tests/
│       ├── conftest.py
│       ├── test_beams.py
│       ├── test_media.py
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

Linting is configured in `pyproject.toml`:

```bash
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
