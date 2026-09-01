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

## Conventions

Two things are worth knowing before writing any code against this package. Most
mistakes trace back to one of them.

**The solver propagates an envelope, not the field.** The physical field is
`E = psi * exp(1j * k * z)` with `k = 2 * pi * n0 / wavelength`. The solver works
in `psi`. Analytical beams return the full field by default; pass
`envelope_only=True` when comparing them against solver output.

**Refractive index is supplied as a perturbation.** `delta_n_fn(z, medium)` must
return `delta_n = n - n0`, so **vacuum is zero, not one**. Omit `delta_n_fn`
entirely and the solver treats the domain as vacuum and drops the refraction term
altogether. Returning `1.0` for vacuum injects a spurious `exp(1j * k0 * lz)`,
which stays invisible whenever `k0 * lz` happens to be a multiple of `2 * pi` —
as it is for `wavelength=1.0` with an integer propagation distance.

The envelope obeys

```
d(psi)/dz = (1j / (2 * k0 * n0)) * lap_perp(psi) + 1j * k0 * delta_n * psi - sigma * psi
```

where `sigma` is the PML absorption profile.

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
solve(psi_0, z_0=0.0, medium=None, save_every=1, return_history=True)
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
