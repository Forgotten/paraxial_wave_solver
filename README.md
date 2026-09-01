# Paraxial Wave Solver

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Forgotten/paraxial_wave_solver/blob/main/examples/demo.ipynb)

A JAX-based numerical solver for the Paraxial Wave Equation in 3D, with included examples for simulating optical beam propagation in vacuum and inhomogeneous media (e.g., turbulence).

## Features

- **Modular Design:** Configurable simulation, solver, and PML settings via Python native dataclasses, validated on construction.
- **High Performance:** JIT-compiled propagation loop using `jax.lax.scan` and `jax.jit`, with every z-independent operator precomputed once.
- **Multiple Solvers:**
  - **Finite Difference:** 2nd, 4th, 6th order, and compact 9-point stencils.
  - **Spectral:** Pseudo-spectral method using FFTs (Split-Step Fourier).
- **Boundary Conditions:** Perfectly Matched Layers (PML), either as an absorbing potential or via complex coordinate stretching.
- **Analytical Beams:** Gaussian, Laguerre-Gaussian and Hermite-Gaussian modes, branchless in `z` so they are `jit`- and `vmap`-able.
- **Type Safety:** Comprehensive type hinting and PEP 8 compliance.

## Conventions

These are the two things worth knowing before writing any code against this
package; most mistakes trace back to one of them.

**The solver propagates an envelope, not the field.** The physical field is
`E = psi * exp(1j * k * z)` with `k = 2 * pi * n0 / wavelength`. The solver
works in `psi`. Analytical beams return the full field by default; pass
`envelope_only=True` to compare them against solver output.

**Refractive index is supplied as a perturbation.** `delta_n_fn(z, medium)`
must return `delta_n = n - n0`, so **vacuum is zero, not one**. Omit
`delta_n_fn` entirely and the solver treats the domain as vacuum and skips the
refraction term. Returning `1.0` for vacuum injects a spurious
`exp(1j * k0 * lz)`, which is invisible whenever `k0 * lz` happens to be a
multiple of `2 * pi`.

The envelope obeys

```
d(psi)/dz = (1j / (2 * k0 * n0)) * lap_perp(psi) + 1j * k0 * delta_n * psi - sigma * psi
```

where `sigma` is the PML absorption profile.

## Project Structure

```
paraxial_wave_solver/
├── paraxial_wave_solver/   # Main package
│   ├── __init__.py         # Exposes API
│   ├── src/                # Source code
│   │   ├── config.py       # Configuration dataclasses and conventions
│   │   ├── operators.py    # Laplacian operators
│   │   ├── pml.py          # PML profile generation
│   │   ├── solvers.py      # Propagation kernels and the solver
│   │   └── utils.py        # Analytical beams and random media
│   └── tests/              # Unit tests
│       ├── conftest.py
│       ├── test_beams.py
│       ├── test_media.py
│       ├── test_operators.py
│       └── test_solvers.py
├── examples/               # Example scripts
│   ├── _plotting.py            # Shared figure helpers
│   ├── demo.ipynb              # Jupyter notebook demo
│   ├── simple_beam.py          # Gaussian beam in vacuum
│   ├── hermite_gaussian.py     # HG_11 mode
│   ├── laguerre_gaussian.py    # LG modes at a HeNe wavelength
│   ├── aperture_diffraction.py # Square aperture vs Fresnel integrals
│   ├── random_media.py         # Random media
│   ├── pml_comparison.py       # Absorbing layer vs coordinate stretching
│   └── turbulence_propagation.py  # Von Karman turbulence
├── pyproject.toml          # Project metadata
├── requirements.txt        # Python dependencies
└── LICENSE                 # MIT License
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Forgotten/paraxial_wave_solver.git
    cd paraxial_wave_solver
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```
    Or install the package in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

### Running Examples

The examples import the installed package, so run `pip install -e .` first.

1.  **Simple Gaussian Beam (Vacuum):** a Gaussian beam in free space, compared
    against the analytical mode. Writes `gaussian_beam_comparison.png`.
    ```bash
    python examples/simple_beam.py
    ```

2.  **Random Media Propagation:** a beam through random refractive index
    fluctuations. Writes `random_media.png`.
    ```bash
    python examples/random_media.py
    ```

3.  **Laguerre-Gaussian Modes:** four LG modes at 632.8 nm, run in float64.
    Writes one benchmark figure per mode.
    ```bash
    python examples/laguerre_gaussian.py
    ```

### Basic Usage Snippet

```python
import jax
import jax.numpy as jnp
import paraxial_wave_solver as pws

# Configure simulation.
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

`psi_history[j]` is the field at `z_0 + j * save_every * dz`, so index 0 is
`psi_0` itself.

### Inhomogeneous media

Pass the medium to `solve` rather than closing over it, so that one compiled
solver serves any number of realizations:

```python
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

### Controlling memory

A full history is `(nz, nx, ny)` complex values, which reaches gigabytes for
long runs. Store fewer planes, or none at all:

```python
psi_final, history = solver.solve(psi_0, save_every=10)      # nz // 10 planes
psi_final, _ = solver.solve(psi_0, return_history=False)     # no allocation
```

### Precision

JAX defaults to float32, which caps achievable accuracy. Switch before
creating any arrays:

```python
pws.enable_x64()
```

## Testing

Run the unit tests using `pytest` to verify the installation and correctness of the solvers:

```bash
pytest
```

Linting is configured in `pyproject.toml`:

```bash
ruff check .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
