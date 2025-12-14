# Paraxial Wave Solver

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Forgotten/paraxial_wave_solver/blob/main/examples/demo.ipynb)

A JAX-based numerical solver for the Paraxial Wave Equation in 3D, with included examples for simulating optical beam propagation in vacuum and inhomogeneous media (e.g., turbulence).

## Features

- **Modular Design:** Configurable simulation, solver, and PML settings via Python native dataclasses.
- **High Performance:** JIT-compiled propagation loop using `jax.lax.scan` and `jax.jit`.
- **Multiple Solvers:**
  - **Finite Difference:** 2nd, 4th, 6th order, and compact 9-point stencils.
  - **Spectral:** Pseudo-spectral method using FFTs (Split-Step Fourier).
- **Boundary Conditions:** Perfectly Matched Layers (PML) to absorb outgoing waves.
- **Type Safety:** Comprehensive type hinting and PEP 8 compliance.

## Project Structure

```
paraxial_wave_solver/
├── paraxial_wave_solver/   # Main package
│   ├── __init__.py         # Exposes API
│   ├── src/                # Source code
│   │   ├── config.py       # Configuration dataclasses
│   │   ├── operators.py    # Laplacian operators
│   │   ├── pml.py          # PML profile generation
│   │   ├── solvers.py      # Solver logic
│   │   └── utils.py        # Utilities
│   └── tests/              # Unit tests
│       ├── test_operators.py
│       └── test_solvers.py
├── examples/               # Example scripts
│   ├── demo.ipynb          # Jupyter notebook demo
│   ├── simple_beam.py      # Gaussian beam example
│   └── random_media.py     # Random media example
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

Two example scripts are provided to demonstrate the solver's capabilities.

1.  **Simple Gaussian Beam (Vacuum):**
    Simulates a Gaussian beam propagating in free space.
    ```bash
    python examples/simple_beam.py
    ```
    This will generate `gaussian_beam_xz.png`.

2.  **Random Media Propagation:**
    Simulates a beam propagating through a medium with random refractive index fluctuations.
    ```bash
    python examples/random_media.py
    ```
    This will generate `random_media_xz.png`.

### Basic Usage Snippet

```python
import jax.numpy as jnp
import paraxial_wave_solver as pws

# Configure simulation.
sim_config = pws.SimulationConfig(
    nx=256, ny=256, dx=0.5, dy=0.5, dz=1.0, nz=100, wavelength=1.0
)
solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
pml_config = pws.PMLConfig(width_x=20, width_y=20, strength=2.0)

# Define refractive index (vacuum).
def n_ref_fn(z):
    return jnp.ones((sim_config.nx, sim_config.ny))

# Initialize solver.
solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config, n_ref_fn)

# Create initial condition.
psi_0 = pws.gaussian_beam(sim_config, w0=10.0)

# Run simulation.
psi_final, psi_history = solver.solve(psi_0)
```

## Testing

Run the unit tests using `pytest` to verify the installation and correctness of the solvers:

```bash
pytest tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
