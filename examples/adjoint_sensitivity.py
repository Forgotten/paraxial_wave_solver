"""Gradients through the solver: sensitivity maps and inverse design.

Two halves, both of which need a derivative of the output field with respect
to something upstream.

1. A sensitivity volume. `d(on-axis intensity at z=L) / d(delta_n(x, y, z))`
   has the same shape as the medium and comes out of a single reverse pass.
   Finite differences would need one forward solve per voxel - for the grid
   here, over three million of them - which is the argument for the adjoint in
   one sentence.

2. Inverse design. A phase mask at z=0 is optimized to concentrate power into
   a small spot at z=L, by plain gradient descent straight through the
   propagation. The objective is `encircled_power` from the diagnostics
   module, which works unchanged because the diagnostics are pure JAX.

Run after installing the package (`pip install -e .`):

    python examples/adjoint_sensitivity.py
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import paraxial_wave_solver as pws

GRID = dict(nx=128, ny=128, dx=0.15, dy=0.15, dz=0.05, nz=200, wavelength=1.0)
WAIST = 3.0
TARGET_RADIUS = 1.5
DESCENT_STEPS = 150
LEARNING_RATE = 40.0


def build() -> tuple[pws.SimulationConfig, pws.ParaxialWaveSolver, pws.Field]:
  """Returns the configuration, a solver and the unperturbed input beam."""
  sim_config = pws.SimulationConfig(**GRID)
  solver_config = pws.SolverConfig(method='spectral', stepper='split_step')
  pml_config = pws.PMLConfig(width_x=16, width_y=16, strength=3.0)
  solver = pws.ParaxialWaveSolver(sim_config, solver_config, pml_config)
  psi_0 = pws.gaussian_beam(sim_config, w0=WAIST)
  return sim_config, solver, psi_0


def _on_axis_weight(sim_config: pws.SimulationConfig) -> pws.Field:
  """A soft on-axis window, so the objective is smooth rather than one pixel."""
  x = jnp.arange(sim_config.nx) * sim_config.dx - sim_config.lx / 2
  y = jnp.arange(sim_config.ny) * sim_config.dy - sim_config.ly / 2
  r2 = x[:, None]**2 + y[None, :]**2
  return jnp.exp(-r2 / TARGET_RADIUS**2)


# --------------------------------------------------------------------------
# Part A: where in the volume does a perturbation matter?
# --------------------------------------------------------------------------

def sensitivity_volume(
  sim_config: pws.SimulationConfig,
  solver: pws.ParaxialWaveSolver,
  psi_0: pws.Field,
) -> tuple[pws.Field, pws.Field]:
  """Returns d(on-axis power)/d(delta_n) over the whole volume.

  Args:
    sim_config: Simulation configuration.
    solver: A solver whose delta_n_fn indexes into the medium.
    psi_0: Input field.

  Returns:
    A tuple (sensitivity, history) where sensitivity has the shape of the
    medium and history is the saved intensity along the propagation.
  """
  weight = _on_axis_weight(sim_config)

  def on_axis_power(medium):
    psi, _ = solver.solve(psi_0, medium=medium, return_history=False)
    return jnp.sum(jnp.abs(psi)**2 * weight)

  medium = jnp.zeros((sim_config.nx, sim_config.ny, sim_config.nz))
  sensitivity = jax.grad(on_axis_power)(medium)

  _, history = solver.solve(psi_0, medium=medium, save_every=4)
  return sensitivity, history


# --------------------------------------------------------------------------
# Part B: design a phase mask
# --------------------------------------------------------------------------

def optimize_phase_mask(
  sim_config: pws.SimulationConfig,
  solver: pws.ParaxialWaveSolver,
  psi_0: pws.Field,
) -> tuple[pws.Field, list[float]]:
  """Gradient-descends a z=0 phase mask to focus the beam at z=L.

  Args:
    sim_config: Simulation configuration.
    solver: Solver to propagate with.
    psi_0: Input field the mask is applied to.

  Returns:
    A tuple (phase, history) with the optimized mask and the fraction of
    power inside the target radius at each iteration.
  """
  def enclosed(phase):
    psi, _ = solver.solve(
      psi_0 * jnp.exp(1j * phase), return_history=False
    )
    return pws.encircled_power(psi, sim_config, radius=TARGET_RADIUS)

  value_and_grad = jax.jit(jax.value_and_grad(enclosed))
  phase = jnp.zeros((sim_config.nx, sim_config.ny))
  history = []
  for step in range(DESCENT_STEPS):
    value, gradient = value_and_grad(phase)
    history.append(float(value))
    # Ascent: the objective is a power fraction to be maximized.
    phase = phase + LEARNING_RATE * gradient
    if step % 30 == 0:
      print(f"    step {step:4d}   encircled power {float(value):.4f}")
  return phase, history


def main():
  pws.enable_x64()
  sim_config, solver, psi_0 = build()

  # A solver that reads the medium slice by slice, for part A.
  def delta_n_fn(z, medium):
    index = jnp.clip(
      jnp.floor(z / sim_config.dz).astype(int), 0, sim_config.nz - 1
    )
    return medium[:, :, index]

  medium_solver = pws.ParaxialWaveSolver(
    sim_config,
    pws.SolverConfig(method='spectral', stepper='split_step'),
    pws.PMLConfig(width_x=16, width_y=16, strength=3.0),
    delta_n_fn,
  )

  print("Part A: sensitivity of the on-axis power to the medium...")
  voxels = sim_config.nx * sim_config.ny * sim_config.nz
  print(f"  one reverse pass returns all {voxels:,} partial derivatives;")
  print(f"  finite differences would need {2 * voxels:,} forward solves.")
  sensitivity, history = sensitivity_volume(sim_config, medium_solver, psi_0)
  per_plane = jnp.linalg.norm(
    sensitivity.reshape(-1, sim_config.nz), axis=0
  )
  peak_plane = int(jnp.argmax(per_plane))
  print(f"  sensitivity peaks at step {peak_plane} of {sim_config.nz} "
        f"(z = {peak_plane * sim_config.dz:.2f} of {sim_config.lz:.2f})")

  print("\nPart B: optimizing a phase mask...")
  before = float(
    pws.encircled_power(
      solver.solve(psi_0, return_history=False)[0],
      sim_config, radius=TARGET_RADIUS,
    )
  )
  phase, descent = optimize_phase_mask(sim_config, solver, psi_0)
  focused, _ = solver.solve(
    psi_0 * jnp.exp(1j * phase), return_history=False
  )
  after = float(
    pws.encircled_power(focused, sim_config, radius=TARGET_RADIUS)
  )
  print(f"  encircled power {before:.4f} -> {after:.4f} "
        f"({after / before:.2f}x)")

  # ---------------- figure ----------------
  centre = sim_config.ny // 2
  fig, axes = plt.subplots(2, 3, figsize=(16, 9))

  ax = axes[0, 0]
  image = ax.imshow(
    jnp.abs(history[:, :, centre].T)**2, origin='lower', cmap='inferno',
    aspect='auto', extent=[0, sim_config.lz, 0, sim_config.lx],
  )
  plt.colorbar(image, ax=ax, fraction=0.046)
  ax.set_title('Beam intensity (xz)')
  ax.set_xlabel('z')
  ax.set_ylabel('x')

  ax = axes[0, 1]
  limit = float(jnp.abs(sensitivity).max())
  image = ax.imshow(
    sensitivity[:, centre, :], origin='lower', cmap='RdBu_r',
    aspect='auto', vmin=-limit, vmax=limit,
    extent=[0, sim_config.lz, 0, sim_config.lx],
  )
  plt.colorbar(image, ax=ax, fraction=0.046)
  ax.set_title(r'Sensitivity $\partial P_{axis} / \partial\, \delta n$ (xz)')
  ax.set_xlabel('z')
  ax.set_ylabel('x')

  ax = axes[0, 2]
  z = jnp.arange(sim_config.nz) * sim_config.dz
  ax.plot(z, per_plane, 'k-')
  ax.axvline(peak_plane * sim_config.dz, color='tab:red', ls=':',
             label=f'peak at z={peak_plane * sim_config.dz:.2f}')
  ax.set_title('Sensitivity magnitude per plane')
  ax.set_xlabel('z')
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)

  ax = axes[1, 0]
  image = ax.imshow(phase.T, origin='lower', cmap='twilight')
  plt.colorbar(image, ax=ax, fraction=0.046)
  ax.set_title('Optimized phase mask at z=0')
  ax.axis('off')

  ax = axes[1, 1]
  unfocused, _ = solver.solve(psi_0, return_history=False)
  x = jnp.arange(sim_config.nx) * sim_config.dx
  ax.plot(x, jnp.abs(unfocused[:, centre])**2, 'b-', label='no mask')
  ax.plot(x, jnp.abs(focused[:, centre])**2, 'r-', label='optimized')
  ax.set_title(f'Intensity at z={sim_config.lz:g} (x-cut)')
  ax.set_xlabel('x')
  ax.legend()
  ax.grid(True, alpha=0.3)

  ax = axes[1, 2]
  ax.plot(descent, 'k-')
  ax.set_title('Gradient ascent on encircled power')
  ax.set_xlabel('iteration')
  ax.set_ylabel(f'power within r={TARGET_RADIUS}')
  ax.grid(True, alpha=0.3)

  fig.tight_layout()
  fig.savefig('adjoint_sensitivity.png')
  print("\nSaved figure to adjoint_sensitivity.png")


if __name__ == "__main__":
  main()
