"""Shared plotting helpers for the example scripts.

The examples were each carrying their own copy of the same six-panel
benchmark figure. They now describe what they simulate and call in here.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt


def _show(ax, field, title, cmap='inferno', colorbar=False, label=None):
  """Draws a transverse field on `ax` with the x-axis horizontal."""
  image = ax.imshow(jnp.asarray(field).T, origin='lower', cmap=cmap)
  ax.set_title(title)
  ax.axis('off')
  if colorbar:
    plt.colorbar(image, ax=ax, label=label, fraction=0.046)
  return image


def relative_l2_error(psi, reference):
  """Returns ||psi - reference|| / ||reference||."""
  return float(
    jnp.linalg.norm(psi - reference) / jnp.linalg.norm(reference)
  )


def plot_benchmark(
  sim_config,
  psi_0,
  psi_final,
  psi_analytical,
  output_path,
  initial_title='Initial Intensity (z=0)',
  compare='cuts',
):
  """Draws the standard numerical-versus-analytical benchmark figure.

  Panels: the initial intensity, the numerical and analytical intensities at
  the final plane, the absolute error field, and two comparison panels.

  Args:
    sim_config: Simulation configuration, for the transverse axis.
    psi_0: Initial field.
    psi_final: Numerical field at the final plane.
    psi_analytical: Analytical field at the final plane.
    output_path: File to write the figure to.
    initial_title: Title for the first panel.
    compare: 'cuts' for intensity and phase profiles along x, or 'phase' for
             full numerical and analytical phase maps. Phase maps are the
             informative choice for modes carrying orbital angular momentum.

  Returns:
    The relative L2 error between psi_final and psi_analytical.
  """
  z_final = sim_config.lz
  error_field = jnp.abs(psi_final - psi_analytical)
  l2_error = relative_l2_error(psi_final, psi_analytical)

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))

  _show(axes[0, 0], jnp.abs(psi_0), initial_title)
  _show(axes[0, 1], jnp.abs(psi_final), f'Numerical Intensity (z={z_final:g})')
  _show(axes[0, 2], jnp.abs(psi_analytical),
        f'Analytical Intensity (z={z_final:g})')
  _show(axes[1, 0], error_field, f'Absolute Error (rel L2 = {l2_error:.2e})',
        cmap='viridis', colorbar=True, label='|Error|')

  if compare == 'phase':
    _show(axes[1, 1], jnp.angle(psi_final), 'Numerical Phase', cmap='hsv')
    _show(axes[1, 2], jnp.angle(psi_analytical), 'Analytical Phase',
          cmap='hsv')
  else:
    x = jnp.arange(sim_config.nx) * sim_config.dx
    centre = sim_config.ny // 2

    ax = axes[1, 1]
    ax.plot(x, jnp.abs(psi_final[:, centre])**2, 'b-', label='Numerical')
    ax.plot(x, jnp.abs(psi_analytical[:, centre])**2, 'r--',
            label='Analytical')
    ax.set_title('Intensity Profile (x-cut)')
    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(x, jnp.unwrap(jnp.angle(psi_final[:, centre])), 'b-',
            label='Numerical')
    ax.plot(x, jnp.unwrap(jnp.angle(psi_analytical[:, centre])), 'r--',
            label='Analytical')
    ax.set_title('Phase Profile (x-cut)')
    ax.set_xlabel('x')
    ax.set_ylabel('Phase (rad)')
    ax.legend()
    ax.grid(True, alpha=0.3)

  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)
  print(f"Saved benchmark plot to {output_path}")
  return l2_error


def plot_xz_intensity(ax, sim_config, psi_history, title, z_extent=None):
  """Draws an xz slice of the intensity history through the beam centre.

  Args:
    ax: Axis to draw on.
    sim_config: Simulation configuration.
    psi_history: History array of shape (n_saved, nx, ny).
    title: Panel title.
    z_extent: Propagation distance spanned by the history; defaults to lz.
  """
  centre = sim_config.ny // 2
  intensity = jnp.abs(psi_history[:, :, centre].T)**2
  extent = [0, z_extent if z_extent is not None else sim_config.lz,
            0, sim_config.lx]
  image = ax.imshow(intensity, extent=extent, origin='lower', cmap='inferno',
                    aspect='auto')
  plt.colorbar(image, ax=ax, label='Intensity', fraction=0.046)
  ax.set_xlabel('z')
  ax.set_ylabel('x')
  ax.set_title(title)
  return image
