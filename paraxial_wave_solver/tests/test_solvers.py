import jax
import jax.numpy as jnp
import pytest

from paraxial_wave_solver.src.config import (
  PMLConfig,
  SimulationConfig,
  SolverConfig,
)
from paraxial_wave_solver.src.solvers import ParaxialWaveSolver, propagate
from paraxial_wave_solver.src.utils import gaussian_beam

NO_PML = PMLConfig(width_x=0, width_y=0, strength=0.0)


def _centered_gaussian(sim_config, width=1.0):
  x = jnp.arange(sim_config.nx) * sim_config.dx
  y = jnp.arange(sim_config.ny) * sim_config.dy
  X, Y = jnp.meshgrid(x, y, indexing='ij')
  r2 = (X - sim_config.lx / 2)**2 + (Y - sim_config.ly / 2)**2
  return jnp.exp(-r2 / width**2).astype(complex)


# --------------------------------------------------------------------------
# Conservation and basic behaviour
# --------------------------------------------------------------------------

def test_energy_conservation_vacuum():
  """Split-step propagation in vacuum without a PML is unitary."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.1, dy=0.1, dz=0.1, nz=10, wavelength=1.0
  )
  solver_config = SolverConfig(method='spectral', stepper='split_step')
  psi_0 = _centered_gaussian(sim_config)

  solver = ParaxialWaveSolver(sim_config, solver_config, NO_PML)
  psi_final, _ = solver.solve(psi_0)

  e_in = jnp.sum(jnp.abs(psi_0)**2)
  e_out = jnp.sum(jnp.abs(psi_final)**2)
  assert jnp.abs(e_in - e_out) / e_in < 1e-6


def test_energy_conservation_finite_difference():
  """The FD + RK4 path also conserves energy in vacuum without a PML."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.02, nz=50, wavelength=1.0
  )
  solver_config = SolverConfig(method='finite_difference', fd_order=4,
                               stepper='rk4')
  psi_0 = _centered_gaussian(sim_config, width=2.0)

  solver = ParaxialWaveSolver(sim_config, solver_config, NO_PML)
  psi_final, _ = solver.solve(psi_0)

  e_in = jnp.sum(jnp.abs(psi_0)**2)
  e_out = jnp.sum(jnp.abs(psi_final)**2)
  assert jnp.abs(e_in - e_out) / e_in < 1e-5


def test_jit_consistency():
  """The JIT-compiled solver matches the same computation with JIT disabled."""
  sim_config = SimulationConfig(
    nx=32, ny=32, dx=0.2, dy=0.2, dz=0.02, nz=5, wavelength=1.0
  )
  solver_config = SolverConfig(method='finite_difference', fd_order=2,
                               stepper='rk4')
  psi_0 = _centered_gaussian(sim_config, width=2.0)

  solver = ParaxialWaveSolver(sim_config, solver_config, NO_PML)
  psi_jit, _ = solver.solve(psi_0)
  with jax.disable_jit():
    psi_no_jit, _ = solver.solve(psi_0)

  assert jnp.linalg.norm(psi_jit - psi_no_jit) < 1e-6


# --------------------------------------------------------------------------
# Accuracy of each solver family against the analytical vacuum solution
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
  "solver_config",
  [
    SolverConfig(method='spectral', stepper='split_step'),
    SolverConfig(method='spectral', stepper='rk4'),
    SolverConfig(method='finite_difference', fd_order=2, stepper='rk4'),
    SolverConfig(method='finite_difference', fd_order=4, stepper='rk4'),
    SolverConfig(method='finite_difference', fd_order=6, stepper='rk4'),
    SolverConfig(method='finite_difference', compact=True, stepper='rk4'),
  ],
  ids=['spectral-split', 'spectral-rk4', 'fd2', 'fd4', 'fd6', 'compact9'],
)
def test_vacuum_matches_analytical(solver_config):
  """Every solver family reproduces the analytical Gaussian in vacuum.

  The compact case is the regression test for the 9-point stencil, which used
  to raise TracerBoolConversionError for every configuration because its grid
  spacings reached it as traced values.
  """
  sim_config = SimulationConfig(
    nx=128, ny=128, dx=0.1, dy=0.1, dz=0.005, nz=200, wavelength=1.0
  )
  w0 = 1.5
  psi_0 = gaussian_beam(sim_config, w0=w0)
  psi_analytical = gaussian_beam(
    sim_config, w0=w0, z=sim_config.lz, envelope_only=True
  )

  solver = ParaxialWaveSolver(sim_config, solver_config, NO_PML)
  psi_final, _ = solver.solve(psi_0, return_history=False)

  rel_err = (jnp.linalg.norm(psi_final - psi_analytical)
             / jnp.linalg.norm(psi_analytical))
  assert jnp.isfinite(rel_err)
  assert rel_err < 1e-2


def test_split_step_is_second_order_in_z(x64):
  """Strang splitting converges at 2nd order in dz through a medium."""
  nx = ny = 64
  dx = dy = 0.2
  lz = 1.0
  solver_config = SolverConfig(method='spectral', stepper='split_step')

  def make(nz):
    return SimulationConfig(nx=nx, ny=ny, dx=dx, dy=dy, dz=lz / nz, nz=nz,
                            wavelength=1.0)

  # A transverse structure that does not commute with the Laplacian, so the
  # splitting error is actually exercised.
  def delta_n_fn(z, medium):
    x = jnp.arange(nx) * dx
    y = jnp.arange(ny) * dy
    return 0.02 * jnp.cos(x[:, None] * 0.5) * jnp.cos(y[None, :] * 0.5)

  psi_0 = _centered_gaussian(make(1), width=2.0)

  def run(nz):
    cfg = make(nz)
    solver = ParaxialWaveSolver(cfg, solver_config, NO_PML, delta_n_fn)
    return solver.solve(psi_0, return_history=False)[0]

  reference = run(2048)
  errors = [
    float(jnp.linalg.norm(run(nz) - reference) / jnp.linalg.norm(reference))
    for nz in (32, 64, 128)
  ]

  # Successive halvings of dz should each cut the error by about four.
  for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
    rate = jnp.log2(coarse / fine)
    assert rate > 1.8, f"measured order {rate}, errors {errors}"


# --------------------------------------------------------------------------
# z_0 handling
# --------------------------------------------------------------------------

def test_chained_solves_match_single_run(x64):
  """Two chained half-runs equal one full run through a z-dependent medium.

  `solve` used to build its z grid with linspace(z_0, lz, nz), which both
  mis-spaced the steps and ignored z_0 in the span: with z_0 >= lz every step
  landed on the same z, freezing the medium.
  """
  nx = ny = 32
  dx = dy = 0.2
  dz = 0.01
  solver_config = SolverConfig(method='spectral', stepper='split_step')

  def delta_n_fn(z, medium):
    # Varies with z, so an incorrect z grid cannot go unnoticed.
    return 0.05 * jnp.cos(3.0 * z) * jnp.ones((nx, ny))

  full = SimulationConfig(nx=nx, ny=ny, dx=dx, dy=dy, dz=dz, nz=80,
                          wavelength=1.0)
  half = SimulationConfig(nx=nx, ny=ny, dx=dx, dy=dy, dz=dz, nz=40,
                          wavelength=1.0)

  psi_0 = _centered_gaussian(full, width=2.0)

  psi_full, _ = ParaxialWaveSolver(
    full, solver_config, NO_PML, delta_n_fn
  ).solve(psi_0, return_history=False)

  half_solver = ParaxialWaveSolver(half, solver_config, NO_PML, delta_n_fn)
  psi_mid, _ = half_solver.solve(psi_0, z_0=0.0, return_history=False)
  psi_chained, _ = half_solver.solve(
    psi_mid, z_0=40 * dz, return_history=False
  )

  rel = (jnp.linalg.norm(psi_chained - psi_full)
         / jnp.linalg.norm(psi_full))
  assert rel < 1e-10, f"chained run differs by {rel}"


def test_z_grid_uses_exact_dz_spacing():
  """The z values fed to delta_n_fn are z_0 + dz * arange(nz), exactly."""
  sim_config = SimulationConfig(
    nx=8, ny=8, dx=0.5, dy=0.5, dz=0.1, nz=10, wavelength=1.0
  )
  seen = []

  def delta_n_fn(z, medium):
    seen.append(z)
    return jnp.zeros((8, 8))

  solver = ParaxialWaveSolver(
    sim_config, SolverConfig(method='spectral', stepper='split_step'),
    NO_PML, delta_n_fn,
  )
  with jax.disable_jit():
    # checkpoint=False so the recorded z values are concrete:
    # jax.checkpoint traces its body even under disable_jit.
    solver.solve(_centered_gaussian(sim_config), z_0=2.0,
                 return_history=False, checkpoint=False)

  # Split-step evaluates the medium at step midpoints.
  expected = 2.0 + 0.1 * jnp.arange(10) + 0.05
  assert jnp.allclose(jnp.array(seen), expected)


# --------------------------------------------------------------------------
# PML
# --------------------------------------------------------------------------

def test_pml_absorbs_more_than_it_reflects(x64):
  """Quantify PML reflection against a domain large enough to avoid the edge.

  A beam tilted into the +x boundary is propagated on a small domain with a
  PML, and on a domain twice as wide whose own boundary it never reaches. In
  the shared interior the two must agree; the residual is the reflection.
  """
  nx, ny = 128, 64
  dx = dy = 0.25
  dz, nz = 0.25, 120
  pml_width = 24

  def build(nx_total):
    return SimulationConfig(nx=nx_total, ny=ny, dx=dx, dy=dy, dz=dz, nz=nz,
                            wavelength=1.0)

  small = build(nx)
  large = build(2 * nx)  # Twice as wide; its far boundary is never reached.
  solver_config = SolverConfig(method='spectral', stepper='split_step')

  # Same absolute beam position on both grids, tilted so that it leaves the
  # small domain through +x well before the run ends.
  w0, theta = 3.0, 0.35
  x0, y0 = nx * dx / 2.0, ny * dy / 2.0

  def initial(cfg):
    x = jnp.arange(cfg.nx) * cfg.dx
    y = jnp.arange(cfg.ny) * cfg.dy
    r2 = (x[:, None] - x0)**2 + (y[None, :] - y0)**2
    tilt = cfg.k0 * jnp.sin(theta) * x[:, None]
    return jnp.exp(-r2 / w0**2) * jnp.exp(1j * tilt)

  interior = slice(pml_width, nx - pml_width)

  # The reference keeps the same PML on the left; on the right the boundary is
  # a full domain width away from anything the beam reaches.
  reference, _ = ParaxialWaveSolver(
    large, solver_config,
    PMLConfig(width_x=pml_width, width_y=pml_width, strength=4.0),
  ).solve(initial(large), return_history=False)
  reference = reference[interior, :]

  def residual(strength):
    psi, _ = ParaxialWaveSolver(
      small, solver_config,
      PMLConfig(width_x=pml_width, width_y=pml_width, strength=strength),
    ).solve(initial(small), return_history=False)
    return float(jnp.linalg.norm(psi[interior, :] - reference)
                 / jnp.linalg.norm(reference))

  absorbing = residual(4.0)
  reflecting = residual(0.0)

  assert absorbing < 0.05, f"PML reflection too large: {absorbing}"
  assert absorbing < 0.2 * reflecting, (
    f"PML barely helps: with PML {absorbing}, without {reflecting}"
  )


def test_pml_dissipates_energy():
  """A PML removes energy from a spreading beam."""
  sim_config = SimulationConfig(
    nx=100, ny=100, dx=0.1, dy=0.1, dz=0.1, nz=50, wavelength=1.0
  )
  pml_config = PMLConfig(width_x=20, width_y=20, strength=10.0)
  solver_config = SolverConfig(method='spectral', stepper='split_step')
  psi_0 = _centered_gaussian(sim_config)

  solver = ParaxialWaveSolver(sim_config, solver_config, pml_config)
  psi_final, _ = solver.solve(psi_0, return_history=False)

  assert jnp.sum(jnp.abs(psi_final)**2) < jnp.sum(jnp.abs(psi_0)**2)


def test_complex_stretching_runs_and_absorbs():
  """The stretched-coordinate FD operator is stable and dissipative."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.25, dy=0.25, dz=0.02, nz=50, wavelength=1.0
  )
  solver_config = SolverConfig(method='finite_difference', fd_order=4,
                               stepper='rk4')
  pml_config = PMLConfig(width_x=12, width_y=12, strength=2.0,
                         use_complex_stretching=True)
  psi_0 = _centered_gaussian(sim_config, width=2.0)

  solver = ParaxialWaveSolver(sim_config, solver_config, pml_config)
  psi_final, _ = solver.solve(psi_0, return_history=False)

  assert bool(jnp.all(jnp.isfinite(jnp.abs(psi_final))))
  assert jnp.sum(jnp.abs(psi_final)**2) <= jnp.sum(jnp.abs(psi_0)**2) * (1 + 1e-6)


# --------------------------------------------------------------------------
# History control
# --------------------------------------------------------------------------

def test_history_indexing_and_save_every():
  """history[j] is the field at z_0 + j * save_every * dz, starting at psi_0."""
  sim_config = SimulationConfig(
    nx=32, ny=32, dx=0.2, dy=0.2, dz=0.05, nz=20, wavelength=1.0
  )
  solver = ParaxialWaveSolver(
    sim_config, SolverConfig(method='spectral', stepper='split_step'), NO_PML
  )
  psi_0 = _centered_gaussian(sim_config, width=2.0)

  psi_final, history = solver.solve(psi_0)
  assert history.shape == (20, 32, 32)
  assert jnp.allclose(history[0], psi_0)

  psi_strided, strided = solver.solve(psi_0, save_every=5)
  assert strided.shape == (4, 32, 32)
  assert jnp.allclose(psi_strided, psi_final)
  # Strided entries must coincide with every fifth full-history entry.
  assert jnp.allclose(strided, history[::5], atol=1e-6)

  psi_none, none_history = solver.solve(psi_0, return_history=False)
  assert none_history is None
  assert jnp.allclose(psi_none, psi_final)


def test_save_every_must_divide_nz():
  sim_config = SimulationConfig(
    nx=16, ny=16, dx=0.2, dy=0.2, dz=0.1, nz=10, wavelength=1.0
  )
  solver = ParaxialWaveSolver(
    sim_config, SolverConfig(method='spectral', stepper='split_step'), NO_PML
  )
  psi_0 = _centered_gaussian(sim_config)
  with pytest.raises(ValueError, match="must divide"):
    solver.solve(psi_0, save_every=3)


# --------------------------------------------------------------------------
# Medium handling
# --------------------------------------------------------------------------

def test_medium_argument_avoids_recompilation():
  """Swapping media reuses the compiled computation instead of retracing.

  Closing over the medium instead made every new realization a fresh cache
  key, so a chunked or ensemble run recompiled once per chunk.
  """
  from paraxial_wave_solver.src import solvers

  sim_config = SimulationConfig(
    nx=32, ny=32, dx=0.2, dy=0.2, dz=0.05, nz=10, wavelength=1.0
  )

  def delta_n_fn(z, medium):
    idx = jnp.clip(jnp.floor(z / sim_config.dz).astype(int), 0,
                   sim_config.nz - 1)
    return medium[:, :, idx]

  solver = ParaxialWaveSolver(
    sim_config, SolverConfig(method='spectral', stepper='split_step'),
    NO_PML, delta_n_fn,
  )
  psi_0 = _centered_gaussian(sim_config, width=2.0)

  key = jax.random.PRNGKey(0)
  media = [
    jax.random.normal(k, (32, 32, 10)) * 1e-3
    for k in jax.random.split(key, 3)
  ]

  solver.solve(psi_0, medium=media[0], return_history=False)
  cache_after_first = solvers._propagate._cache_size()

  results = [
    solver.solve(psi_0, medium=m, return_history=False)[0] for m in media
  ]
  assert solvers._propagate._cache_size() == cache_after_first

  # Different media must still give different answers.
  assert not jnp.allclose(results[0], results[1])


def test_vacuum_default_delta_n_is_zero():
  """Omitting delta_n_fn means vacuum, not a unit index perturbation.

  Passing jnp.ones for 'vacuum' injects exp(1j * k0 * lz); it cancels only
  when k0 * lz happens to be a multiple of 2 pi.
  """
  # Wavelength chosen so the spurious carrier would not cancel.
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.2, dy=0.2, dz=0.05, nz=40, wavelength=0.7
  )
  w0 = 1.5
  psi_0 = gaussian_beam(sim_config, w0=w0)
  psi_analytical = gaussian_beam(
    sim_config, w0=w0, z=sim_config.lz, envelope_only=True
  )

  solver_config = SolverConfig(method='spectral', stepper='split_step')
  psi_default, _ = ParaxialWaveSolver(
    sim_config, solver_config, NO_PML
  ).solve(psi_0, return_history=False)
  psi_explicit, _ = ParaxialWaveSolver(
    sim_config, solver_config, NO_PML,
    lambda z, medium: jnp.zeros((sim_config.nx, sim_config.ny)),
  ).solve(psi_0, return_history=False)

  assert jnp.allclose(psi_default, psi_explicit, atol=1e-6)
  rel = (jnp.linalg.norm(psi_default - psi_analytical)
         / jnp.linalg.norm(psi_analytical))
  assert rel < 1e-2, f"vacuum propagation off by {rel}"


def test_propagate_matches_solver():
  """The functional wrapper agrees with the class."""
  sim_config = SimulationConfig(
    nx=32, ny=32, dx=0.2, dy=0.2, dz=0.05, nz=10, wavelength=1.0
  )
  solver_config = SolverConfig(method='spectral', stepper='split_step')
  psi_0 = _centered_gaussian(sim_config, width=2.0)

  direct, _ = ParaxialWaveSolver(
    sim_config, solver_config, NO_PML
  ).solve(psi_0, return_history=False)
  wrapped, _ = propagate(
    psi_0, 0.0, sim_config, solver_config, NO_PML, return_history=False
  )
  assert jnp.allclose(direct, wrapped)


# --------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------

def test_split_step_requires_spectral():
  with pytest.raises(ValueError, match="requires method='spectral'"):
    SolverConfig(method='finite_difference', stepper='split_step')


def test_compact_requires_finite_difference():
  with pytest.raises(ValueError, match="requires method='finite_difference'"):
    SolverConfig(method='spectral', compact=True)


def test_unsupported_fd_order_rejected():
  with pytest.raises(ValueError, match="Unsupported fd_order"):
    SolverConfig(method='finite_difference', fd_order=99)


def test_compact_does_not_bypass_order_validation():
  """compact=True used to skip fd_order validation entirely."""
  # compact ignores fd_order, so an unusual value is accepted deliberately...
  SolverConfig(method='finite_difference', fd_order=99, compact=True)
  # ...but the non-compact path must still reject it.
  with pytest.raises(ValueError):
    SolverConfig(method='finite_difference', fd_order=99, compact=False)


@pytest.mark.parametrize("kwargs", [
  dict(nx=0), dict(ny=-4), dict(nz=0), dict(dx=0.0), dict(dz=-1.0),
  dict(wavelength=0.0), dict(n0=-1.0),
])
def test_simulation_config_rejects_nonphysical(kwargs):
  base = dict(nx=8, ny=8, dx=0.1, dy=0.1, dz=0.1, nz=4, wavelength=1.0)
  with pytest.raises(ValueError):
    SimulationConfig(**{**base, **kwargs})


def test_pml_config_rejects_negative_width():
  with pytest.raises(ValueError):
    PMLConfig(width_x=-1, width_y=0)


def test_pml_wider_than_domain_rejected():
  sim_config = SimulationConfig(
    nx=32, ny=32, dx=0.1, dy=0.1, dz=0.1, nz=4, wavelength=1.0
  )
  with pytest.raises(ValueError, match="no interior domain"):
    ParaxialWaveSolver(
      sim_config, SolverConfig(method='spectral', stepper='split_step'),
      PMLConfig(width_x=16, width_y=4),
    )


def test_compact_requires_square_cells():
  sim_config = SimulationConfig(
    nx=32, ny=32, dx=0.1, dy=0.2, dz=0.01, nz=4, wavelength=1.0
  )
  with pytest.raises(ValueError, match="dx == dy"):
    ParaxialWaveSolver(
      sim_config, SolverConfig(method='finite_difference', compact=True),
      NO_PML,
    )


def test_unstable_rk4_step_warns():
  """An RK4 step above the stability limit is flagged, not silently diverged."""
  sim_config = SimulationConfig(
    nx=64, ny=64, dx=0.1, dy=0.1, dz=0.5, nz=10, wavelength=1.0
  )
  with pytest.warns(RuntimeWarning, match="stability limit"):
    ParaxialWaveSolver(
      sim_config, SolverConfig(method='finite_difference', fd_order=4),
      NO_PML,
    )


# --------------------------------------------------------------------------
# The documented equation
# --------------------------------------------------------------------------

def _documented_rhs(psi, sim_config, delta_n, sigma):
  """d(psi)/dz exactly as written in the config.py docstring and the README.

  Deliberately written out longhand from the documentation rather than reusing
  anything from solvers.py, so that this is a check of the docs against the
  code and not of the code against itself.
  """
  from paraxial_wave_solver.src.operators import get_spectral_k_grids
  kx, ky = get_spectral_k_grids(
    sim_config.nx, sim_config.ny, sim_config.dx, sim_config.dy
  )
  laplacian = jnp.fft.ifft2(-(kx**2 + ky**2) * jnp.fft.fft2(psi))
  return (
    (1j / (2 * sim_config.k0 * sim_config.n0)) * laplacian
    + 1j * sim_config.k0
    * (delta_n + sim_config.n2 * jnp.abs(psi)**2) * psi
    - sigma * psi
  )


@pytest.mark.parametrize("stepper,splitting_order", [
  ('split_step', 2),
  ('split_step', 4),
  ('rk4', 2),
])
@pytest.mark.parametrize("n2,with_medium,with_pml", [
  (0.0, False, False),
  (0.0, True, False),
  (0.05, True, False),
  (0.05, True, True),
])
def test_solver_integrates_the_documented_equation(
  stepper, splitting_order, n2, with_medium, with_pml, x64
):
  """One tiny step reproduces the documented right-hand side.

  Every scheme is supposed to integrate the same equation; this pins the
  documentation to the implementation, so that a change to either without the
  other is caught.
  """
  nx = ny = 64
  dx = dy = 0.2
  dz = 1e-7

  sim_config = SimulationConfig(
    nx=nx, ny=ny, dx=dx, dy=dy, dz=dz, nz=1, wavelength=1.0, n0=1.0, n2=n2
  )
  solver_config = SolverConfig(
    method='spectral', stepper=stepper,
    **({'splitting_order': splitting_order} if stepper == 'split_step' else {}),
  )
  pml_config = (PMLConfig(width_x=10, width_y=10, strength=3.0)
                if with_pml else NO_PML)

  x = jnp.arange(nx) * dx
  y = jnp.arange(ny) * dy
  r2 = (x[:, None] - nx * dx / 2)**2 + (y[None, :] - ny * dy / 2)**2
  # A tilt, so the field is genuinely complex and the Laplacian is exercised.
  psi_0 = (jnp.exp(-r2 / 4.0) * jnp.exp(0.7j * x[:, None])).astype(complex)

  delta_n = (0.03 * jnp.cos(0.6 * x[:, None]) * jnp.cos(0.4 * y[None, :])
             if with_medium else 0.0)
  delta_n_fn = (lambda z, medium: delta_n) if with_medium else None

  solver = ParaxialWaveSolver(
    sim_config, solver_config, pml_config, delta_n_fn
  )
  psi_1, _ = solver.solve(psi_0, return_history=False)

  measured = (psi_1 - psi_0) / dz
  expected = _documented_rhs(
    psi_0, sim_config, delta_n, solver.pml_profile if with_pml else 0.0
  )
  error = jnp.linalg.norm(measured - expected) / jnp.linalg.norm(expected)
  assert float(error) < 1e-6, f"rel error {error}"
