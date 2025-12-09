from dataclasses import dataclass
from typing import Optional, Tuple, Literal, TypeAlias
import jax

# Custom Type Alias for 2D/3D Fields
Field: TypeAlias = jax.Array

@dataclass(frozen=True)
class SimulationConfig:
  """
  Configuration for the simulation grid and domain.
  
  Attributes:
    nx: Number of grid points in the x-direction.
    ny: Number of grid points in the y-direction.
    dx: Grid spacing in the x-direction (physical units).
    dy: Grid spacing in the y-direction (physical units).
    dz: Step size for propagation in the z-direction (physical units).
    nz: Number of steps to propagate in the z-direction.
    wavelength: Wavelength of the optical field in vacuum.
  """
  nx: int
  ny: int
  dx: float
  dy: float
  dz: float
  nz: int
  wavelength: float

  @property
  def k0(self) -> float:
    """Wavenumber in vacuum (2*pi/wavelength)."""
    return 2 * 3.141592653589793 / self.wavelength

  @property
  def lx(self) -> float:
    """Total physical length of the domain in the x-direction."""
    return self.nx * self.dx

  @property
  def ly(self) -> float:
    """Total physical length of the domain in the y-direction."""
    return self.ny * self.dy
  
  @property
  def lz(self) -> float:
    """Total physical propagation distance in the z-direction."""
    return self.nz * self.dz


@dataclass(frozen=True, slots=True)
class PMLConfig:
  """
  Configuration for Perfectly Matched Layers (PML).
  
  Attributes:
    width_x: Number of grid points for the PML region at the x-boundaries 
             (both sides).
    width_y: Number of grid points for the PML region at the y-boundaries 
             (both sides).
    strength: Maximum absorption strength of the PML profile.
    order: Polynomial order of the PML absorption profile (typically 2.0).
    profile_type: Type of PML profile (currently only 'polynomial' is 
                  supported).
  """
  width_x: int
  width_y: int
  strength: float = 1.0
  order: int = 2
  profile_type: Literal['polynomial'] = 'polynomial'


@dataclass(frozen=True, slots=True)
class SolverConfig:
  """
  Configuration for the numerical solver method.
  
  Attributes:
    method: Spatial discretization method ('finite_difference' or 'spectral').
    fd_order: Order of accuracy for finite difference method (2, 4, or 6). 
              Ignored if method is 'spectral'.
    compact: Whether to use compact finite difference stencils (only for 
             fd_order=4).
    stepper: Z-propagation stepping scheme ('rk4' or 'split_step'). 
             'split_step' is typically used with 'spectral' method.
  """
  method: Literal['finite_difference', 'spectral']
  fd_order: int = 2
  compact: bool = False
  stepper: Literal['rk4', 'split_step'] = 'rk4'
