from dataclasses import dataclass
from typing import Optional, Tuple, Literal

@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the simulation grid and domain."""
    Nx: int
    Ny: int
    dx: float
    dy: float
    dz: float
    Nz: int
    wavelength: float

    @property
    def k0(self) -> float:
        """Wavenumber in vacuum."""
        return 2 * 3.141592653589793 / self.wavelength

    @property
    def Lx(self) -> float:
        """Domain size in x."""
        return self.Nx * self.dx

    @property
    def Ly(self) -> float:
        """Domain size in y."""
        return self.Ny * self.dy
    
    @property
    def Lz(self) -> float:
        """Domain size in z."""
        return self.Nz * self.dz


@dataclass(frozen=True)
class PMLConfig:
    """Configuration for Perfectly Matched Layers."""
    width_x: int  # Number of grid points for PML in x
    width_y: int  # Number of grid points for PML in y
    strength: float = 1.0
    order: float = 2.0
    profile_type: Literal['polynomial'] = 'polynomial'


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the solver method."""
    method: Literal['finite_difference', 'spectral']
    fd_order: int = 2  # 2, 4, or 6
    stepper: Literal['rk4', 'split_step'] = 'rk4'
