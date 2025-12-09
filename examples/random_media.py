import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SimulationConfig, SolverConfig, PMLConfig
from src.solvers import propagate
from src.utils import gaussian_beam, random_medium

def main():
    # 1. Setup Configuration
    sim_config = SimulationConfig(
        Nx=256, Ny=256, 
        dx=0.2, dy=0.2, dz=0.5, 
        Nz=200, 
        wavelength=1.0
    )
    
    pml_config = PMLConfig(width_x=20, width_y=20, strength=5.0)
    solver_config = SolverConfig(method='spectral', stepper='split_step')
    
    # 2. Initial Condition
    w0 = 5.0
    psi_0 = gaussian_beam(sim_config, w0=w0)
    
    # 3. Random Medium
    key = jax.random.PRNGKey(42)
    delta_n = random_medium(sim_config, correlation_length=2.0, strength=0.05, key=key)
    
    # Define refractive index function
    # We need to interpolate or pick the slice closest to z
    # Since we use fixed dz grid, we can just index
    
    # Pre-compute n grid to avoid re-computation (though random_medium returns full grid)
    # But propagate expects a function of z.
    # We can capture delta_n and index it.
    
    def n_ref_fn(z):
        # Find index
        idx = jnp.clip(jnp.round(z / sim_config.dz).astype(int), 0, sim_config.Nz - 1)
        return 1.0 + delta_n[:, :, idx]
        
    # 4. Run Simulation
    print("Running simulation in random media...")
    psi_final, psi_history = propagate(psi_0, sim_config, solver_config, pml_config, n_ref_fn)
    print("Simulation complete.")
    
    # 5. Visualize
    center_y = sim_config.Ny // 2
    field_xz = psi_history[:, :, center_y].T
    intensity_xz = jnp.abs(field_xz)**2
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    extent = [0, sim_config.Lz, 0, sim_config.Lx]
    plt.imshow(intensity_xz, extent=extent, origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(label='Intensity')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.title('Propagation in Random Media (XZ)')
    
    plt.subplot(1, 2, 2)
    # Plot refractive index slice
    n_xz = delta_n[:, center_y, :].T # (Nz, Nx) -> (Nx, Nz) ? No, delta_n is (Nx, Ny, Nz)
    # delta_n[:, center_y, :] is (Nx, Nz). T is (Nz, Nx). 
    # Wait, imshow expects (rows, cols). Rows=x, Cols=z.
    # So (Nx, Nz) is correct for origin='lower' if x is vertical.
    plt.imshow(delta_n[:, center_y, :], extent=extent, origin='lower', cmap='gray', aspect='auto')
    plt.colorbar(label='delta n')
    plt.xlabel('z')
    plt.ylabel('x')
    plt.title('Refractive Index Perturbation (XZ)')
    
    plt.tight_layout()
    plt.savefig('random_media_xz.png')
    print("Saved plot to random_media_xz.png")

if __name__ == "__main__":
    main()
