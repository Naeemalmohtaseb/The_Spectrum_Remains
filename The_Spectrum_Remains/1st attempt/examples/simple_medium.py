"""
Simple Monte Carlo simulation of photon transport through a homogeneous medium.
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.photon import Photon
from src.medium import Medium
from src.geometry import GeometryManager, Plane, Sphere
from src.simulation import Simulation
from src.visualization import (
    plot_photon_paths, 
    plot_energy_deposition, 
    create_summary_plots,
    save_plots
)

def main():
    # Create a medium with specified optical properties
    medium = Medium(
        mu_a=0.1,   # Absorption coefficient [1/length]
        mu_s=10.0,  # Scattering coefficient [1/length]
        g=0.9,      # Anisotropy factor (0: isotropic, 1: forward scattering)
        n=1.0       # Refractive index
    )
    
    print(f"Medium properties: {medium}")
    print(f"  Mean free path: {medium.mfp():.2f}")
    print(f"  Albedo: {medium.albedo():.2f}")
    
    # Create geometry (optional)
    geometry = GeometryManager()
    
    # Add boundaries (e.g., a back reflector plane)
    back_wall = Plane(
        point=[0, 0, 100],   # Point on the plane
        normal_vector=[0, 0, -1],  # Normal vector (pointing into the simulation domain)
        name="Back Wall"
    )
    geometry.add_boundary(back_wall)
    
    # Setup simulation
    grid_size = (100, 100, 100)  # Grid dimensions
    simulation = Simulation(medium, geometry, grid_size, grid_spacing=1.0)
    
    # Run simulation with progress reporting
    num_photons = 1000
    print(f"Starting simulation with {num_photons} photons...")
    start_time = time.time()
    
    def progress_callback(percentage):
        sys.stdout.write(f"\rProgress: {percentage:.1f}%")
        sys.stdout.flush()
    
    results = simulation.run(
        num_photons=num_photons,
        weight_threshold=0.001,  # Minimum photon weight before Russian roulette
        max_steps=1000,          # Maximum steps per photon
        progress_callback=progress_callback
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print(f"Photons per second: {num_photons/elapsed_time:.2f}")
    
    # Display results
    print("\nResults summary:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
    
    # Create and save visualization plots
    print("\nGenerating visualization plots...")
    save_plots(results, output_dir='results')
    
    # Show interactive plots
    plt.figure(figsize=(12, 10))
    
    # Plot photon paths in 3D
    plt.subplot(221, projection='3d')
    paths = results.get('photon_paths', [])
    num_paths = min(len(paths), 20)
    colors = plt.cm.jet(np.linspace(0, 1, num_paths))
    
    for i, path in enumerate(paths[:num_paths]):
        plt.plot(path[:, 0], path[:, 1], path[:, 2], '-', color=colors[i], alpha=0.5)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Photon Trajectories')
    
    # Plot energy deposition
    plt.subplot(222)
    energy_grid = results.get('deposited_energy', np.zeros((1, 1, 1)))
    slice_z = energy_grid.shape[2] // 2  # Middle Z slice
    plt.imshow(energy_grid[:, :, slice_z].T, origin='lower', cmap='hot')
    plt.colorbar(label='Deposited Energy')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Energy Deposition (Z={slice_z})')
    
    # Plot depth profile
    plt.subplot(223)
    profile = np.sum(energy_grid, axis=(0, 1))
    plt.plot(profile, '-o')
    plt.xlabel('Z Position')
    plt.ylabel('Integrated Energy')
    plt.title('Depth Profile')
    plt.grid(True, alpha=0.3)
    
    # Plot statistics
    plt.subplot(224)
    stats = {
        'Absorbed': results.get('absorbed', 0),
        'Transmitted': results.get('transmitted', 0),
        'Reflected': results.get('reflected', 0),
        'Escaped': results.get('escaped', 0)
    }
    
    # Filter out zero values
    labels = []
    values = []
    for label, value in stats.items():
        if value > 0:
            labels.append(label)
            values.append(value)
    
    if values:
        plt.pie(values, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Photon Fate Statistics')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDone!")

if __name__ == "__main__":
    main()