"""
Monte Carlo simulation of light transport through skin tissue layers.
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Dict, Tuple, List, Optional, Union, Any

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.photon import Photon
from src.medium import Medium
from src.geometry import GeometryManager, Plane
from src.simulation import Simulation
from src.visualization import plot_photon_paths, plot_energy_deposition, save_plots

class LayeredSimulation(Simulation):
    """
    Extended simulation for handling multiple media layers.
    """
    def __init__(self, media_layers, boundaries, grid_size=(100, 100, 100), grid_spacing=0.05):
        """
        Initialize a simulation with multiple layers.
        
        Args:
            media_layers: List of Medium objects, one per layer
            boundaries: List of boundary planes between layers
            grid_size: Size of the simulation grid
            grid_spacing: Spacing between grid points
        """
        # Initialize with the first medium (will be updated during simulation)
        super().__init__(media_layers[0], GeometryManager(), grid_size, grid_spacing)
        
        self.media_layers = media_layers
        self.geometry = GeometryManager()
        
        # Add boundaries to geometry manager
        for boundary in boundaries:
            self.geometry.add_boundary(boundary)
            
        # Track which medium each photon is in
        self.current_medium_index = 0
        
    def _trace_photon(self, photon: Photon, weight_threshold: float, max_steps: int) -> None:
        """
        Trace a single photon through layered media.
        
        This overrides the parent method to handle different media layers.
        """
        step_count = 0
        # Start in the top layer (index 0)
        current_medium_index = 0
        
        while photon.is_alive() and step_count < max_steps:
            step_count += 1
            
            # Get current medium
            medium = self.media_layers[current_medium_index]
            
            # Sample step size for current medium
            step_size = medium.sample_step_size()
            
            # Check for boundary interactions
            hit, distance, boundary = self.geometry.find_intersection(photon)
            
            if hit and distance < step_size:
                # Photon will hit boundary before next scattering event
                
                # Move photon to boundary
                photon.move(distance)
                
                # Get surface normal at intersection point
                normal = boundary.normal(photon.position)
                
                # Determine which layers are on either side of the boundary
                # This is simplified; in practice you'd need more sophisticated logic
                # Find the z-coordinate of each boundary
                boundary_z = boundary.point[2]
                
                # Determine if photon is moving downward (into next layer) or upward
                moving_downward = photon.direction[2] > 0
                
                # Check if photon crosses to next medium or reflects
                if moving_downward and current_medium_index < len(self.media_layers) - 1:
                    next_medium_index = current_medium_index + 1
                elif not moving_downward and current_medium_index > 0:
                    next_medium_index = current_medium_index - 1
                else:
                    next_medium_index = current_medium_index  # Same medium (outer boundary)
                
                # Get refractive indices
                n1 = self.media_layers[current_medium_index].n
                n2 = self.media_layers[next_medium_index].n
                
                # Calculate cosines of incident and transmitted angles
                cos_theta_i = abs(np.dot(photon.direction, normal))
                
                # Handle reflection/refraction based on Fresnel equations (simplified)
                if n1 == n2:  # Same refractive index - no reflection
                    # Just pass through
                    current_medium_index = next_medium_index
                else:
                    # Calculate transmission angle using Snell's law
                    sin_theta_t = (n1 / n2) * np.sqrt(1 - cos_theta_i**2)
                    
                    # Check for total internal reflection
                    if sin_theta_t >= 1.0:
                        # Total internal reflection
                        dot_product = np.dot(photon.direction, normal)
                        photon.direction = photon.direction - 2 * dot_product * normal
                        self.results['reflected'] += 1
                    else:
                        # Calculate Fresnel reflection coefficient (simplified)
                        cos_theta_t = np.sqrt(1 - sin_theta_t**2)
                        r_s = ((n1 * cos_theta_i - n2 * cos_theta_t) / 
                              (n1 * cos_theta_i + n2 * cos_theta_t))**2
                        r_p = ((n1 * cos_theta_t - n2 * cos_theta_i) / 
                              (n1 * cos_theta_t + n2 * cos_theta_i))**2
                        r = 0.5 * (r_s + r_p)  # Average for unpolarized light
                        
                        # Probabilistic handling of reflection vs. refraction
                        if np.random.random() < r:
                            # Reflect
                            dot_product = np.dot(photon.direction, normal)
                            photon.direction = photon.direction - 2 * dot_product * normal
                            self.results['reflected'] += 1
                        else:
                            # Refract - update direction using Snell's law
                            # For simplicity, this implementation doesn't fully handle vector refraction
                            # But it gives the correct behavior for simple cases
                            if cos_theta_i > 0.99999:  # Near-normal incidence
                                # Direction stays the same
                                pass
                            else:
                                # Full vector refraction would be implemented here
                                # This is a simplified version
                                dot_product = np.dot(photon.direction, normal)
                                photon.direction = (photon.direction - 
                                                  normal * dot_product) * (n1 / n2) - normal * cos_theta_t
                                photon.direction = photon.direction / np.linalg.norm(photon.direction)
                            
                            # Update medium
                            current_medium_index = next_medium_index
            
            else:
                # No boundary hit, regular step through current medium
                
                # Move photon
                photon.move(step_size)
                
                # Handle absorption
                albedo = medium.albedo()
                if albedo < 1.0:
                    # Absorb energy
                    absorbed_weight = photon.absorb(1.0 - albedo)
                    
                    # Record absorbed energy in grid
                    grid_pos = self._world_to_grid(photon.position)
                    if self._is_in_grid(grid_pos):
                        self.results['deposited_energy'][tuple(grid_pos)] += absorbed_weight
                
                # Check if photon weight is below threshold
                if photon.weight < weight_threshold:
                    # Russian roulette for weight below threshold
                    if np.random.random() < 0.1:  # 10% chance to survive
                        photon.weight *= 10.0  # Boost weight
                    else:
                        # Count as fully absorbed
                        self.results['absorbed'] += 1
                        photon.terminate()
                        continue
                
                # Scatter photon
                photon.scatter(medium.g)
            
            # Check if photon is outside simulation domain
            if not self._is_position_in_domain(photon.position):
                self.results['escaped'] += 1
                photon.terminate()
        
        # Count photons that reached max steps as escaped
        if step_count >= max_steps and photon.is_alive():
            self.results['escaped'] += 1
            photon.terminate()

def main():
    # Create media for different skin layers with realistic optical properties
    # Values based on literature for 633nm light
    epidermis = Medium(mu_a=0.8, mu_s=40.0, g=0.9, n=1.4, name="Epidermis")
    dermis = Medium(mu_a=0.2, mu_s=20.0, g=0.8, n=1.38, name="Dermis")
    subcutaneous = Medium(mu_a=0.1, mu_s=10.0, g=0.75, n=1.36, name="Subcutaneous Fat")
    
    # Define layer thicknesses in mm
    epidermis_thickness = 0.1  # 0.1 mm
    dermis_thickness = 0.6     # 0.6 mm
    
    # Define layer boundaries
    epidermis_dermis_boundary = Plane(
        point=[0, 0, epidermis_thickness], 
        normal_vector=[0, 0, 1], 
        name="Epidermis-Dermis Boundary"
    )
    
    dermis_subcutaneous_boundary = Plane(
        point=[0, 0, epidermis_thickness + dermis_thickness], 
        normal_vector=[0, 0, 1], 
        name="Dermis-Subcutaneous Boundary"
    )
    
    back_boundary = Plane(
        point=[0, 0, epidermis_thickness + dermis_thickness + 1.0], 
        normal_vector=[0, 0, 1], 
        name="Back Boundary"
    )
    
    boundaries = [epidermis_dermis_boundary, dermis_subcutaneous_boundary, back_boundary]
    media_layers = [epidermis, dermis, subcutaneous]
    
    # Setup simulation with high resolution grid
    grid_size = (100, 100, 100)
    grid_spacing = 0.02  # 0.02 mm per grid cell - high resolution
    
    simulation = LayeredSimulation(
        media_layers=media_layers,
        boundaries=boundaries,
        grid_size=grid_size,
        grid_spacing=grid_spacing
    )
    
    # Run simulation with progress reporting
    num_photons = 10000  # Use a large number for smooth results
    print(f"Starting skin tissue simulation with {num_photons} photons...")
    start_time = time.time()
    
    def progress_callback(percentage):
        sys.stdout.write(f"\rProgress: {percentage:.1f}%")
        sys.stdout.flush()
    
    results = simulation.run(
        num_photons=num_photons,
        weight_threshold=0.001,
        max_steps=1000,
        progress_callback=progress_callback
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print(f"Photons per second: {num_photons/elapsed_time:.2f}")
    
    # Create visualization for skin layers
    plt.figure(figsize=(12, 8))
    
    # Plot energy deposition with layer boundaries
    ax = plt.subplot(121)
    energy_grid = results['deposited_energy']
    # Take central slice in Y direction
    central_slice = energy_grid[:, energy_grid.shape[1]//2, :]
    
    # Use log scale for better visualization
    epsilon = 1e-10
    masked_data = np.where(central_slice > epsilon, central_slice, epsilon)
    im = ax.imshow(
        masked_data.T,  # Transpose to make Z vertical
        origin='lower',
        cmap='hot',
        norm=LogNorm(vmin=masked_data.min(), vmax=masked_data.max()),
        extent=[0, grid_size[0]*grid_spacing, 0, grid_size[2]*grid_spacing]
    )
    
    # Add layer boundary lines
    ax.axhline(y=epidermis_thickness, color='cyan', linestyle='--', alpha=0.7)
    ax.axhline(y=epidermis_thickness + dermis_thickness, color='cyan', linestyle='--', alpha=0.7)
    
    # Add layer labels
    ax.text(grid_size[0]*grid_spacing*0.8, epidermis_thickness/2, "Epidermis", color='white')
    ax.text(grid_size[0]*grid_spacing*0.8, epidermis_thickness + dermis_thickness/2, "Dermis", color='white')
    ax.text(grid_size[0]*grid_spacing*0.8, epidermis_thickness + dermis_thickness + 0.2, "Subcut.", color='white')
    
    plt.colorbar(im, label='Energy Deposition (Log Scale)')
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Z Position (mm)')
    ax.set_title('Light Penetration Through Skin Layers')
    
    # Plot depth profile
    ax2 = plt.subplot(122)
    # Sum over X and Y to get depth profile
    depth_profile = np.sum(energy_grid, axis=(0, 1))
    z_positions = np.arange(len(depth_profile)) * grid_spacing
    
    ax2.plot(depth_profile, z_positions, 'r-', linewidth=2)
    ax2.set_ylim([0, grid_size[2]*grid_spacing])
    
    # Add layer boundary lines
    ax2.axhline(y=epidermis_thickness, color='cyan', linestyle='--', alpha=0.7)
    ax2.axhline(y=epidermis_thickness + dermis_thickness, color='cyan', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Deposited Energy')
    ax2.set_ylabel('Depth (mm)')
    ax2.set_title('Depth Penetration Profile')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save and display results
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/skin_tissue_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAdditional Results Summary:")
    total_photons = sum(v for k, v in results.items() if k in ['absorbed', 'reflected', 'transmitted', 'escaped'])
    
    # Calculate statistics
    for key in ['absorbed', 'reflected', 'transmitted', 'escaped']:
        if key in results:
            percentage = results[key] / total_photons * 100 if total_photons > 0 else 0
            print(f"  {key.capitalize()}: {results[key]} ({percentage:.1f}%)")
    
    # Show 3D visualization of photon paths
    plot_paths_fig = plot_photon_paths(results.get('photon_paths', []), max_paths=20)
    plot_paths_fig.savefig('results/skin_photon_paths.png', dpi=300, bbox_inches='tight')
    
    print("\nVisualization complete! Results saved in 'results' folder.")

if __name__ == "__main__":
    main()