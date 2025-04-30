"""
Monte Carlo simulation of light reflection from a spherical mirror.
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, List, Optional, Union, Any

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.photon import Photon
from src.medium import Medium
from src.geometry import GeometryManager, Sphere
from src.simulation import Simulation
from src.visualization import plot_photon_paths, plot_energy_deposition, save_plots

class MirrorSimulation(Simulation):
    """
    Extended simulation for handling mirror reflections.
    """
    def __init__(self, medium, mirror, grid_size=(100, 100, 100), grid_spacing=1.0):
        """
        Initialize a simulation with a mirror.
        
        Args:
            medium: Medium for photon propagation
            mirror: Mirror boundary
            grid_size: Size of the simulation grid
            grid_spacing: Spacing between grid points
        """
        geometry = GeometryManager()
        geometry.add_boundary(mirror)
        
        super().__init__(medium, geometry, grid_size, grid_spacing)
        
        # Track reflection points for visualization
        self.reflection_points = []
        
    def _trace_photon(self, photon: Photon, weight_threshold: float, max_steps: int) -> None:
        """
        Trace a single photon with perfect mirror reflection.
        
        This overrides the parent method to handle mirror reflections differently.
        """
        step_count = 0
        
        while photon.is_alive() and step_count < max_steps:
            step_count += 1
            
            # Sample step size
            step_size = self.medium.sample_step_size()
            
            # Check for boundary interactions
            hit, distance, boundary = self.geometry.find_intersection(photon)
            
            if hit and distance < step_size:
                # Photon will hit boundary (mirror) before next scattering event
                
                # Move photon to boundary
                photon.move(distance)
                
                # Get surface normal at intersection point
                normal = boundary.normal(photon.position)
                
                # Perfect mirror reflection
                dot_product = np.dot(photon.direction, normal)
                photon.direction = photon.direction - 2 * dot_product * normal
                
                # Count as a reflection and store reflection point
                self.results['reflected'] += 1
                self.reflection_points.append(photon.position.copy())
                
            else:
                # No boundary hit, regular step through medium
                
                # Move photon
                photon.move(step_size)
                
                # Handle absorption
                albedo = self.medium.albedo()
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
                photon.scatter(self.medium.g)
            
            # Check if photon is outside simulation domain
            if not self._is_position_in_domain(photon.position):
                self.results['escaped'] += 1
                photon.terminate()
        
        # Count photons that reached max steps as escaped
        if step_count >= max_steps and photon.is_alive():
            self.results['escaped'] += 1
            photon.terminate()

class DirectionalPhotonSource:
    """Generates photons with a specific direction pattern."""
    
    def __init__(self, position=(0, 0, 0), direction=(0, 0, 1), beam_width=1.0, divergence=0.1):
        """
        Initialize a directional photon source.
        
        Args:
            position: Source position [x, y, z]
            direction: Central beam direction
            beam_width: Width of the beam
            divergence: Angular divergence (radians)
        """
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float) / np.linalg.norm(np.array(direction))
        self.beam_width = beam_width
        self.divergence = divergence
    
    def generate_photon(self):
        """Generate a photon with position and direction based on source properties."""
        # Random position within beam width
        r = self.beam_width * np.sqrt(np.random.random())  # Radial distance
        theta = 2 * np.pi * np.random.random()  # Azimuthal angle
        
        # Create vector perpendicular to direction
        if abs(self.direction[2]) > 0.9:
            # If direction is close to z-axis, use x-axis as reference
            perp1 = np.array([1, 0, 0])
        else:
            # Otherwise use z-axis
            perp1 = np.array([0, 0, 1])
        
        # Get two perpendicular vectors to create a plane
        perp1 = perp1 - self.direction * np.dot(perp1, self.direction)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(self.direction, perp1)
        
        # Position offset from center
        offset = r * (perp1 * np.cos(theta) + perp2 * np.sin(theta))
        position = self.position + offset
        
        # Add random divergence to direction
        if self.divergence > 0:
            # Random angle within divergence
            phi = 2 * np.pi * np.random.random()  # Azimuthal angle
            div_angle = self.divergence * np.sqrt(np.random.random())  # Polar angle
            
            # Calculate divergence vector
            sin_div = np.sin(div_angle)
            div_vector = (
                sin_div * np.cos(phi) * perp1 +
                sin_div * np.sin(phi) * perp2 +
                np.cos(div_angle) * self.direction
            )
            
            direction = div_vector / np.linalg.norm(div_vector)
        else:
            direction = self.direction.copy()
        
        return Photon(position=position, direction=direction)

def main():
    # Create a medium with low scattering (nearly transparent)
    air = Medium(mu_a=0.001, mu_s=0.01, g=0.0, n=1.0, name="Air")
    
    # Define a spherical mirror
    mirror_radius = 10.0
    mirror_center = [0, 0, 30.0]  # 30 units along z-axis
    
    mirror = Sphere(
        center=mirror_center, 
        radius=mirror_radius, 
        name="Spherical Mirror"
    )
    
    # Setup simulation
    grid_size = (100, 100, 100)
    grid_spacing = 0.5
    
    simulation = MirrorSimulation(
        medium=air,
        mirror=mirror,
        grid_size=grid_size,
        grid_spacing=grid_spacing
    )
    
    # Create a directional photon source aimed at the mirror
    # Position it to point at the mirror center
    source = DirectionalPhotonSource(
        position=[0, 0, 0],
        direction=[0, 0, 1],  # Pointing along z-axis
        beam_width=5.0,       # Wide beam to illuminate the mirror
        divergence=0.05       # Slight divergence
    )
    
    # Run simulation with progress reporting
    num_photons = 1000
    print(f"Starting spherical mirror simulation with {num_photons} photons...")
    start_time = time.time()
    
    def progress_callback(percentage):
        sys.stdout.write(f"\rProgress: {percentage:.1f}%")
        sys.stdout.flush()
    
    # Override the default photon initialization to use our source
    original_photons = []
    
    for _ in range(num_photons):
        original_photons.append(source.generate_photon())
    
    # Run the simulation with our custom photons
    results = simulation.run(
        num_photons=num_photons,
        weight_threshold=0.001,
        max_steps=1000,
        progress_callback=progress_callback
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print(f"Photons per second: {num_photons/elapsed_time:.2f}")
    
    # Create 3D visualization with mirror
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot photon paths
    paths = results.get('photon_paths', [])
    num_paths = min(len(paths), 50)  # Show more paths for better visualization
    
    # Use spectrum colors
    colors = plt.cm.jet(np.linspace(0, 1, num_paths))
    
    for i, path in enumerate(paths[:num_paths]):
        # Plot full path
        ax.plot(path[:, 0], path[:, 1], path[:, 2], '-', color=colors[i], alpha=0.5)
        
        # Plot start point
        ax.plot([path[0, 0]], [path[0, 1]], [path[0, 2]], 'go', markersize=4)
    
    # Plot reflection points
    reflection_points = np.array(simulation.reflection_points)
    if len(reflection_points) > 0:
        ax.scatter(
            reflection_points[:, 0], 
            reflection_points[:, 1], 
            reflection_points[:, 2], 
            c='r', 
            s=5, 
            alpha=0.3
        )
    
    # Plot the mirror outline
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    
    x = mirror_center[0] + mirror_radius * np.outer(np.cos(u), np.sin(v))
    y = mirror_center[1] + mirror_radius * np.outer(np.sin(u), np.sin(v))
    z = mirror_center[2] + mirror_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the mirror as a wireframe for visibility
    ax.plot_wireframe(x, y, z, color='cyan', alpha=0.2)
    
    # Calculate theoretical focal point (for a parabolic mirror it would be radius/2)
    # For spherical mirror with paraxial approximation, it's radius/2
    focal_point = [0, 0, mirror_center[2] - mirror_radius/2]
    ax.plot([focal_point[0]], [focal_point[1]], [focal_point[2]], 'ro', markersize=10, label='Theoretical Focus')
    
    # Add axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Photon Reflection from Spherical Mirror')
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, mid_z + max_range)  # Keep the bottom at 0
    
    ax.legend()
    
    # Save and display
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/spherical_mirror_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate heat map of reflection density to see focusing effect
    plt.figure(figsize=(10, 8))
    
    # Filter reflection points to those near the focal region
    if len(reflection_points) > 0:
        focal_z = mirror_center[2] - mirror_radius/2
        near_focal = reflection_points[np.abs(reflection_points[:, 2] - focal_z) < 5]
        
        if len(near_focal) > 0:
            plt.hist2d(
                near_focal[:, 0], 
                near_focal[:, 1], 
                bins=50, 
                cmap='hot'
            )
            plt.colorbar(label='Photon Density')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title(f'Photon Density near Focal Region (Z ≈ {focal_z:.1f})')
            plt.grid(alpha=0.3)
            plt.axis('equal')
            
            plt.savefig('results/mirror_focal_region.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    print("\nAdditional Results Summary:")
    total_photons = sum(v for k, v in results.items() if k in ['absorbed', 'reflected', 'transmitted', 'escaped'])
    
    for key in ['absorbed', 'reflected', 'transmitted', 'escaped']:
        if key in results:
            percentage = results[key] / total_photons * 100 if total_photons > 0 else 0
            print(f"  {key.capitalize()}: {results[key]} ({percentage:.1f}%)")
    
    print("\nVisualization complete! Results saved in 'results' folder.")

if __name__ == "__main__":
    main()