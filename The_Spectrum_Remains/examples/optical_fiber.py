"""
Monte Carlo simulation of light propagation in an optical fiber.
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
from src.geometry import GeometryManager, Boundary
from src.simulation import Simulation
from src.visualization import plot_photon_paths, plot_energy_deposition, save_plots

class Cylinder(Boundary):
    """
    Cylindrical boundary defined by center axis and radius.
    """
    def __init__(self, center_line, radius, name=""):
        """
        Initialize a cylinder.
        
        Args:
            center_line: Two points defining the cylinder axis [[x1,y1,z1], [x2,y2,z2]]
            radius: Radius of the cylinder
            name: Optional name
        """
        super().__init__(name)
        self.p1 = np.array(center_line[0], dtype=float)
        self.p2 = np.array(center_line[1], dtype=float)
        self.radius = float(radius)
        
        # Calculate axis direction vector
        self.axis = self.p2 - self.p1
        self.axis_length = np.linalg.norm(self.axis)
        self.axis_unit = self.axis / self.axis_length if self.axis_length > 0 else np.array([0, 0, 1])
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """
        Check ray-cylinder intersection.
        
        Args:
            position: Current position [x, y, z]
            direction: Direction vector
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
        """
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Vector from p1 to position
        dp = position - self.p1
        
        # Projects of dp and direction onto axis
        dp_on_axis = np.dot(dp, self.axis_unit) * self.axis_unit
        dir_on_axis = np.dot(direction, self.axis_unit) * self.axis_unit
        
        # Components perpendicular to axis
        dp_perp = dp - dp_on_axis
        dir_perp = direction - dir_on_axis
        
        # Quadratic equation coefficients for intersection
        a = np.dot(dir_perp, dir_perp)
        b = 2 * np.dot(dp_perp, dir_perp)
        c = np.dot(dp_perp, dp_perp) - self.radius**2
        
        # Handle special case of ray parallel to cylinder axis
        if abs(a) < 1e-10:
            # Ray is parallel to cylinder axis
            # Check if ray is inside or outside cylinder
            if c > 0:
                # Outside cylinder, no intersection
                return False, float('inf')
            else:
                # Inside cylinder, find intersections with end caps
                # (Not implementing end caps in this simplified version)
                return False, float('inf')
        
        # Solve quadratic equation
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            # No real roots, no intersection
            return False, float('inf')
        
        # Calculate both intersection points
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        # Get closest intersection in front of the ray
        if t1 > 1e-10:  # Small epsilon to avoid self-intersections
            # Check if intersection is within cylinder length
            hit_point = position + t1 * direction
            dist_along_axis = np.dot(hit_point - self.p1, self.axis_unit)
            
            if 0 <= dist_along_axis <= self.axis_length:
                return True, t1
        
        if t2 > 1e-10:
            # Check if intersection is within cylinder length
            hit_point = position + t2 * direction
            dist_along_axis = np.dot(hit_point - self.p1, self.axis_unit)
            
            if 0 <= dist_along_axis <= self.axis_length:
                return True, t2
        
        return False, float('inf')
    
    def normal(self, position: np.ndarray) -> np.ndarray:
        """
        Return normal vector at position on cylinder surface.
        
        Args:
            position: Position on the cylinder surface
            
        Returns:
            Unit normal vector
        """
        # Project position onto axis
        pos_to_p1 = position - self.p1
        dist_along_axis = np.dot(pos_to_p1, self.axis_unit)
        closest_axis_point = self.p1 + dist_along_axis * self.axis_unit
        
        # Normal points from axis to surface
        normal_vec = position - closest_axis_point
        return normal_vec / np.linalg.norm(normal_vec)

class FiberSimulation(Simulation):
    """
    Extended simulation for handling optical fiber propagation.
    """
    def __init__(self, core_medium, cladding_medium, fiber_boundary, grid_size=(100, 100, 500), grid_spacing=0.1):
        """
        Initialize a fiber simulation.
        
        Args:
            core_medium: Medium for fiber core
            cladding_medium: Medium for fiber cladding
            fiber_boundary: Cylinder boundary between core and cladding
            grid_size: Size of the simulation grid
            grid_spacing: Spacing between grid points
        """
        # Initialize with core medium
        geometry = GeometryManager()
        geometry.add_boundary(fiber_boundary)
        
        super().__init__(core_medium, geometry, grid_size, grid_spacing)
        
        self.core_medium = core_medium
        self.cladding_medium = cladding_medium
        self.fiber_boundary = fiber_boundary
        
        # Track total internal reflections
        self.total_internal_reflections = 0
        # Track refraction events
        self.refractions = 0
        # Track photon distance traveled
        self.distances_traveled = []
    
    def _trace_photon(self, photon: Photon, weight_threshold: float, max_steps: int) -> None:
        """
        Trace a single photon in the fiber with proper handling of TIR.
        
        Args:
            photon: Photon to trace
            weight_threshold: Minimum photon weight before termination
            max_steps: Maximum steps per photon
        """
        step_count = 0
        distance_traveled = 0
        
        # Determine if photon starts in core or cladding
        # Here we'll just assume all photons start in the core
        in_core = True
        
        while photon.is_alive() and step_count < max_steps:
            step_count += 1
            
            # Use appropriate medium
            current_medium = self.core_medium if in_core else self.cladding_medium
            
            # Sample step size for current medium
            step_size = current_medium.sample_step_size()
            
            # Check for boundary interactions
            hit, distance, boundary = self.geometry.find_intersection(photon)
            
            if hit and distance < step_size:
                # Photon will hit fiber boundary before next scattering event
                
                # Move photon to boundary
                photon.move(distance)
                distance_traveled += distance
                
                # Get surface normal at intersection point
                normal = boundary.normal(photon.position)
                
                # Ensure normal points outward if in core, inward if in cladding
                if in_core and np.dot(normal, photon.direction) < 0:
                    # Leaving core, normal is correct
                    pass
                elif not in_core and np.dot(normal, photon.direction) > 0:
                    # Entering core, normal should be flipped
                    normal = -normal
                else:
                    # Normal needs to be flipped
                    normal = -normal
                
                # Get dot product for angle calculation
                cos_theta_i = abs(np.dot(photon.direction, normal))
                
                # Get refractive indices based on which side we're on
                if in_core:
                    n1 = self.core_medium.n
                    n2 = self.cladding_medium.n
                else:
                    n1 = self.cladding_medium.n
                    n2 = self.core_medium.n
                
                # Calculate critical angle for total internal reflection
                if n1 > n2:
                    # Total internal reflection is possible
                    critical_angle = np.arcsin(n2 / n1)
                    incident_angle = np.arccos(cos_theta_i)
                    
                    if incident_angle > critical_angle:
                        # Total internal reflection
                        self.total_internal_reflections += 1
                        # Reflect
                        dot_product = np.dot(photon.direction, normal)
                        photon.direction = photon.direction - 2 * dot_product * normal
                        continue
                
                # Calculate transmission angle using Snell's law
                sin_theta_t = (n1 / n2) * np.sqrt(1 - cos_theta_i**2)
                
                # Check for total internal reflection
                if sin_theta_t >= 1.0:
                    # Total internal reflection (should be caught above, but double-check)
                    self.total_internal_reflections += 1
                    dot_product = np.dot(photon.direction, normal)
                    photon.direction = photon.direction - 2 * dot_product * normal
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
                    else:
                        # Refract
                        self.refractions += 1
                        # Change medium
                        in_core = not in_core
                        
                        # Update direction using Snell's law
                        # For non-normal incidence, need full vector formula
                        if cos_theta_i > 0.99999:  # Near-normal incidence
                            # Direction stays the same
                            pass
                        else:
                            # Full vector refraction
                            dot_product = np.dot(photon.direction, normal)
                            sign = 1.0 if dot_product < 0 else -1.0
                            photon.direction = (n1 / n2) * (photon.direction - normal * dot_product) - normal * sign * cos_theta_t
                            photon.direction = photon.direction / np.linalg.norm(photon.direction)
            
            else:
                # No boundary hit, regular step through current medium
                
                # Move photon
                photon.move(step_size)
                distance_traveled += step_size
                
                # Handle absorption
                albedo = current_medium.albedo()
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
                photon.scatter(current_medium.g)
            
            # Check if photon is outside simulation domain
            if not self._is_position_in_domain(photon.position):
                self.results['escaped'] += 1
                photon.terminate()
        
        # Record distance traveled
        self.distances_traveled.append(distance_traveled)
        
        # Count photons that reached max steps as escaped
        if step_count >= max_steps and photon.is_alive():
            self.results['escaped'] += 1
            photon.terminate()

class FiberPhotonSource:
    """Generates photons for fiber optic simulation."""
    
    def __init__(self, fiber_radius, numerical_aperture=0.22, position=(0, 0, 0)):
        """
        Initialize a fiber photon source.
        
        Args:
            fiber_radius: Radius of the fiber core
            numerical_aperture: NA of the fiber, determines acceptance angle
            position: Source position [x, y, z]
        """
        self.fiber_radius = fiber_radius
        self.numerical_aperture = numerical_aperture
        self.position = np.array(position, dtype=float)
        
        # Calculate acceptance angle from NA
        self.acceptance_angle = np.arcsin(numerical_aperture)
    
    def generate_photon(self):
        """Generate a photon within the fiber core with appropriate direction."""
        # Random position within fiber core
        r = self.fiber_radius * np.sqrt(np.random.random())  # Radial distance
        theta = 2 * np.pi * np.random.random()  # Azimuthal angle
        
        # Position within circular cross-section
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Set initial position at fiber entrance
        position = self.position + np.array([x, y, 0.0])
        
        # Generate direction within acceptance cone
        # Random angle within acceptance cone
        theta_max = self.acceptance_angle
        cos_theta = 1.0 - np.random.random() * (1.0 - np.cos(theta_max))  # Importance sampling
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2 * np.pi * np.random.random()
        
        # Direction vector
        direction = np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ])
        
        return Photon(position=position, direction=direction)

def main():
    # Create media for fiber core and cladding
    # Using silica fiber values for 1550 nm light
    core = Medium(mu_a=0.0001, mu_s=0.001, g=0.0, n=1.48, name="Fiber Core")
    cladding = Medium(mu_a=0.0001, mu_s=0.001, g=0.0, n=1.46, name="Fiber Cladding")
    
    # Define fiber parameters
    fiber_radius = 4.5  # 4.5 μm (single-mode fiber at 1550 nm)
    fiber_length = 50.0  # 50 μm for visualization purposes
    numerical_aperture = 0.14  # Typical value for single-mode fiber
    
    # Create cylinder for fiber
    fiber_cylinder = Cylinder(
        center_line=[[0, 0, 0], [0, 0, fiber_length]],
        radius=fiber_radius,
        name="Fiber Core-Cladding Boundary"
    )
    
    # Setup simulation
    grid_size = (50, 50, 500)  # Higher resolution in Z direction
    grid_spacing = 0.5  # 0.5 μm per grid cell
    
    simulation = FiberSimulation(
        core_medium=core,
        cladding_medium=cladding,
        fiber_boundary=fiber_cylinder,
        grid_size=grid_size,
        grid_spacing=grid_spacing
    )
    
    # Create a fiber photon source
    source = FiberPhotonSource(
        fiber_radius=fiber_radius,
        numerical_aperture=numerical_aperture,
        position=[0, 0, 0]  # At the start of the fiber
    )
    
    # Run simulation with progress reporting
    num_photons = 1000
    print(f"Starting optical fiber simulation with {num_photons} photons...")
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
    
    # Create visualizations
    plt.figure(figsize=(16, 6))
    
    # 1. Plot 3D photon paths in the fiber
    ax1 = plt.subplot(121, projection='3d')
    
    # Plot photon paths
    paths = results.get('photon_paths', [])
    num_paths = min(len(paths), 30)  # Show a limited number for clarity
    colors = plt.cm.jet(np.linspace(0, 1, num_paths))
    
    for i, path in enumerate(paths[:num_paths]):
        ax1.plot(path[:, 0], path[:, 1], path[:, 2], '-', color=colors[i], alpha=0.5)
    
    # Plot fiber outline
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.array([0, fiber_length])
    
    for zi in z:
        x = fiber_radius * np.cos(theta)
        y = fiber_radius * np.sin(theta)
        z_array = np.ones_like(x) * zi
        ax1.plot(x, y, z_array, 'k--', alpha=0.3)
    
    # Plot fiber sides
    for i in range(0, 100, 10):
        ax1.plot([fiber_radius * np.cos(theta[i]), fiber_radius * np.cos(theta[i])],
                [fiber_radius * np.sin(theta[i]), fiber_radius * np.sin(theta[i])],
                [0, fiber_length], 'k--', alpha=0.1)
    
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_zlabel('Z (μm)')
    ax1.set_title('Photon Paths in Optical Fiber')
    
    # Set equal aspect for X and Y
    ax1.set_box_aspect([1, 1, fiber_length/fiber_radius/3])
    
    # 2. Plot cross-section of energy deposition
    ax2 = plt.subplot(122)
    energy_grid = results.get('deposited_energy', np.zeros(grid_size))
    
    # Integrate along Z-axis to see cross-sectional pattern
    cross_section = np.sum(energy_grid, axis=2)
    
    # Display with log scale
    epsilon = 1e-10
    masked_data = np.where(cross_section > epsilon, cross_section, epsilon)
    
    im = ax2.imshow(
        masked_data.T,  # Transpose to match physical orientation
        origin='lower',
        extent=[-grid_size[0]//2 * grid_spacing, grid_size[0]//2 * grid_spacing,
                -grid_size[1]//2 * grid_spacing, grid_size[1]//2 * grid_spacing],
        cmap='hot',
        interpolation='bicubic'
    )
    
    # Plot fiber outline
    circle = plt.Circle((0, 0), fiber_radius, fill=False, color='cyan', linestyle='--')
    ax2.add_artist(circle)
    
    plt.colorbar(im, ax=ax2, label='Deposited Energy')
    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Y (μm)')
    ax2.set_title('Cross-sectional Energy Distribution')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save and display
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/optical_fiber_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis plots
    plt.figure(figsize=(16, 6))
    
    # 1. Plot longitudinal energy distribution
    ax1 = plt.subplot(121)
    # Integrate over X and Y to see pattern along fiber length
    z_profile = np.sum(energy_grid, axis=(0, 1))
    z_positions = np.arange(len(z_profile)) * grid_spacing
    
    ax1.plot(z_positions, z_profile, 'r-', linewidth=2)
    ax1.set_xlabel('Z Position (μm)')
    ax1.set_ylabel('Deposited Energy')
    ax1.set_title('Energy Distribution Along Fiber Length')
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot histogram of distances traveled
    ax2 = plt.subplot(122)
    ax2.hist(simulation.distances_traveled, bins=30, color='blue', alpha=0.7)
    ax2.set_xlabel('Distance Traveled (μm)')
    ax2.set_ylabel('Number of Photons')
    ax2.set_title('Distribution of Photon Path Lengths')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/optical_fiber_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nFiber Simulation Statistics:")
    print(f"  Total internal reflections: {simulation.total_internal_reflections}")
    print(f"  Refractions: {simulation.refractions}")
    
    avg_distance = np.mean(simulation.distances_traveled) if simulation.distances_traveled else 0
    print(f"  Average distance traveled: {avg_distance:.2f} μm")
    
    # Calculate acceptance angle from numerical aperture
    acceptance_angle_deg = np.degrees(np.arcsin(numerical_aperture))
    print(f"  Fiber numerical aperture: {numerical_aperture:.3f}")
    print(f"  Acceptance angle: {acceptance_angle_deg:.2f}°")
    
    print("\nVisualization complete! Results saved in 'results' folder.")

if __name__ == "__main__":
    main()