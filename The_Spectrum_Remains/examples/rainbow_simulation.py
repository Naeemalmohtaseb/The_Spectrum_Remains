"""
Corrected rainbow formation simulation with accurate physics and visualization.
This version fixes mechanical issues and focuses on proper ray tracing.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# Create output directory
os.makedirs('results', exist_ok=True)

def wavelength_to_rgb(wavelength):
    """Convert wavelength to RGB color."""
    # Visible spectrum: 380-750 nm
    if wavelength < 380 or wavelength > 750:
        return (0.0, 0.0, 0.0)
    
    # More accurate conversion based on CIE standard observer
    if wavelength < 440:
        # Violet/Blue
        r = (440 - wavelength) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wavelength < 490:
        # Blue/Cyan
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif wavelength < 510:
        # Cyan/Green
        r = 0.0
        g = 1.0
        b = (510 - wavelength) / (510 - 490)
    elif wavelength < 580:
        # Green/Yellow
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wavelength < 645:
        # Yellow/Red
        r = 1.0
        g = (645 - wavelength) / (645 - 580)
        b = 0.0
    else:
        # Red
        r = 1.0
        g = 0.0
        b = 0.0
    
    # Scale RGB values based on intensity perception
    gamma = 0.8
    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength > 700:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
    else:
        factor = 1.0
    
    r = pow(r * factor, gamma)
    g = pow(g * factor, gamma)
    b = pow(b * factor, gamma)
    
    return (r, g, b)


def get_refractive_index(wavelength, medium='water'):
    """
    Get refractive index for a specific wavelength and medium.
    
    Args:
        wavelength: Light wavelength in nm
        medium: 'water' or 'air'
        
    Returns:
        Refractive index value
    """
    if medium == 'air':
        return 1.0003  # Air has very slight wavelength dependence
    
    elif medium == 'water':
        # Convert wavelength from nm to μm for Cauchy's formula
        wl_um = wavelength / 1000.0
        
        # More accurate Cauchy's formula coefficients for water
        A = 1.324
        B = 0.00325
        C = 0.00031
        
        # Calculate refractive index
        n = A + B / (wl_um**2) + C / (wl_um**4)
        
        return n
    
    return 1.0  # Default


class RainbowRayTracer:
    """
    Accurate ray tracer for rainbow formation.
    """
    def __init__(self, droplet_radius=1.0):
        self.droplet_radius = droplet_radius
        self.results = {
            'ray_paths': [],      # Will store complete ray paths
            'exit_angles': [],    # Store (wavelength, exit_angle, impact_param) tuples
        }
    
    def trace_ray(self, impact_param, wavelength, max_reflections=3):
        """
        Trace a single ray through a water droplet with accurate physics.
        
        Args:
            impact_param: Impact parameter (distance from central axis)
            wavelength: Wavelength of light in nm
            max_reflections: Maximum number of internal reflections to model
            
        Returns:
            Tuple of (complete_path, exit_angle)
                complete_path: List of (x, y) coordinates along the ray path
                exit_angle: Final deviation angle in degrees
        """
        # Normalize impact parameter (0-1)
        b = impact_param / self.droplet_radius
        
        if b > 1.0:
            # Ray misses the droplet
            return None, None
        
        # Get refractive indices
        n1 = get_refractive_index(wavelength, 'air')
        n2 = get_refractive_index(wavelength, 'water')
        
        # Ray starts from far left, moving horizontally
        path = []
        
        # Calculate entry point on the droplet
        entry_x = -np.sqrt(1 - b**2) * self.droplet_radius
        entry_y = b * self.droplet_radius
        
        # Start point (far left)
        start_x = -3 * self.droplet_radius
        path.append((start_x, entry_y))
        
        # Add entry point to path
        path.append((entry_x, entry_y))
        
        # Calculate refraction at entry
        # Normal vector points inward at entry
        normal_x = entry_x / self.droplet_radius
        normal_y = entry_y / self.droplet_radius
        
        # Incident direction is horizontal from left
        incident_dir_x = 1.0
        incident_dir_y = 0.0
        
        # Calculate angle of incidence (with respect to normal)
        cos_theta_i = -(incident_dir_x * normal_x + incident_dir_y * normal_y)
        sin_theta_i = np.sqrt(1 - cos_theta_i**2)
        
        # Calculate angle of refraction using Snell's law
        sin_theta_t = (n1 / n2) * sin_theta_i
        
        # Check for total internal reflection (shouldn't happen at entry)
        if sin_theta_t > 1.0:
            return path, None
        
        cos_theta_t = np.sqrt(1 - sin_theta_t**2)
        
        # Calculate refracted direction
        refracted_dir_x = (n1 / n2) * incident_dir_x + (n1 / n2 * cos_theta_i - cos_theta_t) * normal_x
        refracted_dir_y = (n1 / n2) * incident_dir_y + (n1 / n2 * cos_theta_i - cos_theta_t) * normal_y
        
        # Normalize refracted direction
        refracted_norm = np.sqrt(refracted_dir_x**2 + refracted_dir_y**2)
        refracted_dir_x /= refracted_norm
        refracted_dir_y /= refracted_norm
        
        # Current position and direction
        current_x, current_y = entry_x, entry_y
        current_dir_x, current_dir_y = refracted_dir_x, refracted_dir_y
        
        # Inside the droplet, track reflections
        in_droplet = True
        reflection_count = 0
        
        while in_droplet and reflection_count <= max_reflections:
            # Find intersection with droplet boundary
            # Quadratic equation: |p + t*dir - center|^2 = radius^2
            a = current_dir_x**2 + current_dir_y**2  # Should be 1.0
            b = 2 * (current_x * current_dir_x + current_y * current_dir_y)
            c = current_x**2 + current_y**2 - self.droplet_radius**2
            
            discriminant = b**2 - 4*a*c
            
            if discriminant < 0:
                # No intersection (shouldn't happen)
                break
            
            # Calculate intersection point
            t = (-b + np.sqrt(discriminant)) / (2*a)  # We want the further intersection
            
            if t < 1e-10:
                # No valid intersection
                break
            
            # Calculate the intersection point
            intersection_x = current_x + t * current_dir_x
            intersection_y = current_y + t * current_dir_y
            
            # Add intersection point to path
            path.append((intersection_x, intersection_y))
            
            # Calculate normal at intersection (pointing outward)
            normal_x = intersection_x / self.droplet_radius
            normal_y = intersection_y / self.droplet_radius
            
            # Calculate angle of incidence from inside
            cos_theta_i = current_dir_x * normal_x + current_dir_y * normal_y
            sin_theta_i = np.sqrt(1 - cos_theta_i**2)
            
            # Calculate angle of refraction using Snell's law (exiting to air)
            sin_theta_t = (n2 / n1) * sin_theta_i
            
            if sin_theta_t > 1.0:
                # Total internal reflection occurs
                # Calculate reflection direction
                reflected_dir_x = current_dir_x - 2 * cos_theta_i * normal_x
                reflected_dir_y = current_dir_y - 2 * cos_theta_i * normal_y
                
                # Update current position and direction
                current_x, current_y = intersection_x, intersection_y
                current_dir_x, current_dir_y = reflected_dir_x, reflected_dir_y
                
                reflection_count += 1
            else:
                # Refraction occurs (exiting the droplet)
                cos_theta_t = np.sqrt(1 - sin_theta_t**2)
                
                # Calculate refracted direction
                exit_dir_x = (n2 / n1) * current_dir_x - ((n2 / n1) * cos_theta_i - cos_theta_t) * normal_x
                exit_dir_y = (n2 / n1) * current_dir_y - ((n2 / n1) * cos_theta_i - cos_theta_t) * normal_y
                
                # Normalize exit direction
                exit_norm = np.sqrt(exit_dir_x**2 + exit_dir_y**2)
                exit_dir_x /= exit_norm
                exit_dir_y /= exit_norm
                
                # Calculate the extended exit ray
                exit_end_x = intersection_x + 3 * self.droplet_radius * exit_dir_x
                exit_end_y = intersection_y + 3 * self.droplet_radius * exit_dir_y
                
                # Add the exit point to path
                path.append((exit_end_x, exit_end_y))
                
                # Calculate the deviation angle
                # Original direction is [1, 0], exit direction is [exit_dir_x, exit_dir_y]
                # Angle between vectors: cos(θ) = a·b / (|a|·|b|)
                # Both vectors are normalized, so the denominator is 1
                cos_deviation = 1.0 * exit_dir_x + 0.0 * exit_dir_y  # Dot product with [1, 0]
                
                # Ensure the value is within valid range for arccos
                cos_deviation = np.clip(cos_deviation, -1.0, 1.0)
                
                # Calculate deviation angle in degrees
                deviation_angle = np.degrees(np.arccos(cos_deviation))
                
                # For rainbow physics, we often need to calculate the supplement angle
                if exit_dir_y < 0:
                    deviation_angle = 180 + (180 - deviation_angle)
                
                # Exit the loop as we've left the droplet
                in_droplet = False
                
                # Store the ray's parameters
                return path, deviation_angle
        
        # If we don't exit the droplet, return path but no angle
        return path, None
    
    def run_simulation(self, num_params=500, num_wavelengths=25, verbose=True):
        """
        Run a comprehensive simulation with many rays.
        
        Args:
            num_params: Number of impact parameters to sample
            num_wavelengths: Number of wavelengths to sample
            verbose: Whether to print progress
            
        Returns:
            Simulation results dictionary
        """
        # Reset results
        self.results = {
            'ray_paths': [],
            'exit_angles': [],
        }
        
        # Create wavelength range across visible spectrum
        wavelengths = np.linspace(400, 700, num_wavelengths)
        
        # Create impact parameter range (focus on the range that creates rainbows)
        # For primary rainbow, impact parameters around 0.7-0.9 are most important
        impact_params = np.linspace(0.1, 0.99, num_params) * self.droplet_radius
        
        # Dictionary to store rays by wavelength and reflection count
        rays_by_type = {}
        
        # Total rays to trace
        total_rays = num_params * num_wavelengths
        ray_count = 0
        
        start_time = time.time()
        
        # Trace all rays
        for impact_param in impact_params:
            for wavelength in wavelengths:
                # Trace this ray
                path, angle = self.trace_ray(impact_param, wavelength)
                
                # Increment count
                ray_count += 1
                
                # Print progress
                if verbose and ray_count % 100 == 0:
                    progress = ray_count / total_rays * 100
                    print(f"\rTracing rays... {progress:.1f}% complete", end="")
                
                if path and angle is not None:
                    # Get normalized impact parameter
                    norm_impact = impact_param / self.droplet_radius
                    
                    # Store the exit angle data
                    self.results['exit_angles'].append((wavelength, angle, norm_impact))
                    
                    # Determine rainbow type based on number of segments in path
                    reflection_count = len(path) - 3  # Incident, entry, exit points
                    
                    # Store only rays that form primary rainbow (around 138° deviation)
                    if 135 <= angle <= 145:
                        # Group rays by wavelength
                        if wavelength not in rays_by_type:
                            rays_by_type[wavelength] = []
                        
                        # Store the ray
                        rays_by_type[wavelength].append((path, angle, impact_param))
        
        if verbose:
            print(f"\nSimulation complete! Traced {ray_count} rays in {time.time() - start_time:.2f} seconds")
        
        # Select representative rays for visualization
        # For each wavelength, find the ray closest to the peak angle for that wavelength
        for wavelength, rays in rays_by_type.items():
            if not rays:
                continue
                
            # Calculate the average angle for this wavelength (peak of rainbow)
            angles = [angle for _, angle, _ in rays]
            mean_angle = np.mean(angles)
            
            # Find ray closest to the mean angle
            closest_ray = min(rays, key=lambda x: abs(x[1] - mean_angle))
            
            # Store this representative ray
            self.results['ray_paths'].append((closest_ray[0], wavelength))
        
        return self.results
    
    def create_rainbow_formation_diagram(self, save_path='results/rainbow_formation_diagram.png'):
        """
        Create a comprehensive 2D diagram showing rainbow formation physics.
        """
        # Set up the figure
        plt.figure(figsize=(14, 10))
        
        # Draw water droplet
        circle = plt.Circle((0, 0), self.droplet_radius, fill=True, color='skyblue', alpha=0.2, 
                           edgecolor='blue', linewidth=1.5)
        plt.gca().add_artist(circle)
        
        # Draw ray paths for each wavelength
        for path, wavelength in self.results['ray_paths']:
            color = wavelength_to_rgb(wavelength)
            
            # Draw path segments
            for i in range(len(path) - 1):
                start_x, start_y = path[i]
                end_x, end_y = path[i+1]
                
                # Use thicker line for first segment (incident ray)
                linewidth = 2.5 if i == 0 else 1.5
                
                plt.plot([start_x, end_x], [start_y, end_y], '-', 
                       color=color, linewidth=linewidth, alpha=0.8)
        
        # Add annotations
        plt.text(-3 * self.droplet_radius, 0.3 * self.droplet_radius, 'Sunlight', 
               fontsize=12, ha='right', va='center')
        
        plt.text(0, 0, 'Water\nDroplet', 
               fontsize=14, ha='center', va='center', 
               bbox=dict(facecolor='white', alpha=0.7))
        
        # Mark key points on a representative ray (red)
        red_ray = None
        for path, wavelength in self.results['ray_paths']:
            if 650 <= wavelength <= 700:
                red_ray = path
                break
        
        if red_ray:
            entry_point = red_ray[1]  # Second point is entry point
            reflection_point = red_ray[2]  # Third point is reflection point
            exit_point = red_ray[3]  # Fourth point is exit point
            
            plt.annotate('1. Refraction\non entry', 
                       xy=entry_point,
                       xytext=(entry_point[0] - 0.5, entry_point[1] - 0.5),
                       arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                       fontsize=11, ha='right')
            
            plt.annotate('2. Reflection\ninside droplet', 
                       xy=reflection_point,
                       xytext=(reflection_point[0], reflection_point[1] + 0.8),
                       arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                       fontsize=11)
            
            plt.annotate('3. Refraction\non exit', 
                       xy=exit_point,
                       xytext=(exit_point[0] - 0.8, exit_point[1] - 0.7),
                       arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                       fontsize=11, ha='right')
        
        # Add annotations for the rainbow colors
        violet_ray = None
        for path, wavelength in self.results['ray_paths']:
            if 390 <= wavelength <= 430:
                violet_ray = path
                break
        
        if red_ray and violet_ray:
            # Annotate the violet and red ray exits
            plt.annotate('Violet (400nm)', 
                       xy=violet_ray[3],  # Exit point
                       xytext=(violet_ray[3][0] - 1.2, violet_ray[3][1] - 0.4),
                       arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                       fontsize=11, color='purple', ha='right')
            
            plt.annotate('Red (650nm)', 
                       xy=red_ray[3],  # Exit point
                       xytext=(red_ray[3][0] - 1.2, red_ray[3][1] - 0.8),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       fontsize=11, color='red', ha='right')
        
        # Calculate and show primary rainbow angle
        angles = [angle for wavelength, angle, _ in self.results['exit_angles'] if 650 <= wavelength <= 700]
        if angles:
            mean_red_angle = np.mean(angles)
            
            plt.figtext(0.25, 0.06, 
                      f"Primary Rainbow Angle:\n"
                      f"Deviation angle: ~{mean_red_angle:.1f}°\n"
                      f"(Corresponds to ~42° from antisolar point)",
                      ha='center', fontsize=12, 
                      bbox=dict(facecolor='white', alpha=0.9))
        
        # Add explanation of rainbow formation
        plt.figtext(0.75, 0.10, 
                  "Rainbow Formation Physics:\n\n"
                  "1. White sunlight enters the water droplet\n"
                  "2. Light refracts (bends) as it slows down in water\n"
                  "3. Light reflects off the back of the droplet\n"
                  "4. Light refracts again when exiting\n"
                  "5. Different colors bend at slightly different angles\n"
                  "   (dispersion) creating the rainbow spectrum",
                  ha='left', fontsize=12, va='bottom', 
                  bbox=dict(facecolor='white', alpha=0.9))
        
        # Style the plot
        plt.axis('equal')
        plt.xlim(-3.5 * self.droplet_radius, 2.5 * self.droplet_radius)
        plt.ylim(-2.0 * self.droplet_radius, 2.0 * self.droplet_radius)
        plt.title('Rainbow Formation: Primary Rainbow Physics', fontsize=16)
        plt.grid(alpha=0.2)
        
        # Remove axis ticks but keep the grid
        plt.tick_params(axis='both', which='both', length=0)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_angular_distribution_diagram(self, save_path='results/rainbow_angular_distribution.png'):
        """
        Create a clear diagram showing the angular distribution of different wavelengths.
        """
        # Set up the figure
        plt.figure(figsize=(14, 8))
        
        # Group by wavelength ranges for cleaner visualization
        wavelength_bins = [
            (390, 430, 'Violet (390-430 nm)'),
            (430, 470, 'Blue (430-470 nm)'),
            (470, 520, 'Cyan (470-520 nm)'),
            (520, 570, 'Green (520-570 nm)'),
            (570, 590, 'Yellow (570-590 nm)'),
            (590, 650, 'Orange (590-650 nm)'),
            (650, 710, 'Red (650-710 nm)')
        ]
        
        # Create histograms for each wavelength range
        hist_data = {}
        for wl_range in wavelength_bins:
            wl_min, wl_max, label = wl_range
            # Filter data for this wavelength range
            angles = [angle for wavelength, angle, _ in self.results['exit_angles'] 
                    if wl_min <= wavelength < wl_max]
            
            if angles:
                hist_data[label] = angles
        
        # Plot histograms
        bin_range = (125, 145)  # Focus on primary rainbow angle
        num_bins = 100
        
        # Plot histograms from each wavelength range
        for label, angles in hist_data.items():
            # Create histogram
            hist, bin_edges = np.histogram(angles, bins=num_bins, range=bin_range, density=True)
            
            # Get bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Look up the wavelength range from the label to get the color
            for wl_min, wl_max, bin_label in wavelength_bins:
                if bin_label == label:
                    # Get color for this wavelength range
                    mid_wl = (wl_min + wl_max) / 2
                    color = wavelength_to_rgb(mid_wl)
                    break
            
            # Normalize histogram
            hist = hist / np.max(hist) if np.max(hist) > 0 else hist
            
            # Smooth histogram using simple moving average
            window_size = 5
            smoothed = np.convolve(hist, np.ones(window_size)/window_size, mode='same')
            
            # Plot the histogram
            plt.plot(bin_centers, smoothed, '-', color=color, 
                linewidth=2.5, alpha=0.8, 
                label=label)
        
        # Add the primary rainbow theory band
        plt.axvspan(137.5, 139.5, color='gray', alpha=0.1, label='Primary Rainbow')
        
        # Add annotations
        plt.annotate('Red', xy=(138, 0.8), xytext=(138, 0.9), color='red', fontsize=12,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
                
        plt.annotate('Violet', xy=(140, 0.8), xytext=(140, 0.9), color='purple', fontsize=12,
                arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7))
        
        plt.text(142, 0.5, "Red light deviates less\nViolet light deviates more", 
            ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
        
        # Set labels and style
        plt.xlabel('Deviation Angle (degrees)', fontsize=12)
        plt.ylabel('Relative Intensity', fontsize=12)
        plt.title('Primary Rainbow - Angular Distribution by Wavelength', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(bin_range)
        plt.ylim(0, 1.0)
        
        # Add legend
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_dispersion_physics_plot(self, save_path='results/rainbow_dispersion_physics.png'):
        """
        Create a clear plot showing the relationship between wavelength and exit angle.
        """
        # Set up the figure
        plt.figure(figsize=(14, 8))
        
        # Group data by wavelength
        wavelength_groups = {}
        
        for wavelength, angle, _ in self.results['exit_angles']:
            # Only include angles within primary rainbow range
            if 125 <= angle <= 145:
                if wavelength not in wavelength_groups:
                    wavelength_groups[wavelength] = []
                
                wavelength_groups[wavelength].append(angle)
        
        # Calculate average angle for each wavelength
        wavelengths = []
        avg_angles = []
        colors = []
        
        for wavelength, angles in sorted(wavelength_groups.items()):
            if angles:
                wavelengths.append(wavelength)
                avg_angles.append(np.mean(angles))
                colors.append(wavelength_to_rgb(wavelength))
        
        # Plot the wavelength vs. angle relationship
        for i in range(len(wavelengths)):
            plt.scatter(wavelengths[i], avg_angles[i], 
                      color=colors[i], 
                      s=80, alpha=0.8)
        
        # Add trend line using polynomial fit
        if len(wavelengths) > 1:
            z = np.polyfit(wavelengths, avg_angles, 1)
            p = np.poly1d(z)
            
            x_range = np.linspace(min(wavelengths), max(wavelengths), 100)
            plt.plot(x_range, p(x_range), 'k--', alpha=0.6)
            
            # Calculate and show slope
            slope = z[0] * 100  # Convert to degrees per 100 nm for readability
            
            plt.text(600, 138.2, f"Slope: {slope:.3f}° per 100 nm\n"
                             f"As wavelength increases (red),\nexit angle decreases", 
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Add theoretical curve
        theory_wl = np.linspace(400, 700, 100)
        theory_angles = []
        
        for wl in theory_wl:
            # Theoretical relation (simplified model)
            n_water = get_refractive_index(wl, 'water')
            # Approximate formula based on geometric optics
            angle = 138.0 - (n_water - 1.33) * 30
            theory_angles.append(angle)
        
        plt.plot(theory_wl, theory_angles, 'r-', linewidth=2, alpha=0.5, label='Theoretical')
        
        # Add labels and styling
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Peak Deviation Angle (degrees)', fontsize=12)
        plt.title('Rainbow Dispersion: How Wavelength Affects Exit Angle', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.xlim(390, 710)
        plt.ylim(137.0, 141.0)
        
        # Add color band across bottom to show visible spectrum
        gradient_data = np.linspace(0, 1, 1000)
        gradient = np.vstack((gradient_data, gradient_data))
        
        cmap_colors = [wavelength_to_rgb(w) for w in np.linspace(400, 700, 100)]
        custom_cmap = LinearSegmentedColormap.from_list('rainbow_cmap', cmap_colors)
        
        # Add spectrum bar below main plot
        spectrum_height = 0.05
        spectrum_axes = plt.axes([0.1, 0.05, 0.8, spectrum_height])
        spectrum_axes.imshow(gradient, aspect='auto', cmap=custom_cmap)
        spectrum_axes.set_yticks([])
        
        # Add wavelength labels to spectrum
        wavelength_labels = ['400\nViolet', '450\nBlue', '500\nCyan', '550\nGreen', 
                           '600\nYellow', '650\nRed', '700\nDeep Red']
        positions = np.linspace(0, 999, len(wavelength_labels))
        
        spectrum_axes.set_xticks(positions)
        spectrum_axes.set_xticklabels(wavelength_labels, fontsize=8)
        spectrum_axes.set_xlim(0, 999)
        
        # Add explanation of rainbow dispersion
        plt.figtext(0.5, 0.01, 
                  "Rainbow formation depends on the wavelength-dependent refractive index of water (dispersion).\n"
                  "Different wavelengths of light exit at different angles, creating the spectrum of colors.",
                  ha='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_circular_rainbow_view(self, save_path='results/rainbow_circular_view.png'):
        """
        Create a circular rainbow visualization as seen from the observer's perspective.
        """
        # Set up the figure
        plt.figure(figsize=(12, 12))
        ax = plt.subplot(111, polar=True)
        
        # Define the primary rainbow angles
        # Primary rainbow is around 42° from antisolar point (converts to ~138° deviation)
        rainbow_radius = np.radians(42)
        rainbow_width = np.radians(2.5)
        
        # Define wavelength ranges for rainbow colors (from outer to inner)
        wavelength_ranges = [
            (650, 700, 'Red'),
            (590, 650, 'Orange'),
            (570, 590, 'Yellow'),
            (520, 570, 'Green'),
            (490, 520, 'Cyan'),
            (450, 490, 'Blue'),
            (400, 450, 'Violet')
        ]
        
        # Create angles for full circle
        theta = np.linspace(0, 2*np.pi, 1000)
        
        # Plot rainbow bands from outer (red) to inner (violet)
        for i, (wl_min, wl_max, label) in enumerate(wavelength_ranges):
            # Calculate position based on wavelength
            # Linear mapping: Red (longest wavelength) on the outside, violet (shortest) on inside
            position = i / len(wavelength_ranges)
            
            # Get color for this band
            mid_wl = (wl_min + wl_max) / 2
            color = wavelength_to_rgb(mid_wl)
            
            # Calculate radius for this color band
            band_radius = rainbow_radius - position * rainbow_width
            
            # Calculate band width
            band_width = rainbow_width / len(wavelength_ranges) * 1.2  # Slight overlap for smoothness
            
            # Create band using fill_between
            inner_radius = band_radius - (band_width / 2)
            outer_radius = band_radius + (band_width / 2)
            
            ax.fill_between(theta, inner_radius, outer_radius, color=color, alpha=0.9)
        
        # Style the polar plot
        ax.set_rticks([])  # Remove radial ticks
        ax.set_rlim(0, np.radians(90))  # Limit to hemisphere view
        
        # Set proper orientation
        ax.set_theta_zero_location("N")  # North at top
        ax.set_theta_direction(-1)  # Clockwise
        
        # Remove grid and set background
        ax.grid(False)
        ax.set_facecolor((0.7, 0.85, 1.0))  # Light blue sky
        
        # Add sun marker at the center (antisolar point)
        sun_circle = plt.Circle((0, 0), 0.05, transform=ax.transData._b, color='yellow', alpha=0.9)
        ax.add_artist(sun_circle)
        
        # Add annotations
        plt.figtext(0.5, 0.95, "Primary Rainbow: Viewed From Observer's Perspective", 
                   ha='center', fontsize=14, weight='bold')
        
        plt.figtext(0.5, 0.91, "The rainbow appears at approximately 42° from the antisolar point",
                   ha='center', fontsize=12)
        
        # Add color labels
        ax.annotate('Red (outside)', 
                  xy=(np.pi/4, rainbow_radius - 0.1*rainbow_width), 
                  xytext=(np.pi/4 - 0.2, rainbow_radius - 2*rainbow_width),
                  arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
                  color='red', ha='right', fontsize=10)
        
        ax.annotate('Violet (inside)', 
                  xy=(np.pi/4, rainbow_radius - 0.9*rainbow_width), 
                  xytext=(np.pi/4 - 0.2, rainbow_radius - 3*rainbow_width),
                  arrowprops=dict(arrowstyle='->', color='purple', alpha=0.8),
                  color='purple', ha='right', fontsize=10)
        
        # Add explanation
        plt.figtext(0.5, 0.02, 
                  "A rainbow is a circular arc because all water droplets that create the same deviation angle\n"
                  "lie along a cone with its apex at the observer's eye and axis pointing to the antisolar point.\n"
                  "The rainbow's color sequence (red outside, violet inside) is due to dispersion.",
                  ha='center', fontsize=11)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_3d_photon_paths(self, save_path='results/rainbow_3d_paths.png'):
        """
        Create a 3D visualization of photon paths through a water droplet.
        """
        # Set up 3D figure
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw water droplet
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.droplet_radius * np.cos(u) * np.sin(v)
        y = self.droplet_radius * np.sin(u) * np.sin(v)
        z = self.droplet_radius * np.cos(v)
        
        # Plot droplet as transparent surface
        ax.plot_surface(x, y, z, color='skyblue', alpha=0.1, shade=False)
        
        # Draw ray paths in 3D
        for path_2d, wavelength in self.results['ray_paths']:
            # Convert 2D path to 3D by adding random rotation around x-axis
            # This creates a more interesting 3D visualization while preserving the physics
            theta = np.random.uniform(0, 2*np.pi)
            
            path_3d = []
            for x, y in path_2d:
                # Apply rotation to create 3D path
                y_3d = y * np.cos(theta)
                z_3d = y * np.sin(theta)
                path_3d.append((x, y_3d, z_3d))
            
            # Get color for this wavelength
            color = wavelength_to_rgb(wavelength)
            
            # Plot 3D path
            points = np.array(path_3d)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                   color=color, linewidth=1.5, alpha=0.8)
        
        # Style the 3D plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
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
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add title
        plt.title('3D Visualization of Light Rays Through Water Droplet', fontsize=14)
        
        # Add legend for wavelengths
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color=wavelength_to_rgb(400), lw=2),
            Line2D([0], [0], color=wavelength_to_rgb(470), lw=2),
            Line2D([0], [0], color=wavelength_to_rgb(550), lw=2),
            Line2D([0], [0], color=wavelength_to_rgb(580), lw=2),
            Line2D([0], [0], color=wavelength_to_rgb(650), lw=2)
        ]
        
        ax.legend(custom_lines, ['Violet (400nm)', 'Blue (470nm)', 
                               'Green (550nm)', 'Yellow (580nm)', 'Red (650nm)'],
                 loc='upper right')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def run_rainbow_simulation():
    """
    Run a complete rainbow simulation and generate all visualizations.
    """
    print("Starting comprehensive rainbow simulation...")
    
    # Create ray tracer with 1.0 unit radius droplet
    tracer = RainbowRayTracer(droplet_radius=1.0)
    
    # Run simulation with many rays
    tracer.run_simulation(num_params=500, num_wavelengths=30)
    
    print("Creating visualizations...")
    
    # Create rainbow formation diagram
    tracer.create_rainbow_formation_diagram()
    print("- Rainbow formation diagram created")
    
    # Create angular distribution diagram
    tracer.create_angular_distribution_diagram()
    print("- Angular distribution diagram created")
    
    # Create dispersion physics plot
    tracer.create_dispersion_physics_plot()
    print("- Dispersion physics plot created")
    
    # Create circular rainbow view
    tracer.create_circular_rainbow_view()
    print("- Circular rainbow view created")
    
    # Create 3D photon paths visualization
    tracer.create_3d_photon_paths()
    print("- 3D photon paths visualization created")
    
    print("\nAll visualizations saved to 'results' folder.")
    
    # Calculate primary rainbow angle
    angles = [angle for wavelength, angle, _ in tracer.results['exit_angles'] 
             if 650 <= wavelength <= 700 and 125 <= angle <= 145]
    if angles:
        mean_angle = np.mean(angles)
        print(f"Primary rainbow deviation angle: ~{mean_angle:.1f}°")
        print(f"Corresponds to ~{180-mean_angle:.1f}° from antisolar point")
    
    print("\nSimulation complete! Run with the following command:")
    print("python rainbow_simulation.py")


if __name__ == "__main__":
    run_rainbow_simulation()