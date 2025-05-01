import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

class WhiteLight:
    """Represents a white light ray containing multiple wavelengths."""
    
    def __init__(self, position=(0,0), direction=(1,0), num_wavelengths=30, intensity=1.0):
        """
        Initialize a white light ray with spectral components.
        
        Args:
            position: Starting position as (x,y) tuple
            direction: Direction vector as (dx,dy) tuple
            num_wavelengths: Number of wavelength components
            intensity: Initial intensity of the ray
        """
        self.position = np.array(position)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))
        self.intensity = intensity
        
        # Create spectral components covering visible spectrum
        self.min_wavelength = 380  # violet
        self.max_wavelength = 750  # red
        self.wavelengths = np.linspace(self.min_wavelength, self.max_wavelength, num_wavelengths)
        
        # Each component has its own path
        self.paths = {wl: [self.position.copy()] for wl in self.wavelengths}
        
        # Track component directions (initially all the same)
        self.directions = {wl: self.direction.copy() for wl in self.wavelengths}
        
        # Track component intensities
        self.intensities = {wl: self.intensity for wl in self.wavelengths}
        
        # Track if component is active
        self.active = {wl: True for wl in self.wavelengths}

    def get_refractive_index(self, wavelength, medium='water'):
        """Get refractive index for a wavelength in a medium."""
        if medium == 'air':
            return 1.0003  # Approximately constant for air
        elif medium == 'water':
            # Cauchy's equation for water (simplified)
            wl_microns = wavelength / 1000.0
            return 1.324 + 0.00325/(wl_microns**2) + 0.00031/(wl_microns**4)
        else:
            return 1.0  # Default

    def calculate_fresnel(self, n1, n2, cos_theta_i):
        """
        Calculate Fresnel coefficients for reflection/transmission.
        
        Args:
            n1: Refractive index of first medium
            n2: Refractive index of second medium
            cos_theta_i: Cosine of incident angle
            
        Returns:
            Tuple of (reflection coefficient, transmission coefficient)
        """
        # Ensure cos_theta_i is in valid range
        cos_theta_i = np.clip(cos_theta_i, -1.0, 1.0)
        
        # Calculate sin²(θ_t) using Snell's law
        sin2_theta_i = 1.0 - cos_theta_i**2
        sin2_theta_t = (n1/n2)**2 * sin2_theta_i
        
        # Check for total internal reflection
        if sin2_theta_t >= 1.0:
            return 1.0, 0.0
        
        cos_theta_t = np.sqrt(1.0 - sin2_theta_t)
        
        # Calculate reflection coefficients for s and p polarizations
        r_s = ((n1*cos_theta_i - n2*cos_theta_t) / 
               (n1*cos_theta_i + n2*cos_theta_t))**2
        
        r_p = ((n1*cos_theta_t - n2*cos_theta_i) / 
               (n1*cos_theta_t + n2*cos_theta_i))**2
        
        # Average for unpolarized light
        r = (r_s + r_p) / 2.0
        
        # Transmittance from conservation of energy
        t = 1.0 - r
        
        return r, t

    def process_interface(self, wavelength, normal, is_entering):
        """
        Process refraction/reflection at an interface for a specific wavelength.
        
        Args:
            wavelength: Wavelength to process
            normal: Surface normal vector at interface
            is_entering: True if entering the droplet, False if exiting
            
        Returns:
            Tuple of (reflected_dir, transmitted_dir, r_coeff, t_coeff)
        """
        # Skip if component is no longer active
        if not self.active[wavelength]:
            return None, None, 0, 0
        
        # Get current direction
        incident = self.directions[wavelength]
        
        # Ensure normal points against incident direction
        if np.dot(normal, incident) > 0:
            normal = -normal
        
        # Get refractive indices
        if is_entering:
            n1 = self.get_refractive_index(wavelength, 'air')
            n2 = self.get_refractive_index(wavelength, 'water')
        else:
            n1 = self.get_refractive_index(wavelength, 'water')
            n2 = self.get_refractive_index(wavelength, 'air')
        
        # Calculate cosine of incident angle
        cos_theta_i = -np.dot(normal, incident)
        
        # Calculate Fresnel coefficients
        r_coeff, t_coeff = self.calculate_fresnel(n1, n2, cos_theta_i)
        
        # Calculate reflection direction
        reflected_dir = incident + 2.0 * cos_theta_i * normal
        
        # Calculate refraction direction using Snell's law
        sin_theta_i = np.sqrt(1.0 - cos_theta_i**2)
        sin_theta_t = (n1 / n2) * sin_theta_i
        
        # Check for total internal reflection
        if sin_theta_t >= 1.0:
            return reflected_dir, None, 1.0, 0.0
        
        cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)
        
        # Calculate transmitted direction
        transmitted_dir = (n1 / n2) * incident + ((n1 / n2) * cos_theta_i - cos_theta_t) * normal
        transmitted_dir = transmitted_dir / np.linalg.norm(transmitted_dir)
        
        return reflected_dir, transmitted_dir, r_coeff, t_coeff

    def trace_through_droplet(self, droplet_radius=1.0, max_depth=5, min_intensity=0.01):
        """
        Trace all wavelength components through a water droplet.
        
        Args:
            droplet_radius: Radius of the water droplet
            max_depth: Maximum number of ray segments to trace
            min_intensity: Minimum intensity to keep tracing a component
        """
        # For each wavelength component
        for wavelength in self.wavelengths:
            # Skip if already inactive
            if not self.active[wavelength]:
                continue
            
            # Current position and direction for this component
            position = self.paths[wavelength][-1]
            direction = self.directions[wavelength]
            
            # Flag tracking if we're inside the droplet
            inside_droplet = False
            
            # Track if we've had an internal reflection yet
            had_reflection = False
            
            # Trace up to max_depth segments
            for depth in range(max_depth):
                # Check if intensity is too low to continue
                if self.intensities[wavelength] < min_intensity:
                    self.active[wavelength] = False
                    break
                
                if inside_droplet:
                    # Inside droplet - find intersection with surface
                    a = np.sum(direction**2)
                    b = 2.0 * np.sum(position * direction)
                    c = np.sum(position**2) - droplet_radius**2
                    
                    discriminant = b**2 - 4*a*c
                    
                    if discriminant < 0:
                        self.active[wavelength] = False
                        break
                    
                    # Calculate intersection distance
                    t = (-b + np.sqrt(discriminant)) / (2.0 * a)
                    
                    # Calculate intersection point
                    intersection = position + t * direction
                    
                    # Calculate normal at intersection (pointing outward)
                    normal = intersection / droplet_radius
                    
                    # Add intersection to path
                    self.paths[wavelength].append(intersection)
                    
                    if not had_reflection:
                        # First internal hit - reflect instead of exiting
                        reflected_dir = direction - 2.0 * np.dot(direction, normal) * normal
                        
                        # Continue with reflection
                        position = intersection
                        direction = reflected_dir
                        self.directions[wavelength] = reflected_dir
                        
                        # Mark that we've had internal reflection
                        had_reflection = True
                    else:
                        # We've already had our reflection - now exit the droplet
                        # Process interface (exiting droplet)
                        reflected_dir, transmitted_dir, r, t = self.process_interface(
                            wavelength, normal, is_entering=False)
                        
                        # Exit droplet with transmission
                        if transmitted_dir is not None:
                            position = intersection
                            direction = transmitted_dir
                            self.directions[wavelength] = transmitted_dir
                            self.intensities[wavelength] *= t
                            
                            # Add extended exit ray
                            exit_point = position + 3.0 * direction
                            self.paths[wavelength].append(exit_point)
                        
                        # Done with this component
                        break
                else:
                    # Outside droplet - find entry point
                    a = np.sum(direction**2)
                    b = 2.0 * np.sum(position * direction)
                    c = np.sum(position**2) - droplet_radius**2
                    
                    discriminant = b**2 - 4*a*c
                    
                    if discriminant < 0:
                        # No intersection with droplet
                        self.active[wavelength] = False
                        break
                    
                    # Calculate intersection distances
                    t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
                    t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
                    
                    # We want the closest intersection in the forward direction
                    if t1 > 1e-10:  # Small epsilon to avoid numerical issues
                        t = t1
                    elif t2 > 1e-10:
                        t = t2
                    else:
                        # No forward intersection
                        self.active[wavelength] = False
                        break
                    
                    # Calculate intersection point
                    intersection = position + t * direction
                    
                    # Calculate normal at intersection (pointing outward)
                    normal = intersection / droplet_radius
                    
                    # Process interface (entering droplet)
                    reflected_dir, transmitted_dir, r, t = self.process_interface(
                        wavelength, normal, is_entering=True)
                    
                    # Add intersection to path
                    self.paths[wavelength].append(intersection)
                    
                    # Enter droplet with transmission
                    if transmitted_dir is not None:
                        position = intersection
                        direction = transmitted_dir
                        self.directions[wavelength] = transmitted_dir
                        self.intensities[wavelength] *= t
                        # Now inside
                        inside_droplet = True
                    else:
                        # Ray didn't enter droplet
                        self.active[wavelength] = False
                        break

    def wavelength_to_rgb(self, wavelength):
        """Convert wavelength to RGB color for visualization."""
        gamma = 0.8
        intensity = 1.0
        
        if wavelength < 380 or wavelength > 750:
            return (0, 0, 0)
            
        if wavelength < 440:
            r = (440 - wavelength) / (440 - 380)
            g = 0.0
            b = 1.0
        elif wavelength < 490:
            r = 0.0
            g = (wavelength - 440) / (490 - 440)
            b = 1.0
        elif wavelength < 510:
            r = 0.0
            g = 1.0
            b = (510 - wavelength) / (510 - 490)
        elif wavelength < 580:
            r = (wavelength - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif wavelength < 645:
            r = 1.0
            g = (645 - wavelength) / (645 - 580)
            b = 0.0
        else:
            r = 1.0
            g = 0.0
            b = 0.0
            
        # Adjust for lower intensity at spectrum edges
        if wavelength < 420:
            intensity = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif wavelength > 700:
            intensity = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
            
        # Apply gamma and intensity
        r = intensity * pow(r, gamma)
        g = intensity * pow(g, gamma)
        b = intensity * pow(b, gamma)
        
        return (r, g, b)

def simulate_rainbow():
    """Simulate rainbow formation from white light through a droplet."""
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Draw water droplet
    droplet = plt.Circle((0, 0), 1.0, color='skyblue', alpha=0.3, 
                        edgecolor='blue', linewidth=1)
    plt.gca().add_artist(droplet)
    
    # Set plot limits
    plt.xlim(-3, 4)
    plt.ylim(-2, 2)
    
    # Create a white light ray starting from left side
    light = WhiteLight(
        position=(-3.0, 0.8), 
        direction=(1.0, 0.0),
        num_wavelengths=30,
        intensity=1.0
    )
    
    # Trace through droplet
    light.trace_through_droplet(droplet_radius=1.0, max_depth=7, min_intensity=0.001)
    
    # Plot incoming white light ray before it hits the droplet
    entry_point = light.paths[light.wavelengths[0]][0]
    droplet_edge = light.paths[light.wavelengths[0]][1]
    plt.plot([entry_point[0], droplet_edge[0]], 
             [entry_point[1], droplet_edge[1]], 
             '-', color='white', linewidth=2.5)
    
    # Add an arrow to show direction
    arrow_pos = entry_point + 0.7 * (droplet_edge - entry_point)
    plt.arrow(arrow_pos[0], arrow_pos[1], 0.2, 0, 
              head_width=0.05, head_length=0.1, fc='white', ec='white')
    
    # Plot all wavelength paths separately with appropriate colors
    exit_angles = []
    
    for wavelength in light.wavelengths:
        path = np.array(light.paths[wavelength])
        if len(path) >= 3:  # Only plot if path has at least entry, internal, and exit points
            color = light.wavelength_to_rgb(wavelength)
            
            # Plot the path through and after the droplet
            plt.plot(path[1:, 0], path[1:, 1], '-', 
                     color=color, alpha=0.8, linewidth=1.5)
            
            # Calculate exit angle if the ray exited
            if len(path) >= 4:
                exit_dir = path[-1] - path[-2]
                exit_angle = np.arctan2(exit_dir[1], exit_dir[0])
                exit_angles.append((wavelength, np.degrees(exit_angle)))
    
    # Add title and labels
    plt.title('Rainbow Formation: White Light Splitting into Spectrum', fontsize=14)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Y Position', fontsize=12)
    plt.grid(alpha=0.3)
    
    # Display the rainbow angles
    if exit_angles:
        # Calculate rainbow angle
        min_wavelength = min(wl for wl, _ in exit_angles)
        max_wavelength = max(wl for wl, _ in exit_angles)
        
        min_angle = next((angle for wl, angle in exit_angles if wl == min_wavelength), None)
        max_angle = next((angle for wl, angle in exit_angles if wl == max_wavelength), None)
        
        if min_angle is not None and max_angle is not None:
            angle_span = abs(max_angle - min_angle)
            print(f"Rainbow angular span: {angle_span:.2f}° (from {min_angle:.2f}° to {max_angle:.2f}°)")
            
            plt.figtext(0.5, 0.01, 
                      f"Rainbow angular span: {angle_span:.2f}° from violet ({min_wavelength:.0f}nm) to red ({max_wavelength:.0f}nm)",
                      ha='center', fontsize=12)
    
    # Create a spectrum legend
    ax_legend = plt.axes([0.25, 0.025, 0.5, 0.03], frameon=False)
    
    # Create a gradient for the spectrum
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    
    # Create colors for the gradient
    spectrum_colors = [light.wavelength_to_rgb(wl) for wl in 
                      np.linspace(light.min_wavelength, light.max_wavelength, 256)]
    spectrum_cmap = LinearSegmentedColormap.from_list('spectrum', spectrum_colors)
    
    # Display the spectrum
    ax_legend.imshow(gradient, aspect='auto', cmap=spectrum_cmap)
    ax_legend.set_yticks([])
    
    # Add wavelength markers
    ax_legend.set_xticks([0, 64, 128, 192, 255])
    ax_legend.set_xticklabels(['400nm\n(Violet)', '500nm\n(Blue/Green)', 
                              '600nm\n(Yellow)', '700nm\n(Red)', '750nm'])
    
    plt.savefig('rainbow_spectrum.png', dpi=300, bbox_inches='tight')
    print("Rainbow simulation saved to 'rainbow_spectrum.png'")
    plt.show()

if __name__ == "__main__":
    simulate_rainbow()