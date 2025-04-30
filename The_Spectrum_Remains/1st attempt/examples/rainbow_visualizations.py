"""
Focusing on creating an accurate 2D rainbow formation diagram
that clearly shows the ray path for primary rainbow formation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

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
    Get refractive index based on wavelength.
    
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
        
        # Cauchy's formula coefficients for water - adjusted for better dispersion
        A = 1.324
        B = 0.00325
        C = 0.00031
        
        # Calculate refractive index
        n = A + B / (wl_um**2) + C / (wl_um**4)
        
        return n
    
    return 1.0  # Default


def trace_rainbow_ray(impact_param, wavelength, droplet_radius=1.0):
    """
    Trace a single ray through a water droplet to show rainbow formation.
    
    Args:
        impact_param: Impact parameter (distance from central axis)
        wavelength: Wavelength of light in nm
        droplet_radius: Radius of the water droplet
        
    Returns:
        Tuple of (ray_segments, exit_angle)
            ray_segments: List of (start_point, end_point) pairs
            exit_angle: Final deviation angle in degrees
    """
    # Normalized impact parameter (0-1)
    b = impact_param / droplet_radius
    
    # Get refractive indices
    n1 = get_refractive_index(wavelength, 'air')
    n2 = get_refractive_index(wavelength, 'water')
    
    # Ray starts from far left, moving horizontally
    ray_segments = []
    
    # Calculate entry point on the droplet
    entry_x = -np.sqrt(1 - b**2) * droplet_radius
    entry_y = b * droplet_radius
    
    # Add the incident ray segment
    start_point = [-3 * droplet_radius, entry_y]
    end_point = [entry_x, entry_y]
    ray_segments.append((start_point, end_point))
    
    # Calculate refraction at entry
    # Normal vector points inward
    normal = [-entry_x, -entry_y]
    normal = normal / np.linalg.norm(normal)
    
    # Incident direction (from left to right)
    incident_dir = [1, 0]
    
    # Calculate angle of incidence (with respect to normal)
    cos_theta_i = -np.dot(incident_dir, normal)
    sin_theta_i = np.sqrt(1 - cos_theta_i**2)
    
    # Calculate angle of refraction using Snell's law
    sin_theta_t = (n1 / n2) * sin_theta_i
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)
    
    # Calculate refracted direction
    refracted_dir = [(n1 / n2) * incident_dir[0] + (n1 / n2 * cos_theta_i - cos_theta_t) * normal[0],
                     (n1 / n2) * incident_dir[1] + (n1 / n2 * cos_theta_i - cos_theta_t) * normal[1]]
    refracted_dir = refracted_dir / np.linalg.norm(refracted_dir)
    
    # Now trace the ray inside the droplet until it hits the far side
    # Set up a ray equation: p(t) = entry_point + t * refracted_dir
    # Intersect with circle: |p(t) - center|^2 = radius^2
    
    # Quadratic equation coefficients
    # a = dot(refracted_dir, refracted_dir) = 1 (normalized)
    # b = 2 * dot(entry_point - center, refracted_dir)
    # c = dot(entry_point - center, entry_point - center) - radius^2
    
    b_coef = 2 * (entry_x * refracted_dir[0] + entry_y * refracted_dir[1])
    c_coef = entry_x**2 + entry_y**2 - droplet_radius**2
    
    # Solve quadratic equation: a*t^2 + b*t + c = 0
    discriminant = b_coef**2 - 4 * c_coef
    
    if discriminant >= 0:
        # Two intersection points; we want the farther one
        t1 = (-b_coef - np.sqrt(discriminant)) / 2
        t2 = (-b_coef + np.sqrt(discriminant)) / 2
        t = max(t1, t2)  # Take the larger value of t
        
        # Calculate the intersection point
        intersection_x = entry_x + t * refracted_dir[0]
        intersection_y = entry_y + t * refracted_dir[1]
        
        # Add the refracted ray segment inside the droplet
        ray_segments.append(([entry_x, entry_y], [intersection_x, intersection_y]))
        
        # Calculate reflection at the back of the droplet
        # Normal points outward from the center
        normal = [intersection_x, intersection_y]
        normal = normal / np.linalg.norm(normal)
        
        # Calculate reflection direction
        dot_product = np.dot(refracted_dir, normal)
        reflected_dir = [refracted_dir[0] - 2 * dot_product * normal[0],
                         refracted_dir[1] - 2 * dot_product * normal[1]]
        
        # Find the next intersection with the droplet surface
        b_coef = 2 * (intersection_x * reflected_dir[0] + intersection_y * reflected_dir[1])
        c_coef = intersection_x**2 + intersection_y**2 - droplet_radius**2
        
        discriminant = b_coef**2 - 4 * c_coef
        
        if discriminant >= 0:
            # Two intersection points; we want the farther one
            t1 = (-b_coef - np.sqrt(discriminant)) / 2
            t2 = (-b_coef + np.sqrt(discriminant)) / 2
            t = max(t1, t2)  # Take the larger value of t
            
            # Calculate the exit point
            exit_x = intersection_x + t * reflected_dir[0]
            exit_y = intersection_y + t * reflected_dir[1]
            
            # Add the reflected ray segment inside the droplet
            ray_segments.append(([intersection_x, intersection_y], [exit_x, exit_y]))
            
            # Calculate refraction at exit
            # Normal points outward
            normal = [exit_x, exit_y]
            normal = normal / np.linalg.norm(normal)
            
            # Calculate angle of incidence (with respect to normal)
            cos_theta_i = np.dot(reflected_dir, normal)
            sin_theta_i = np.sqrt(1 - cos_theta_i**2)
            
            # Calculate angle of refraction using Snell's law
            sin_theta_t = (n2 / n1) * sin_theta_i
            
            # Check for total internal reflection
            if sin_theta_t > 1.0:
                # Total internal reflection occurs
                return ray_segments, None
            
            cos_theta_t = np.sqrt(1 - sin_theta_t**2)
            
            # Calculate refracted direction (exiting the droplet)
            exit_dir = [(n2 / n1) * reflected_dir[0] + ((n2 / n1) * cos_theta_i - cos_theta_t) * normal[0],
                        (n2 / n1) * reflected_dir[1] + ((n2 / n1) * cos_theta_i - cos_theta_t) * normal[1]]
            exit_dir = exit_dir / np.linalg.norm(exit_dir)
            
            # Add the exit ray segment
            ray_segments.append(([exit_x, exit_y], [exit_x + 2 * droplet_radius * exit_dir[0], 
                                                   exit_y + 2 * droplet_radius * exit_dir[1]]))
            
            # Calculate the deviation angle
            incident_angle = 0  # In radians (horizontal from left to right)
            exit_angle = np.arctan2(exit_dir[1], exit_dir[0])
            
            # Convert to degrees and calculate total deviation
            deviation_angle = (exit_angle - incident_angle) * 180 / np.pi
            
            # Ensure the angle is in the expected range for a rainbow
            if deviation_angle < 0:
                deviation_angle += 360
            
            return ray_segments, deviation_angle
    
    # If we get here, something went wrong
    return ray_segments, None


def create_rainbow_formation_diagram(save_path='results/rainbow_formation_diagram.png'):
    """
    Create a comprehensive 2D diagram showing rainbow formation physics.
    """
    # Set up the figure with a larger size for better detail
    plt.figure(figsize=(14, 10))
    
    # Draw water droplet
    droplet_radius = 1.0
    circle = plt.Circle((0, 0), droplet_radius, fill=True, color='skyblue', alpha=0.2, 
                       edgecolor='blue', linewidth=1.5)
    plt.gca().add_artist(circle)
    
    # Define wavelengths for rainbow colors
    wavelengths = [660, 610, 580, 520, 480, 430, 400]  # Red to violet
    
    # Trace rays with multiple impact parameters
    impact_params = np.linspace(0.3, 0.95, 15)
    
    # First - trace a single white light ray to show the entry
    midpoint_ip = 0.75  # Choose a good impact parameter that shows clear rainbow effect
    plt.arrow(-3 * droplet_radius, midpoint_ip * droplet_radius, 
             1.5 * droplet_radius, 0, 
             head_width=0.05 * droplet_radius, 
             head_length=0.1 * droplet_radius, 
             fc='white', ec='black', linewidth=2)
    
    # Store deviation angles for each wavelength
    deviation_angles = {wl: [] for wl in wavelengths}
    
    # Special impact parameter for detailed illustration
    special_ip = 0.8
    special_rays = {}
    all_exit_points = []
    
    # Trace rays for the special impact parameter with all wavelengths
    for wavelength in wavelengths:
        ray_segments, angle = trace_rainbow_ray(special_ip * droplet_radius, wavelength, droplet_radius)
        if angle is not None:
            special_rays[wavelength] = ray_segments
            color = wavelength_to_rgb(wavelength)
            
            # Draw each segment of the ray
            for i, (start, end) in enumerate(ray_segments):
                if i == 0:  # Skip the incident ray (we'll draw it as white)
                    continue
                plt.plot([start[0], end[0]], [start[1], end[1]], '-', 
                       color=color, linewidth=2.5, alpha=0.8)
                
                # Store exit points for annotation
                if i == len(ray_segments) - 1:  # This is the exit ray
                    exit_point = start  # The start of the exit ray is the exit point on the droplet
                    all_exit_points.append((exit_point, wavelength))
    
    # Add white incident ray for the special impact parameter
    if 660 in special_rays:  # Check if we have the red wavelength ray
        start, end = special_rays[660][0]  # Get the incident ray segment
        plt.plot([start[0], end[0]], [start[1], end[1]], '-', 
               color='white', linewidth=2.5, alpha=0.8)
    
    # Add subtle rays for multiple impact parameters to show the distribution
    for impact_param in impact_params:
        for wavelength in wavelengths:
            ray_segments, angle = trace_rainbow_ray(impact_param * droplet_radius, wavelength, droplet_radius)
            if angle is not None:
                color = wavelength_to_rgb(wavelength)
                deviation_angles[wavelength].append(angle)
                
                # Only show subtle hints of these rays for the exit segment
                if len(ray_segments) > 2:  # If we have an exit ray
                    start, end = ray_segments[-1]  # Get the exit ray segment
                    plt.plot([start[0], end[0]], [start[1], end[1]], '-', 
                           color=color, linewidth=0.8, alpha=0.2)
    
    # Find average deviation angle for each wavelength
    avg_angles = {}
    for wl, angles in deviation_angles.items():
        if angles:  # If we have any valid angles
            avg_angles[wl] = np.mean(angles)
    
    # Add labels and annotations
    plt.text(-3.3 * droplet_radius, 0.7 * droplet_radius, 'Sunlight', 
           fontsize=12, ha='right', va='center')
    
    plt.text(0, 0, 'Water\nDroplet', 
           fontsize=14, ha='center', va='center', 
           bbox=dict(facecolor='white', alpha=0.7))
    
    # Annotate the entry, reflection, and exit points on the special ray path
    if len(special_rays) > 0:
        # Use the red ray for annotations
        ref_wavelength = 660
        if ref_wavelength in special_rays:
            ray_path = special_rays[ref_wavelength]
            
            if len(ray_path) >= 3:
                entry_point = ray_path[0][1]  # End of the incident ray
                reflection_point = ray_path[1][1]  # End of the first internal ray
                exit_point = ray_path[2][0]  # Start of the exit ray
                
                plt.annotate('1. Refraction\non entry', 
                           xy=(entry_point[0], entry_point[1]),
                           xytext=(entry_point[0] - 0.3, entry_point[1] - 0.5),
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                           fontsize=11, ha='right')
                
                plt.annotate('2. Reflection\ninside droplet', 
                           xy=(reflection_point[0], reflection_point[1]),
                           xytext=(reflection_point[0] + 0.5, reflection_point[1] + 0.5),
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                           fontsize=11)
                
                plt.annotate('3. Refraction\non exit', 
                           xy=(exit_point[0], exit_point[1]),
                           xytext=(exit_point[0] - 0.5, exit_point[1] - 0.6),
                           arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                           fontsize=11, ha='right')
    
    # Add annotations for the rainbow colors
    # Find the two most extreme wavelengths for annotation
    if all_exit_points:
        # Sort by wavelength
        all_exit_points.sort(key=lambda x: x[1])
        
        # Get violet point (lowest wavelength)
        violet_point, violet_wl = all_exit_points[0]
        
        # Get red point (highest wavelength)
        red_point, red_wl = all_exit_points[-1]
        
        # Annotate the violet ray exit
        plt.annotate('Violet (400nm)', 
                   xy=(violet_point[0], violet_point[1]),
                   xytext=(violet_point[0] - 0.8, violet_point[1] - 0.7),
                   arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                   fontsize=11, color='purple', ha='right')
        
        # Annotate the red ray exit
        plt.annotate('Red (660nm)', 
                   xy=(red_point[0], red_point[1]),
                   xytext=(red_point[0] - 0.8, red_point[1] - 0.9),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   fontsize=11, color='red', ha='right')
    
    # Add information about the primary rainbow angle
    if avg_angles:
        # Get the average angles for red and violet
        if 660 in avg_angles and 400 in avg_angles:
            red_angle = avg_angles[660]
            violet_angle = avg_angles[400]
            
            plt.figtext(0.25, 0.06, 
                      f"Primary Rainbow (~138° deviation)\n"
                      f"Red: ~{red_angle:.1f}°\n"
                      f"Violet: ~{violet_angle:.1f}°",
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
    plt.xlim(-3.5 * droplet_radius, 3.5 * droplet_radius)
    plt.ylim(-2 * droplet_radius, 2 * droplet_radius)
    plt.title('Rainbow Formation: Primary Rainbow Physics', fontsize=16)
    plt.grid(alpha=0.2)
    
    # Add subtle grid for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.2)
    
    # Remove axis ticks but keep the grid
    plt.tick_params(axis='both', which='both', length=0)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_primary_rainbow_simulation(num_impact_params=200, num_wavelengths=20, 
                                     save_path='results/primary_rainbow_simulation.png'):
    """
    Create a detailed simulation of primary rainbow formation with many rays.
    
    Args:
        num_impact_params: Number of different impact parameters to simulate
        num_wavelengths: Number of wavelengths to simulate
        save_path: Path to save the visualization
    """
    # Set up the figure
    plt.figure(figsize=(14, 10))
    
    # Draw water droplet
    droplet_radius = 1.0
    circle = plt.Circle((0, 0), droplet_radius, fill=True, color='skyblue', alpha=0.1, 
                       edgecolor='blue', linewidth=1)
    plt.gca().add_artist(circle)
    
    # Generate wavelengths across the visible spectrum
    wavelengths = np.linspace(400, 660, num_wavelengths)
    
    # Generate impact parameters
    impact_params = np.linspace(0.4, 0.99, num_impact_params)
    
    # Store exit angles and colors
    exit_rays = []
    
    # Trace rays for all combinations
    for impact_param in impact_params:
        for wavelength in wavelengths:
            ray_segments, angle = trace_rainbow_ray(impact_param * droplet_radius, wavelength, droplet_radius)
            
            if angle is not None:
                color = wavelength_to_rgb(wavelength)
                
                # Store exit ray for later
                if len(ray_segments) > 2:
                    start, end = ray_segments[-1]
                    exit_rays.append((start, end, angle, color, wavelength))
    
    # Create histogram of exit angles to find the primary rainbow angle
    if exit_rays:
        angles = [angle for _, _, angle, _, _ in exit_rays]
        
        # Sort rays by angle
        exit_rays.sort(key=lambda x: x[2])
        
        # Plot exit rays, focusing on those within primary rainbow angle range
        for start, end, angle, color, wavelength in exit_rays:
            # Only show exit rays that form the rainbow
            if 136 <= angle <= 142:
                # Calculate vector for the line
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                
                # Normalize and extend the line
                length = np.sqrt(dx**2 + dy**2)
                dx = dx / length * 3 * droplet_radius
                dy = dy / length * 3 * droplet_radius
                
                # Draw the exit ray with color based on wavelength
                plt.plot([start[0], start[0] + dx], [start[1], start[1] + dy], '-', 
                       color=color, linewidth=0.8, alpha=0.6)
    
    # Style the plot
    plt.axis('equal')
    plt.xlim(-3.5 * droplet_radius, 3.5 * droplet_radius)
    plt.ylim(-3 * droplet_radius, 3 * droplet_radius)
    plt.title('Primary Rainbow Simulation: Multiple Light Rays', fontsize=16)
    
    # Add subtle grid for reference
    plt.grid(alpha=0.1)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.2)
    
    # Add annotations explaining the rainbow formation
    plt.figtext(0.5, 0.02, 
              "This simulation shows how thousands of light rays exit a water droplet.\n"
              "Notice how they concentrate at specific angles, creating the rainbow.\n"
              "Different colors (wavelengths) exit at slightly different angles due to dispersion.",
              ha='center', fontsize=12, 
              bbox=dict(facecolor='white', alpha=0.9))
    
    # Add a spectrum bar for reference
    axes_pos = plt.gca().get_position()
    cbar_ax = plt.axes([axes_pos.x0, axes_pos.y0 - 0.08, axes_pos.width, 0.02])
    
    gradient = np.linspace(0, 1, 1000)
    gradient = np.vstack((gradient, gradient))
    
    custom_cmap = LinearSegmentedColormap.from_list(
        'rainbow', 
        [wavelength_to_rgb(w) for w in np.linspace(400, 700, 100)]
    )
    
    cbar_ax.imshow(gradient, aspect='auto', cmap=custom_cmap)
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([0, 250, 500, 750, 999])
    cbar_ax.set_xticklabels(['400nm\n(Violet)', '480nm\n(Blue)', '550nm\n(Green)', 
                           '600nm\n(Orange)', '700nm\n(Red)'])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_angular_distribution_diagram(save_path='results/rainbow_angular_distribution.png'):
    """
    Create a clear diagram showing the angular distribution of different wavelengths.
    """
    # Set up the figure
    plt.figure(figsize=(14, 8))
    
    # Generate wavelengths across the visible spectrum
    wavelengths = np.linspace(400, 660, 7)
    
    # Define impact parameters - use a good range for primary rainbow
    impact_params = np.linspace(0.4, 0.99, 100)
    
    # Store deviation angles for each wavelength
    angles_by_wavelength = {wl: [] for wl in wavelengths}
    
    # Trace rays for all combinations
    for impact_param in impact_params:
        for wavelength in wavelengths:
            _, angle = trace_rainbow_ray(impact_param, wavelength)
            
            if angle is not None:
                angles_by_wavelength[wavelength].append(angle)
    
    # Create histograms for each wavelength
    bin_range = (135, 145)  # Focus on primary rainbow angle
    num_bins = 100
    
    # Plot histograms
    for wavelength in wavelengths:
        angles = angles_by_wavelength[wavelength]
        
        if angles:
            # Create histogram
            hist, bin_edges = np.histogram(angles, bins=num_bins, range=bin_range, density=True)
            
            # Get bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Get color for this wavelength
            color = wavelength_to_rgb(wavelength)
            
            # Plot the histogram
            plt.plot(bin_centers, hist, '-', color=color, 
                   linewidth=2.5, alpha=0.8, 
                   label=f"{wavelength:.0f} nm")
    
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


def create_dispersion_physics_plot(save_path='results/rainbow_dispersion_physics.png'):
    """
    Create a clear plot showing the relationship between wavelength and exit angle.
    """
    # Set up the figure
    plt.figure(figsize=(14, 8))
    
    # Generate wavelengths across the visible spectrum
    wavelengths = np.linspace(400, 660, 30)
    
    # Define impact parameters for optimal rainbow formation
    impact_params = np.linspace(0.7, 0.9, 50)
    
    # Store average deviation angle for each wavelength
    avg_angles = {}
    
    # Trace rays for all combinations
    for wavelength in wavelengths:
        angles = []
        
        for impact_param in impact_params:
            _, angle = trace_rainbow_ray(impact_param, wavelength)
            
            if angle is not None and 137 <= angle <= 141:
                angles.append(angle)
        
        if angles:
            avg_angles[wavelength] = np.mean(angles)
    
    # Extract data for plotting
    plot_wavelengths = []
    plot_angles = []
    plot_colors = []
    
    for wl in sorted(avg_angles.keys()):
        plot_wavelengths.append(wl)
        plot_angles.append(avg_angles[wl])
        plot_colors.append(wavelength_to_rgb(wl))
    
    # Plot the wavelength vs. angle relationship
    for i in range(len(plot_wavelengths)):
        plt.scatter(plot_wavelengths[i], plot_angles[i], 
                  color=plot_colors[i], 
                  s=80, alpha=0.8)
    
    # Add trend line
    if len(plot_wavelengths) > 1:
        # Use polynomial fit for trend line
        z = np.polyfit(plot_wavelengths, plot_angles, 1)
        p = np.poly1d(z)
        
        x_range = np.linspace(min(plot_wavelengths), max(plot_wavelengths), 100)
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
        # Simple theoretical model based on dispersion
        n_water = get_refractive_index(wl, 'water')
        # Approximate relation between refractive index and exit angle
        angle = 138.5 - (n_water - 1.33) * 25
        theory_angles.append(angle)
    
    plt.plot(theory_wl, theory_angles, 'r-', linewidth=2, alpha=0.5, label='Theoretical')
    
    # Add labels and styling
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Peak Deviation Angle (degrees)', fontsize=12)
    plt.title('Rainbow Dispersion: How Wavelength Affects Exit Angle', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.xlim(390, 710)
    plt.ylim(137.5, 140.5)
    
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


def create_circular_rainbow_view(save_path='results/rainbow_circular_view.png'):
    """
    Create a circular rainbow visualization as seen from the observer's perspective.
    """
    # Set up the figure
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    
    # Define the primary rainbow angles
    rainbow_radius = np.radians(42)  # Corresponds to ~138° deviation
    rainbow_width = np.radians(2.5)
    
    # Define wavelength ranges for rainbow colors (from outer to inner)
    wavelength_ranges = [
        (620, 700, 'Red'),
        (590, 620, 'Orange'),
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


def run_rainbow_visualizations():
    """
    Run all rainbow visualization functions.
    """
    print("Creating accurate rainbow visualizations...")
    
    # Create the detailed rainbow formation diagram
    create_rainbow_formation_diagram()
    print("- Rainbow formation diagram created")
    
    # Create the multi-ray simulation
    create_primary_rainbow_simulation()
    print("- Primary rainbow ray simulation created")
    
    # Create the angular distribution diagram
    create_angular_distribution_diagram()
    print("- Angular distribution diagram created")
    
    # Create the dispersion physics plot
    create_dispersion_physics_plot()
    print("- Dispersion physics plot created")
    
    # Create the circular rainbow view
    create_circular_rainbow_view()
    print("- Circular rainbow view created")
    
    print("\nAll visualizations saved to 'results' folder.")
    print("These diagrams show the correct physics of rainbow formation.")


if __name__ == "__main__":
    run_rainbow_visualizations()