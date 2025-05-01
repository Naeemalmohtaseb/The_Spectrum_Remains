import numpy as np
import matplotlib.pyplot as plt

def trace_rainbow_ray(impact_param, wavelength, droplet_radius=1.0):
    """Trace a single ray through a water droplet."""
    
    # Calculate entry point based on impact parameter
    entry_x = -np.sqrt(droplet_radius**2 - impact_param**2)
    entry_y = impact_param
    
    # Get refractive indices (simplified Cauchy equation)
    n_air = 1.0003
    n_water = 1.324 + 0.00325 * (550/wavelength)**2
    
    # Initialize path with initial position and entry point
    path = [[-3.0, impact_param, 0.0], [entry_x, entry_y, 0.0]]
    
    # Calculate refraction at entry
    normal = [-entry_x/droplet_radius, -entry_y/droplet_radius, 0.0]
    incident = [1.0, 0.0, 0.0]  # Horizontal ray
    
    # Calculate refracted direction using Snell's law
    cos_theta_i = -np.dot(normal, incident)
    sin_theta_i = np.sqrt(1 - cos_theta_i**2)
    sin_theta_t = (n_air / n_water) * sin_theta_i
    
    if sin_theta_t > 1.0:  # Check for total internal reflection
        return path, None
        
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)
    refracted = [(n_air / n_water) * incident[0] + ((n_air / n_water) * cos_theta_i - cos_theta_t) * normal[0],
                 (n_air / n_water) * incident[1] + ((n_air / n_water) * cos_theta_i - cos_theta_t) * normal[1],
                 0.0]
    refracted = refracted / np.linalg.norm(refracted)
    
    # Find internal reflection point
    # Solve quadratic equation for intersection with circle
    a = refracted[0]**2 + refracted[1]**2
    b = 2 * (entry_x * refracted[0] + entry_y * refracted[1])
    c = entry_x**2 + entry_y**2 - droplet_radius**2
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return path, None
        
    t = (-b + np.sqrt(discriminant)) / (2*a)  # Farther intersection
    reflection_x = entry_x + t * refracted[0]
    reflection_y = entry_y + t * refracted[1]
    
    # Add reflection point to path
    path.append([reflection_x, reflection_y, 0.0])
    
    # Calculate reflection
    normal = [reflection_x/droplet_radius, reflection_y/droplet_radius, 0.0]
    dot_product = np.dot(refracted, normal)
    reflected = [refracted[0] - 2 * dot_product * normal[0],
                 refracted[1] - 2 * dot_product * normal[1],
                 0.0]
    
    # Find exit point
    a = reflected[0]**2 + reflected[1]**2
    b = 2 * (reflection_x * reflected[0] + reflection_y * reflected[1])
    c = reflection_x**2 + reflection_y**2 - droplet_radius**2
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return path, None
        
    t = (-b + np.sqrt(discriminant)) / (2*a)  # Farther intersection
    exit_x = reflection_x + t * reflected[0]
    exit_y = reflection_y + t * reflected[1]
    
    # Add exit point to path
    path.append([exit_x, exit_y, 0.0])
    
    # Calculate refraction at exit
    normal = [exit_x/droplet_radius, exit_y/droplet_radius, 0.0]
    
    # Calculate refracted direction using Snell's law
    cos_theta_i = np.dot(normal, reflected)
    sin_theta_i = np.sqrt(1 - cos_theta_i**2)
    sin_theta_t = (n_water / n_air) * sin_theta_i
    
    if sin_theta_t > 1.0:  # Check for total internal reflection
        return path, None
        
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)
    exit_dir = [(n_water / n_air) * reflected[0] - ((n_water / n_air) * cos_theta_i - cos_theta_t) * normal[0],
                (n_water / n_air) * reflected[1] - ((n_water / n_air) * cos_theta_i - cos_theta_t) * normal[1],
                0.0]
    exit_dir = exit_dir / np.linalg.norm(exit_dir)
    
    # Add extended exit ray
    path.append([exit_x + 3.0 * exit_dir[0], exit_y + 3.0 * exit_dir[1], 0.0])
    
    # Calculate exit angle
    exit_angle = np.arctan2(exit_dir[1], exit_dir[0])
    
    return np.array(path), exit_angle

def wavelength_to_rgb(wavelength):
    """Convert wavelength to RGB color for visualization."""
    if wavelength < 380 or wavelength > 750:
        return (0.0, 0.0, 0.0)
    
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
    
    # Scale RGB values
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

def improved_rainbow_test():
    """Create proper visualization of rainbow formation with multiple rays."""
    
    plt.figure(figsize=(10, 8))
    
    # Draw water droplet
    circle = plt.Circle((0, 0), 1.0, color='skyblue', alpha=0.3)
    plt.gca().add_artist(circle)
    
    # Define wavelengths and impact parameters
    wavelengths = [410, 450, 500, 550, 600, 650]  # violet to red
    impact_params = np.linspace(0.5, 0.99, 15)  # Range of impact parameters
    
    # Store exit angles for analysis
    exit_angles = {wl: [] for wl in wavelengths}
    
    # Trace rays
    for wavelength in wavelengths:
        color = wavelength_to_rgb(wavelength)
        
        for impact in impact_params:
            path, exit_angle = trace_rainbow_ray(impact, wavelength)
            
            if path is not None and exit_angle is not None:
                # Only show rays that form rainbow (after one internal reflection)
                plt.plot(path[:, 0], path[:, 1], '-', color=color, alpha=0.7, linewidth=1.5)
                exit_angles[wavelength].append(exit_angle)
    
    # Print average exit angles
    for wl in wavelengths:
        if exit_angles[wl]:
            avg_angle = np.degrees(np.mean(exit_angles[wl]))
            print(f"λ={wl}nm: Average exit angle = {avg_angle:.2f}°")
    
    # Styling
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.xlim(-1.5, 3.0)
    plt.ylim(-1.5, 1.5)
    plt.title('Rainbow Formation - Water Droplet')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Create legend for wavelengths
    legend_handles = []
    for wl in wavelengths:
        color = wavelength_to_rgb(wl)
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=f'{wl}nm'))
    plt.legend(handles=legend_handles, loc='upper right')
    
    plt.savefig('rainbow_formation.png', dpi=300)
    print("Rainbow formation visualization saved to 'rainbow_formation.png'")

if __name__ == "__main__":
    improved_rainbow_test()