"""
Animation of rainbow formation through water droplets.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import os
from tqdm import tqdm

def wavelength_to_rgb(wavelength):
    """Convert wavelength to RGB color."""
    # Function defined previously
    # Visible spectrum is approximately 380-750 nm
    if wavelength < 380 or wavelength > 750:
        return (0.0, 0.0, 0.0)  # Outside visible spectrum
    
    # Approximate conversion based on the CIE standard observer
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

def create_rainbow_formation_animation(num_frames=180, output_file='rainbow_animation.mp4', dpi=100):
    """
    Create an animation showing the formation of a rainbow.
    """
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Water droplet properties
    droplet_radius = 1.0
    droplet_center = np.array([0, 0])
    
    # Create wavelengths
    visible_min = 380
    visible_max = 750
    num_wavelengths = 7  # Classic rainbow colors
    
    wavelengths = np.linspace(visible_min, visible_max, num_wavelengths)
    
    # Initialize rays
    num_rays = 15  # Number of incident rays
    
    # Ray starting positions (evenly spaced across droplet)
    positions = []
    for i in range(num_rays):
        # Parameterize starting position by impact parameter
        # Impact parameter is distance from central axis
        b = (i / (num_rays - 1)) * droplet_radius
        positions.append(np.array([-3, b - droplet_radius/2]))
    
    # Direction is always to the right
    direction = np.array([1, 0])
    
    # Function to update animation
    def animate(frame):
        ax.clear()
        
        # Draw water droplet
        circle = plt.Circle(droplet_center, droplet_radius, fill=True, color='skyblue', alpha=0.3)
        ax.add_artist(circle)
        circle_outline = plt.Circle(droplet_center, droplet_radius, fill=False, color='blue', linewidth=1)
        ax.add_artist(circle_outline)
        
        # Relative progress for this frame
        progress = frame / num_frames
        
        # For each ray and wavelength
        for pos in positions:
            for wavelength in wavelengths:
                color = wavelength_to_rgb(wavelength)
                
                # Calculate ray path for this wavelength
                n_air = 1.0
                n_water = 1.33 + 0.02 * (750 - wavelength) / (750 - 380)  # Simplified dispersion
                
                # Initial ray
                ray_length = min(3 * progress, 3 * 0.3)  # Limit initial ray length
                ax.arrow(pos[0], pos[1], ray_length * direction[0], ray_length * direction[1],
                        head_width=0, head_length=0, fc=color, ec=color, linewidth=1.5, alpha=0.7)
                
                # Calculate intersection with droplet
                # Solve quadratic equation for intersection with circle
                a = direction[0]**2 + direction[1]**2  # Should be 1.0 for unit direction
                b = 2 * ((pos[0] - droplet_center[0]) * direction[0] + 
                         (pos[1] - droplet_center[1]) * direction[1])
                c = ((pos[0] - droplet_center[0])**2 + 
                     (pos[1] - droplet_center[1])**2 - droplet_radius**2)
                
                discriminant = b**2 - 4*a*c
                
                if discriminant > 0:
                    # Ray intersects the droplet
                    t1 = (-b + np.sqrt(discriminant)) / (2*a)
                    t2 = (-b - np.sqrt(discriminant)) / (2*a)
                    t = min(t1, t2)  # First intersection
                    
                    if t > 0:
                        # Calculate intersection point
                        intersection = pos + t * direction
                        
                        # If we've reached this part of the animation
                        if progress > 0.3:
                            # Draw the ray to the intersection
                            ax.arrow(pos[0], pos[1], (intersection[0] - pos[0]), (intersection[1] - pos[1]),
                                    head_width=0, head_length=0, fc=color, ec=color, linewidth=1.5, alpha=0.7)
                            
                            # Calculate refraction at entry point
                            normal = (intersection - droplet_center) / droplet_radius
                            incident = direction
                            
                            # Snell's law for refraction
                            cos_theta_i = -np.dot(normal, incident)
                            sin_theta_i = np.sqrt(1 - cos_theta_i**2)
                            sin_theta_t = (n_air / n_water) * sin_theta_i
                            
                            if sin_theta_t <= 1.0:  # Check for total internal reflection
                                cos_theta_t = np.sqrt(1 - sin_theta_t**2)
                                refracted = (n_air / n_water) * incident + (
                                    (n_air / n_water) * cos_theta_i - cos_theta_t) * normal
                                refracted = refracted / np.linalg.norm(refracted)
                                
                                # Calculate internal ray length for this frame
                                internal_progress = max(0, (progress - 0.3) / 0.2)
                                internal_ray_length = min(internal_progress * 2 * droplet_radius, 
                                                       2 * droplet_radius)
                                
                                # Find the second intersection point (from inside the droplet)
                                a_internal = 1  # Normalized direction
                                b_internal = 2 * np.dot(refracted, intersection - droplet_center)
                                c_internal = np.sum((intersection - droplet_center)**2) - droplet_radius**2
                                
                                discriminant_internal = b_internal**2 - 4*a_internal*c_internal
                                
                                if discriminant_internal > 0 and progress > 0.5:
                                    t1_internal = (-b_internal + np.sqrt(discriminant_internal)) / (2*a_internal)
                                    t2_internal = (-b_internal - np.sqrt(discriminant_internal)) / (2*a_internal)
                                    t_internal = max(t1_internal, t2_internal)  # Second intersection
                                    
                                    if t_internal > 1e-6:  # Avoid self-intersection
                                        # Calculate exit point
                                        exit_point = intersection + t_internal * refracted
                                        
                                        # Draw internal ray
                                        internal_endpoint = intersection + min(t_internal, internal_ray_length) * refracted
                                        ax.arrow(intersection[0], intersection[1], 
                                                (internal_endpoint[0] - intersection[0]), 
                                                (internal_endpoint[1] - intersection[1]),
                                                head_width=0, head_length=0, fc=color, ec=color, 
                                                linewidth=1.5, alpha=0.7)
                                        
                                        # If we've reached the reflection part
                                        if internal_ray_length >= t_internal and progress > 0.5:
                                            # Calculate reflection at back of droplet
                                            normal_internal = (exit_point - droplet_center) / droplet_radius
                                            
                                            # Internal reflection
                                            dot_product = np.dot(refracted, normal_internal)
                                            reflected = refracted - 2 * dot_product * normal_internal
                                            reflected = reflected / np.linalg.norm(reflected)
                                            
                                            # Calculate reflected ray length for this frame
                                            reflection_progress = max(0, (progress - 0.5) / 0.2)
                                            reflection_ray_length = min(reflection_progress * 2 * droplet_radius,
                                                                      2 * droplet_radius)
                                            
                                            # Find the third intersection point (after reflection)
                                            a_reflect = 1  # Normalized direction
                                            b_reflect = 2 * np.dot(reflected, exit_point - droplet_center)
                                            c_reflect = np.sum((exit_point - droplet_center)**2) - droplet_radius**2
                                            
                                            discriminant_reflect = b_reflect**2 - 4*a_reflect*c_reflect
                                            
                                            if discriminant_reflect > 0 and progress > 0.7:
                                                t1_reflect = (-b_reflect + np.sqrt(discriminant_reflect)) / (2*a_reflect)
                                                t2_reflect = (-b_reflect - np.sqrt(discriminant_reflect)) / (2*a_reflect)
                                                t_reflect = max(t1_reflect, t2_reflect)  # Third intersection
                                                
                                                if t_reflect > 1e-6:  # Avoid self-intersection
                                                    # Calculate final exit point
                                                    final_exit = exit_point + t_reflect * reflected
                                                    
                                                    # Draw reflected ray
                                                    reflected_endpoint = exit_point + min(t_reflect, reflection_ray_length) * reflected
                                                    ax.arrow(exit_point[0], exit_point[1], 
                                                            (reflected_endpoint[0] - exit_point[0]), 
                                                            (reflected_endpoint[1] - exit_point[1]),
                                                            head_width=0, head_length=0, fc=color, ec=color, 
                                                            linewidth=1.5, alpha=0.7)
                                                    
                                                    # If we've completed the internal rays
                                                    if reflection_ray_length >= t_reflect and progress > 0.7:
                                                        # Calculate final refraction at exit
                                                        normal_exit = -(final_exit - droplet_center) / droplet_radius
                                                        incident_exit = reflected
                                                        
                                                        # Snell's law for refraction
                                                        cos_theta_i_exit = -np.dot(normal_exit, incident_exit)
                                                        sin_theta_i_exit = np.sqrt(1 - cos_theta_i_exit**2)
                                                        sin_theta_t_exit = (n_water / n_air) * sin_theta_i_exit
                                                        
                                                        if sin_theta_t_exit <= 1.0:  # Check for total internal reflection
                                                            cos_theta_t_exit = np.sqrt(1 - sin_theta_t_exit**2)
                                                            refracted_exit = (n_water / n_air) * incident_exit + (
                                                                (n_water / n_air) * cos_theta_i_exit - cos_theta_t_exit) * normal_exit
                                                            refracted_exit = refracted_exit / np.linalg.norm(refracted_exit)
                                                            
                                                            # Calculate exit ray length for this frame
                                                            exit_progress = max(0, (progress - 0.7) / 0.3)
                                                            exit_ray_length = min(exit_progress * 4, 4)
                                                            
# Draw exit ray
                                                            ax.arrow(final_exit[0], final_exit[1], 
                                                                    exit_ray_length * refracted_exit[0], 
                                                                    exit_ray_length * refracted_exit[1],
                                                                    head_width=0, head_length=0, fc=color, ec=color, 
                                                                    linewidth=1.5, alpha=0.7)
        
        # Add title showing progress
        if progress <= 0.3:
            title = "Step 1: Incoming Light"
        elif progress <= 0.5:
            title = "Step 2: Refraction Upon Entry"
        elif progress <= 0.7:
            title = "Step 3: Internal Reflection"
        else:
            title = "Step 4: Refraction Upon Exit — Rainbow Formation"
        
        ax.set_title(title, fontsize=14)
        
        # Set axis limits and labels
        ax.set_xlim(-3, 5)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add explanatory text
        if progress <= 0.3:
            ax.text(3, 1.5, "Parallel rays of sunlight\napproach a water droplet", 
                   fontsize=10, ha='center')
        elif progress <= 0.5:
            ax.text(3, 1.5, "Light bends (refracts) when\nentering the water droplet\ndue to change in speed", 
                   fontsize=10, ha='center')
        elif progress <= 0.7:
            ax.text(3, 1.5, "Light reflects off the\nback surface of the droplet", 
                   fontsize=10, ha='center')
        else:
            ax.text(3, 1.5, "Light refracts again when leaving\nthe droplet, with different wavelengths\nbending at slightly different angles", 
                   fontsize=10, ha='center')
            
            # Add rainbow explanation in final frames
            if progress > 0.9:
                ax.text(-2, -1.5, "Red light (~650nm): Bends less", fontsize=10, color='red')
                ax.text(-2, -1.7, "Violet light (~400nm): Bends more", fontsize=10, color='purple')
                ax.text(3, -1.5, "This wavelength-dependent refraction\ncreates the rainbow's color spectrum", 
                       fontsize=10, ha='center')
        
        return ax
    
    # Create animation
    print("Generating rainbow formation animation...")
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, blit=False)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Monte Carlo Simulation'),
                                   bitrate=5000)
    output_path = os.path.join('results', output_file)
    ani.save(output_path, writer=writer, dpi=dpi)
    
    print(f"Animation saved to {output_path}")
    
    return ani

def create_full_rainbow_animation(num_frames=240, output_file='full_rainbow_animation.mp4', dpi=150):
    """
    Create a more comprehensive animation showing the overall rainbow formation in the sky.
    """
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Function to update animation
    def animate(frame):
        plt.clf()
        
        # Relative progress for this frame
        progress = frame / num_frames
        
        if progress < 0.25:
            # Phase 1: Show water droplets in the sky
            phase_progress = progress / 0.25
            
            # Draw sky background
            plt.fill_between([-5, 5], [-5, -5], [5, 5], color='skyblue')
            
            # Draw ground
            plt.fill_between([-5, 5], [-5, -5], [-3, -3], color='green')
            
            # Draw sun
            sun_x = -4
            sun_y = 3
            sun_circle = plt.Circle((sun_x, sun_y), 0.5, color='yellow')
            plt.gca().add_artist(sun_circle)
            
            # Draw sun rays
            num_rays = 12
            ray_length = 0.8
            for i in range(num_rays):
                angle = i * (2 * np.pi / num_rays)
                dx = ray_length * np.cos(angle)
                dy = ray_length * np.sin(angle)
                plt.plot([sun_x, sun_x + dx], [sun_y, sun_y + dy], 'y-', linewidth=2, alpha=0.7)
            
            # Draw water droplets gradually appearing
            num_droplets = int(100 * phase_progress)
            droplet_positions = []
            
            for i in range(num_droplets):
                x = np.random.uniform(-4, 4)
                y = np.random.uniform(-2, 4)
                droplet_positions.append((x, y))
                
                # Draw water droplet
                droplet = plt.Circle((x, y), 0.1, color='lightskyblue', alpha=0.5)
                plt.gca().add_artist(droplet)
            
            # Title and explanation
            plt.title("Phase 1: Water Droplets in the Air", fontsize=14)
            plt.text(0, -4, "When sunlight encounters water droplets in the air\n(such as after rainfall), a rainbow can form", 
                   ha='center', fontsize=12)
            
        elif progress < 0.5:
            # Phase 2: Show light interacting with a single droplet
            phase_progress = (progress - 0.25) / 0.25
            
            # Set up subplot for single droplet
            ax = plt.subplot(111)
            
            # Draw water droplet
            droplet_radius = 1.0
            droplet_center = np.array([0, 0])
            circle = plt.Circle(droplet_center, droplet_radius, fill=True, color='skyblue', alpha=0.3)
            ax.add_artist(circle)
            circle_outline = plt.Circle(droplet_center, droplet_radius, fill=False, color='blue', linewidth=1)
            ax.add_artist(circle_outline)
            
            # Create wavelengths (colors)
            visible_min = 380
            visible_max = 750
            num_wavelengths = 7  # Classic rainbow colors
            wavelengths = np.linspace(visible_min, visible_max, num_wavelengths)
            
            # Draw ray paths
            if phase_progress > 0.1:
                # Draw incoming ray
                ray_start = np.array([-3, 0])
                ray_direction = np.array([1, 0])
                
                # Entry point
                entry_point = np.array([-droplet_radius, 0])
                
                # For each wavelength
                for i, wavelength in enumerate(wavelengths):
                    color = wavelength_to_rgb(wavelength)
                    
                    # Calculate refraction angles based on wavelength
                    n_air = 1.0
                    n_water = 1.33 + 0.02 * (750 - wavelength) / (750 - 380)  # Simplified dispersion
                    
                    # Draw initial ray
                    ax.arrow(ray_start[0], ray_start[1],
                            entry_point[0] - ray_start[0], entry_point[1] - ray_start[1],
                            head_width=0, head_length=0, fc=color, ec=color, linewidth=1.5, alpha=0.7)
                    
                    if phase_progress > 0.3:
                        # Calculate angles for internal path
                        # This is a simplified calculation, not physically accurate
                        offset = 0.2 * (i / (num_wavelengths - 1) - 0.5)
                        internal_point1 = np.array([0, -0.7 + offset])
                        
                        # Draw refraction into the droplet
                        ax.arrow(entry_point[0], entry_point[1],
                                internal_point1[0] - entry_point[0], internal_point1[1] - entry_point[1],
                                head_width=0, head_length=0, fc=color, ec=color, linewidth=1.5, alpha=0.7)
                        
                        if phase_progress > 0.6:
                            # Internal reflection
                            exit_offset = 0.3 * (i / (num_wavelengths - 1) - 0.5)
                            internal_point2 = np.array([0.7, -0.3 + exit_offset])
                            
                            # Draw internal reflection
                            ax.arrow(internal_point1[0], internal_point1[1],
                                    internal_point2[0] - internal_point1[0], internal_point2[1] - internal_point1[1],
                                    head_width=0, head_length=0, fc=color, ec=color, linewidth=1.5, alpha=0.7)
                            
                            if phase_progress > 0.8:
                                # Exit ray
                                # Calculate exit point
                                ray_angle = 0.7 - 0.15 * i / (num_wavelengths - 1)  # Different angles for different colors
                                exit_point = np.array([droplet_radius * np.cos(ray_angle), 
                                                     droplet_radius * np.sin(ray_angle)])
                                
                                # Draw path to exit point
                                ax.arrow(internal_point2[0], internal_point2[1],
                                        exit_point[0] - internal_point2[0], exit_point[1] - internal_point2[1],
                                        head_width=0, head_length=0, fc=color, ec=color, linewidth=1.5, alpha=0.7)
                                
                                # Draw exit ray
                                exit_direction = exit_point - internal_point2
                                exit_direction = exit_direction / np.linalg.norm(exit_direction)
                                ray_length = 3.0
                                
                                ax.arrow(exit_point[0], exit_point[1],
                                        ray_length * exit_direction[0], ray_length * exit_direction[1],
                                        head_width=0, head_length=0, fc=color, ec=color, linewidth=1.5, alpha=0.7)
            
            # Title and explanation
            plt.title("Phase 2: Light Interaction with a Single Droplet", fontsize=14)
            explanation = ""
            if phase_progress <= 0.1:
                explanation = "A single water droplet"
            elif phase_progress <= 0.3:
                explanation = "Sunlight enters the droplet"
            elif phase_progress <= 0.6:
                explanation = "Light bends (refracts) as it enters the droplet"
            elif phase_progress <= 0.8:
                explanation = "Light reflects from the back of the droplet"
            else:
                explanation = "Different wavelengths of light exit at different angles,\nseparating white light into its component colors"
            
            plt.text(0, -3, explanation, ha='center', fontsize=12)
            
            # Set limits
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            plt.axis('equal')
            
        elif progress < 0.75:
            # Phase 3: Show multiple droplets creating rainbow effect
            phase_progress = (progress - 0.5) / 0.25
            
            # Draw sky background
            plt.fill_between([-5, 5], [-5, -5], [5, 5], color='skyblue')
            
            # Draw ground
            plt.fill_between([-5, 5], [-5, -5], [-3, -3], color='green')
            
            # Draw sun
            sun_x = -4
            sun_y = 3
            sun_circle = plt.Circle((sun_x, sun_y), 0.5, color='yellow')
            plt.gca().add_artist(sun_circle)
            
            # Draw many water droplets
            num_droplets = 100
            droplet_positions = []
            
            for i in range(num_droplets):
                x = np.random.uniform(-4, 4)
                y = np.random.uniform(-2, 4)
                droplet_positions.append((x, y))
                
                # Draw water droplet
                droplet = plt.Circle((x, y), 0.05, color='lightskyblue', alpha=0.3)
                plt.gca().add_artist(droplet)
            
            # Draw light rays from multiple droplets
            if phase_progress > 0.3:
                # For each droplet
                wavelengths = np.linspace(380, 750, 7)  # Rainbow colors
                
                for i, (x, y) in enumerate(droplet_positions):
                    if i % 5 == 0:  # Only show some rays for clarity
                        # Calculate observer position
                        observer_x = 0
                        observer_y = -4
                        
                        # Calculate angle to observer
                        dx = observer_x - x
                        dy = observer_y - y
                        angle = np.arctan2(dy, dx)
                        
                        # For this droplet, only one color is visible to the observer
                        # based on the specific angle
                        # This is highly simplified, but demonstrates the concept
                        relative_angle = (angle + np.pi) % (2 * np.pi)
                        color_index = int(relative_angle * 7 / (2 * np.pi)) % 7
                        wavelength = wavelengths[color_index]
                        color = wavelength_to_rgb(wavelength)
                        
                        # Draw ray from droplet to observer
                        ray_length = 1.0 * phase_progress
                        plt.plot([x, x + dx * ray_length], [y, y + dy * ray_length], 
                                '-', color=color, linewidth=1, alpha=0.7)
            
            # Draw rainbow outline gradually
            if phase_progress > 0.6:
                fade_in = min(1.0, (phase_progress - 0.6) / 0.4)
                
                # Primary rainbow
                rainbow_center_x = 0
                rainbow_center_y = -10
                rainbow_radius = 7
                
                # Draw rainbow arcs
                for i, wavelength in enumerate(wavelengths):
                    color = wavelength_to_rgb(wavelength)
                    thickness = 0.2
                    radius = rainbow_radius - i * thickness
                    
                    # Create arc
                    theta = np.linspace(np.pi/4, 3*np.pi/4, 100)
                    x = rainbow_center_x + radius * np.cos(theta)
                    y = rainbow_center_y + radius * np.sin(theta)
                    
                    plt.plot(x, y, '-', color=color, linewidth=5, alpha=fade_in)
            
            # Title and explanation
            plt.title("Phase 3: Multiple Droplets Create a Rainbow", fontsize=14)
            explanation = ""
            if phase_progress <= 0.3:
                explanation = "Each water droplet in the air disperses light"
            elif phase_progress <= 0.6:
                explanation = "From each droplet, only specific colors reach the observer's eye,\ndepending on the angle"
            else:
                explanation = "When millions of droplets are present,\nthey collectively form a continuous rainbow arc"
            
            plt.text(0, -4, explanation, ha='center', fontsize=12)
            
            # Set axis equal and limits
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            
        else:
            # Phase 4: Complete rainbow visualization
            phase_progress = (progress - 0.75) / 0.25
            
            # Draw sky background with gradient
            y = np.linspace(-5, 5, 100)
            for i in range(len(y)-1):
                color_val = 0.6 + 0.4 * (y[i] + 5) / 10
                plt.fill_between([-5, 5], [y[i], y[i]], [y[i+1], y[i+1]], 
                               color=(color_val*0.6, color_val*0.8, color_val*1.0))
            
            # Draw ground
            plt.fill_between([-5, 5], [-5, -5], [-3, -3], color='green')
            
            # Draw sun behind the viewer
            if phase_progress > 0.2:
                # Sun's position is implied to be behind the viewer
                # Add some sun rays coming from behind
                for i in range(8):
                    angle = np.pi/4 + i * np.pi/4
                    x_start = -4.5 + 0.5 * np.cos(angle)
                    y_start = -4.5 + 0.5 * np.sin(angle)
                    x_end = x_start + 0.7 * np.cos(angle)
                    y_end = y_start + 0.7 * np.sin(angle)
                    plt.plot([x_start, x_end], [y_start, y_end], 'y-', linewidth=3)
            
            # Draw full rainbow
            rainbow_center_x = 0
            rainbow_center_y = -10
            rainbow_radius = 7
            
            # Draw rainbow arcs with fade-in
            fade_in = min(1.0, phase_progress / 0.5)
            
            # Primary rainbow
            wavelengths = np.linspace(380, 750, 100)  # More colors for smoother rainbow
            
            for i, wavelength in enumerate(reversed(wavelengths)):  # Reverse for correct rainbow order
                color = wavelength_to_rgb(wavelength)
                thickness = 0.03
                radius = rainbow_radius - i * thickness / 3
                
                # Create arc
                theta = np.linspace(np.pi/4, 3*np.pi/4, 100)
                x = rainbow_center_x + radius * np.cos(theta)
                y = rainbow_center_y + radius * np.sin(theta)
                
                plt.plot(x, y, '-', color=color, linewidth=2, alpha=fade_in * 0.8)
            
            # Secondary rainbow (if progress is far enough)
            if phase_progress > 0.5:
                secondary_fade = min(1.0, (phase_progress - 0.5) / 0.5)
                
                # Secondary rainbow is about 51° from antisolar point (vs 42° for primary)
                secondary_radius = 8
                
                for i, wavelength in enumerate(wavelengths):  # Not reversed for secondary rainbow
                    color = wavelength_to_rgb(wavelength)
                    thickness = 0.03
                    radius = secondary_radius + i * thickness / 3
                    
                    # Create arc
                    theta = np.linspace(np.pi/4, 3*np.pi/4, 100)
                    x = rainbow_center_x + radius * np.cos(theta)
                    y = rainbow_center_y + radius * np.sin(theta)
                    
                    plt.plot(x, y, '-', color=color, linewidth=2, alpha=secondary_fade * 0.4)
            
            # Title and explanation
            plt.title("The Rainbow", fontsize=16)
            
            # Add explanation
            if phase_progress < 0.3:
                plt.text(0, -4, "A rainbow forms 40-42° away from the antisolar point\n(opposite direction from the sun)", 
                       ha='center', fontsize=12)
            elif phase_progress < 0.6:
                plt.text(0, -4, "The primary rainbow shows colors from red (outer) to violet (inner)\ndue to the angles at which light exits water droplets", 
                       ha='center', fontsize=12)
            else:
                plt.text(0, -4, "A secondary rainbow sometimes appears at 51° with colors in reverse order\ndue to light undergoing two internal reflections in the droplets", 
                       ha='center', fontsize=12)
            
            # Rain drops if early in this phase
            if phase_progress < 0.3:
                for i in range(50):
                    x = np.random.uniform(-4, 4)
                    y = np.random.uniform(-3, 4)
                    plt.plot([x, x-0.1], [y, y-0.3], 'b-', alpha=0.3)
            
            # Set limits
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
        
        # Remove axes
        plt.axis('off')
        
        return fig
    
    # Create animation
    print("Generating full rainbow formation animation...")
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50, blit=False)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Monte Carlo Simulation'),
                                   bitrate=5000)
    output_path = os.path.join('results', output_file)
    ani.save(output_path, writer=writer, dpi=dpi)
    
    print(f"Animation saved to {output_path}")
    
    return ani

def main():
    """Main function to run the rainbow animation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rainbow Formation Animation')
    parser.add_argument('--type', type=str, default='both',
                       choices=['simple', 'full', 'both'],
                       help='Animation type to generate')
    parser.add_argument('--frames', type=int, default=180,
                       help='Number of frames for animation')
    parser.add_argument('--dpi', type=int, default=100,
                       help='DPI for output video')
    
    args = parser.parse_args()
    
    if args.type in ['simple', 'both']:
        create_rainbow_formation_animation(
            num_frames=args.frames,
            output_file='rainbow_droplet_animation.mp4',
            dpi=args.dpi
        )
    
    if args.type in ['full', 'both']:
        create_full_rainbow_animation(
            num_frames=args.frames,
            output_file='full_rainbow_animation.mp4',
            dpi=args.dpi
        )

if __name__ == "__main__":
    main()