import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Import your custom modules
from photon import Photon, WhiteLight
from medium import Medium, MediumManager
from geometry import Boundary, Sphere, GeometryManager
from detector import CubeDetector


def simulate_3d_rainbow_monte_carlo():
    """Simulate 3D rainbow formation with Monte Carlo ray tracing."""
    
    # Create medium manager
    medium_manager = MediumManager()
    
    # Create geometry manager
    geometry_manager = GeometryManager()
    
    # Create water droplet
    droplet = Sphere(
        center=[0, 0, 0],
        radius=1.0,
        name="water_droplet",
        inside_medium="water",
        outside_medium="air"
    )
    geometry_manager.add_boundary(droplet)
    
    # Create cube detector
    detector = CubeDetector(
        center=[0, 0, 0],
        size=5.0,
        resolution=100
    )
    
    # Generate a cylinder of rays
    num_rays = 400  # Increase this for better resolution
    
    # Generate rays in a circle on the negative x-axis
    ray_positions = []
    radius = 0.9  # Slightly less than droplet radius for good coverage
    
    # Top ray for visualization
    top_ray_position = [-3, 0, 0.8]
    ray_positions.append(top_ray_position)
    
    # Generate circle of rays
    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        y = radius * np.sin(angle) 
        z = radius * np.cos(angle)
        ray_positions.append([-3, y, z])
    
    # Direction is always along +X axis
    ray_direction = [1, 0, 0]
    
    # Create and trace white light rays
    all_photons = []
    
    # Flag for special rays to visualize
    visualization_photons = []
    
    # Process each ray
    for i, pos in enumerate(ray_positions):
        # Create white light
        white_light = WhiteLight(
            position=pos,
            direction=ray_direction,
            num_wavelengths=30
        )
        
        # Trace through scene
        traced_photons = white_light.trace(medium_manager, geometry_manager)
        
        # Mark photons from the top ray for visualization
        if i == 0:  # The top ray
            for photon in traced_photons:
                photon.for_visualization = True
                visualization_photons.append(photon)
        
        # Add all photons to the list
        all_photons.extend(traced_photons)
    
    # Register hits on detector
    for photon in all_photons:
        detector.register_hit(photon)
    
    # Generate rainbow images for each face
    images = {}
    for i, name in enumerate(detector.face_names):
        rgb_image = detector.get_rgb_image(i)
        if rgb_image is not None:
            images[name] = rgb_image
    
    # Visualize the 3D scene with rainbow
    visualize_3d_rainbow_with_cube(droplet, visualization_photons, images)
    
    # Create zoomed out view
    visualize_rainbow_zoomed_out(droplet, visualization_photons, images)
    
    return detector

def visualize_3d_rainbow_with_cube(droplet, photons, images):
    """
    Create a 3D visualization showing the droplet, light paths, and cube detector.
    
    Args:
        droplet: The water droplet
        photons: List of traced photons
        images: Dictionary of face RGB images
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches
    
    # Create 3D figure
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw water droplet (transparent sphere)
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = droplet.radius * np.cos(u) * np.sin(v)
    y = droplet.radius * np.sin(u) * np.sin(v)
    z = droplet.radius * np.cos(v)
    ax.plot_surface(x, y, z, color='skyblue', alpha=0.2, edgecolor='lightblue', linewidth=0.2)
    
    # Draw coordinate axes with labels
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', arrow_length_ratio=0.1, label='X')
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='r', arrow_length_ratio=0.1, label='Y') 
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='r', arrow_length_ratio=0.1, label='Z')
    
    # Add "X", "Y", "Z" labels at the end of each axis
    ax.text(1.7, 0, 0, "X", color='red', fontsize=12)
    ax.text(0, 1.7, 0, "Y", color='red', fontsize=12)
    ax.text(0, 0, 1.7, "Z", color='red', fontsize=12)
    
    # Create and draw cube (using wireframe)
    cube_size = 3.0
    r = cube_size / 2
    
    # Define cube vertices
    vertices = np.array([
        [-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],
        [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]
    ])
    
    # Define cube edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
    ]
    
    # Draw cube edges
    for edge in edges:
        ax.plot3D(
            [vertices[edge[0]][0], vertices[edge[1]][0]],
            [vertices[edge[0]][1], vertices[edge[1]][1]],
            [vertices[edge[0]][2], vertices[edge[1]][2]],
            color='gray', linestyle='--', alpha=0.7
        )
    
    # Draw cube faces with rainbow projections
    # Define the faces of the cube
    cube_faces = [
        [0, 1, 2, 3],  # Back face (-Z)
        [4, 5, 6, 7],  # Front face (+Z)
        [0, 1, 5, 4],  # Bottom face (-Y)
        [2, 3, 7, 6],  # Top face (+Y)
        [0, 3, 7, 4],  # Left face (-X)
        [1, 2, 6, 5]   # Right face (+X)
    ]
    
    face_names = ["-Z", "+Z", "-Y", "+Y", "-X", "+X"]
    
    # Draw each face with semi-transparency
    for i, face in enumerate(cube_faces):
        # Get the 4 corners of this face
        x = [vertices[j][0] for j in face]
        y = [vertices[j][1] for j in face]
        z = [vertices[j][2] for j in face]
        
        # Draw face with light gray color (for spots without light)
        ax.plot_surface(
            np.array([[x[0], x[1]], [x[3], x[2]]]),
            np.array([[y[0], y[1]], [y[3], y[2]]]),
            np.array([[z[0], z[1]], [z[3], z[2]]]),
            color='lightgray', alpha=0.3
        )
        
        # Add face label
        face_center = np.mean(vertices[face], axis=0)
        ax.text(face_center[0]*1.1, face_center[1]*1.1, face_center[2]*1.1, 
               face_names[i], color='black', fontsize=10)
    
    # Define only the top incoming ray to visualize
    top_position = [-3, 0, 0.8]  # top center ray

    wavelength_groups = [
        (380, 420, 'violet', 'Violet (400nm)'),
        (420, 470, 'blue', 'Blue (450nm)'),
        (470, 520, 'cyan', 'Cyan (500nm)'),
        (520, 570, 'green', 'Green (550nm)'),
        (570, 590, 'yellow', 'Yellow (580nm)'),
        (590, 650, 'orange', 'Orange (620nm)'),
        (650, 750, 'red', 'Red (650nm)')
    ]
    for min_wl, max_wl, color, label in wavelength_groups:
        # Find photons in this wavelength range that originated from the top position
        group_photons = [
            p for p in photons
            if min_wl <= p.wavelength < max_wl and
            np.linalg.norm(np.array(p.path[0]) - np.array(top_position)) < 0.1
        ]
        
        for photon in group_photons:
            if len(photon.path) < 3:
                continue  # skip short paths
            path = np.array(photon.path)
            xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]
            ax.plot(xs, ys, zs, color=color, linewidth=1.5, alpha=0.7)
    
    # Create legend handles
    legend_handles = []
    
    # Add entries for photon paths by color
    for min_wl, max_wl, color, label in wavelength_groups:
        # Find photons in this wavelength range
        group_photons = [p for p in photons if min_wl <= p.wavelength < max_wl]
        
        # Plot paths for these photons
        for i, photon in enumerate(group_photons):
            if i % 5 != 0:  # Only plot every 5th photon for clarity
                continue
                
            # Skip very short paths
            if len(photon.path) < 3:
                continue
                
            # Get path coordinates
            path = np.array(photon.path)
            xs, ys, zs = path[:, 0], path[:, 1], path[:, 2]
            
            # Plot path with appropriate color
            ax.plot(xs, ys, zs, color=color, linewidth=1.5, alpha=0.7)
        
        # Add to legend
        legend_handles.append(mpatches.Patch(color=color, label=label))
    
    # Add white light to legend
    legend_handles.append(mpatches.Patch(color='white', label='White Light'))
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper right')
    
    # Set equal aspect ratio and limits
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    # Set plot limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    
    # Add title
    plt.title('3D Rainbow Formation in Water Droplet', fontsize=14)
    
    # Save figure
    plt.savefig('rainbow_3d_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_rainbow_zoomed_out(droplet, photons, face_images):
    """
    Create a zoomed-out visualization focusing on the rainbow pattern.
    
    Args:
        droplet: The water droplet
        photons: List of traced photons
        face_images: Dictionary of face RGB images
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create 3D figure
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw water droplet (very small now)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = droplet.radius * np.cos(u) * np.sin(v)
    y = droplet.radius * np.sin(u) * np.sin(v)
    z = droplet.radius * np.cos(v)
    ax.plot_surface(x, y, z, color='skyblue', alpha=0.2)
    
    # Create and draw cube (using wireframe)
    cube_size = 5.0
    r = cube_size / 2
    
    # Define cube vertices
    vertices = np.array([
        [-r, -r, -r], [r, -r, -r], [r, r, -r], [-r, r, -r],
        [-r, -r, r], [r, -r, r], [r, r, r], [-r, r, r]
    ])
    
    # Define cube faces
    faces = [
        [0, 1, 2, 3],  # Back face (-Z)
        [4, 5, 6, 7],  # Front face (+Z)
        [0, 1, 5, 4],  # Bottom face (-Y)
        [2, 3, 7, 6],  # Top face (+Y)
        [0, 3, 7, 4],  # Left face (-X)
        [1, 2, 6, 5]   # Right face (+X)
    ]
    face_names = ["-Z", "+Z", "-Y", "+Y", "-X", "+X"]
    
    # Draw each face with its rainbow image
    for i, (face, name) in enumerate(zip(faces, face_names)):
        # Get the 4 corners of this face
        x = [vertices[j][0] for j in face]
        y = [vertices[j][1] for j in face]
        z = [vertices[j][2] for j in face]
        
        # Draw face with light gray color
        ax.plot_surface(
            np.array([[x[0], x[1]], [x[3], x[2]]]),
            np.array([[y[0], y[1]], [y[3], y[2]]]),
            np.array([[z[0], z[1]], [z[3], z[2]]]),
            color='lightgray', alpha=0.3
        )
        
        # If we have an image for this face, visualize it in the corner
        if name in face_images:
            # Add a small inset image in the corner
            img = face_images[name]
            # Code to inset an image would go here if matplotlib allowed it easily
    
    # Set view to focus on the -X face where rainbow appears
    ax.view_init(elev=20, azim=-110)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set plot limits (much larger)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_zlim(-6, 6)
    
    # Add title
    plt.title('Rainbow Formation (Zoomed Out)', fontsize=14)
    
    # Save figure
    plt.savefig('rainbow_zoomed_out.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Also show the -X face image separately in high resolution
    if "-X" in face_images:
        plt.figure(figsize=(10, 10))
        plt.imshow(face_images["-X"])
        plt.title('Rainbow on -X Face (High Resolution)')
        plt.colorbar(label='Intensity')
        plt.savefig('rainbow_face_detail.png', dpi=300)
        plt.show()

if __name__ == "__main__":
    simulate_3d_rainbow_monte_carlo()
