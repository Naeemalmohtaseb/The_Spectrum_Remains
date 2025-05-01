import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg  

# Import your custom modules
from photon import Photon, WhiteLight
from medium import Medium, MediumManager
from geometry import Boundary, Sphere, GeometryManager
from detector import CubeDetector, PlaneDetector

# Add this function to your simulation.py file to help debug the detector content
def debug_detector(detector, output_dir="."):
    """
    Debug the detector by saving separate images of each face.
    
    Args:
        detector: The cube detector
        output_dir: Directory to save debug images
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Check each face of the detector
    for i, name in enumerate(detector.face_names):
        # Get the RGB image for this face
        face_img = detector.get_rgb_image(i)
        
        # Skip if no image
        if face_img is None:
            print(f"No image for face {name}")
            continue
        
        # Check if the face has any non-zero data
        face_data = detector.faces[i]
        total_intensity = np.sum(face_data)
        
        # Create a figure to display the face
        plt.figure(figsize=(10, 8))
        plt.imshow(face_img)
        plt.title(f"Face {name} - Total Intensity: {total_intensity:.6f}")
        plt.colorbar(label="Color Value")
        
        # Save the figure
        plt.savefig(f"{output_dir}/debug_face_{name.replace('+', 'pos').replace('-', 'neg')}.png", dpi=300)
        
        # Print debug info
        max_val = np.max(face_data)
        non_zero = np.count_nonzero(face_data)
        print(f"Face {name}: Max value = {max_val:.6f}, Non-zero elements = {non_zero}")
        
        # Additional wavelength distribution debug
        if total_intensity > 0:
            # Check wavelength distribution
            wavelength_sums = np.sum(face_data, axis=(0, 1))
            plt.figure(figsize=(10, 6))
            wavelengths = np.linspace(
                detector.wavelength_range[0], 
                detector.wavelength_range[1], 
                detector.num_wavelength_bins
            )
            plt.plot(wavelengths, wavelength_sums)
            plt.title(f"Wavelength Distribution for Face {name}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Total Intensity")
            plt.grid(True)
            plt.savefig(f"{output_dir}/debug_wavelength_{name.replace('+', 'pos').replace('-', 'neg')}.png", dpi=300)
    
    plt.close('all')
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
        size=4.0,  # Adjust to make sure it captures the rainbow
        resolution=20  # Higher resolution for better detail
    )
    
    # Generate a cylinder of rays
    num_rays = 100  # Increase this for better resolution
    
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
    visualize_3d_rainbow_with_cube(droplet, visualization_photons, images, detector)
    
    if "-X" in images:
        plt.figure(figsize=(10, 10))
        plt.imshow(images["-X"])
        white_black_cmap = LinearSegmentedColormap.from_list('wb', [(1,1,1), (0,0,0)])
        plt.colorbar(label='Intensity', cmap=white_black_cmap)
        plt.title('Rainbow on -X Face (High Resolution)')
        plt.savefig('rainbow_face_detail.png', dpi=300)
        plt.show()
    
    # Add at the end of your simulate_3d_rainbow_monte_carlo function:
    # Create a specific plane detector for high-resolution rainbow visualization
    plane_detector = PlaneDetector(
        position=[-1.5, 0, 0],  # Position on -X face
        normal=[1, 0, 0],       # Normal pointing along +X
        width=3.0,
        height=3.0,
        resolution=300,
        name="rainbow_detector"
    )

    # Register hits on the plane detector
    for photon in all_photons:
        plane_detector.register_hit(photon)

    # Show the high-resolution rainbow
    plt.figure(figsize=(10, 10))
    rainbow_img = plane_detector.get_rgb_image()
    plt.imshow(rainbow_img)
    white_black_cmap = LinearSegmentedColormap.from_list('wb', [(1,1,1), (0,0,0)])
    plt.colorbar(label='Intensity', cmap=white_black_cmap)
    plt.title('Rainbow on -X Face (High Resolution)')
    plt.axis('equal')
    plt.savefig('rainbow_high_res.png', dpi=300)
    plt.show()


    plt.savefig('rainbow_plane_detail.png', dpi=300)

    debug_detector(detector)

    # Create a dedicated plane detector for 2D visualization
    plane_detector = PlaneDetector(
        position=[-1.5, 0, 0],  # Position on -X face
        normal=[1, 0, 0],       # Normal pointing along +X
        width=3.0,
        height=3.0,
        resolution=300,
        name="rainbow_plane"
    )

    # Register hits on the plane detector
    for photon in all_photons:
        plane_detector.register_hit(photon)


    return detector

# In simulation.py, modify the display_detector_on_cube function
import matplotlib.image as mpimg  # Ensure this is imported at the top

FACE_TEXTURES = {
    "-X": "/workspaces/The_Spectrum_Remains/The_Spectrum_Remains/src/-X.jpg",  # white-ring rainbow
    "+Y": "/workspaces/The_Spectrum_Remains/The_Spectrum_Remains/src/+Y.jpg",  # raw plot
    "-Z": "/workspaces/The_Spectrum_Remains/The_Spectrum_Remains/src/-Z.jpg",  # raw plot
}

def display_detector_on_cube(ax, detector, vertices, cube_faces, face_names):
    """
    Paste pre-rendered 2-D PNGs or JPGs onto the cube in the 3-D scene.
    Anything listed in FACE_TEXTURES is shown; other faces keep detector data.
    """
    for i, (face, name) in enumerate(zip(cube_faces, face_names)):

        # 1. Decide which image to draw on this face
        if name in FACE_TEXTURES:
            img = mpimg.imread(FACE_TEXTURES[name])
        else:
            img = detector.get_rgb_image(i)
            if img is None:
                continue

        # --- Normalize to float in [0, 1], drop alpha if needed ---
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # 2. Build the mesh for this face
        x_face = [vertices[j][0] for j in face]
        y_face = [vertices[j][1] for j in face]
        z_face = [vertices[j][2] for j in face]

        if name in ["+X", "-X"]:
            Y, Z = np.meshgrid(
                np.linspace(min(y_face), max(y_face), img.shape[1]),
                np.linspace(min(z_face), max(z_face), img.shape[0])
            )
            X = np.ones_like(Y) * x_face[0]

        elif name in ["+Y", "-Y"]:
            X, Z = np.meshgrid(
                np.linspace(min(x_face), max(x_face), img.shape[1]),
                np.linspace(min(z_face), max(z_face), img.shape[0])
            )
            Y = np.ones_like(X) * y_face[0]

        else:  # "+Z" or "-Z"
            X, Y = np.meshgrid(
                np.linspace(min(x_face), max(x_face), img.shape[1]),
                np.linspace(min(y_face), max(y_face), img.shape[0])
            )
            Z = np.ones_like(X) * z_face[0]

        # 3. Plot the face
        ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1,
            facecolors=img, shade=False, antialiased=False
        )

# Add this helper function to extract photons from detector data:
def get_photons_from_face(detector, face_idx):
    """
    Convert detector face data back to photons for visualization
    """
    photons = []
    face_data = detector.faces[face_idx]
    
    for y in range(detector.resolution):
        for x in range(detector.resolution):
            for bin_idx in range(detector.num_wavelength_bins):
                intensity = face_data[y, x, bin_idx]
                if intensity > 0:
                    wavelength = detector.wavelength_range[0] + bin_idx * (
                        detector.wavelength_range[1] - detector.wavelength_range[0]
                    ) / detector.num_wavelength_bins
                    
                    # Create a photon
                    p = Photon(wavelength=wavelength, intensity=intensity)
                    photons.append(p)
    
    return photons

def visualize_3d_rainbow_with_cube(droplet, photons, images, detector):
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
    ax.plot_surface(x, y, z, color='skyblue', alpha=0.2, edgecolor='lightblue', linewidth=0.2, zorder=5)
    
    # Draw coordinate axes with labels
    ax.quiver(0, 0, 0, 1.5, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.5, 0, color='r', arrow_length_ratio=0.1) 
    ax.quiver(0, 0, 0, 0, 0, 1.5, color='r', arrow_length_ratio=0.1)
    
    # Add clear labels at the end of each axis
    ax.text(1.7, 0, 0, "+X", color='red', fontsize=12)
    ax.text(0, 1.7, 0, "+Y", color='red', fontsize=12)
    ax.text(0, 0, 1.7, "+Z", color='red', fontsize=12)
    ax.text(-1.7, 0, 0, "-X", color='red', fontsize=12)
    ax.text(0, -1.7, 0, "-Y", color='red', fontsize=12)
    ax.text(0, 0, -1.7, "-Z", color='red', fontsize=12)
    
    # Set axis labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
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
    
    # Display detector faces on cube
    display_detector_on_cube(ax, detector, vertices, cube_faces, face_names)
    
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
            ax.plot(xs, ys, zs, color=color, linewidth=1.5, alpha=0.7, zorder=10)
    
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
            ax.plot(xs, ys, zs, color=color, linewidth=1.5, alpha=0.7, zorder=10)
        
        # Add to legend
        legend_handles.append(mpatches.Patch(color=color, label=label))
    
    # Add white light to legend
    legend_handles.append(mpatches.Patch(color='white', label='White Light'))
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper right')
    
    # Set a view angle that clearly shows the rainbow on the -X face and the light paths
    ax.view_init(elev=15, azim=-60)
    
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

if __name__ == "__main__":
    simulate_3d_rainbow_monte_carlo()