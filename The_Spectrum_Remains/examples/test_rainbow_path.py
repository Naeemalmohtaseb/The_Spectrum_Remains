"""
Test suite for rainbow physics simulation.

This file contains tests for the fundamental physics required for rainbow simulation:
- Photon propagation and direction
- Wavelength-dependent refraction
- Reflection at boundaries
- Medium properties
- Detector functionality
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import our modules
from src.photon import Photon
from src.medium import Medium, water, air
from src.geometry import Sphere, Plane, Detector, GeometryManager


def test_photon_creation():
    """Test basic photon creation and properties."""
    print("\n--- Testing Photon Creation ---")
    
    # Create photons with different wavelengths
    red = Photon(wavelength=650.0)
    green = Photon(wavelength=550.0)
    blue = Photon(wavelength=450.0)
    
    # Test position and direction initialization
    assert np.allclose(red.position, [0, 0, 0]), "Default position should be origin"
    assert np.allclose(red.direction, [0, 0, 1]), "Default direction should be z-axis"
    
    # Test custom position and direction
    custom = Photon(
        position=[1, 2, 3],
        direction=[-1, 0, 0],
        wavelength=600.0
    )
    assert np.allclose(custom.position, [1, 2, 3]), "Custom position not set correctly"
    assert np.allclose(custom.direction, [-1, 0, 0]), "Custom direction not normalized correctly"
    
    # Test RGB conversion
    r_rgb = red.get_rgb_color()
    g_rgb = green.get_rgb_color()
    b_rgb = blue.get_rgb_color()
    
    print(f"Red (650nm) → RGB: {r_rgb}")
    print(f"Green (550nm) → RGB: {g_rgb}")
    print(f"Blue (450nm) → RGB: {b_rgb}")
    
    # Check if colors are generally correct (red has more R than G or B, etc.)
    assert r_rgb[0] > r_rgb[1] and r_rgb[0] > r_rgb[2], "Red wavelength should produce more R than G or B"
    assert g_rgb[1] > g_rgb[0] and g_rgb[1] > g_rgb[2], "Green wavelength should produce more G than R or B"
    assert b_rgb[2] > b_rgb[0] and b_rgb[2] > b_rgb[1], "Blue wavelength should produce more B than R or G"
    
    print("Photon creation tests passed!")


def test_photon_movement():
    """Test photon movement and path tracking."""
    print("\n--- Testing Photon Movement ---")
    
    # Create a photon
    photon = Photon(position=[0, 0, 0], direction=[1, 0, 0], wavelength=550.0)
    
    # Move the photon several times
    photon.move(1.0)
    photon.move(2.0)
    photon.move(3.0)
    
    # Check final position
    assert np.allclose(photon.position, [6, 0, 0]), "Photon should move along x-axis"
    
    # Check path history
    path = photon.get_path()
    assert len(path) == 4, "Path should contain 4 points"
    assert np.allclose(path[0], [0, 0, 0]), "Path should start at origin"
    assert np.allclose(path[-1], [6, 0, 0]), "Path should end at final position"
    
    print("Photon movement tests passed!")


def test_reflection():
    """Test photon reflection mechanics."""
    print("\n--- Testing Reflection ---")
    
    # Create a photon approaching a surface
    photon = Photon(position=[0, 0, -1], direction=[0, 0, 1], wavelength=550.0)
    
    # Reflect off a horizontal surface (normal = [0, 0, 1])
    normal = np.array([0, 0, 1])
    
    # Before reflection
    print(f"Before reflection: direction = {photon.direction}")
    
    # Apply reflection
    photon.reflect(normal)
    
    # After reflection
    print(f"After reflection: direction = {photon.direction}")
    
    # Check reflection result (should be [0, 0, -1])
    assert np.allclose(photon.direction, [0, 0, -1]), "Reflection should reverse z direction"
    
    # Test reflection at an angle
    photon = Photon(position=[0, 0, 0], direction=[1, 1, 1], wavelength=550.0)
    normal = np.array([0, 0, 1])
    
    # Normalize the direction
    photon.direction = photon.direction / np.linalg.norm(photon.direction)
    
    # Before reflection
    dir_before = photon.direction.copy()
    print(f"Before angled reflection: direction = {dir_before}")
    
    # Apply reflection
    photon.reflect(normal)
    
    # After reflection
    print(f"After angled reflection: direction = {photon.direction}")
    
    # Check reflection result (should maintain x,y components but negate z)
    assert np.allclose(photon.direction[0], dir_before[0]), "X component shouldn't change"
    assert np.allclose(photon.direction[1], dir_before[1]), "Y component shouldn't change"
    assert np.allclose(photon.direction[2], -dir_before[2]), "Z component should be negated"
    
    print("Reflection tests passed!")


def test_refraction():
    """Test wavelength-dependent refraction."""
    print("\n--- Testing Refraction ---")
    
    # Create media
    air_medium = air()
    water_medium = water()
    
    # Get refractive indices for different wavelengths
    wavelengths = [400, 500, 600, 700]  # violet, green, orange, red
    n_air = [air_medium.get_refractive_index(wl) for wl in wavelengths]
    n_water = [water_medium.get_refractive_index(wl) for wl in wavelengths]
    
    print("Refractive indices:")
    for i, wl in enumerate(wavelengths):
        print(f"  λ={wl}nm: Air n={n_air[i]:.6f}, Water n={n_water[i]:.6f}, Ratio={n_water[i]/n_air[i]:.6f}")
    
    # Verify dispersion (different wavelengths should have different indices)
    n_diffs = [n_water[i] - n_water[i+1] for i in range(len(wavelengths)-1)]
    assert all(diff > 0 for diff in n_diffs), "Water should have wavelength-dependent index (n_violet > n_red)"
    
    # Test refraction at air-water interface
    # Create photons at different wavelengths
    incident_angle = np.pi/4  # 45 degrees
    direction = [np.sin(incident_angle), 0, np.cos(incident_angle)]  # Angled incidence
    
    # Surface normal pointing up
    normal = np.array([0, 0, 1])
    
    # Create an array to store refraction angles
    refraction_angles = []
    
    # Test refraction for each wavelength
    for wl in wavelengths:
        photon = Photon(position=[0, 0, 0], direction=direction, wavelength=wl)
        n1 = air_medium.get_refractive_index(wl)
        n2 = water_medium.get_refractive_index(wl)
        
        # Apply refraction
        result = photon.refract(normal, n1, n2)
        
        # Calculate angle after refraction
        refracted_angle = np.arccos(abs(photon.direction[2]))
        refraction_angles.append(refracted_angle)
        
        print(f"  λ={wl}nm: Incident angle = {incident_angle:.6f} rad, "
              f"Refracted angle = {refracted_angle:.6f} rad")
    
    # Verify wavelength-dependent refraction (different wavelengths should refract differently)
    # Violet (400nm) should bend more than red (700nm)
    assert refraction_angles[0] < refraction_angles[-1], "Shorter wavelengths should bend more"
    
    # Calculate angular separation between violet and red
    angular_separation = refraction_angles[-1] - refraction_angles[0]
    print(f"Angular separation between red and violet: {angular_separation:.6f} rad "
          f"({np.degrees(angular_separation):.6f} degrees)")
    
    print("Refraction tests passed!")


def test_sphere_interaction():
    """Test photon interactions with a spherical boundary (like a raindrop)."""
    print("\n--- Testing Sphere Interaction ---")
    
    # Create a sphere (water droplet)
    droplet = Sphere(center=[0, 0, 0], radius=1.0, 
                   name="water_droplet", inside_medium="water", outside_medium="air")
    
    # Create media
    air_medium = air()
    water_medium = water()
    
    # Register media with geometry manager
    manager = GeometryManager()
    manager.add_boundary(droplet)
    manager.register_medium("air", air_medium)
    manager.register_medium("water", water_medium)
    
    # Create photons at different impact parameters
    # Impact parameter = distance from central axis
    impact_params = [0.0, 0.3, 0.6, 0.9]
    wavelength = 550.0  # green light
    
    for impact in impact_params:
        # Position photon to left of droplet
        photon = Photon(
            position=[-3.0, impact, 0.0],  
            direction=[1.0, 0.0, 0.0],  # Moving to the right
            wavelength=wavelength
        )
        
        # Find intersection with droplet
        hit, distance, boundary = manager.find_intersection(photon.position, photon.direction)
        
        # Check if intersection was found
        assert hit, f"Photon at impact parameter {impact} should hit the droplet"
        assert boundary is droplet, "Intersection should be with the droplet"
        
        # Move photon to intersection point
        photon.move(distance)
        
        # Check if photon is on the sphere surface
        distance_to_center = np.linalg.norm(photon.position)
        assert abs(distance_to_center - 1.0) < 1e-10, "Photon should be on sphere surface"
        
        # Get normal at intersection
        normal = droplet.normal(photon.position, photon.direction)
        
        # Determine media
        current_medium, next_medium = manager.get_medium_pair(boundary, photon.position, photon.direction)
        
        # Should be entering from air to water
        assert current_medium is air_medium, "Current medium should be air"
        assert next_medium is water_medium, "Next medium should be water"
        
        # Refract into droplet
        n1 = current_medium.get_refractive_index(wavelength)
        n2 = next_medium.get_refractive_index(wavelength)
        photon.refract(normal, n1, n2)
        
        print(f"Impact={impact}: Entry position={photon.position}, Refracted direction={photon.direction}")
    
    print("Sphere interaction tests passed!")


def test_rainbow_path():
    """Trace a single ray through a water droplet to demonstrate rainbow formation."""
    print("\n--- Testing Rainbow Path ---")
    
    # Create a sphere (water droplet)
    droplet = Sphere(center=[0, 0, 0], radius=1.0, 
                   name="water_droplet", inside_medium="water", outside_medium="air")
    
    # Create media
    air_medium = air()
    water_medium = water()
    
    # Register media with geometry manager
    manager = GeometryManager()
    manager.add_boundary(droplet)
    manager.register_medium("air", air_medium)
    manager.register_medium("water", water_medium)
    
    # Create photons at different wavelengths with the same impact parameter
    wavelengths = [650.0, 550.0, 450.0]  # red, green, blue
    impact_param = 0.8  # good value for rainbow formation
    
    # Store paths for plotting
    paths = []
    exit_angles = []
    
    for wavelength in wavelengths:
        # Position photon to left of droplet
        photon = Photon(
            position=[-3.0, impact_param, 0.0],  
            direction=[1.0, 0.0, 0.0],  # Moving to the right
            wavelength=wavelength
        )
        
        # Track path
        path = [photon.position.copy()]
        
        # Flag to track if we're inside the droplet
        inside = False
        
        # Maximum number of steps to prevent infinite loops
        max_steps = 10
        step = 0
        
        # While photon is active and hasn't exceeded max steps
        while photon.is_alive() and step < max_steps:
            step += 1
            
            # Find next intersection
            hit, distance, boundary = manager.find_intersection(photon.position, photon.direction)
            
            if hit:
                # Move photon to intersection
                photon.move(distance)
                
                # Record position
                path.append(photon.position.copy())
                
                # Get normal at intersection
                normal = boundary.normal(photon.position, photon.direction)
                
                # Determine media
                current_medium, next_medium = manager.get_medium_pair(boundary, photon.position, photon.direction)
                
                # Get refractive indices
                n1 = current_medium.get_refractive_index(wavelength)
                n2 = next_medium.get_refractive_index(wavelength)
                
                # Toggle inside flag
                inside = not inside
                
                # Refract or reflect
                if inside and boundary is droplet:
                    # Internal reflection if needed (will automatically do refraction if possible)
                    result = photon.refract(normal, n1, n2)
                    if not result:
                        # Total internal reflection occurred
                        print(f"λ={wavelength}nm: Total internal reflection at {photon.position}")
                else:
                    # External refraction
                    photon.refract(normal, n1, n2)
            else:
                # No intersection found, end the trace
                break
        
        # Calculate exit angle from the +x axis
        exit_direction = photon.direction
        exit_angle = np.arctan2(exit_direction[1], exit_direction[0])
        exit_angles.append(exit_angle)
        
        print(f"λ={wavelength}nm: Final direction={exit_direction}, Exit angle={np.degrees(exit_angle):.2f} degrees")
        
        # Convert path to numpy array
        paths.append((np.array(path), wavelength))
    
    # Calculate angular dispersion (difference between red and blue)
    angular_dispersion = abs(np.degrees(exit_angles[0] - exit_angles[2]))
    print(f"Angular dispersion between red and blue: {angular_dispersion:.3f} degrees")
    
    # Plot the paths
    plt.figure(figsize=(10, 6))
    
    # Draw the droplet
    circle = plt.Circle((0, 0), 1.0, color='skyblue', alpha=0.3)
    plt.gca().add_artist(circle)
    
    # Draw the paths
    for path, wavelength in paths:
        # Get RGB color
        color = Photon(wavelength=wavelength).get_rgb_color()
        
        # Plot path
        plt.plot(path[:, 0], path[:, 1], '-', color=color, 
                 linewidth=2, label=f'{wavelength}nm')
    
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.title('Rainbow Formation - Single Droplet')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('rainbow_test_path.png', dpi=300)
    
    print(f"Path visualization saved to 'rainbow_test_path.png'")
    print("Rainbow path test complete!")


def test_detector():
    """Test detector functionality for recording spectral information."""
    print("\n--- Testing Detector ---")
    
    # Create a detector
    detector = Detector(
        position=[5.0, 0.0, 0.0],  # 5 units to the right
        normal=[-1.0, 0.0, 0.0],   # Facing left
        size=(4.0, 4.0),           # 4x4 units
        resolution=(40, 40),       # 40x40 pixels
        name="screen"
    )
    
    # Create photons at different wavelengths
    wavelengths = np.linspace(400, 700, 7)  # violet to red
    
    # Send photons to different parts of the detector
    for i, wavelength in enumerate(wavelengths):
        # Vertical position varies with wavelength
        y_pos = -1.5 + 3.0 * i / (len(wavelengths) - 1)
        
        # Create photon moving toward detector
        photon = Photon(
            position=[0.0, y_pos, 0.0],
            direction=[1.0, 0.0, 0.0],  # Moving to the right
            wavelength=wavelength,
            intensity=1.0
        )
        
        # Register hit on detector
        hit = detector.register_hit(photon)
        
        # Should hit the detector
        assert hit, f"Photon at wavelength {wavelength}nm should hit the detector"
    
    # Get RGB image from detector
    rgb_image = detector.get_rgb_image()
    
    # Check image dimensions
    assert rgb_image.shape == (40, 40, 3), "RGB image should have correct dimensions"
    
    # Save the detector image
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image, origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Detector Image')
    plt.savefig('detector_test.png', dpi=300)
    
    print(f"Detector image saved to 'detector_test.png'")
    print("Detector test complete!")


def run_all_tests():
    """Run all tests."""
    print("=== Running Rainbow Physics Tests ===")
    
    test_photon_creation()
    test_photon_movement()
    test_reflection()
    test_refraction()
    test_sphere_interaction()
    test_rainbow_path()
    test_detector()
    
    print("\n=== All tests completed! ===")


if __name__ == "__main__":
    run_all_tests()