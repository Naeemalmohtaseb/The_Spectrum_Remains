import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any

class Photon:
    """
    Represents a spectral photon in 3D space with wavelength-dependent properties.
    
    This enhanced photon model supports:
    - 3D position and direction vectors
    - Wavelength-dependent properties
    - Intensity tracking for reflections/refractions
    - Path history for visualization
    """
        
    def __init__(self, 
                position: Union[List[float], np.ndarray] = None, 
                direction: Union[List[float], np.ndarray] = None, 
                wavelength: float = 550.0,
                intensity: float = 1.0,
                polarization: Union[List[float], np.ndarray] = None,
                is_spectral_component: bool = False):
        """
        Initialize a photon with position, direction, wavelength and other properties.
        
        Args:
            position: 3D position vector [x, y, z]
            direction: 3D direction vector
            wavelength: Wavelength in nanometers (visible range ~380-750nm)
            intensity: Initial intensity (1.0 = full)
            polarization: Polarization vector (optional)
            is_spectral_component: If True, photon is part of a white light ray
        """
        self.for_visualization = False  # <--- correctly indented
        # Initialize position
        if position is None:
            self.position = np.array([0.0, 0.0, 0.0])
        else:
            self.position = np.array(position, dtype=float)
        
        # Initialize direction (normalize it)
        if direction is None:
            self.direction = np.array([0.0, 0.0, 1.0])
        else:
            self.direction = np.array(direction, dtype=float)
            self.direction = self.direction / np.linalg.norm(self.direction)
        
        # Wavelength and intensity
        self.wavelength = float(wavelength)
        self.intensity = float(intensity)
        self.is_spectral_component = is_spectral_component
        
        # Initialize polarization if provided
        if polarization is not None:
            self.polarization = np.array(polarization, dtype=float)
            self.polarization = self.polarization / np.linalg.norm(self.polarization)
        else:
            # Default polarization perpendicular to direction
            if abs(self.direction[2]) < 0.9:
                self.polarization = np.cross([0, 0, 1], self.direction)
            else:
                self.polarization = np.cross([1, 0, 0], self.direction)
            self.polarization = self.polarization / np.linalg.norm(self.polarization)
        
        # Path history
        self.path = [self.position.copy()]
        self.active = True
        
        # For rainbow simulation
        self.reflection_count = 0
    
    def move(self, distance: float) -> None:
        """
        Move the photon along its current direction.
        
        Args:
            distance: Distance to move
        """
        self.position = self.position + distance * self.direction
        self.path.append(self.position.copy())
    
    def reflect(self, normal: np.ndarray) -> None:
        """
        Reflect the photon at a surface with the given normal.
        
        Args:
            normal: Normal vector at reflection point (should be normalized)
        """
        # Ensure normal is normalized
        normal = normal / np.linalg.norm(normal)
        
        # Calculate reflection direction: r = d - 2(d·n)n
        dot_product = np.dot(self.direction, normal)
        self.direction = self.direction - 2.0 * dot_product * normal
        
        # Update polarization
        dot_pol = np.dot(self.polarization, normal)
        if abs(dot_pol) > 1e-10:
            self.polarization = self.polarization - 2.0 * dot_pol * normal
            self.polarization = self.polarization / np.linalg.norm(self.polarization)
        
        # Increment reflection count
        self.reflection_count += 1
    
    def refract(self, normal: np.ndarray, n1: float, n2: float) -> bool:
        """
        Refract the photon at a medium interface.
        
        Args:
            normal: Surface normal (normalized)
            n1: Refractive index of current medium
            n2: Refractive index of new medium
            
        Returns:
            True if refraction occurred, False if total internal reflection
        """
        # Ensure normal is normalized
        normal = normal / np.linalg.norm(normal)
        
        # Make sure normal points against incident direction
        dot_product = np.dot(self.direction, normal)
        if dot_product > 0:
            normal = -normal
            dot_product = -dot_product
        
        # Calculate sin(theta_t) using Snell's law
        sin_theta_i = np.sqrt(1.0 - dot_product**2)
        sin_theta_t = (n1 / n2) * sin_theta_i
        
        # Check for total internal reflection
        if sin_theta_t >= 1.0:
            self.reflect(normal)
            return False
        
        # Calculate refracted direction
        cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)
        self.direction = (n1 / n2) * self.direction + \
                         ((n1 / n2) * dot_product - cos_theta_t) * normal
        self.direction = self.direction / np.linalg.norm(self.direction)
        
        # Update polarization (simplified)
        return True
    
    def calculate_fresnel(self, normal: np.ndarray, n1: float, n2: float) -> Tuple[float, float]:
        """
        Calculate Fresnel coefficients for reflection and transmission.
        
        Args:
            normal: Surface normal (normalized)
            n1: Refractive index of current medium
            n2: Refractive index of new medium
            
        Returns:
            Tuple of (reflection coefficient, transmission coefficient)
        """
        # Ensure normal is normalized and points against incident direction
        normal = normal / np.linalg.norm(normal)
        cos_theta_i = -np.dot(self.direction, normal)
        
        if cos_theta_i < 0:
            normal = -normal
            cos_theta_i = -cos_theta_i
        
        # Calculate sin(theta_t) using Snell's law
        sin_theta_i = np.sqrt(1.0 - cos_theta_i**2)
        sin_theta_t = (n1 / n2) * sin_theta_i
        
        # Check for total internal reflection
        if sin_theta_t >= 1.0:
            return 1.0, 0.0
        
        cos_theta_t = np.sqrt(1.0 - sin_theta_t**2)
        
        # Calculate reflection coefficients for s and p polarizations
        r_s = ((n1 * cos_theta_i - n2 * cos_theta_t) / 
               (n1 * cos_theta_i + n2 * cos_theta_t))**2
        
        r_p = ((n1 * cos_theta_t - n2 * cos_theta_i) / 
               (n1 * cos_theta_t + n2 * cos_theta_i))**2
        
        # Calculate average for unpolarized light
        r = (r_s + r_p) / 2.0
        
        # Transmission coefficient from conservation of energy
        t = 1.0 - r
        
        return r, t
    
    def split(self, normal: np.ndarray, n1: float, n2: float) -> Optional['Photon']:
        """
        Split the photon into reflected and transmitted components.
        
        Args:
            normal: Surface normal
            n1: Refractive index of current medium
            n2: Refractive index of new medium
            
        Returns:
            New photon representing the reflected component if split occurs,
            None otherwise
        """
        r, t = self.calculate_fresnel(normal, n1, n2)
        
        # If reflection is negligible, don't split
        if r < 0.01:
            # Just refract the current photon
            self.refract(normal, n1, n2)
            return None
        
        # If transmission is negligible, just reflect
        if t < 0.01:
            self.reflect(normal)
            return None
        
        # Create new photon for reflected component
        reflected = Photon(
            position=self.position.copy(),
            direction=self.direction.copy(),
            wavelength=self.wavelength,
            intensity=self.intensity * r,
            polarization=self.polarization.copy(),
            is_spectral_component=self.is_spectral_component
        )
        
        # Reflect the new photon
        reflected.reflect(normal)
        reflected.path = self.path.copy()
        reflected.reflection_count = self.reflection_count + 1
        
        # Refract the current photon
        self.refract(normal, n1, n2)
        self.intensity *= t
        
        return reflected
    
    def get_rgb_color(self) -> Tuple[float, float, float]:
        """
        Convert wavelength to RGB color for visualization.
        
        Returns:
            Tuple of (R, G, B) values in range [0, 1]
        """
        # Visible spectrum is approximately 380-750 nm
        if self.wavelength < 380 or self.wavelength > 750:
            return (0.0, 0.0, 0.0)
        
        # Approximate conversion based on wavelength
        if self.wavelength < 440:
            # Violet/Blue
            r = (440 - self.wavelength) / (440 - 380)
            g = 0.0
            b = 1.0
        elif self.wavelength < 490:
            # Blue/Cyan
            r = 0.0
            g = (self.wavelength - 440) / (490 - 440)
            b = 1.0
        elif self.wavelength < 510:
            # Cyan/Green
            r = 0.0
            g = 1.0
            b = (510 - self.wavelength) / (510 - 490)
        elif self.wavelength < 580:
            # Green/Yellow
            r = (self.wavelength - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif self.wavelength < 645:
            # Yellow/Red
            r = 1.0
            g = (645 - self.wavelength) / (645 - 580)
            b = 0.0
        else:
            # Red
            r = 1.0
            g = 0.0
            b = 0.0
        
        # Scale intensity at spectrum edges
        gamma = 0.8
        if self.wavelength < 420:
            factor = 0.3 + 0.7 * (self.wavelength - 380) / (420 - 380)
        elif self.wavelength > 700:
            factor = 0.3 + 0.7 * (750 - self.wavelength) / (750 - 700)
        else:
            factor = 1.0
        
        # Apply gamma correction
        r = pow(r * factor, gamma)
        g = pow(g * factor, gamma)
        b = pow(b * factor, gamma)
        
        return (r, g, b)
    
    def terminate(self) -> None:
        """Mark the photon as terminated."""
        self.active = False
    
    def is_alive(self) -> bool:
        """Check if the photon is still active."""
        return self.active and self.intensity > 0.001

class WhiteLight:
    """
    Represents a white light ray composed of multiple wavelengths in 3D space.
    
    This class handles the generation and management of multiple spectral components
    for simulating dispersion effects like rainbows.
    """
    
    def __init__(self, 
                 position: Union[List[float], np.ndarray],
                 direction: Union[List[float], np.ndarray],
                 num_wavelengths: int = 30,
                 intensity: float = 1.0):
        """
        Initialize a white light ray with spectral components.
        
        Args:
            position: 3D position [x, y, z]
            direction: 3D direction vector
            num_wavelengths: Number of wavelength components to simulate
            intensity: Initial intensity of the light
        """
        self.position = np.array(position)
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))
        self.intensity = intensity
        
        # Generate wavelength components across visible spectrum
        self.min_wl = 380  # violet
        self.max_wl = 750  # red
        self.wavelengths = np.linspace(self.min_wl, self.max_wl, num_wavelengths)
        
        # Create individual photons for each wavelength
        self.spectral_components = []
        for wl in self.wavelengths:
            photon = Photon(
                position=self.position.copy(),
                direction=self.direction.copy(),
                wavelength=wl,
                intensity=self.intensity,
                is_spectral_component=True
            )
            self.spectral_components.append(photon)
    
    def trace(self, medium_manager, geometry_manager, max_depth: int = 10, min_intensity: float = 0.001):
        """
        Trace all spectral components through the scene.
        
        Args:
            medium_manager: Manager for handling media properties
            geometry_manager: Manager for handling geometry intersections
            max_depth: Maximum recursion depth for tracing
            min_intensity: Minimum intensity threshold
        """
        # Keep track of all photons to process, including split photons
        all_photons = self.spectral_components.copy()
        
        # Keep tracking photons that are generated through splitting
        new_photons = []
        
        # Trace each photon up to max_depth
        for depth in range(max_depth):
            new_photons.clear()
            
            # Process each active photon
            for photon in all_photons:
                if not photon.is_alive() or photon.intensity < min_intensity:
                    photon.terminate()
                    continue
                
                # Find closest intersection
                hit, distance, boundary = geometry_manager.find_intersection(
                    photon.position, photon.direction)
                
                if not hit:
                    # No intersection, photon escapes
                    # Add a point far along the ray direction for visualization
                    far_point = photon.position + 10.0 * photon.direction
                    photon.path.append(far_point)
                    photon.terminate()
                    continue
                
                # Move to intersection point
                photon.move(distance)
                
                # Get medium information
                current_medium, next_medium = geometry_manager.get_medium_pair(
                    boundary, photon.position, photon.direction)
                
                # Get refractive indices
                n1 = medium_manager.get_refractive_index(current_medium, photon.wavelength)
                n2 = medium_manager.get_refractive_index(next_medium, photon.wavelength)
                
                # Get normal at intersection
                normal = boundary.get_normal(photon.position, photon.direction)
                
                # Handle reflection/refraction with possible splitting
                split_photon = photon.split(normal, n1, n2)
                
                if split_photon is not None:
                    new_photons.append(split_photon)
            
            # Add newly created photons to the processing list
            all_photons.extend(new_photons)
            
            # Check if all photons are terminated
            if all(not photon.is_alive() for photon in all_photons):
                break
        
        # Return all photons including original and split ones
        return all_photons