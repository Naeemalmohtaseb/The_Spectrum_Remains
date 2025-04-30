import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any

class Photon:
    """
    Represents a spectral photon in the rainbow simulation with wavelength-dependent properties.
    
    Attributes:
        position (np.ndarray): 3D position vector [x, y, z]
        direction (np.ndarray): 3D normalized direction vector
        wavelength (float): Wavelength in nanometers
        intensity (float): Current intensity of the photon (1.0 = full energy)
        polarization (np.ndarray): Polarization vector (optional)
        path (List[np.ndarray]): List of positions for path tracking
        alive (bool): Whether the photon is still active in the simulation
    """
    
    def __init__(self, 
                 position: List[float] = None, 
                 direction: List[float] = None, 
                 wavelength: float = 550.0,
                 intensity: float = 1.0,
                 polarization: List[float] = None):
        """
        Initialize a new spectral photon.
        
        Args:
            position: Initial position [x, y, z], defaults to [0, 0, 0]
            direction: Initial direction vector, defaults to [0, 0, 1] (z-axis)
            wavelength: Wavelength in nanometers (visible range ~380-750nm)
            intensity: Initial intensity, defaults to 1.0
            polarization: Polarization vector (optional)
        """
        self.position = np.array(position if position is not None else [0, 0, 0], dtype=float)
        self.direction = self._normalize(np.array(direction if direction is not None else [0, 0, 1], dtype=float))
        self.wavelength = float(wavelength)
        self.intensity = float(intensity)
        
        # Initialize polarization if provided (important for accurate rainbow physics)
        if polarization is not None:
            self.polarization = self._normalize(np.array(polarization, dtype=float))
        else:
            # Default polarization perpendicular to direction
            if abs(self.direction[2]) < 0.9:
                # Not aligned with z-axis, use cross product with z-axis
                self.polarization = self._normalize(np.cross([0, 0, 1], self.direction))
            else:
                # Aligned with z-axis, use cross product with x-axis
                self.polarization = self._normalize(np.cross([1, 0, 0], self.direction))
        
        self.path = [self.position.copy()]  # Track path for visualization
        self.alive = True
        
        # Additional attributes for rainbow simulation
        self.num_reflections = 0  # Track number of internal reflections
        self.phase = 0.0  # For modeling interference effects (optional)
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:  # Avoid division by zero
            return np.array([0, 0, 1])  # Default direction if zero vector
        return vector / norm
    
    def move(self, step_size: float) -> None:
        """
        Move the photon along its current direction.
        
        Args:
            step_size: Distance to move
        """
        self.position += step_size * self.direction
        self.path.append(self.position.copy())
    
    def reflect(self, normal: np.ndarray) -> None:
        """
        Reflect the photon direction based on a surface normal.
        
        Args:
            normal: Surface normal vector (must be normalized)
        """
        # Calculate reflection direction: r = d - 2(d·n)n
        dot_product = np.dot(self.direction, normal)
        self.direction = self.direction - 2 * dot_product * normal
        
        # Update polarization (reflect polarization vector as well)
        # For accurate rainbow physics, polarization changes must be modeled
        # This is a simplified approach
        dot_product_pol = np.dot(self.polarization, normal)
        if abs(dot_product_pol) > 1e-10:  # Only update if not perpendicular
            self.polarization = self.polarization - 2 * dot_product_pol * normal
            self.polarization = self._normalize(self.polarization)
        
        self.num_reflections += 1
    
    def refract(self, normal: np.ndarray, n1: float, n2: float) -> bool:
        """
        Refract the photon at a medium boundary.
        
        Args:
            normal: Surface normal vector (must be normalized)
            n1: Refractive index of current medium
            n2: Refractive index of medium being entered
            
        Returns:
            True if refraction occurred, False if total internal reflection
        """
        # Calculate refraction direction using Snell's law
        # Make sure normal points in the right direction (against incident ray)
        dot_product = np.dot(self.direction, normal)
        if dot_product > 0:
            normal = -normal  # Flip normal if it's not against incident ray
            dot_product = -dot_product
        
        # Calculate sin(theta_t) using Snell's law: n1*sin(theta_i) = n2*sin(theta_t)
        sin_theta_i = np.sqrt(1 - dot_product**2)  # sin(theta_i) = sqrt(1 - cos^2(theta_i))
        sin_theta_t = (n1 / n2) * sin_theta_i
        
        # Check for total internal reflection
        if sin_theta_t >= 1.0:
            # Total internal reflection occurs
            self.reflect(normal)
            return False
        
        # Calculate refracted direction
        cos_theta_t = np.sqrt(1 - sin_theta_t**2)
        refracted_dir = (n1 / n2) * self.direction - ((n1 / n2) * dot_product + cos_theta_t) * normal
        
        # Update direction
        self.direction = self._normalize(refracted_dir)
        
        # Update polarization (simplified model)
        # For a complete model, implement Fresnel equations for polarization components
        # This is important for accurate rainbow intensity patterns
        
        return True
    
    def get_rgb_color(self) -> Tuple[float, float, float]:
        """
        Convert photon wavelength to RGB color for visualization.
        
        Returns:
            Tuple of (R, G, B) values in range [0, 1]
        """
        # Visible spectrum is approximately 380-750 nm
        if self.wavelength < 380 or self.wavelength > 750:
            return (0.0, 0.0, 0.0)  # Outside visible spectrum
        
        # Approximate conversion based on the CIE standard observer
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
        
        # Scale RGB values based on intensity perception
        gamma = 0.8
        if self.wavelength < 420:
            factor = 0.3 + 0.7 * (self.wavelength - 380) / (420 - 380)
        elif self.wavelength > 700:
            factor = 0.3 + 0.7 * (750 - self.wavelength) / (750 - 700)
        else:
            factor = 1.0
        
        r = pow(r * factor, gamma)
        g = pow(g * factor, gamma)
        b = pow(b * factor, gamma)
        
        return (r, g, b)
    
    def get_spectral_bin(self, num_bins: int = 64) -> int:
        """
        Map wavelength to a spectral bin for color combination.
        
        Args:
            num_bins: Number of spectral bins to use
            
        Returns:
            Bin index in range [0, num_bins-1]
        """
        # Visible spectrum roughly 380-750nm
        min_wl, max_wl = 380, 750
        normalized = (self.wavelength - min_wl) / (max_wl - min_wl)
        bin_index = int(normalized * num_bins)
        return max(0, min(bin_index, num_bins-1))  # Clamp to valid range
    
    def terminate(self) -> None:
        """Mark the photon as terminated (no longer active in simulation)."""
        self.alive = False
    
    def is_alive(self) -> bool:
        """Check if the photon is still active."""
        return self.alive and self.intensity > 0.0
    
    def get_path(self) -> np.ndarray:
        """Return the complete path history of the photon."""
        return np.array(self.path)
    
    def split(self, split_factor: float = 0.5) -> 'Photon':
        """
        Split the photon into two for handling partial reflection/refraction.
        
        Args:
            split_factor: Factor determining intensity split between original and new photon
            
        Returns:
            New photon with portion of the intensity
        """
        # Create a new photon with the same properties
        new_photon = Photon(
            position=self.position.copy(),
            direction=self.direction.copy(),
            wavelength=self.wavelength,
            intensity=self.intensity * split_factor,
            polarization=self.polarization.copy() if hasattr(self, 'polarization') else None
        )
        
        # Reduce the intensity of the original photon
        self.intensity *= (1.0 - split_factor)
        
        # Copy history
        new_photon.path = [pos.copy() for pos in self.path]
        new_photon.num_reflections = self.num_reflections
        
        return new_photon
    
    def __str__(self) -> str:
        """String representation of the photon."""
        return (f"Photon(λ={self.wavelength:.1f}nm, pos={self.position}, "
                f"dir={self.direction}, intensity={self.intensity:.3f})")