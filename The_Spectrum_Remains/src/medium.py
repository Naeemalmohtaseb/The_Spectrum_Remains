import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Callable

class Medium:
    """
    Represents a medium with wavelength-dependent optical properties in 3D.
    """
    
    def __init__(self, name: str, refr_index_func=None, absorption_func=None):
        """
        Initialize a medium.
        
        Args:
            name: Name of the medium
            refr_index_func: Function to calculate refractive index based on wavelength
            absorption_func: Function to calculate absorption based on wavelength
        """
        self.name = name
        self.refr_index_func = refr_index_func
        self.absorption_func = absorption_func
        
        # Default values if functions not provided
        self.default_refr_index = 1.0
        self.default_absorption = 0.0
    
    def get_refractive_index(self, wavelength: float) -> float:
        """
        Get refractive index for a specific wavelength.
        
        Args:
            wavelength: Light wavelength in nanometers
            
        Returns:
            Refractive index value
        """
        if self.refr_index_func is not None:
            return self.refr_index_func(wavelength)
        return self.default_refr_index
    
    def get_absorption(self, wavelength: float) -> float:
        """
        Get absorption coefficient for a specific wavelength.
        
        Args:
            wavelength: Light wavelength in nanometers
            
        Returns:
            Absorption coefficient
        """
        if self.absorption_func is not None:
            return self.absorption_func(wavelength)
        return self.default_absorption


class MediumManager:
    """
    Manages different media in the simulation.
    """
    
    def __init__(self):
        """Initialize the medium manager."""
        self.media = {}
        self.add_default_media()
    
    def add_default_media(self):
        """Add default media (air, water, etc.)."""
        # Air medium
        def air_index(wavelength):
            """Simplified model for air refractive index."""
            # Air has very little dispersion, but we include a small wavelength dependence
            return 1.0003 - 1.0e-8 * (wavelength - 550)**2
            
        self.add_medium("air", Medium(
            name="Air",
            refr_index_func=air_index
        ))
        
        # Water medium
        def water_index(wavelength):
            """Cauchy's equation for water's refractive index."""
            # Convert wavelength from nm to μm for Cauchy's formula
            wl_um = wavelength / 1000.0
            
            # Cauchy's formula coefficients for water
            A = 1.324
            B = 0.0065   # double
            C = 0.00062  # double

            
            # Calculate refractive index
            n = A + B / (wl_um**2) + C / (wl_um**4)
            
            return n
            
        self.add_medium("water", Medium(
            name="Water",
            refr_index_func=water_index
        ))
    
    def add_medium(self, key: str, medium: Medium) -> None:
        """
        Add a medium to the manager.
        
        Args:
            key: Key for the medium
            medium: Medium object
        """
        self.media[key] = medium
    
    def get_refractive_index(self, medium_key: str, wavelength: float) -> float:
        """
        Get refractive index for a specific medium and wavelength.
        
        Args:
            medium_key: Key for the medium
            wavelength: Light wavelength in nanometers
            
        Returns:
            Refractive index value
        """
        if medium_key in self.media:
            return self.media[medium_key].get_refractive_index(wavelength)
        
        # Default to air if medium not found
        return 1.0003