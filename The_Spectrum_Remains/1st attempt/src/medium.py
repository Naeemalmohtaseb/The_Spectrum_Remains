from typing import Tuple, List, Optional, Dict, Any
import numpy as np

class Medium:
    """
    Represents a medium with optical properties for light transport simulation.
    
    Attributes:
        mu_a (float): Absorption coefficient [1/length]
        mu_s (float): Scattering coefficient [1/length]
        mu_t (float): Total interaction coefficient (mu_a + mu_s) [1/length]
        g (float): Anisotropy factor [-1 to 1]
        n (float): Refractive index
        name (str): Optional name for the medium
    """
    
    def __init__(self, mu_a: float = 0.1, mu_s: float = 10.0, g: float = 0.9, n: float = 1.0, name: str = ""):
        """
        Initialize a medium with optical properties.
        
        Args:
            mu_a: Absorption coefficient [1/length]
            mu_s: Scattering coefficient [1/length]
            g: Anisotropy factor (-1: backscattering, 0: isotropic, 1: forward)
            n: Refractive index
            name: Optional name for the medium
        """
        self.mu_a = float(mu_a)
        self.mu_s = float(mu_s)
        self.mu_t = mu_a + mu_s  # Total interaction coefficient
        self.g = float(g)
        self.n = float(n)
        self.name = name
    
    def albedo(self) -> float:
        """
        Calculate the single-scattering albedo.
        
        Returns:
            Albedo value [0 to 1], where 0 means pure absorption and 1 means pure scattering
        """
        if self.mu_t == 0:
            return 0.0
        return self.mu_s / self.mu_t
    
    def mfp(self) -> float:
        """
        Calculate the mean free path (average distance between interactions).
        
        Returns:
            Mean free path [length units]
        """
        if self.mu_t == 0:
            return float('inf')
        return 1.0 / self.mu_t
    
    def sample_step_size(self) -> float:
        """
        Sample a step size from exponential distribution based on the total interaction coefficient.
        
        Returns:
            Step size [length units]
        """
        if self.mu_t == 0:
            return float('inf')
        return -np.log(np.random.random()) / self.mu_t

    def __str__(self) -> str:
        """String representation of the medium."""
        name_str = f" ({self.name})" if self.name else ""
        return (f"Medium{name_str}: μₐ={self.mu_a:.4f}, μₛ={self.mu_s:.4f}, "
                f"g={self.g:.4f}, n={self.n:.4f}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert medium properties to dictionary for serialization."""
        return {
            'mu_a': self.mu_a,
            'mu_s': self.mu_s,
            'g': self.g,
            'n': self.n,
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Medium':
        """Create a Medium instance from a dictionary."""
        return cls(
            mu_a=data.get('mu_a', 0.1),
            mu_s=data.get('mu_s', 10.0),
            g=data.get('g', 0.9),
            n=data.get('n', 1.0),
            name=data.get('name', '')
        )


# Common predefined media
def water() -> Medium:
    """Return optical properties of water."""
    return Medium(mu_a=0.05, mu_s=0.1, g=0.9, n=1.33, name="Water")

def skin() -> Medium:
    """Return optical properties of human skin (approximate)."""
    return Medium(mu_a=0.5, mu_s=20.0, g=0.9, n=1.4, name="Skin")

def fog() -> Medium:
    """Return optical properties of fog (approximate)."""
    return Medium(mu_a=0.01, mu_s=80.0, g=0.97, n=1.0, name="Fog")

def clear_tissue() -> Medium:
    """Return optical properties of clear biological tissue (approximate)."""
    return Medium(mu_a=0.1, mu_s=10.0, g=0.9, n=1.4, name="Clear Tissue")