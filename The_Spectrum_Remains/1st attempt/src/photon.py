import numpy as np
from typing import List, Tuple, Optional, Union

class Photon:
    """
    Represents a photon in the Monte Carlo simulation with position, direction, and weight.
    
    Attributes:
        position (np.ndarray): 3D position vector [x, y, z]
        direction (np.ndarray): 3D normalized direction vector
        weight (float): Current weight of the photon (1.0 = full energy)
        history (List[np.ndarray]): List of positions for path tracking
        alive (bool): Whether the photon is still active in the simulation
    """
    
    def __init__(self, position: List[float] = None, direction: List[float] = None, weight: float = 1.0):
        """
        Initialize a new photon.
        
        Args:
            position: Initial position [x, y, z], defaults to [0, 0, 0]
            direction: Initial direction vector, defaults to [0, 0, 1] (z-axis)
            weight: Initial weight/energy, defaults to 1.0
        """
        self.position = np.array(position if position is not None else [0, 0, 0], dtype=float)
        self.direction = self._normalize(np.array(direction if direction is not None else [0, 0, 1], dtype=float))
        self.weight = weight
        self.history = [self.position.copy()]  # Track path for visualization
        self.alive = True
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.array([0, 0, 1])  # Default direction if zero vector
        return vector / norm
    
    def move(self, step_size: float) -> None:
        """
        Move the photon along its current direction.
        
        Args:
            step_size: Distance to move
        """
        self.position += step_size * self.direction
        self.history.append(self.position.copy())
    
    def scatter(self, g: float = 0.0) -> None:
        """
        Change the photon's direction according to a scattering phase function.
        
        Args:
            g: Anisotropy factor for Henyey-Greenstein phase function (-1 to 1)
               g=0: isotropic, g=1: forward, g=-1: backward scattering
        """
        if g == 0:
            # Isotropic scattering
            phi = 2 * np.pi * np.random.random()  # Azimuthal angle [0, 2π]
            cos_theta = 2 * np.random.random() - 1  # Polar angle cosine [-1, 1]
        else:
            # Henyey-Greenstein phase function
            phi = 2 * np.pi * np.random.random()  # Azimuthal angle [0, 2π]
            
            # Sample cos_theta from H-G phase function
            if abs(g) < 1e-10:  # Handle g ≈ 0 case
                cos_theta = 2 * np.random.random() - 1
            else:
                temp = (1 - g * g) / (1 - g + 2 * g * np.random.random())
                cos_theta = (1 + g * g - temp * temp) / (2 * g)
                # Ensure cos_theta is in valid range due to floating point errors
                cos_theta = np.clip(cos_theta, -1, 1)
        
        sin_theta = np.sqrt(1 - cos_theta * cos_theta)
        
        # Calculate new direction based on current direction and scattering angles
        # First, construct a local coordinate system
        if abs(self.direction[2]) > 0.99999:
            # Special case: direction is along z-axis
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, 1.0, 0.0])
            w = self.direction
        else:
            # General case
            u = self._normalize(np.cross(np.array([0, 0, 1]), self.direction))
            v = np.cross(self.direction, u)
            w = self.direction
        
        # Calculate new direction in local system, then transform to global
        new_direction = (
            sin_theta * np.cos(phi) * u +
            sin_theta * np.sin(phi) * v +
            cos_theta * w
        )
        
        self.direction = self._normalize(new_direction)
    
    def absorb(self, fraction: float) -> float:
        """
        Absorb a fraction of the photon's weight and return the absorbed amount.
        
        Args:
            fraction: Fraction of current weight to absorb [0, 1]
        
        Returns:
            Amount of weight/energy absorbed
        """
        absorbed = self.weight * fraction
        self.weight -= absorbed
        return absorbed
    
    def terminate(self) -> None:
        """Mark the photon as terminated (no longer active in simulation)."""
        self.alive = False
    
    def is_alive(self) -> bool:
        """Check if the photon is still active."""
        return self.alive and self.weight > 0
    
    def get_path(self) -> np.ndarray:
        """Return the complete path history of the photon."""
        return np.array(self.history)