import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from .photon import Photon

class Boundary:
    """
    Abstract base class for geometric boundaries in the simulation.
    """
    def __init__(self, name: str = ""):
        """Initialize a boundary with an optional name."""
        self.name = name
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a ray intersects this boundary and return distance to intersection.
        
        Args:
            position: Current position [x, y, z]
            direction: Direction vector
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
            If no hit, distance will be infinity
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def normal(self, position: np.ndarray) -> np.ndarray:
        """
        Return the surface normal at the given position.
        
        Args:
            position: Position on the boundary
            
        Returns:
            Unit normal vector
        """
        raise NotImplementedError("Subclasses must implement this method")


class Plane(Boundary):
    """
    Infinite plane boundary defined by a point and normal vector.
    """
    def __init__(self, point: List[float], normal_vector: List[float], name: str = ""):
        """
        Initialize a plane.
        
        Args:
            point: A point on the plane [x, y, z]
            normal_vector: Normal vector to the plane
            name: Optional name
        """
        super().__init__(name)
        self.point = np.array(point, dtype=float)
        self.normal_vector = self._normalize(np.array(normal_vector, dtype=float))
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.array([0, 0, 1])
        return vector / norm
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """Check ray-plane intersection."""
        direction = self._normalize(direction)
        
        # Calculate denominator (dot product of direction and normal)
        denom = np.dot(direction, self.normal_vector)
        
        # If denominator is close to 0, ray is parallel to plane
        if abs(denom) < 1e-10:
            return False, float('inf')
        
        # Calculate distance along ray to intersection
        t = np.dot(self.point - position, self.normal_vector) / denom
        
        # Only positive distances are valid (in front of the ray)
        if t > 1e-10:  # Small epsilon to avoid self-intersections
            return True, t
        else:
            return False, float('inf')
    
    def normal(self, position: np.ndarray) -> np.ndarray:
        """Return normal vector (constant for a plane)."""
        return self.normal_vector


class Sphere(Boundary):
    """
    Sphere boundary defined by center and radius.
    """
    def __init__(self, center: List[float], radius: float, name: str = ""):
        """
        Initialize a sphere.
        
        Args:
            center: Center point of the sphere [x, y, z]
            radius: Radius of the sphere
            name: Optional name
        """
        super().__init__(name)
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """Check ray-sphere intersection."""
        # Normalize direction vector
        direction = direction / np.linalg.norm(direction)
        
        # Vector from position to sphere center
        oc = position - self.center
        
        # Quadratic equation coefficients: at² + bt + c = 0
        a = np.dot(direction, direction)  # Should be 1.0 if direction is normalized
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        
        # Calculate discriminant
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            # No real roots, no intersection
            return False, float('inf')
        
        # Calculate both intersection points
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        # Get closest intersection in front of the ray
        if t1 > 1e-10:  # Small epsilon to avoid self-intersections
            return True, t1
        elif t2 > 1e-10:
            return True, t2
        else:
            return False, float('inf')
    
    def normal(self, position: np.ndarray) -> np.ndarray:
        """Return normal vector at position on sphere surface."""
        # Normal points from center to position
        normal_vec = position - self.center
        return normal_vec / np.linalg.norm(normal_vec)


class Box(Boundary):
    """
    Axis-aligned box boundary defined by minimum and maximum corners.
    """
    def __init__(self, min_corner: List[float], max_corner: List[float], name: str = ""):
        """
        Initialize an axis-aligned box.
        
        Args:
            min_corner: Minimum corner coordinates [x_min, y_min, z_min]
            max_corner: Maximum corner coordinates [x_max, y_max, z_max]
            name: Optional name
        """
        super().__init__(name)
        self.min_corner = np.array(min_corner, dtype=float)
        self.max_corner = np.array(max_corner, dtype=float)
        
        # Ensure min is actually min and max is actually max
        for i in range(3):
            if self.min_corner[i] > self.max_corner[i]:
                self.min_corner[i], self.max_corner[i] = self.max_corner[i], self.min_corner[i]
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """Check ray-box intersection using slab method."""
        # Small number to avoid division by zero
        eps = 1e-10
        
        t_min = -float('inf')
        t_max = float('inf')
        
        for i in range(3):
            # Handle case where direction component is near zero
            if abs(direction[i]) < eps:
                # Ray is parallel to slab. No hit if origin not within slab
                if position[i] < self.min_corner[i] or position[i] > self.max_corner[i]:
                    return False, float('inf')
            else:
                # Calculate intersection with the two planes perpendicular to axis i
                t1 = (self.min_corner[i] - position[i]) / direction[i]
                t2 = (self.max_corner[i] - position[i]) / direction[i]
                
                # Ensure t1 <= t2
                if t1 > t2:
                    t1, t2 = t2, t1
                
                # Update bounds
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                
                if t_min > t_max:
                    return False, float('inf')
        
        # Check if intersection is in front of the ray
        if t_min > 1e-10:  # Small epsilon to avoid self-intersections
            return True, t_min
        elif t_max > 1e-10:
            return True, t_max
        else:
            return False, float('inf')
    
    def normal(self, position: np.ndarray) -> np.ndarray:
        """Return normal vector at position on box surface."""
        # Find which face the position is on by checking which component is closest to the boundary
        normal = np.zeros(3)
        min_dist = float('inf')
        
        for i in range(3):
            dist_min = abs(position[i] - self.min_corner[i])
            dist_max = abs(position[i] - self.max_corner[i])
            
            if dist_min < min_dist:
                min_dist = dist_min
                normal = np.zeros(3)
                normal[i] = -1.0  # Normal points in negative direction
                
            if dist_max < min_dist:
                min_dist = dist_max
                normal = np.zeros(3)
                normal[i] = 1.0  # Normal points in positive direction
        
        return normal


class GeometryManager:
    """
    Manages all geometric boundaries and interfaces in the simulation.
    """
    def __init__(self):
        """Initialize an empty geometry manager."""
        self.boundaries = []
    
    def add_boundary(self, boundary: Boundary) -> None:
        """Add a boundary to the manager."""
        self.boundaries.append(boundary)
    
    def find_intersection(self, photon: Photon) -> Tuple[bool, float, Optional[Boundary]]:
        """
        Find the closest boundary intersection for a photon.
        
        Args:
            photon: Photon to check
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit, boundary_hit)
        """
        closest_dist = float('inf')
        closest_boundary = None
        hit_occurred = False
        
        for boundary in self.boundaries:
            hit, distance = boundary.intersect(photon.position, photon.direction)
            if hit and distance < closest_dist:
                hit_occurred = True
                closest_dist = distance
                closest_boundary = boundary
        
        return hit_occurred, closest_dist, closest_boundary