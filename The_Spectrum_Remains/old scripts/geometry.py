import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any

class Boundary:
    """
    Abstract base class for geometric boundaries in the rainbow simulation.
    """
    def __init__(self, name: str = "", inside_medium: str = "", outside_medium: str = ""):
        """
        Initialize a boundary with an optional name and medium identifiers.
        
        Args:
            name: Name identifier for the boundary
            inside_medium: Name of the medium inside this boundary
            outside_medium: Name of the medium outside this boundary
        """
        self.name = name
        self.inside_medium = inside_medium
        self.outside_medium = outside_medium
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a ray intersects this boundary and return distance to intersection.
        
        Args:
            position: Current position [x, y, z]
            direction: Direction vector (will be normalized)
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
            If no hit, distance will be infinity
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def normal(self, position: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Return the surface normal at the given position.
        
        Args:
            position: Position on the boundary
            direction: Incident direction (needed to determine inside/outside)
            
        Returns:
            Unit normal vector pointing against the incident direction
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_entering(self, position: np.ndarray, direction: np.ndarray) -> bool:
        """
        Determine if a ray is entering or exiting the boundary.
        
        Args:
            position: Position on the boundary
            direction: Incident direction
            
        Returns:
            True if entering the boundary, False if exiting
        """
        # Get normal at intersection point
        norm = self.normal(position, direction)
        
        # If dot product is negative, ray is entering
        return np.dot(direction, norm) < 0


class Sphere(Boundary):
    """
    Spherical boundary defined by center and radius.
    
    Important for raindrop simulation.
    """
    def __init__(self, center: List[float], radius: float, 
                 name: str = "", inside_medium: str = "", outside_medium: str = ""):
        """
        Initialize a sphere.
        
        Args:
            center: Center point of the sphere [x, y, z]
            radius: Radius of the sphere
            name: Optional name
            inside_medium: Name of the medium inside this boundary
            outside_medium: Name of the medium outside this boundary
        """
        super().__init__(name, inside_medium, outside_medium)
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """
        Check ray-sphere intersection using quadratic formula.
        
        Args:
            position: Current position [x, y, z]
            direction: Direction vector
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
        """
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
    
    def normal(self, position: np.ndarray, direction: np.ndarray = None) -> np.ndarray:
        """
        Return normal vector at position on sphere surface.
        
        Args:
            position: Position on the sphere surface
            direction: Incident direction (optional)
            
        Returns:
            Unit normal vector
        """
        # Normal points from center to position
        normal_vec = position - self.center
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        
        # Ensure normal points against the incident direction if provided
        if direction is not None and np.dot(normal_vec, direction) > 0:
            normal_vec = -normal_vec
            
        return normal_vec


class Plane(Boundary):
    """
    Infinite plane boundary defined by a point and normal vector.
    
    Useful for creating flat surfaces like ground or detector planes.
    """
    def __init__(self, point: List[float], normal_vector: List[float], 
                 name: str = "", inside_medium: str = "", outside_medium: str = ""):
        """
        Initialize a plane.
        
        Args:
            point: A point on the plane [x, y, z]
            normal_vector: Normal vector to the plane
            name: Optional name
            inside_medium: Name of the medium inside this boundary
            outside_medium: Name of the medium outside this boundary
        """
        super().__init__(name, inside_medium, outside_medium)
        self.point = np.array(point, dtype=float)
        self.normal_vector = self._normalize(np.array(normal_vector, dtype=float))
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.array([0, 0, 1])
        return vector / norm
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """
        Check ray-plane intersection.
        
        Args:
            position: Current position [x, y, z]
            direction: Direction vector
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
        """
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
    
    def normal(self, position: np.ndarray, direction: np.ndarray = None) -> np.ndarray:
        """
        Return normal vector (constant for a plane).
        
        Args:
            position: Position on the plane (unused for planes)
            direction: Incident direction (optional)
            
        Returns:
            Unit normal vector
        """
        # Ensure normal points against the incident direction if provided
        if direction is not None and np.dot(self.normal_vector, direction) > 0:
            return -self.normal_vector
        return self.normal_vector


class Cylinder(Boundary):
    """
    Cylindrical boundary defined by axis, radius, and height.
    
    Useful for detector surfaces and other shapes.
    """
    def __init__(self, base: List[float], axis: List[float], radius: float, height: float,
                 name: str = "", inside_medium: str = "", outside_medium: str = ""):
        """
        Initialize a cylinder.
        
        Args:
            base: Center of the base of the cylinder [x, y, z]
            axis: Direction vector of the cylinder axis
            radius: Radius of the cylinder
            height: Height of the cylinder
            name: Optional name
            inside_medium: Name of the medium inside this boundary
            outside_medium: Name of the medium outside this boundary
        """
        super().__init__(name, inside_medium, outside_medium)
        self.base = np.array(base, dtype=float)
        self.axis = self._normalize(np.array(axis, dtype=float))
        self.radius = float(radius)
        self.height = float(height)
        
        # Calculate top center point
        self.top = self.base + self.height * self.axis
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.array([0, 0, 1])
        return vector / norm
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """
        Check ray-cylinder intersection.
        
        Args:
            position: Current position [x, y, z]
            direction: Direction vector
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
        """
        direction = self._normalize(direction)
        
        # Transform to cylinder coordinates
        rel_pos = position - self.base
        
        # Project onto axis
        pos_on_axis = np.dot(rel_pos, self.axis) * self.axis
        
        # Vector from axis to position
        radial_vec = rel_pos - pos_on_axis
        
        # Project direction onto axis
        dir_on_axis = np.dot(direction, self.axis) * self.axis
        
        # Vector perpendicular to axis
        radial_dir = direction - dir_on_axis
        
        # Quadratic equation coefficients
        a = np.dot(radial_dir, radial_dir)
        b = 2.0 * np.dot(radial_vec, radial_dir)
        c = np.dot(radial_vec, radial_vec) - self.radius**2
        
        # Check if ray is parallel to cylinder axis
        if abs(a) < 1e-10:
            if abs(c) < 1e-10:  # Ray inside cylinder and parallel to axis
                # Check cap intersections
                t_base = -np.dot(rel_pos, self.axis) / np.dot(direction, self.axis)
                t_top = (self.height - np.dot(rel_pos, self.axis)) / np.dot(direction, self.axis)
                
                if t_base > 1e-10 and t_top > 1e-10:
                    return True, min(t_base, t_top)
                elif t_base > 1e-10:
                    return True, t_base
                elif t_top > 1e-10:
                    return True, t_top
                else:
                    return False, float('inf')
            else:
                return False, float('inf')
        
        # Calculate discriminant
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return False, float('inf')
        
        # Calculate cylinder intersection points
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        # Check if intersection points are on the cylinder (not infinite)
        t_values = []
        
        for t in [t1, t2]:
            if t > 1e-10:
                # Calculate intersection point
                p = position + t * direction
                
                # Calculate height on cylinder axis
                h = np.dot(p - self.base, self.axis)
                
                # Check if point is within cylinder height
                if 0 <= h <= self.height:
                    t_values.append(t)
        
        # Check caps
        if np.dot(direction, self.axis) != 0:
            # Bottom cap
            t_base = -np.dot(rel_pos, self.axis) / np.dot(direction, self.axis)
            if t_base > 1e-10:
                p = position + t_base * direction
                radial_dist = np.linalg.norm(p - self.base - np.dot(p - self.base, self.axis) * self.axis)
                if radial_dist <= self.radius:
                    t_values.append(t_base)
            
            # Top cap
            t_top = (self.height - np.dot(rel_pos, self.axis)) / np.dot(direction, self.axis)
            if t_top > 1e-10:
                p = position + t_top * direction
                radial_dist = np.linalg.norm(p - self.top - np.dot(p - self.top, self.axis) * self.axis)
                if radial_dist <= self.radius:
                    t_values.append(t_top)
        
        if t_values:
            return True, min(t_values)
        else:
            return False, float('inf')
    
    def normal(self, position: np.ndarray, direction: np.ndarray = None) -> np.ndarray:
        """
        Return the surface normal at a position on the cylinder.
        
        Args:
            position: Position on the cylinder surface
            direction: Incident direction (optional)
            
        Returns:
            Unit normal vector
        """
        # Check if position is on one of the caps
        height_on_axis = np.dot(position - self.base, self.axis)
        
        if abs(height_on_axis) < 1e-10:
            # Bottom cap
            normal_vec = -self.axis
        elif abs(height_on_axis - self.height) < 1e-10:
            # Top cap
            normal_vec = self.axis
        else:
            # Side of cylinder
            # Project position onto axis
            closest_on_axis = self.base + height_on_axis * self.axis
            
            # Normal points from axis to position
            normal_vec = position - closest_on_axis
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
        
        # Ensure normal points against the incident direction if provided
        if direction is not None and np.dot(normal_vec, direction) > 0:
            normal_vec = -normal_vec
            
        return normal_vec


class Detector:
    """
    Special detector boundary for recording photon hits.
    
    This is a planar detector that records photon positions and properties.
    """
    def __init__(self, position: List[float], normal: List[float], 
                 size: Tuple[float, float] = (1.0, 1.0),
                 resolution: Tuple[int, int] = (100, 100),
                 name: str = "detector"):
        """
        Initialize a detector plane.
        
        Args:
            position: Center position of the detector [x, y, z]
            normal: Normal vector of the detector plane
            size: Physical size of the detector (width, height)
            resolution: Grid resolution for recording hits (pixels_x, pixels_y)
            name: Name of the detector
        """
        self.position = np.array(position, dtype=float)
        self.normal = self._normalize(np.array(normal, dtype=float))
        self.size = size
        self.resolution = resolution
        self.name = name
        
        # Calculate local coordinate system for the detector plane
        self._setup_coordinate_system()
        
        # Initialize detector grid for recording hits
        # Grid will store intensity values for each spectral bin
        self.num_spectral_bins = 64  # Number of wavelength bins
        self.grid = np.zeros((resolution[0], resolution[1], self.num_spectral_bins))
        
        # Create the actual plane for intersection testing
        self.plane = Plane(position, normal, name=name)
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.array([0, 0, 1])
        return vector / norm
    
    def _setup_coordinate_system(self):
        """Create local coordinate system for detector plane."""
        # Create two orthogonal vectors in the plane
        if abs(self.normal[2]) > 0.9:
            # If normal is close to z-axis, use x-axis for u_axis
            self.u_axis = self._normalize(np.cross(np.array([1, 0, 0]), self.normal))
        else:
            # Otherwise use z-axis
            self.u_axis = self._normalize(np.cross(np.array([0, 0, 1]), self.normal))
            
        # v_axis is perpendicular to both normal and u_axis
        self.v_axis = self._normalize(np.cross(self.normal, self.u_axis))
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a ray intersects the detector.
        
        Args:
            position: Current position [x, y, z]
            direction: Direction vector
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
        """
        # First check intersection with the infinite plane
        hit, distance = self.plane.intersect(position, direction)
        
        if not hit:
            return False, float('inf')
        
        # Calculate intersection point
        hit_point = position + distance * direction
        
        # Calculate local coordinates on the detector plane
        local_u = np.dot(hit_point - self.position, self.u_axis)
        local_v = np.dot(hit_point - self.position, self.v_axis)
        
        # Check if hit point is within detector bounds
        half_width = self.size[0] / 2
        half_height = self.size[1] / 2
        
        if -half_width <= local_u <= half_width and -half_height <= local_v <= half_height:
            return True, distance
        else:
            return False, float('inf')
    
    def register_hit(self, photon) -> bool:
        """
        Register a photon hit on the detector.
        
        Args:
            photon: Photon that hit the detector
            
        Returns:
            True if hit was successful and within detector bounds
        """
        # Check intersection
        hit, distance = self.intersect(photon.position, photon.direction)
        
        if not hit:
            return False
        
        # Calculate hit point
        hit_point = photon.position + distance * photon.direction
        
        # Calculate local coordinates on the detector
        local_u = np.dot(hit_point - self.position, self.u_axis)
        local_v = np.dot(hit_point - self.position, self.v_axis)
        
        # Convert to grid coordinates
        half_width = self.size[0] / 2
        half_height = self.size[1] / 2
        
        # Map from [-half_size, +half_size] to [0, resolution]
        grid_x = int((local_u + half_width) / self.size[0] * self.resolution[0])
        grid_y = int((local_v + half_height) / self.size[1] * self.resolution[1])
        
        # Ensure coordinates are within grid bounds
        if 0 <= grid_x < self.resolution[0] and 0 <= grid_y < self.resolution[1]:
            # Get spectral bin for this photon
            spectral_bin = self._wavelength_to_bin(photon.wavelength)
            
            # Add photon intensity to the grid
            self.grid[grid_x, grid_y, spectral_bin] += photon.intensity
            return True
        
        return False
    
    def _wavelength_to_bin(self, wavelength: float) -> int:
        """
        Convert wavelength to spectral bin index.
        
        Args:
            wavelength: Wavelength in nm
            
        Returns:
            Bin index in range [0, num_spectral_bins-1]
        """
        # Visible spectrum roughly 380-750nm
        min_wl, max_wl = 380, 750
        normalized = (wavelength - min_wl) / (max_wl - min_wl)
        bin_index = int(normalized * self.num_spectral_bins)
        return max(0, min(bin_index, self.num_spectral_bins-1))  # Clamp to valid range
    
    def get_rgb_image(self) -> np.ndarray:
        """
        Convert spectral data to RGB image.
        
        Returns:
            RGB image as numpy array (resolution_x, resolution_y, 3)
        """
        from photon import Photon  # Import here to avoid circular import
        
        # Create RGB image
        rgb_image = np.zeros((self.resolution[0], self.resolution[1], 3))
        
        # For each spectral bin, add its contribution to the RGB image
        for bin_idx in range(self.num_spectral_bins):
            # Get approximate wavelength from bin index
            wavelength = 380 + (750-380) * bin_idx / self.num_spectral_bins
            
            # Create a temporary photon to use its get_rgb_color method
            temp_photon = Photon(wavelength=wavelength)
            r, g, b = temp_photon.get_rgb_color()
            
            # Add contribution to each pixel
            for x in range(self.resolution[0]):
                for y in range(self.resolution[1]):
                    intensity = self.grid[x, y, bin_idx]
                    rgb_image[x, y, 0] += r * intensity
                    rgb_image[x, y, 1] += g * intensity
                    rgb_image[x, y, 2] += b * intensity
        
        # Normalize image if needed
        max_val = np.max(rgb_image)
        if max_val > 0:
            rgb_image = rgb_image / max_val
        
        return rgb_image
    
    def clear(self) -> None:
        """Reset the detector grid."""
        self.grid = np.zeros((self.resolution[0], self.resolution[1], self.num_spectral_bins))


class GeometryManager:
    """
    Manager for all geometric objects in the simulation.
    
    Handles boundary lookups and medium interfaces.
    """
    def __init__(self):
        """Initialize an empty geometry manager."""
        self.boundaries = []
        self.detectors = []
        self.media = {}  # Map of medium name to Medium objects
    
    def add_boundary(self, boundary: Boundary) -> None:
        """Add a boundary to the manager."""
        self.boundaries.append(boundary)
    
    def add_detector(self, detector: Detector) -> None:
        """Add a detector to the manager."""
        self.detectors.append(detector)
    
    def register_medium(self, name: str, medium) -> None:
        """
        Register a medium with the geometry manager.
        
        Args:
            name: Name of the medium
            medium: Medium object
        """
        self.media[name] = medium
    
    def find_intersection(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float, Optional[Boundary]]:
        """
        Find the closest boundary intersection.
        
        Args:
            position: Current position
            direction: Direction vector
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit, boundary_hit)
        """
        closest_dist = float('inf')
        closest_boundary = None
        hit_occurred = False
        
        # Check all regular boundaries
        for boundary in self.boundaries:
            hit, distance = boundary.intersect(position, direction)
            if hit and distance < closest_dist:
                hit_occurred = True
                closest_dist = distance
                closest_boundary = boundary
        
        # Check all detectors
        for detector in self.detectors:
            hit, distance = detector.intersect(position, direction)
            if hit and distance < closest_dist:
                hit_occurred = True
                closest_dist = distance
                closest_boundary = detector
        
        return hit_occurred, closest_dist, closest_boundary
    
    def get_medium_pair(self, boundary: Boundary, position: np.ndarray, direction: np.ndarray) -> Tuple[Any, Any]:
        """
        Get the media pair for a boundary intersection.
        
        Args:
            boundary: Boundary that was intersected
            position: Position of intersection
            direction: Direction of photon
            
        Returns:
            Tuple of (current_medium, next_medium)
        """
        is_entering = boundary.is_entering(position, direction)
        
        if is_entering:
            current_medium = self.media.get(boundary.outside_medium)
            next_medium = self.media.get(boundary.inside_medium)
        else:
            current_medium = self.media.get(boundary.inside_medium)
            next_medium = self.media.get(boundary.outside_medium)
        
        return current_medium, next_medium
    
    def register_detector_hits(self, photon) -> bool:
        """
        Register hits on all detectors.
        
        Args:
            photon: Photon to check for detector hits
            
        Returns:
            True if photon hit any detector
        """
        for detector in self.detectors:
            if detector.register_hit(photon):
                return True
        return False