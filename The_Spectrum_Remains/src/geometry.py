class Boundary:
    """Base class for geometric boundaries in 3D space."""
    
    def __init__(self, name: str = "", inside_medium: str = "", outside_medium: str = ""):
        """
        Initialize a boundary.
        
        Args:
            name: Name identifier
            inside_medium: Key for the medium inside the boundary
            outside_medium: Key for the medium outside the boundary
        """
        self.name = name
        self.inside_medium = inside_medium
        self.outside_medium = outside_medium
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a ray intersects this boundary.
        
        Args:
            position: Starting position
            direction: Ray direction
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
        """
        raise NotImplementedError("Subclass must implement this method")
    
    def get_normal(self, position: np.ndarray, direction: np.ndarray = None) -> np.ndarray:
        """
        Get the normal vector at a point on the boundary.
        
        Args:
            position: Position on boundary
            direction: Incident ray direction (optional)
            
        Returns:
            Normal vector (normalized)
        """
        raise NotImplementedError("Subclass must implement this method")
    
    def is_inside(self, position: np.ndarray) -> bool:
        """
        Check if a point is inside the boundary.
        
        Args:
            position: Position to check
            
        Returns:
            True if inside, False otherwise
        """
        raise NotImplementedError("Subclass must implement this method")


class Sphere(Boundary):
    """Spherical boundary in 3D space."""
    
    def __init__(self, center: Union[List[float], np.ndarray], radius: float, 
                 name: str = "", inside_medium: str = "water", outside_medium: str = "air"):
        """
        Initialize a sphere.
        
        Args:
            center: Center point [x, y, z]
            radius: Radius of the sphere
            name: Name identifier
            inside_medium: Key for the medium inside the sphere
            outside_medium: Key for the medium outside the sphere
        """
        super().__init__(name, inside_medium, outside_medium)
        self.center = np.array(center)
        self.radius = radius
    
    def intersect(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a ray intersects this sphere.
        
        Args:
            position: Starting position
            direction: Ray direction
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit)
        """
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Vector from origin to center
        oc = position - self.center
        
        # Quadratic equation coefficients
        a = np.dot(direction, direction)  # Should be 1.0
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius**2
        
        # Calculate discriminant
        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0:
            return False, float('inf')
        
        # Find the nearer of the two intersections
        sqrtd = np.sqrt(discriminant)
        
        # Check first intersection
        t1 = (-b - sqrtd) / (2.0 * a)
        if t1 > 1e-6:  # Avoid self-intersection
            return True, t1
        
        # Check second intersection
        t2 = (-b + sqrtd) / (2.0 * a)
        if t2 > 1e-6:
            return True, t2
        
        # No valid intersection
        return False, float('inf')
    
    def get_normal(self, position: np.ndarray, direction: np.ndarray = None) -> np.ndarray:
        """
        Get the normal vector at a point on the sphere.
        
        Args:
            position: Position on sphere
            direction: Incident ray direction (optional)
            
        Returns:
            Normal vector (normalized)
        """
        normal = position - self.center
        normal = normal / np.linalg.norm(normal)
        
        # Ensure normal points against incident direction if provided
        if direction is not None and np.dot(normal, direction) > 0:
            normal = -normal
            
        return normal
    
    def is_inside(self, position: np.ndarray) -> bool:
        """
        Check if a point is inside the sphere.
        
        Args:
            position: Position to check
            
        Returns:
            True if inside, False otherwise
        """
        distance_squared = np.sum((position - self.center)**2)
        return distance_squared < self.radius**2


class GeometryManager:
    """Manages geometric objects and boundaries in 3D space."""
    
    def __init__(self):
        """Initialize the geometry manager."""
        self.boundaries = []
    
    def add_boundary(self, boundary: Boundary) -> None:
        """
        Add a boundary to the manager.
        
        Args:
            boundary: Boundary object
        """
        self.boundaries.append(boundary)
    
    def find_intersection(self, position: np.ndarray, direction: np.ndarray) -> Tuple[bool, float, Optional[Boundary]]:
        """
        Find the closest boundary intersection.
        
        Args:
            position: Starting position
            direction: Ray direction
            
        Returns:
            Tuple of (hit_occurred, distance_to_hit, boundary_hit)
        """
        closest_dist = float('inf')
        closest_boundary = None
        hit_occurred = False
        
        for boundary in self.boundaries:
            hit, distance = boundary.intersect(position, direction)
            if hit and distance < closest_dist:
                hit_occurred = True
                closest_dist = distance
                closest_boundary = boundary
        
        return hit_occurred, closest_dist, closest_boundary
    
    def get_medium_pair(self, boundary: Boundary, position: np.ndarray, direction: np.ndarray) -> Tuple[str, str]:
        """
        Get the media pair for a boundary intersection.
        
        Args:
            boundary: Boundary that was intersected
            position: Position of intersection
            direction: Direction of ray
            
        Returns:
            Tuple of (current_medium_key, next_medium_key)
        """
        normal = boundary.get_normal(position, direction)
        is_entering = np.dot(direction, normal) < 0
        
        if is_entering:
            return boundary.outside_medium, boundary.inside_medium
        else:
            return boundary.inside_medium, boundary.outside_medium