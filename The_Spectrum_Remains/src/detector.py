import numpy as np
from typing import Tuple, List, Dict


# Helper function for wavelength to RGB conversion
def wavelength_to_rgb(wavelength):
    """Convert wavelength to RGB color for visualization."""
    # Visible spectrum is approximately 380-750 nm
    if wavelength < 380 or wavelength > 750:
        return (0.0, 0.0, 0.0)
    
    # Approximate conversion based on wavelength
    if wavelength < 440:
        # Violet/Blue
        r = (440 - wavelength) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wavelength < 490:
        # Blue/Cyan
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif wavelength < 510:
        # Cyan/Green
        r = 0.0
        g = 1.0
        b = (510 - wavelength) / (510 - 490)
    elif wavelength < 580:
        # Green/Yellow
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wavelength < 645:
        # Yellow/Red
        r = 1.0
        g = (645 - wavelength) / (645 - 580)
        b = 0.0
    else:
        # Red
        r = 1.0
        g = 0.0
        b = 0.0
    
    # Scale intensity at spectrum edges
    gamma = 0.8
    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength > 700:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
    else:
        factor = 1.0
    
    # Apply gamma correction
    r = pow(r * factor, gamma)
    g = pow(g * factor, gamma)
    b = pow(b * factor, gamma)
    
    return (r, g, b)


class CubeDetector:
    """Detector shaped as a cube surrounding the droplet."""
    
    def __init__(self, center=[0, 0, 0], size=5.0, resolution=100):
        """
        Initialize a cube detector.
        
        Args:
            center: Center of the cube
            size: Side length of the cube
            resolution: Grid resolution per side
        """
        self.center = np.array(center)
        self.size = size
        self.resolution = resolution
        
        # Half-size for calculations
        self.half_size = size / 2.0
        
        # Create detector grids for each face of the cube
        # Each grid stores spectral information
        # Format: [face_index][y][x][wavelength_bin]
        self.faces = []
        self.face_names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
        
        # Number of wavelength bins (for visible spectrum)
        self.num_wavelength_bins = 64
        self.wavelength_range = (380, 750)  # visible spectrum in nm
        
        # Initialize all faces
        for _ in range(6):
            face = np.zeros((resolution, resolution, self.num_wavelength_bins))
            self.faces.append(face)
    
    def intersect(self, position, direction):
        """
        Find intersection with cube.
        
        Args:
            position: Ray position
            direction: Ray direction
            
        Returns:
            Tuple of (hit_occurred, distance, face_index, uv_coords)
        """
        # Normalized direction
        direction = direction / np.linalg.norm(direction)
        
        # Distance to each face
        t_values = []
        face_indices = []
        uv_coords = []
        
        # Check each dimension (x, y, z)
        for dim in range(3):
            for sign in [-1, 1]:
                # Face position
                face_pos = self.center[dim] + sign * self.half_size
                
                # Check if ray is parallel to face
                if abs(direction[dim]) < 1e-6:
                    continue
                
                # Calculate distance to face
                t = (face_pos - position[dim]) / direction[dim]
                
                # Only consider faces in front of the ray
                if t < 0:
                    continue
                
                # Calculate intersection point
                intersection = position + t * direction
                
                # Check if intersection is within face bounds
                in_bounds = True
                u, v = 0, 0
                
                for other_dim in range(3):
                    if other_dim == dim:
                        continue
                    
                    # Check if within cube bounds
                    coord = intersection[other_dim] - self.center[other_dim]
                    if abs(coord) > self.half_size:
                        in_bounds = False
                        break
                    
                    # Calculate UV coordinates (normalized to 0-1)
                    if u == 0:
                        u = (coord + self.half_size) / self.size
                    else:
                        v = (coord + self.half_size) / self.size
                
                if in_bounds:
                    # Determine face index
                    face_idx = 2 * dim + (0 if sign > 0 else 1)
                    
                    t_values.append(t)
                    face_indices.append(face_idx)
                    uv_coords.append((u, v))
        
        if not t_values:
            return False, float('inf'), -1, (0, 0)
        
        # Find closest intersection
        min_idx = np.argmin(t_values)
        return True, t_values[min_idx], face_indices[min_idx], uv_coords[min_idx]
    
    def register_hit(self, photon):
        """
        Register photon hit on the detector.
        
        Args:
            photon: The photon that hit the detector
            
        Returns:
            True if hit was successful
        """
        # Check intersection
        hit, distance, face_idx, (u, v) = self.intersect(photon.position, photon.direction)
        
        if not hit or face_idx < 0:
            return False
        
        # Convert to grid coordinates
        grid_x = min(int(u * self.resolution), self.resolution - 1)
        grid_y = min(int(v * self.resolution), self.resolution - 1)
        
        # Get wavelength bin
        bin_idx = self._wavelength_to_bin(photon.wavelength)
        
        # Add photon intensity to the grid
        self.faces[face_idx][grid_y, grid_x, bin_idx] += photon.intensity
        
        return True
    
    def _wavelength_to_bin(self, wavelength):
        """Convert wavelength to bin index."""
        min_wl, max_wl = self.wavelength_range
        if wavelength < min_wl or wavelength > max_wl:
            return 0
        
        normalized = (wavelength - min_wl) / (max_wl - min_wl)
        bin_idx = int(normalized * self.num_wavelength_bins)
        return min(max(0, bin_idx), self.num_wavelength_bins - 1)
    
    def get_rgb_image(self, face_idx):
        """
        Convert spectral data to RGB image for a specific face.
        
        Args:
            face_idx: Index of the face to visualize
            
        Returns:
            RGB image as numpy array
        """
        if face_idx < 0 or face_idx >= len(self.faces):
            return None
        
        # Create RGB image
        rgb_image = np.zeros((self.resolution, self.resolution, 3))
        
        # For each wavelength bin, add its RGB contribution
        for bin_idx in range(self.num_wavelength_bins):
            # Calculate wavelength from bin index
            wavelength = self.wavelength_range[0] + bin_idx * (
                self.wavelength_range[1] - self.wavelength_range[0]
            ) / self.num_wavelength_bins
            
            # Convert wavelength to RGB
            r, g, b = wavelength_to_rgb(wavelength)
            
            # Add contribution to each pixel
            for y in range(self.resolution):
                for x in range(self.resolution):
                    intensity = self.faces[face_idx][y, x, bin_idx]
                    rgb_image[y, x, 0] += r * intensity
                    rgb_image[y, x, 1] += g * intensity
                    rgb_image[y, x, 2] += b * intensity
        
        # Normalize if needed
        max_val = np.max(rgb_image)
        if max_val > 0:
            rgb_image = rgb_image / max_val
        
        return rgb_image

class PlaneDetector:
    """
    Detector shaped as a plane for capturing the rainbow pattern.
    This version properly handles wavelength addition.
    """
    
    def __init__(self, position, normal, width=5.0, height=5.0, resolution=100, name="detector"):
        """
        Initialize a plane detector.
        
        Args:
            position: Center position of the plane
            normal: Normal vector to the plane
            width: Width of the detector plane
            height: Height of the detector plane
            resolution: Grid resolution
            name: Name of the detector
        """
        self.position = np.array(position)
        self.normal = np.array(normal) / np.linalg.norm(np.array(normal))
        self.width = width
        self.height = height
        self.resolution = resolution
        self.name = name
        
        # Set up local coordinate system
        self._setup_coordinates()
        
        # Number of wavelength bins and range
        self.num_wavelength_bins = 64
        self.wavelength_range = (380, 750)  # visible spectrum in nm
        
        # Initialize spectral grid [y][x][wavelength_bin]
        self.spectral_grid = np.zeros((resolution, resolution, self.num_wavelength_bins))
        
        # Keep track of total intensity at each pixel for proper normalization
        self.total_intensity = np.zeros((resolution, resolution))
        
        # For white light detection
        self.white_light_threshold = 0.9  # If >90% of wavelengths present, treat as white
    
    def _setup_coordinates(self):
        """Create local coordinate system for the plane."""
        # Create basis vectors on the plane
        if abs(self.normal[2]) > 0.9:
            # If normal is close to z-axis, use x-axis for u_axis
            self.u_axis = self._normalize(np.cross(np.array([1, 0, 0]), self.normal))
        else:
            # Otherwise use z-axis
            self.u_axis = self._normalize(np.cross(np.array([0, 0, 1]), self.normal))
            
        # v_axis is perpendicular to both normal and u_axis
        self.v_axis = self._normalize(np.cross(self.normal, self.u_axis))
    
    def _normalize(self, vector):
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return np.array([0, 0, 1])
        return vector / norm
    
    def intersect(self, position, direction):
        """
        Find intersection with the detector plane.
        
        Args:
            position: Ray position
            direction: Ray direction
            
        Returns:
            Tuple of (hit_occurred, distance, uv_coords)
        """
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Calculate dot product for ray-plane intersection
        denom = np.dot(direction, self.normal)
        
        # If ray is parallel to plane, no intersection
        if abs(denom) < 1e-10:
            return False, float('inf'), (0, 0)
        
        # Calculate distance to plane along ray
        t = np.dot(self.position - position, self.normal) / denom
        
        # If plane is behind the ray, no intersection
        if t <= 0:
            return False, float('inf'), (0, 0)
        
        # Calculate intersection point
        intersection = position + t * direction
        
        # Calculate local coordinates on the plane
        relative_pos = intersection - self.position
        u = np.dot(relative_pos, self.u_axis)
        v = np.dot(relative_pos, self.v_axis)
        
        # Check if intersection is within plane bounds
        half_width = self.width / 2
        half_height = self.height / 2
        if -half_width <= u <= half_width and -half_height <= v <= half_height:
            # Convert to UV coordinates (0-1 range)
            u_coord = (u + half_width) / self.width
            v_coord = (v + half_height) / self.height
            return True, t, (u_coord, v_coord)
        
        return False, float('inf'), (0, 0)
    
    def register_hit(self, photon):
        """
        Register a photon hit on the detector.
        
        Args:
            photon: The photon that hit
            
        Returns:
            True if hit was registered
        """
        # Check intersection
        hit, distance, (u, v) = self.intersect(photon.position, photon.direction)
        
        if not hit:
            return False
        
        # Convert to grid coordinates
        grid_x = min(int(u * self.resolution), self.resolution - 1)
        grid_y = min(int(v * self.resolution), self.resolution - 1)
        
        # Get wavelength bin
        bin_idx = self._wavelength_to_bin(photon.wavelength)
        
        # Add photon intensity to the grid
        self.spectral_grid[grid_y, grid_x, bin_idx] += photon.intensity
        self.total_intensity[grid_y, grid_x] += photon.intensity
        
        return True
    
    def _wavelength_to_bin(self, wavelength):
        """Convert wavelength to spectral bin index."""
        min_wl, max_wl = self.wavelength_range
        normalized = (wavelength - min_wl) / (max_wl - min_wl)
        bin_idx = int(normalized * self.num_wavelength_bins)
        return min(max(0, bin_idx), self.num_wavelength_bins - 1)
    
    def get_rgb_image(self):
        """
        Convert spectral data to RGB image.
        
        Returns:
            RGB image as numpy array
        """
        # Create RGB image
        rgb_image = np.zeros((self.resolution, self.resolution, 3))
        
        # For showing wavelength coverage (white vs colored light)
        wavelength_presence = np.zeros((self.resolution, self.resolution))
        
        # For each spectral bin, add its contribution
        for bin_idx in range(self.num_wavelength_bins):
            # Calculate wavelength from bin index
            wavelength = self.wavelength_range[0] + bin_idx * (
                self.wavelength_range[1] - self.wavelength_range[0]
            ) / self.num_wavelength_bins
            
            # Convert wavelength to RGB
            r, g, b = wavelength_to_rgb(wavelength)
            
            # Count which pixels have significant energy in this wavelength
            threshold = 0.01 * np.max(self.spectral_grid[:,:,bin_idx])
            has_wavelength = (self.spectral_grid[:,:,bin_idx] > threshold)
            wavelength_presence += has_wavelength
            
            # Add contribution to each pixel
            for y in range(self.resolution):
                for x in range(self.resolution):
                    intensity = self.spectral_grid[y, x, bin_idx]
                    rgb_image[y, x, 0] += r * intensity
                    rgb_image[y, x, 1] += g * intensity
                    rgb_image[y, x, 2] += b * intensity
        
        # Normalize the image
        max_val = np.max(rgb_image)
        if max_val > 0:
            rgb_image = rgb_image / max_val
        
        # Convert pixels with most wavelengths to white
        normalized_presence = wavelength_presence / self.num_wavelength_bins
        white_mask = normalized_presence > self.white_light_threshold
        
        # Highlight white light areas
        for y in range(self.resolution):
            for x in range(self.resolution):
                if white_mask[y, x]:
                    # Make it white
                    rgb_image[y, x] = [1.0, 1.0, 1.0]
        
        # Create background mask for light gray
        gray_mask = self.total_intensity == 0
        rgb_image[gray_mask] = [0.8, 0.8, 0.8]  # Light gray for no light
        
        return rgb_image