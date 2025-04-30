import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from .photon import Photon
from .medium import Medium
from .geometry import Boundary, GeometryManager

class Simulation:
    """
    Main simulation engine for Monte Carlo ray tracing of light transport.
    
    Attributes:
        medium: Medium in which photons propagate
        geometry: GeometryManager handling boundary interactions
        grid_size: Size of the simulation grid for recording data
        results: Dictionary storing simulation results
    """
    
    def __init__(self, medium: Medium, geometry: Optional[GeometryManager] = None, 
                 grid_size: Tuple[int, int, int] = (100, 100, 100),
                 grid_spacing: float = 1.0):
        """
        Initialize a simulation.
        
        Args:
            medium: Medium for photon propagation
            geometry: Optional geometry manager for boundary interactions
            grid_size: Size of grid for recording data (x, y, z)
            grid_spacing: Spacing between grid points
        """
        self.medium = medium
        self.geometry = geometry if geometry is not None else GeometryManager()
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        
        # Initialize data structures for recording results
        self.reset_results()
    
    def reset_results(self) -> None:
        """Clear and initialize results storage."""
        self.results = {
            'deposited_energy': np.zeros(self.grid_size),
            'photon_paths': [],
            'transmitted': 0,
            'reflected': 0,
            'absorbed': 0,
            'escaped': 0
        }
    
    def run(self, num_photons: int, 
            weight_threshold: float = 0.001, 
            max_steps: int = 1000,
            progress_callback=None) -> Dict[str, Any]:
        """
        Run the simulation with specified number of photons.
        
        Args:
            num_photons: Number of photons to simulate
            weight_threshold: Minimum photon weight before termination
            max_steps: Maximum number of steps per photon
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with simulation results
        """
        self.reset_results()
        start_time = time.time()
        
        for i in range(num_photons):
            # Create a new photon
            photon = Photon()
            
            # Track photon until termination
            self._trace_photon(photon, weight_threshold, max_steps)
            
            # Store photon path for visualization (store every 10th photon to save memory)
            if i % 10 == 0:
                self.results['photon_paths'].append(photon.get_path())
            
            # Update progress if callback provided
            if progress_callback and i % max(1, num_photons // 100) == 0:
                progress = (i + 1) / num_photons * 100
                progress_callback(progress)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Add additional statistics to results
        self.results['elapsed_time'] = elapsed_time
        self.results['photons_per_second'] = num_photons / elapsed_time
        
        return self.results
    
    def _trace_photon(self, photon: Photon, weight_threshold: float, max_steps: int) -> None:
        """
        Trace a single photon through the medium until termination.
        
        Args:
            photon: Photon to trace
            weight_threshold: Minimum weight before termination
            max_steps: Maximum number of steps before forced termination
        """
        step_count = 0
        
        while photon.is_alive() and step_count < max_steps:
            step_count += 1
            
            # Sample step size
            step_size = self.medium.sample_step_size()
            
            # Check for boundary interactions
            hit, distance, boundary = self.geometry.find_intersection(photon)
            
            if hit and distance < step_size:
                # Boundary interaction occurs before next scattering event
                
                # Move photon to boundary
                photon.move(distance)
                
                # Get surface normal at intersection point
                normal = boundary.normal(photon.position)
                
                # Handle reflection/refraction (simplified)
                # For now, just reflect the photon
                dot_product = np.dot(photon.direction, normal)
                photon.direction = photon.direction - 2 * dot_product * normal
                
                # Count as a reflection
                self.results['reflected'] += 1
                
            else:
                # No boundary hit, regular step through medium
                
                # Move photon
                photon.move(step_size)
                
                # Handle absorption
                albedo = self.medium.albedo()
                if albedo < 1.0:
                    # Absorb energy
                    absorbed_weight = photon.absorb(1.0 - albedo)
                    
                    # Record absorbed energy in grid
                    grid_pos = self._world_to_grid(photon.position)
                    if self._is_in_grid(grid_pos):
                        self.results['deposited_energy'][tuple(grid_pos)] += absorbed_weight
                
                # Check if photon weight is below threshold
                if photon.weight < weight_threshold:
                    # Russian roulette for weight below threshold
                    if np.random.random() < 0.1:  # 10% chance to survive
                        photon.weight *= 10.0  # Boost weight
                    else:
                        # Count as fully absorbed
                        self.results['absorbed'] += 1
                        photon.terminate()
                        continue
                
                # Scatter photon
                photon.scatter(self.medium.g)
            
            # Check if photon is outside simulation domain
            if not self._is_position_in_domain(photon.position):
                self.results['escaped'] += 1
                photon.terminate()
        
        # Count photons that reached max steps as escaped
        if step_count >= max_steps and photon.is_alive():
            self.results['escaped'] += 1
            photon.terminate()
    
    def _world_to_grid(self, position: np.ndarray) -> List[int]:
        """
        Convert world coordinates to grid indices.
        
        Args:
            position: Position in world coordinates
            
        Returns:
            Grid indices [i, j, k]
        """
        # Assuming the grid is centered at origin
        half_size = np.array(self.grid_size) / 2
        offset = position / self.grid_spacing + half_size
        return [int(offset[0]), int(offset[1]), int(offset[2])]
    
    def _is_in_grid(self, grid_pos: List[int]) -> bool:
        """Check if grid position is within bounds."""
        for i in range(3):
            if grid_pos[i] < 0 or grid_pos[i] >= self.grid_size[i]:
                return False
        return True
    
    def _is_position_in_domain(self, position: np.ndarray) -> bool:
        """Check if a position is within the simulation domain."""
        # Define domain bounds (can be customized)
        half_domain = np.array(self.grid_size) * self.grid_spacing / 2
        for i in range(3):
            if abs(position[i]) > half_domain[i]:
                return False
        return True
    
    def get_results(self) -> Dict[str, Any]:
        """Return simulation results."""
        return self.results