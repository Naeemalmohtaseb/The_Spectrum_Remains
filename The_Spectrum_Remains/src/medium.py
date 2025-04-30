from typing import Tuple, List, Dict, Any, Optional, Callable
import numpy as np

class Medium:
    """
    Represents a medium with wavelength-dependent optical properties for rainbow simulation.
    
    Attributes:
        name (str): Name of the medium
        n_func (Callable): Function to calculate refractive index based on wavelength
        mu_a_func (Callable): Function to calculate absorption coefficient based on wavelength
        mu_s_func (Callable): Function to calculate scattering coefficient based on wavelength
        g_func (Callable): Function to calculate anisotropy factor based on wavelength
        dispersion_model (str): Name of dispersion model used ('cauchy', 'sellmeier', etc.)
    """
    
    def __init__(self, 
                 name: str = "",
                 n_func: Optional[Callable] = None,
                 mu_a_func: Optional[Callable] = None,
                 mu_s_func: Optional[Callable] = None,
                 g_func: Optional[Callable] = None,
                 n_fixed: float = 1.0,
                 mu_a_fixed: float = 0.0,
                 mu_s_fixed: float = 0.0,
                 g_fixed: float = 0.0,
                 dispersion_model: str = ""):
        """
        Initialize a medium with optical properties.
        
        Args:
            name: Name of the medium
            n_func: Function f(wavelength) returning refractive index
            mu_a_func: Function f(wavelength) returning absorption coefficient [1/length]
            mu_s_func: Function f(wavelength) returning scattering coefficient [1/length]
            g_func: Function f(wavelength) returning anisotropy factor [-1 to 1]
            n_fixed: Fixed refractive index (used if n_func is None)
            mu_a_fixed: Fixed absorption coefficient (used if mu_a_func is None)
            mu_s_fixed: Fixed scattering coefficient (used if mu_s_func is None)
            g_fixed: Fixed anisotropy factor (used if g_func is None)
            dispersion_model: Name of dispersion model used
        """
        self.name = name
        
        # Store functions for wavelength-dependent properties
        self.n_func = n_func
        self.mu_a_func = mu_a_func
        self.mu_s_func = mu_s_func
        self.g_func = g_func
        
        # Store fixed values as fallbacks
        self.n_fixed = float(n_fixed)
        self.mu_a_fixed = float(mu_a_fixed)
        self.mu_s_fixed = float(mu_s_fixed)
        self.g_fixed = float(g_fixed)
        
        # Record dispersion model used
        self.dispersion_model = dispersion_model
        
        # Cache for frequently accessed wavelengths (optimization)
        self._n_cache = {}
    
    def get_refractive_index(self, wavelength: float) -> float:
        """
        Get refractive index for a specific wavelength.
        
        Args:
            wavelength: Light wavelength in nm
            
        Returns:
            Refractive index value
        """
        # Check cache first
        if wavelength in self._n_cache:
            return self._n_cache[wavelength]
        
        # Calculate refractive index using function if provided
        if self.n_func is not None:
            n = self.n_func(wavelength)
        else:
            n = self.n_fixed
        
        # Store in cache and return
        self._n_cache[wavelength] = n
        return n
    
    def get_absorption(self, wavelength: float) -> float:
        """
        Get absorption coefficient for a specific wavelength.
        
        Args:
            wavelength: Light wavelength in nm
            
        Returns:
            Absorption coefficient [1/length]
        """
        if self.mu_a_func is not None:
            return self.mu_a_func(wavelength)
        return self.mu_a_fixed
    
    def get_scattering(self, wavelength: float) -> float:
        """
        Get scattering coefficient for a specific wavelength.
        
        Args:
            wavelength: Light wavelength in nm
            
        Returns:
            Scattering coefficient [1/length]
        """
        if self.mu_s_func is not None:
            return self.mu_s_func(wavelength)
        return self.mu_s_fixed
    
    def get_anisotropy(self, wavelength: float) -> float:
        """
        Get anisotropy factor for a specific wavelength.
        
        Args:
            wavelength: Light wavelength in nm
            
        Returns:
            Anisotropy factor g [-1 to 1]
        """
        if self.g_func is not None:
            return self.g_func(wavelength)
        return self.g_fixed
    
    def get_total_attenuation(self, wavelength: float) -> float:
        """
        Calculate total attenuation coefficient (mu_t = mu_a + mu_s).
        
        Args:
            wavelength: Light wavelength in nm
            
        Returns:
            Total attenuation coefficient [1/length]
        """
        return self.get_absorption(wavelength) + self.get_scattering(wavelength)
    
    def get_mean_free_path(self, wavelength: float) -> float:
        """
        Calculate mean free path (average distance between interactions).
        
        Args:
            wavelength: Light wavelength in nm
            
        Returns:
            Mean free path [length units]
        """
        mu_t = self.get_total_attenuation(wavelength)
        if mu_t < 1e-10:  # Avoid division by zero
            return float('inf')
        return 1.0 / mu_t
    
    def sample_step_size(self, wavelength: float) -> float:
        """
        Sample a step size from exponential distribution based on the total attenuation.
        
        Args:
            wavelength: Light wavelength in nm
            
        Returns:
            Step size [length units]
        """
        mu_t = self.get_total_attenuation(wavelength)
        if mu_t < 1e-10:  # Avoid division by zero
            return float('inf')
        return -np.log(np.random.random()) / mu_t
    
    def calculate_fresnel(self, cos_theta_i: float, n1: float, n2: float) -> Tuple[float, float]:
        """
        Calculate Fresnel reflectance and transmittance coefficients.
        
        Args:
            cos_theta_i: Cosine of incident angle
            n1: Refractive index of first medium
            n2: Refractive index of second medium
            
        Returns:
            Tuple of (reflection_coefficient, transmission_coefficient)
        """
        if abs(cos_theta_i) > 1.0:
            cos_theta_i = np.clip(cos_theta_i, -1.0, 1.0)  # Ensure valid range
        
        # Calculate sin^2(theta_t) using Snell's law
        sin2_theta_i = 1.0 - cos_theta_i**2
        sin2_theta_t = (n1 / n2)**2 * sin2_theta_i
        
        # Check for total internal reflection
        if sin2_theta_t >= 1.0:
            return 1.0, 0.0  # Total reflection
        
        cos_theta_t = np.sqrt(1.0 - sin2_theta_t)
        
        # Calculate Fresnel coefficients for unpolarized light (average of s and p polarizations)
        # For s-polarization (perpendicular)
        r_s = ((n1 * cos_theta_i - n2 * cos_theta_t) / 
               (n1 * cos_theta_i + n2 * cos_theta_t))**2
        
        # For p-polarization (parallel)
        r_p = ((n1 * cos_theta_t - n2 * cos_theta_i) / 
               (n1 * cos_theta_t + n2 * cos_theta_i))**2
        
        # Average reflectance for unpolarized light
        r = (r_s + r_p) / 2.0
        
        # Transmittance from conservation of energy
        t = 1.0 - r
        
        return r, t
    
    def __str__(self) -> str:
        """String representation of the medium."""
        name_str = f" ({self.name})" if self.name else ""
        model_str = f", model={self.dispersion_model}" if self.dispersion_model else ""
        
        if self.n_func is not None:
            return f"Medium{name_str}: wavelength-dependent{model_str}"
        else:
            return (f"Medium{name_str}: n={self.n_fixed:.4f}, "
                   f"μₐ={self.mu_a_fixed:.4f}, μₛ={self.mu_s_fixed:.4f}, "
                   f"g={self.g_fixed:.4f}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert medium properties to dictionary for serialization."""
        # Can't easily serialize functions, so just store the model info
        return {
            'name': self.name,
            'n_fixed': self.n_fixed,
            'mu_a_fixed': self.mu_a_fixed,
            'mu_s_fixed': self.mu_s_fixed,
            'g_fixed': self.g_fixed,
            'dispersion_model': self.dispersion_model,
            'has_n_func': self.n_func is not None,
            'has_mu_a_func': self.mu_a_func is not None,
            'has_mu_s_func': self.mu_s_func is not None,
            'has_g_func': self.g_func is not None
        }


# Define common predefined media with realistic dispersion models

def air() -> Medium:
    """
    Create a medium representing air with wavelength-dependent properties.
    
    Returns:
        Medium instance for air
    """
    def air_refractive_index(wavelength):
        """Simplified model for air refractive index."""
        # Air has very little dispersion, but we include a small wavelength dependence
        # Values are approximate for standard conditions
        return 1.0003 - 1.0e-8 * (wavelength - 550)**2
    
    return Medium(
        name="Air",
        n_func=air_refractive_index,
        mu_a_fixed=0.0,  # Negligible absorption
        mu_s_fixed=0.0,  # Negligible scattering
        g_fixed=0.0,
        dispersion_model="simplified"
    )

def water() -> Medium:
    """
    Create a medium representing water with wavelength-dependent properties.
    
    Returns:
        Medium instance for water
    """
    def water_refractive_index(wavelength):
        """Cauchy's equation for water's refractive index."""
        # Convert wavelength from nm to μm for Cauchy's formula
        wl_um = wavelength / 1000.0
        
        # Cauchy's formula coefficients for water
        A = 1.324
        B = 0.00325
        C = 0.00031
        
        # Calculate refractive index
        n = A + B / (wl_um**2) + C / (wl_um**4)
        
        return n
    
    def water_absorption(wavelength):
        """Approximate absorption coefficient for water."""
        # This is a simplified model - real water absorption is complex
        # and depends on many factors including temperature
        if 380 <= wavelength <= 600:
            # Lower absorption in blue-green region
            return 0.01 + 0.01 * ((wavelength - 480) / 100)**2
        else:
            # Higher absorption in red and UV regions
            return 0.05 + 0.03 * ((wavelength - 600) / 100)**2
    
    return Medium(
        name="Water",
        n_func=water_refractive_index,
        mu_a_func=water_absorption,
        mu_s_fixed=0.01,  # Low scattering for clean water
        g_fixed=0.9,      # Forward scattering
        dispersion_model="cauchy"
    )

def glass_BK7() -> Medium:
    """
    Create a medium representing BK7 optical glass with wavelength-dependent properties.
    
    Returns:
        Medium instance for BK7 glass
    """
    def bk7_refractive_index(wavelength):
        """Sellmeier equation for BK7 glass."""
        # Convert wavelength from nm to μm for Sellmeier formula
        wl_um = wavelength / 1000.0
        
        # Sellmeier coefficients for BK7
        B1, B2, B3 = 1.03961212, 0.231792344, 1.01046945
        C1, C2, C3 = 0.00600069867, 0.0200179144, 103.560653
        
        # Calculate refractive index
        n_squared = 1.0 + (B1 * wl_um**2) / (wl_um**2 - C1) + \
                         (B2 * wl_um**2) / (wl_um**2 - C2) + \
                         (B3 * wl_um**2) / (wl_um**2 - C3)
        
        return np.sqrt(n_squared)
    
    return Medium(
        name="BK7 Glass",
        n_func=bk7_refractive_index,
        mu_a_fixed=0.001,  # Very low absorption for optical glass
        mu_s_fixed=0.0,    # Negligible scattering for clear glass
        g_fixed=0.0,
        dispersion_model="sellmeier"
    )

def diamond() -> Medium:
    """
    Create a medium representing diamond with wavelength-dependent properties.
    
    Returns:
        Medium instance for diamond
    """
    def diamond_refractive_index(wavelength):
        """Sellmeier equation for diamond."""
        # Convert wavelength from nm to μm for Sellmeier formula
        wl_um = wavelength / 1000.0
        
        # Sellmeier coefficients for diamond
        B1, B2, B3 = 4.3356, 0.3306, 0.0
        C1, C2, C3 = 0.1060, 0.1750, 0.0
        
        # Calculate refractive index
        n_squared = 1.0 + (B1 * wl_um**2) / (wl_um**2 - C1) + \
                         (B2 * wl_um**2) / (wl_um**2 - C2) + \
                         (B3 * wl_um**2) / (wl_um**2 - C3)
        
        return np.sqrt(n_squared)
    
    return Medium(
        name="Diamond",
        n_func=diamond_refractive_index,
        mu_a_fixed=0.001,  # Very low absorption for pure diamond
        mu_s_fixed=0.0,    # Negligible scattering
        g_fixed=0.0,
        dispersion_model="sellmeier"
    )

def create_custom_medium(name: str, 
                         refr_index_model: str = "cauchy", 
                         refr_index_params: List[float] = None) -> Medium:
    """
    Create a custom medium with specified dispersion model.
    
    Args:
        name: Name for the medium
        refr_index_model: Model for dispersion ("cauchy", "sellmeier", or "constant")
        refr_index_params: Parameters for the dispersion model
        
    Returns:
        Medium instance with specified properties
    """
    if refr_index_model.lower() == "cauchy":
        # Default Cauchy parameters if none provided
        if refr_index_params is None or len(refr_index_params) < 3:
            refr_index_params = [1.5, 0.003, 0.0]
            
        A, B, C = refr_index_params[:3]
        
        def cauchy_index(wavelength):
            wl_um = wavelength / 1000.0
            return A + B / (wl_um**2) + C / (wl_um**4)
        
        return Medium(
            name=name,
            n_func=cauchy_index,
            dispersion_model="cauchy"
        )
        
    elif refr_index_model.lower() == "sellmeier":
        # Default Sellmeier parameters if none provided
        if refr_index_params is None or len(refr_index_params) < 6:
            refr_index_params = [1.0, 0.5, 0.0, 0.1, 0.1, 0.0]
            
        B1, B2, B3 = refr_index_params[:3]
        C1, C2, C3 = refr_index_params[3:6]
        
        def sellmeier_index(wavelength):
            wl_um = wavelength / 1000.0
            n_squared = 1.0 + (B1 * wl_um**2) / (wl_um**2 - C1) + \
                             (B2 * wl_um**2) / (wl_um**2 - C2) + \
                             (B3 * wl_um**2) / (wl_um**2 - C3)
            return np.sqrt(n_squared)
        
        return Medium(
            name=name,
            n_func=sellmeier_index,
            dispersion_model="sellmeier"
        )
        
    else:  # constant
        n_value = 1.5
        if refr_index_params is not None and len(refr_index_params) > 0:
            n_value = refr_index_params[0]
            
        return Medium(
            name=name,
            n_fixed=n_value,
            dispersion_model="constant"
        )