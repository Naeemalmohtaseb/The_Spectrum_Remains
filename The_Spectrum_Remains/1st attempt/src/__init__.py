"""
Monte Carlo Ray Tracing for Light Transport.

This package implements a Monte Carlo simulation for tracking photons
through scattering and absorbing media.
"""

__version__ = '0.1.0'

from .photon import Photon
from .medium import Medium
from .geometry import Boundary, Plane, Sphere, Box, GeometryManager
from .simulation import Simulation