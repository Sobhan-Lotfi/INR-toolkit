"""
INR model implementations.

Available models:
- SIREN: Sinusoidal activation networks (best for smooth signals)
- FourierFeaturesMLP: Fourier feature encoding + MLP (versatile, good default)
- ReLUMLP: Baseline ReLU network (included for comparison)
"""

from .base import INRModel
from .siren import SIREN
from .fourier import FourierFeaturesMLP
from .mlp import ReLUMLP

__all__ = [
    'INRModel',
    'SIREN',
    'FourierFeaturesMLP',
    'ReLUMLP',
]
