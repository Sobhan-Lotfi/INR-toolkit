"""
INR Toolkit - Educational toolkit for Implicit Neural Representations

Main modules:
- models: INR model implementations (SIREN, Fourier Features, ReLU MLP)
- training: Training utilities
- utils: Coordinate generation, metrics, visualization
"""

__version__ = '0.1.0'

from . import models
from . import training
from . import utils

__all__ = ['models', 'training', 'utils']
