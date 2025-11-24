"""
SIREN: Sinusoidal Representation Networks

Paper: Implicit Neural Representations with Periodic Activation Functions
Authors: Sitzmann et al. (2020)
Paper: https://arxiv.org/abs/2006.09661

Key idea: Use sine activations instead of ReLU to represent high-frequency details.
"""

import torch
import torch.nn as nn
import numpy as np
from .base import INRModel


class SineLayer(nn.Module):
    """Single layer with sine activation and special initialization."""
    
    def __init__(self, in_features: int, out_features: int, is_first: bool = False, omega_0: float = 30.0):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            is_first: Whether this is the first layer (uses different initialization)
            omega_0: Frequency parameter for sine activation
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
    
    @property
    def in_features(self):
        return self.linear.in_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(INRModel):
    """
    SIREN: Sinusoidal Representation Network
    
    Uses sine activations to represent high-frequency functions.
    Best for smooth signals and when you need accurate derivatives.
    
    Example:
        model = SIREN(in_dim=2, out_dim=3, hidden_dim=256, num_layers=4)
        coords = torch.rand(1000, 2)  # Random 2D coordinates
        colors = model(coords)  # Get RGB values
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        omega_0: float = 30.0,
    ):
        """
        Args:
            in_dim: Input coordinate dimension
            out_dim: Output dimension
            hidden_dim: Hidden layer width
            num_layers: Number of hidden layers
            omega_0: Frequency of sine activation (higher = more detail)
        """
        super().__init__(in_dim, out_dim)
        
        # First layer
        layers = [SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0)]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        
        self.net = nn.Sequential(*layers)
        
        # Final layer (linear, no activation)
        self.final_layer = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            self.final_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0,
                np.sqrt(6 / hidden_dim) / omega_0
            )
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Map coordinates to outputs.
        
        Args:
            coords: Shape (batch_size, in_dim)
        
        Returns:
            outputs: Shape (batch_size, out_dim)
        """
        x = self.net(coords)
        return self.final_layer(x)
