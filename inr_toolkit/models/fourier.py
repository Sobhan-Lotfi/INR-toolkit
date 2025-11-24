"""
Fourier Feature Networks

Paper: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
Authors: Tancik et al. (2020)
Paper: https://arxiv.org/abs/2006.10739

Key idea: Map input coordinates through random Fourier features before feeding to MLP.
This helps the network learn high-frequency patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from .base import INRModel


class FourierFeaturesMLP(INRModel):
    """
    MLP with Fourier feature encoding.
    
    Maps input x → [sin(Bx), cos(Bx)] → MLP → output
    where B is a random Gaussian matrix.
    
    This is often the best default choice:
    - Works well for most signals
    - Easy to tune (just adjust fourier_scale)
    - More stable than SIREN
    
    Example:
        model = FourierFeaturesMLP(
            in_dim=2, out_dim=3,
            hidden_dim=256, num_layers=4,
            fourier_scale=10.0  # Adjust for your data
        )
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        fourier_dim: int = 256,
        fourier_scale: float = 10.0,
    ):
        """
        Args:
            in_dim: Input coordinate dimension
            out_dim: Output dimension
            hidden_dim: Hidden layer width
            num_layers: Number of hidden layers
            fourier_dim: Dimension of Fourier feature mapping
            fourier_scale: Scale of random Fourier features (controls frequency range)
                          Higher = capture finer details
                          Lower = smoother outputs
        """
        super().__init__(in_dim, out_dim)
        
        # Random Fourier feature matrix (fixed, not trainable)
        self.register_buffer(
            'B',
            torch.randn(in_dim, fourier_dim) * fourier_scale
        )
        
        # MLP layers
        # Input dimension is 2*fourier_dim because we use both sin and cos
        layers = []
        layers.append(nn.Linear(2 * fourier_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
    
    def fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature mapping.
        
        Args:
            coords: Shape (batch_size, in_dim)
        
        Returns:
            features: Shape (batch_size, 2*fourier_dim)
                     Concatenation of [sin(2π*B*coords), cos(2π*B*coords)]
        """
        # Project coordinates: coords @ B
        projected = coords @ self.B  # (batch_size, fourier_dim)
        
        # Apply sin and cos, then concatenate
        # Multiply by 2π as per the paper
        projected = 2 * np.pi * projected
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Map coordinates to outputs.
        
        Args:
            coords: Shape (batch_size, in_dim)
        
        Returns:
            outputs: Shape (batch_size, out_dim)
        """
        features = self.fourier_features(coords)
        return self.net(features)
