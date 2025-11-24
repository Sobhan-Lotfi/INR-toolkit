"""
Baseline ReLU MLP

This is what happens if you use a standard MLP without any special tricks.
Spoiler: It doesn't work well for INRs!

This is included to demonstrate why SIREN and Fourier Features are needed.
"""

import torch
import torch.nn as nn
from .base import INRModel


class ReLUMLP(INRModel):
    """
    Standard MLP with ReLU activations.
    
    This is a baseline to show why naive approaches fail for INRs.
    Problem: ReLU networks have "spectral bias" - they learn low frequencies first
    and struggle with high-frequency details.
    
    Use this to appreciate why SIREN and Fourier Features exist!
    
    Example:
        model = ReLUMLP(in_dim=2, out_dim=3, hidden_dim=256, num_layers=4)
        # Watch it fail to capture fine details...
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ):
        """
        Args:
            in_dim: Input coordinate dimension
            out_dim: Output dimension
            hidden_dim: Hidden layer width
            num_layers: Number of hidden layers
        """
        super().__init__(in_dim, out_dim)
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Map coordinates to outputs.
        
        Args:
            coords: Shape (batch_size, in_dim)
        
        Returns:
            outputs: Shape (batch_size, out_dim)
        """
        return self.net(coords)
