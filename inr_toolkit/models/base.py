"""
Base class for Implicit Neural Representation models.

All INR models inherit from this base class and implement the forward pass.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class INRModel(nn.Module, ABC):
    """
    Abstract base class for all INR models.
    
    An INR model maps continuous coordinates to output values:
        f: R^in_dim → R^out_dim
    
    Example:
        For 2D image fitting: in_dim=2 (x,y), out_dim=3 (r,g,b)
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        """
        Args:
            in_dim: Input coordinate dimension (e.g., 2 for images)
            out_dim: Output dimension (e.g., 3 for RGB images)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    @abstractmethod
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Map coordinates to output values.
        
        Args:
            coords: Input coordinates of shape (batch_size, in_dim)
        
        Returns:
            outputs: Output values of shape (batch_size, out_dim)
        """
        pass
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
