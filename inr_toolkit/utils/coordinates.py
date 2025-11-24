"""
Coordinate generation utilities.

Functions for creating coordinate grids for different tasks.
"""

import torch
import numpy as np
from typing import Tuple


def get_coordinates(
    shape: Tuple[int, ...],
    normalize: bool = True,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate coordinate grid for given shape.
    
    For 2D images, creates a grid of (x, y) coordinates.
    Coordinates are normalized to [-1, 1] by default.
    
    Args:
        shape: Shape of the coordinate grid (e.g., (H, W) for images)
        normalize: If True, normalize coordinates to [-1, 1]
        device: Device to create tensor on
    
    Returns:
        coords: Coordinate tensor of shape (H*W*..., len(shape))
    
    Example:
        # For a 256x256 image
        coords = get_coordinates((256, 256))
        # coords.shape = (65536, 2)
        # Each row is an (x, y) coordinate in [-1, 1]
    """
    # Create coordinate ranges for each dimension
    ranges = [torch.arange(s, dtype=torch.float32, device=device) for s in shape]
    
    # Normalize to [-1, 1] if requested
    if normalize:
        ranges = [
            (r / (s - 1)) * 2 - 1  # Map [0, s-1] to [-1, 1]
            for r, s in zip(ranges, shape)
        ]
    
    # Create meshgrid and flatten
    grids = torch.meshgrid(*ranges, indexing='ij')
    coords = torch.stack(grids, dim=-1).reshape(-1, len(shape))
    
    return coords


def get_image_coordinates(
    height: int,
    width: int,
    normalize: bool = True,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Convenience function for 2D image coordinates.
    
    Args:
        height: Image height
        width: Image width
        normalize: If True, normalize coordinates to [-1, 1]
        device: Device to create tensor on
    
    Returns:
        coords: Coordinate tensor of shape (H*W, 2)
    """
    return get_coordinates((height, width), normalize=normalize, device=device)


def pixels_to_coordinates(
    pixel_indices: torch.Tensor,
    shape: Tuple[int, int],
    normalize: bool = True
) -> torch.Tensor:
    """
    Convert pixel indices to coordinates.
    
    Args:
        pixel_indices: Integer indices, shape (N, 2) for 2D
        shape: Image shape (H, W)
        normalize: If True, normalize to [-1, 1]
    
    Returns:
        coords: Float coordinates, shape (N, 2)
    """
    coords = pixel_indices.float()
    
    if normalize:
        coords[:, 0] = (coords[:, 0] / (shape[0] - 1)) * 2 - 1
        coords[:, 1] = (coords[:, 1] / (shape[1] - 1)) * 2 - 1
    
    return coords


def random_sample_coordinates(
    shape: Tuple[int, ...],
    num_samples: int,
    normalize: bool = True,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample coordinates from a grid.
    
    Useful for mini-batch training on large images.
    
    Args:
        shape: Shape of the grid
        num_samples: Number of coordinates to sample
        normalize: If True, normalize to [-1, 1]
        device: Device to create tensor on
    
    Returns:
        coords: Sampled coordinates, shape (num_samples, len(shape))
        indices: Integer indices of sampled positions
    
    Example:
        # Sample 1000 random pixels from a 512x512 image
        coords, indices = random_sample_coordinates((512, 512), 1000)
    """
    # Generate all coordinates
    all_coords = get_coordinates(shape, normalize=normalize, device=device)
    
    # Sample random indices
    total_size = np.prod(shape)
    indices = torch.randperm(total_size, device=device)[:num_samples]
    
    return all_coords[indices], indices
