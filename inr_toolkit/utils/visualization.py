"""
Visualization utilities for INR models.

Functions for plotting images, comparisons, and training curves.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from PIL import Image


def load_image(path: str, normalize: bool = True) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        path: Path to image file
        normalize: If True, normalize pixel values to [0, 1]
    
    Returns:
        image: Image array of shape (H, W, 3)
    """
    img = Image.open(path).convert('RGB')
    img_array = np.array(img)
    
    if normalize:
        img_array = img_array.astype(np.float32) / 255.0
    
    return img_array


def save_image(image: np.ndarray, path: str):
    """
    Save an image to file.
    
    Args:
        image: Image array of shape (H, W, 3), values in [0, 1]
        path: Output path
    """
    # Convert to uint8
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def plot_image(image: np.ndarray, title: str = None, ax: plt.Axes = None):
    """
    Plot an image.
    
    Args:
        image: Image array of shape (H, W, 3) or (H, W)
        title: Optional title
        ax: Optional matplotlib axis (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.imshow(np.clip(image, 0, 1))
    ax.axis('off')
    if title:
        ax.set_title(title)


def plot_comparison(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = None
):
    """
    Plot multiple images side by side.
    
    Args:
        images: List of image arrays
        titles: List of titles (one per image)
        figsize: Figure size (auto-calculated if None)
    
    Example:
        plot_comparison(
            [ground_truth, reconstruction],
            ['Ground Truth', 'Reconstruction']
        )
    """
    n = len(images)
    if figsize is None:
        figsize = (6 * n, 6)
    
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for img, title, ax in zip(images, titles, axes):
        plot_image(img, title, ax)
    
    plt.tight_layout()
    return fig


def plot_training_curve(
    losses: List[float],
    val_losses: Optional[List[float]] = None,
    log_scale: bool = False,
    title: str = 'Training Curve'
):
    """
    Plot training loss curve.
    
    Args:
        losses: List of training losses
        val_losses: Optional list of validation losses
        log_scale: If True, use log scale for y-axis
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(losses, label='Train Loss', linewidth=2)
    if val_losses is not None:
        ax.plot(val_losses, label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    return fig


def render_model(
    model: torch.nn.Module,
    shape: Tuple[int, int],
    device: str = 'cpu'
) -> np.ndarray:
    """
    Render an INR model to an image.
    
    Args:
        model: INR model
        shape: Output image shape (H, W)
        device: Device to run on
    
    Returns:
        image: Rendered image of shape (H, W, out_dim)
    
    Example:
        # Render trained model at high resolution
        high_res = render_model(model, (1024, 1024))
        save_image(high_res, 'output.png')
    """
    from .coordinates import get_image_coordinates
    
    model.eval()
    with torch.no_grad():
        coords = get_image_coordinates(shape[0], shape[1], device=device)
        output = model(coords)
        image = output.reshape(shape[0], shape[1], -1).cpu().numpy()
    
    return image


def create_comparison_grid(
    model_outputs: List[np.ndarray],
    model_names: List[str],
    ground_truth: Optional[np.ndarray] = None,
    metrics: Optional[List[dict]] = None
) -> plt.Figure:
    """
    Create a grid comparing multiple model outputs.
    
    Args:
        model_outputs: List of output images
        model_names: List of model names
        ground_truth: Optional ground truth image
        metrics: Optional list of metric dictionaries
    
    Returns:
        fig: Matplotlib figure
    """
    n_models = len(model_outputs)
    n_cols = n_models + (1 if ground_truth is not None else 0)
    
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]
    
    idx = 0
    
    # Plot ground truth if provided
    if ground_truth is not None:
        plot_image(ground_truth, 'Ground Truth', axes[idx])
        idx += 1
    
    # Plot model outputs
    for i, (output, name) in enumerate(zip(model_outputs, model_names)):
        title = name
        if metrics is not None and i < len(metrics):
            psnr_val = metrics[i].get('psnr', 0)
            title += f"\nPSNR: {psnr_val:.2f} dB"
        plot_image(output, title, axes[idx])
        idx += 1
    
    plt.tight_layout()
    return fig
