"""
Evaluation metrics for INR models.

Common metrics for comparing reconstruction quality.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_sklearn
from skimage.metrics import peak_signal_noise_ratio as psnr_sklearn


def psnr(prediction: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Higher is better. Typical values for good reconstructions: 30-40 dB.
    
    Args:
        prediction: Predicted values
        target: Ground truth values
        max_val: Maximum possible value (1.0 for normalized images)
    
    Returns:
        psnr: PSNR in decibels (dB)
    
    Example:
        psnr_value = psnr(model(coords), ground_truth)
        print(f"PSNR: {psnr_value:.2f} dB")
    """
    mse = torch.mean((prediction - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


def mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.
    
    Lower is better.
    
    Args:
        prediction: Predicted values
        target: Ground truth values
    
    Returns:
        mse: Mean squared error
    """
    return torch.mean((prediction - target) ** 2).item()


def mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute Mean Absolute Error.
    
    Lower is better.
    
    Args:
        prediction: Predicted values
        target: Ground truth values
    
    Returns:
        mae: Mean absolute error
    """
    return torch.mean(torch.abs(prediction - target)).item()


def ssim(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Higher is better. Range: [-1, 1], where 1 means perfect similarity.
    Typical values for good reconstructions: 0.9-0.99.
    
    Args:
        prediction: Predicted image (H, W) or (H, W, C)
        target: Ground truth image
        data_range: Range of the data (1.0 for normalized images)
    
    Returns:
        ssim: Structural similarity index
    
    Example:
        pred_img = model(coords).reshape(H, W, 3).cpu().numpy()
        ssim_value = ssim(pred_img, ground_truth_img)
        print(f"SSIM: {ssim_value:.4f}")
    """
    # Ensure we're working with numpy arrays
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Handle multi-channel images
    channel_axis = -1 if prediction.ndim == 3 else None
    
    return ssim_sklearn(
        target,
        prediction,
        data_range=data_range,
        channel_axis=channel_axis
    )


def compute_all_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    image_shape: tuple = None
) -> dict:
    """
    Compute all common metrics.
    
    Args:
        prediction: Predicted values
        target: Ground truth values
        image_shape: If provided, also compute SSIM (requires reshaping to image)
    
    Returns:
        metrics: Dictionary of metric names to values
    
    Example:
        metrics = compute_all_metrics(
            model(coords),
            ground_truth,
            image_shape=(256, 256, 3)
        )
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
    """
    metrics = {
        'mse': mse(prediction, target),
        'mae': mae(prediction, target),
        'psnr': psnr(prediction, target),
    }
    
    # Compute SSIM if image shape is provided
    if image_shape is not None:
        pred_img = prediction.reshape(image_shape).cpu().numpy()
        target_img = target.reshape(image_shape).cpu().numpy()
        metrics['ssim'] = ssim(pred_img, target_img)
    
    return metrics
