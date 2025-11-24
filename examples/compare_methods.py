"""
Compare different INR methods side-by-side on a single image.

Usage:
    python compare_methods.py --image path/to/image.jpg
    python compare_methods.py --image photo.jpg --epochs 1500 --quick
"""

import sys
sys.path.append('..')

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from inr_toolkit.models import SIREN, FourierFeaturesMLP, ReLUMLP
from inr_toolkit.training import Trainer
from inr_toolkit.utils import (
    load_image,
    get_image_coordinates,
    psnr,
    create_comparison_grid
)


def train_and_evaluate(model_class, model_name, config, coords, colors, image, device):
    """Train a model and return results."""
    print(f"\nTraining {model_name}...")
    
    # Create model
    if model_name == 'Fourier Features':
        model = model_class(
            in_dim=2, out_dim=3,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            fourier_scale=config.get('fourier_scale', 10.0)
        )
    else:
        model = model_class(
            in_dim=2, out_dim=3,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers']
        )
    
    # Train
    trainer = Trainer(model, lr=config['lr'], device=device)
    
    start_time = time.time()
    trainer.fit(coords, colors, epochs=config['epochs'], log_every=config['epochs'] // 4)
    train_time = time.time() - start_time
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        output = model(coords.to(device)).cpu().numpy()
        reconstruction = output.reshape(image.shape)
    
    psnr_val = psnr(torch.from_numpy(reconstruction), torch.from_numpy(image))
    
    print(f"  PSNR: {psnr_val:.2f} dB, Time: {train_time:.1f}s, Params: {model.count_parameters():,}")
    
    return {
        'reconstruction': reconstruction,
        'psnr': psnr_val,
        'time': train_time,
        'params': model.count_parameters()
    }


def main():
    parser = argparse.ArgumentParser(description='Compare INR methods')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: smaller network, fewer epochs')
    parser.add_argument('--output', type=str, default='comparison.png',
                       help='Output path for comparison image')
    
    args = parser.parse_args()
    
    # Quick mode settings
    if args.quick:
        args.hidden_dim = 128
        args.num_layers = 3
        args.epochs = 500
        print("🚀 Quick mode: Using smaller network and fewer epochs")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load image
    print(f"\nLoading image: {args.image}")
    image = load_image(args.image)
    h, w = image.shape[:2]
    print(f"Image shape: {h}×{w}")
    
    # Prepare data
    coords = get_image_coordinates(h, w)
    colors = torch.from_numpy(image.reshape(-1, 3))
    
    # Configuration
    config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'lr': args.lr,
        'epochs': args.epochs,
        'fourier_scale': 10.0,
    }
    
    print(f"\nConfiguration:")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Epochs: {config['epochs']}")
    
    # Models to compare
    models = [
        (ReLUMLP, 'ReLU MLP'),
        (FourierFeaturesMLP, 'Fourier Features'),
        (SIREN, 'SIREN'),
    ]
    
    # Train and evaluate all models
    print("\n" + "="*60)
    print("TRAINING ALL MODELS")
    print("="*60)
    
    results = {}
    for model_class, model_name in models:
        results[model_name] = train_and_evaluate(
            model_class, model_name, config,
            coords, colors, image, device
        )
    
    # Create comparison visualization
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n{'Model':<20} {'PSNR (dB)':<12} {'Time (s)':<10} {'Params':<10}")
    print("-"*60)
    for name, res in results.items():
        print(f"{name:<20} {res['psnr']:<12.2f} {res['time']:<10.1f} {res['params']:<10,}")
    print("="*60)
    
    # Determine winner
    best_model = max(results.items(), key=lambda x: x[1]['psnr'])[0]
    print(f"\n🏆 Best PSNR: {best_model}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Ground truth
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Model outputs
    for ax, (name, res) in zip(axes.flat[1:], results.items()):
        ax.imshow(res['reconstruction'].clip(0, 1))
        title = f"{name}\nPSNR: {res['psnr']:.2f} dB"
        if name == best_model:
            title = "🏆 " + title
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved comparison to: {args.output}")
    plt.show()
    
    # Additional detailed comparison
    print("\n💡 Observations:")
    relu_psnr = results['ReLU MLP']['psnr']
    fourier_psnr = results['Fourier Features']['psnr']
    siren_psnr = results['SIREN']['psnr']
    
    print(f"  - Fourier Features improved over ReLU MLP by {fourier_psnr - relu_psnr:.2f} dB")
    print(f"  - SIREN improved over ReLU MLP by {siren_psnr - relu_psnr:.2f} dB")
    
    if fourier_psnr > siren_psnr:
        print(f"  - Fourier Features outperformed SIREN by {fourier_psnr - siren_psnr:.2f} dB")
    else:
        print(f"  - SIREN outperformed Fourier Features by {siren_psnr - fourier_psnr:.2f} dB")
    
    print("\n✨ Done! Check the visualization above.")


if __name__ == '__main__':
    main()
