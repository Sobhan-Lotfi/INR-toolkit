"""
Simple example: Fit an INR to a single image.

Usage:
    python fit_image.py --image path/to/image.jpg
    python fit_image.py --image photo.jpg --model siren --epochs 2000
"""

import sys
sys.path.append('..')

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from inr_toolkit.models import SIREN, FourierFeaturesMLP, ReLUMLP
from inr_toolkit.training import Trainer
from inr_toolkit.utils import (
    load_image,
    get_image_coordinates,
    psnr,
    save_image,
    render_model
)


def main():
    parser = argparse.ArgumentParser(description='Fit INR to an image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='fourier',
                       choices=['relu', 'fourier', 'siren'],
                       help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for reconstruction')
    parser.add_argument('--render_scale', type=int, default=1,
                       help='Render at Nx resolution (1=same, 2=double, etc.)')
    
    args = parser.parse_args()
    
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
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    if args.model == 'relu':
        model = ReLUMLP(
            in_dim=2, out_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        )
    elif args.model == 'fourier':
        model = FourierFeaturesMLP(
            in_dim=2, out_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            fourier_scale=10.0
        )
    else:  # siren
        model = SIREN(
            in_dim=2, out_dim=3,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers
        )
    
    print(f"Parameters: {model.count_parameters():,}")
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    trainer = Trainer(model, lr=args.lr, device=device)
    trainer.fit(coords, colors, epochs=args.epochs, log_every=args.epochs // 10)
    
    # Evaluate at original resolution
    print("\nEvaluating...")
    model.eval()
    with torch.no_grad():
        output = model(coords.to(device)).cpu().numpy()
        reconstruction = output.reshape(h, w, 3)
    
    psnr_val = psnr(torch.from_numpy(reconstruction), torch.from_numpy(image))
    print(f"\nReconstruction PSNR: {psnr_val:.2f} dB")
    
    # Render at higher resolution if requested
    if args.render_scale > 1:
        print(f"\nRendering at {args.render_scale}x resolution...")
        new_h, new_w = h * args.render_scale, w * args.render_scale
        reconstruction = render_model(model, (new_h, new_w), device=device)
        print(f"New resolution: {new_h}×{new_w}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image\n{h}×{w}')
    axes[0].axis('off')
    
    axes[1].imshow(reconstruction.clip(0, 1))
    title = f'{args.model.upper()} Reconstruction\n'
    if args.render_scale > 1:
        title += f'{reconstruction.shape[0]}×{reconstruction.shape[1]} ({args.render_scale}x)'
    else:
        title += f'PSNR: {psnr_val:.2f} dB'
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: reconstruction.png")
    plt.show()
    
    # Save output if requested
    if args.output:
        save_image(reconstruction, args.output)
        print(f"Saved reconstruction to: {args.output}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
