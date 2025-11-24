"""
Benchmark script for image fitting with INRs.

Compares ReLU MLP, Fourier Features, and SIREN on image reconstruction.
"""

import sys
sys.path.append('../..')

import torch
import numpy as np
import argparse
from pathlib import Path
import time
import json

from inr_toolkit.models import ReLUMLP, FourierFeaturesMLP, SIREN
from inr_toolkit.training import Trainer
from inr_toolkit.utils import (
    load_image, 
    get_image_coordinates, 
    compute_all_metrics,
    save_image,
    create_comparison_grid
)
import matplotlib.pyplot as plt


def benchmark_model(model_class, model_name, config, coords, colors, image_shape, device):
    """Run benchmark for a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
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
    
    print(f"Parameters: {model.count_parameters():,}")
    
    # Train
    trainer = Trainer(model, lr=config['lr'], device=device)
    
    start_time = time.time()
    trainer.fit(coords, colors, epochs=config['epochs'], log_every=config['epochs'] // 4)
    train_time = time.time() - start_time
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        output = model(coords.to(device)).cpu()
    
    # Compute metrics
    metrics = compute_all_metrics(output, colors, image_shape=image_shape)
    metrics['train_time'] = train_time
    metrics['params'] = model.count_parameters()
    
    # Reshape output for visualization
    output_image = output.numpy().reshape(image_shape)
    
    print(f"\nResults:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  Training time: {train_time:.1f}s")
    
    return output_image, metrics


def main():
    parser = argparse.ArgumentParser(description='Benchmark INR methods on image fitting')
    parser.add_argument('--data_dir', type=str, default='../data/kodak',
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Resize images to this size')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Training epochs')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of hidden layers')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'lr': 1e-4,
        'epochs': args.epochs,
        'fourier_scale': 10.0,
    }
    
    # Models to benchmark
    models_to_test = [
        (ReLUMLP, 'ReLU MLP'),
        (FourierFeaturesMLP, 'Fourier Features'),
        (SIREN, 'SIREN'),
    ]
    
    # Load test images
    data_dir = Path(args.data_dir)
    image_files = list(data_dir.glob('*.png')) + list(data_dir.glob('*.jpg'))
    
    if len(image_files) == 0:
        print(f"No images found in {data_dir}")
        print("Creating a test image...")
        # Create a test image
        height, width = args.image_size, args.image_size
        x = np.linspace(-2, 2, width)
        y = np.linspace(-2, 2, height)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        test_image = np.stack([
            0.5 + 0.5 * np.sin(5 * R) / (1 + R),
            0.5 + 0.3 * np.cos(3 * X) * np.cos(3 * Y),
            0.5 + 0.4 * np.exp(-R**2 / 2),
        ], axis=-1)
        image_files = ['synthetic']
        images_to_test = [test_image]
    else:
        print(f"Found {len(image_files)} images")
        # Load first image for demo
        images_to_test = [load_image(str(image_files[0]))]
        # Resize if needed
        from PIL import Image as PILImage
        if images_to_test[0].shape[0] != args.image_size:
            img = PILImage.fromarray((images_to_test[0] * 255).astype(np.uint8))
            img = img.resize((args.image_size, args.image_size))
            images_to_test[0] = np.array(img).astype(np.float32) / 255.0
    
    # Benchmark each image
    all_results = []
    
    for img_idx, (image, img_name) in enumerate(zip(images_to_test, image_files)):
        print(f"\n{'#'*60}")
        print(f"# Benchmarking image: {img_name}")
        print(f"# Shape: {image.shape}")
        print(f"{'#'*60}")
        
        # Prepare data
        h, w = image.shape[:2]
        coords = get_image_coordinates(h, w)
        colors = torch.from_numpy(image.reshape(-1, 3).astype(np.float32))
        
        # Benchmark all models
        results = {}
        outputs = []
        model_names = []
        metrics_list = []
        
        for model_class, model_name in models_to_test:
            output_image, metrics = benchmark_model(
                model_class, model_name, config,
                coords, colors, image.shape, device
            )
            
            results[model_name] = metrics
            outputs.append(output_image)
            model_names.append(model_name)
            metrics_list.append(metrics)
            
            # Save individual output
            save_image(
                output_image,
                output_dir / f"{Path(str(img_name)).stem}_{model_name.replace(' ', '_')}.png"
            )
        
        # Create comparison plot
        fig = create_comparison_grid(
            outputs, model_names,
            ground_truth=image,
            metrics=metrics_list
        )
        plt.savefig(output_dir / f"{Path(str(img_name)).stem}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save results
        all_results.append({
            'image': str(img_name),
            'results': results
        })
    
    # Save JSON results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'PSNR (dB)':<12} {'SSIM':<10} {'Time (s)':<10} {'Params':<10}")
    print(f"{'-'*70}")
    
    for model_name in [name for _, name in models_to_test]:
        metrics = all_results[0]['results'][model_name]
        print(f"{model_name:<20} {metrics['psnr']:<12.2f} {metrics['ssim']:<10.4f} "
              f"{metrics['train_time']:<10.1f} {metrics['params']:<10,}")
    
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"View comparison: {output_dir}/comparison.png")


if __name__ == '__main__':
    main()
