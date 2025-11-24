# Benchmarks

Honest, reproducible comparisons of INR methods.

## Philosophy

We believe in **honest benchmarks** that show:
- ✅ What works (and what doesn't)
- ✅ When methods fail
- ✅ Full reproduction instructions
- ❌ No cherry-picking results
- ❌ No hiding failure cases

## Available Benchmarks

### Image Fitting
Compare SIREN, Fourier Features, and ReLU MLP on natural images.

**Results:** [benchmarks/image_fitting/results.md](image_fitting/results.md)  
**Run it:** `python image_fitting/run_benchmark.py`

## Running Benchmarks

```bash
# Run image fitting benchmark
cd benchmarks/image_fitting
python run_benchmark.py

# Use your own images
python run_benchmark.py --data_dir /path/to/images
```

## Datasets

### Kodak PhotoCD
We use 3 images from the Kodak PhotoCD dataset for quick benchmarks.

**Download:** Images are included in `data/kodak/` (add your own images there)

## Adding New Benchmarks

Want to benchmark a new task? Create a new directory with:
1. `run_benchmark.py` - Automated benchmark script
2. `results.md` - Document your findings
3. `README.md` - Instructions to reproduce

PRs welcome!
