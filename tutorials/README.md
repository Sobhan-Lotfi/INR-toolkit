# INR Toolkit Tutorials

Interactive tutorials to learn Implicit Neural Representations by doing.

## Learning Path

These tutorials are designed to be completed in order:

### 1. **Hello INR** (`01_hello_inr.ipynb`) - 15 minutes
**What you'll learn:**
- What is an Implicit Neural Representation?
- Fit your first neural field to a 2D image
- Visualize how the network learns
- Understand the coordinate → value mapping

**Prerequisites:** Basic Python and PyTorch knowledge

---

### 2. **Fourier Features** (`02_fourier_features.ipynb`) - 20 minutes
**What you'll learn:**
- Why standard MLPs fail at high frequencies (spectral bias)
- How Fourier features solve this problem
- Hands-on comparison: with vs without Fourier features
- Tuning the Fourier scale parameter

**Prerequisites:** Tutorial 1

---

### 3. **Comparing Architectures** (`03_comparing_architectures.ipynb`) - 25 minutes
**What you'll learn:**
- SIREN vs Fourier Features vs ReLU MLP
- When to use each architecture
- Quantitative comparison (PSNR, SSIM, training time)
- Visualizing the differences

**Prerequisites:** Tutorials 1 & 2

---

## Running the Tutorials

```bash
# Install dependencies
pip install -r ../requirements.txt

# Launch Jupyter
jupyter notebook

# Open any tutorial and run cells sequentially
```

## Tips for Learning

1. **Run every cell** - Don't just read, execute the code!
2. **Experiment** - Change parameters and see what happens
3. **Visualize** - Look at the plots to build intuition
4. **Ask questions** - Open an issue if anything is unclear

---

## What's Next?

After completing these tutorials:
- Try the [examples](../examples/) on your own images
- Read the [architecture deep dives](../docs/architectures.md)
- Run the [benchmarks](../benchmarks/) to see performance comparisons
- Build your own INR applications!

---

**Total time investment:** ~1 hour to go from zero to confident with INRs.
