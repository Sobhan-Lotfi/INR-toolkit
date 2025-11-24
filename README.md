# 🧠 INR Toolkit

**The hands-on guide to Implicit Neural Representations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Bridge the gap between INR papers and practice. Learn by doing with interactive tutorials, clean implementations, and honest benchmarks.

---

## 🎯 Why This Exists

INR papers are exciting. But going from paper to working code? That's where most people get stuck.

**This toolkit gives you:**
- ✅ **Interactive tutorials** that build intuition (not just theory)
- ✅ **Clean, documented implementations** you can actually understand
- ✅ **Honest benchmarks** showing what works (and what doesn't)
- ✅ **Production-ready code** you can use in your projects

**Target audience:** Researchers, ML engineers, and students who want to actually USE INRs.

---

## 🚀 Quick Start (5 minutes)

```bash
# Install
git clone https://github.com/Sobhan-Lotfi/INR-toolkit.git
cd INR-toolkit
pip install -r requirements.txt

# Fit your first neural field
python examples/fit_image.py --image path/to/image.jpg
```

Or jump into the [interactive tutorial](tutorials/01_hello_inr.ipynb):
```bash
jupyter notebook tutorials/01_hello_inr.ipynb
```

---

## 📚 Learning Path

**Start here if you're new to INRs:**

1. **[Hello INR](tutorials/01_hello_inr.ipynb)** (15 min)
   Fit a 2D image with a neural network. Build intuition for what INRs do.

2. **[Fourier Features](tutorials/02_fourier_features.ipynb)** (20 min)
   Why do simple MLPs fail at high frequencies? See the fix in action.

3. **[Architecture Comparison](tutorials/03_comparing_architectures.ipynb)** (25 min)
   SIREN vs Fourier Features vs ReLU MLP. When to use each?

**Total learning time:** ~1 hour to go from zero to confident

---

## 💡 What Are Implicit Neural Representations?

Traditional approach:
```
Image = 2D array of pixels (H × W × 3)
```

INR approach:
```
Image = Neural network that maps coordinates → colors
f(x, y) → (r, g, b)
```

**Why this matters:**
- 🎨 Resolution-independent (query at any resolution)
- 📦 Compact representation (small network, infinite detail)
- 🔧 Differentiable (optimize with gradients)
- 🌐 General (works for images, 3D shapes, audio, video...)

---

## 🛠️ Implementations

All implementations are **<100 lines** with extensive comments.

| Model | Use Case | Lines of Code |
|-------|----------|---------------|
| **ReLU MLP** | Baseline (spoiler: doesn't work well) | ~50 |
| **Fourier Features** | Most versatile, good default | ~60 |
| **SIREN** | Smooth signals, best for derivatives | ~80 |

```python
from inr_toolkit.models import FourierFeaturesMLP

# Create model
model = FourierFeaturesMLP(
    in_dim=2,           # 2D coordinates (x, y)
    out_dim=3,          # RGB colors
    hidden_dim=256,
    num_layers=4,
    fourier_scale=10.0
)

# Use it
coordinates = ...   # Shape: (N, 2)
colors = model(coordinates)  # Shape: (N, 3)
```

---

## 📊 Benchmarks

We compare methods on standard image fitting tasks. **Honest results** with code to reproduce.

**Preview (Kodak dataset, 512×512 images):**

| Method | PSNR (dB) | Training Time | Parameters |
|--------|-----------|---------------|------------|
| ReLU MLP | 24.3 | 2 min | 200K |
| Fourier Features | **33.8** | 3 min | 200K |
| SIREN | **34.2** | 3 min | 200K |

📈 [Full benchmarks with analysis →](benchmarks/image_fitting/results.md)

---

## 📖 Examples

### Fit a Single Image
```python
from inr_toolkit.models import SIREN
from inr_toolkit.training import Trainer
from inr_toolkit.utils import load_image, get_coordinates

# Load image
image = load_image("photo.jpg")  # Shape: (H, W, 3)
coords = get_coordinates(image.shape[:2])  # Shape: (H*W, 2)
colors = image.reshape(-1, 3)  # Shape: (H*W, 3)

# Train model
model = SIREN(in_dim=2, out_dim=3, hidden_dim=256)
trainer = Trainer(model, lr=1e-4)
trainer.fit(coords, colors, epochs=1000)

# Render at any resolution
new_coords = get_coordinates((1024, 1024))  # 2x resolution
high_res = model(new_coords).reshape(1024, 1024, 3)
```

### Compare Methods
```bash
python examples/compare_methods.py --image photo.jpg
```
Generates side-by-side comparison of all architectures.

---

## 🏗️ Repository Structure

```
inr-toolkit/
├── tutorials/          # 🔥 Start here! Interactive learning
├── inr_toolkit/        # Core library (models, training, utils)
├── benchmarks/         # Reproducible comparisons
├── examples/           # Copy-paste starting points
└── docs/               # Deep dives on architectures
```

---

## 🤝 Contributing

This is an **educational project**. Contributions that help people learn are especially welcome:

- 📝 Better explanations or tutorial improvements
- 🐛 Bug fixes or code clarity improvements
- 📊 New benchmarks or use cases
- 📚 Paper summaries or implementation notes

See issues tagged \`good-first-issue\` to get started.

---

## 📄 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{inr_toolkit2025,
  title = {INR Toolkit: Educational Resource for Implicit Neural Representations},
  author = {INR Toolkit Contributors},
  year = {2025},
  url = {https://github.com/Sobhan-Lotfi/INR-toolkit}
}
```

---

## 🙏 Acknowledgments

Built on foundational work:
- **SIREN** - Sitzmann et al. (2020) - [Paper](https://arxiv.org/abs/2006.09661)
- **Fourier Features** - Tancik et al. (2020) - [Paper](https://arxiv.org/abs/2006.10739)

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## ⭐ Star History

If this helps you, please star the repo! It helps others discover the project.

**Mission: Make INRs accessible to everyone. 500+ stars = we succeeded.**

---

<div align="center">
Made with ❤️ by the INR community | Questions? Open an issue!
</div>
