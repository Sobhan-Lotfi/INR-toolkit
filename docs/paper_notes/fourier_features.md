# Fourier Features Paper Notes

**Title:** Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains  
**Authors:** Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal, Ramamoorthi, Barron, Ng  
**Published:** NeurIPS 2020  
**Paper:** https://arxiv.org/abs/2006.10739  
**Project:** https://people.eecs.berkeley.edu/~bmild/fourfeat/

---

## TL;DR

Map input coordinates through **random Fourier features** before feeding to an MLP. This simple trick lets networks learn high-frequency functions.

**Key transformation:**
```
γ(v) = [sin(2πBv), cos(2πBv)]
```
Where `B ~ N(0, σ²I)` is a random matrix.

---

## Main Contributions

1. **Diagnosis:** Neural networks have spectral bias (learn low frequencies first)
2. **Solution:** Fourier feature mapping fixes this
3. **Theory:** Connection to Neural Tangent Kernel (NTK)
4. **Practice:** Massive improvements on low-dimensional signals

---

## Motivation

### The Spectral Bias Problem

**Observation:** Standard MLPs fail at high-frequency signals.

**Example:**
```python
# Try to fit: f(x) = sin(20πx)
model = ReLU_MLP(...)
# Result: Blurry approximation (learns sin(πx) instead)
```

**Why?** The Neural Tangent Kernel (NTK) of standard MLPs favors low frequencies.

### Low-Dimensional Input

**Key insight:** The problem is worse when inputs are low-dimensional!

- 2D image coordinates: (x, y) ∈ [0,1]²
- 3D positions: (x, y, z) ∈ R³

**Issue:** Low-dimensional inputs → limited frequency content → spectral bias

---

## Method

### Fourier Feature Mapping

**Input:** v ∈ R^d (e.g., 2D coordinates)

**Transform:**
```
γ(v) = [a₁ᵀv, a₂ᵀv, ..., aₘᵀv, b₁ᵀv, b₂ᵀv, ..., bₘᵀv]ᵀ
```
Where:
- aᵢ = sin(2πBᵢv)
- bᵢ = cos(2πBᵢv)
- Bᵢ ~ N(0, σ²I) (random, fixed)

**Output:** γ(v) ∈ R^(2m)

### Practical Implementation

```python
class FourierFeatures:
    def __init__(self, in_dim, fourier_dim, scale=1.0):
        # Random matrix (fixed, not trainable)
        self.B = torch.randn(in_dim, fourier_dim) * scale
    
    def forward(self, v):
        # Project and apply sin/cos
        v_proj = 2 * π * (v @ self.B)
        return torch.cat([torch.sin(v_proj), torch.cos(v_proj)], dim=-1)

# Use it
features = FourierFeatures(in_dim=2, fourier_dim=256, scale=10.0)
mlp = ReLU_MLP(in_dim=512, ...)  # 2*256 = 512
model = lambda x: mlp(features(x))
```

### Key Parameter: σ (scale)

**Controls frequency range:**
- High σ → high frequencies
- Low σ → low frequencies

**How to choose:**
- For signal with max frequency `f_max`, set `σ ≈ f_max`
- Empirically: try `σ = 1, 10, 100` and pick best

---

## Theoretical Analysis

### Neural Tangent Kernel (NTK)

**Standard MLP:** NTK has limited high-frequency components

**With Fourier features:** NTK spectrum matches the Fourier feature distribution

**Implication:** By choosing B's distribution, we control what frequencies the network can learn!

### Random vs Learned Features

**Paper shows:** Random features work as well as learned ones!

**Why?**
- Random features provide a diverse set of frequencies
- Network learns coefficients (weights) to combine them
- No need to learn the features themselves

**Practical benefit:** Faster training, fewer parameters

---

## Experiments (from paper)

### 1. 1D Signal Fitting

**Task:** Fit f(x) = sin(40πx)

**Results:**
| Method | MSE |
|--------|-----|
| ReLU MLP | 0.234 |
| ReLU MLP + Fourier (σ=10) | **0.001** |

**Takeaway:** 200x improvement!

### 2. Image Fitting (Kodak dataset)

**Task:** Represent 512×512 RGB images

**Results:**
| Method | PSNR |
|--------|------|
| ReLU MLP | 25.1 dB |
| Fourier (σ=1) | 28.4 dB |
| Fourier (σ=10) | **33.2 dB** |
| Fourier (σ=100) | 32.8 dB |

**Observation:** σ=10 is optimal for natural images.

### 3. 3D Shape Representation

**Task:** Fit occupancy field for 3D objects

**Results:**
- Standard MLP: Blurry shapes
- Fourier features: Sharp geometric details

### 4. CT Reconstruction

**Task:** Reconstruct 3D volume from X-ray projections

**Results:**
- Fourier features enable faster convergence
- Better reconstruction quality

---

## Critical Analysis

### Strengths

1. ✅ **Simple:** Just map inputs, then use standard MLP
2. ✅ **Effective:** Massive quality improvements (8+ dB PSNR)
3. ✅ **General:** Works for images, 3D, video, etc.
4. ✅ **Theoretically grounded:** NTK analysis explains why it works
5. ✅ **No special initialization:** Unlike SIREN

### Weaknesses

1. ❌ **Random component:** Different runs give slightly different results
2. ❌ **Extra parameters:** Fourier features add to model size
3. ❌ **Hyperparameter:** Need to tune σ (though not too sensitive)

### Comparison to Alternatives

**vs SIREN:**
- Fourier: Easier to tune, more stable
- SIREN: Better derivatives, deterministic

**vs Positional Encoding (NeRF):**
- Same core idea (frequency mapping)
- Fourier: Random features
- NeRF: Fixed frequencies [1, 2, 4, 8, ...]

---

## Tuning Guide

### Choosing σ (scale)

**For natural images:**
- 256×256: σ = 10
- 512×512: σ = 10-20
- 1024×1024: σ = 20-40

**For synthetic signals:**
- Estimate maximum frequency
- Set σ ≈ f_max

**Diagnostic:**
- Too blurry? → Increase σ
- Noisy/overfitting? → Decrease σ

### Choosing fourier_dim

**Typical values:** 128-512

**Trade-off:**
- Higher dim → more expressive, more parameters
- Lower dim → faster, less memory

**Recommendation:** Start with 256.

---

## Implementation Gotchas

### 1. Don't Forget 2π!

**Wrong:**
```python
features = torch.cat([torch.sin(v @ B), torch.cos(v @ B)], dim=-1)
```

**Right:**
```python
features = torch.cat([
    torch.sin(2 * π * (v @ B)),
    torch.cos(2 * π * (v @ B))
], dim=-1)
```

### 2. Fix Random Features

**Important:** B should be **fixed** (not trainable)

```python
# Right way
self.register_buffer('B', torch.randn(in_dim, fourier_dim) * scale)

# Wrong way
self.B = nn.Parameter(torch.randn(in_dim, fourier_dim) * scale)
```

### 3. Coordinate Normalization

**Best practice:** Normalize coordinates to [-1, 1] or [0, 1]

```python
# For images
coords = coords / image_size * 2 - 1  # Map to [-1, 1]
```

---

## Variants

### Learned Fourier Features

**Modification:** Make B trainable

**Results:** Sometimes slightly better, often not worth it

**Cost:** More parameters, slower training

### Positional Encoding (NeRF-style)

**Alternative:** Use fixed frequencies instead of random

```python
# NeRF encoding
frequencies = [1, 2, 4, 8, 16, ...]
features = [sin(2^i * v), cos(2^i * v) for i in range(L)]
```

**Comparison:**
- Fourier: Random, adaptive frequency distribution
- NeRF: Fixed, logarithmic spacing

Both work well in practice!

---

## Ablation Studies (from paper)

### Effect of σ

| σ | PSNR (Kodak) |
|---|--------------|
| 1 | 28.4 dB |
| 10 | **33.2 dB** |
| 100 | 32.8 dB |

**Takeaway:** σ=10 is robust for natural images.

### Random vs Learned

| Features | PSNR |
|----------|------|
| Random (fixed) | 33.2 dB |
| Learned | 33.4 dB |

**Takeaway:** Random features are nearly as good!

### Number of Features

| fourier_dim | PSNR |
|-------------|------|
| 64 | 31.8 dB |
| 256 | 33.2 dB |
| 1024 | 33.5 dB |

**Takeaway:** Diminishing returns after 256.

---

## Key Equations

### Fourier Feature Mapping
```
γ(v) = [sin(2πB₁v), cos(2πB₁v), ..., sin(2πBₘv), cos(2πBₘv)]
```

Where:
```
Bᵢ ~ N(0, σ²I)
```

### Full Model
```
f(v) = W_L(...ReLU(W₂(ReLU(W₁(γ(v))))))
```

---

## Reproducibility

**Paper provides:**
- ✅ Full method description
- ✅ Hyperparameters
- ✅ Code: https://github.com/tancik/fourier-feature-networks

**Our implementation:**
- Simplified for education
- Matches paper performance
- See: [inr_toolkit/models/fourier.py](../../inr_toolkit/models/fourier.py)

---

## Impact

**Citations:** 1500+ (as of 2024)

**Influence:**
- Became standard baseline for INRs
- Inspired NeRF's positional encoding
- Widely used in graphics, vision, robotics

**Applications:**
- Image/video compression
- Novel view synthesis (NeRF)
- 3D reconstruction
- Medical imaging

---

## Key Takeaways

1. **Spectral bias is real** - standard MLPs can't learn high frequencies
2. **Fourier features fix this** - simple and effective
3. **σ controls frequency range** - tune for your signal
4. **Random features work** - no need to learn them
5. **Best default choice** - use this for most INR tasks

---

## Practical Recipe

```python
# 1. Choose parameters
fourier_dim = 256
fourier_scale = 10.0  # Tune this!

# 2. Create model
model = FourierFeaturesMLP(
    in_dim=2,
    out_dim=3,
    fourier_dim=fourier_dim,
    fourier_scale=fourier_scale
)

# 3. Train normally
trainer = Trainer(model, lr=1e-4)
trainer.fit(coords, targets, epochs=1000)

# 4. If blurry, increase fourier_scale
# 5. If noisy, decrease fourier_scale
```

---

## Further Reading

- Paper: https://arxiv.org/abs/2006.10739
- Project: https://people.eecs.berkeley.edu/~bmild/fourfeat/
- Code: https://github.com/tancik/fourier-feature-networks
- [Our implementation](../../inr_toolkit/models/fourier.py)
- [Tutorial 2: Fourier Features](../../tutorials/02_fourier_features.ipynb)
