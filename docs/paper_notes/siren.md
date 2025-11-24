# SIREN Paper Notes

**Title:** Implicit Neural Representations with Periodic Activation Functions  
**Authors:** Sitzmann, Martel, Bergman, Lindell, Wetzstein  
**Published:** NeurIPS 2020  
**Paper:** https://arxiv.org/abs/2006.09661  
**Project:** https://www.vincentsitzmann.com/siren/

---

## TL;DR

Use **sine activations** instead of ReLU to represent signals with complex derivatives (images, 3D shapes, physics simulations). Special weight initialization is crucial.

**Key equation:**
```
Φ(x; θ) = W_n ∘ sin(W_{n-1} ∘ sin(...∘ sin(W_0 x)...))
```

Where `sin` is applied element-wise with frequency parameter ω.

---

## Main Contributions

1. **Periodic activation functions** (sine) for neural implicit representations
2. **Special initialization scheme** that ensures stable training
3. **Theoretical analysis** of derivatives and frequency content
4. **Applications** to images, 3D shapes, video, and PDEs

---

## Motivation

### Problem with ReLU Networks

Standard ReLU networks:
- Learn low frequencies first (spectral bias)
- Poor at representing fine details
- Derivatives are piecewise constant (not smooth)

### Why This Matters for INRs

INRs often need:
- High-frequency details (textures, edges)
- Smooth, accurate derivatives (for physics, normals, etc.)
- Ability to fit complex signals

---

## Method

### Sine Activation

```python
def activation(x, omega_0=30.0):
    return torch.sin(omega_0 * x)
```

**Why sine?**
- Periodic → naturally represents oscillating patterns
- Smooth → derivatives are also smooth
- Mathematical properties:
  - `d/dx sin(x) = cos(x)` (also periodic)
  - `d²/dx² sin(x) = -sin(x)` (bounded)

### Weight Initialization

**Critical for SIREN to work!**

For layer `i` with `n` input features:

1. **First layer:**
   ```
   W₀ ~ Uniform(-1/n, 1/n)
   ```

2. **Hidden layers:**
   ```
   Wᵢ ~ Uniform(-√(6/n)/ω₀, √(6/n)/ω₀)
   ```

**Intuition:** Keep activations in linear regime of sine during initialization.

### Architecture

```
Input → Sin(ω₀W₀) → Sin(ω₀W₁) → ... → Sin(ω₀W_{n-1}) → W_n → Output
```

Note: Final layer is **linear** (no activation).

---

## Theoretical Results

### Derivative Quality

**Claim:** SIREN's derivatives are accurate and smooth.

**Evidence:**
- Derivatives are compositions of sine/cosine (smooth)
- Higher-order derivatives remain bounded
- Experiments show <1% error on gradient computation

**Comparison:** ReLU derivatives are piecewise constant.

### Frequency Spectrum

**Claim:** SIREN can represent higher frequencies than ReLU networks.

**Evidence:**
- Fourier analysis shows broader frequency spectrum
- Experimentally verified on synthetic signals

---

## Experiments (from paper)

### 1. Image Fitting

**Task:** Fit f(x,y) → (r,g,b) to images

**Results:**
| Method | PSNR (Pluto) | PSNR (Text) |
|--------|--------------|-------------|
| ReLU | 21.98 dB | 14.88 dB |
| Tanh | 25.44 dB | 17.72 dB |
| SIREN | **36.45 dB** | **32.41 dB** |

**Takeaway:** Massive improvement over ReLU!

### 2. 3D Shape Representation

**Task:** Represent shapes as signed distance functions

**Results:**
- SIREN captures fine geometric details
- Smooth normals (gradients of SDF)
- Enables applications like shape interpolation

### 3. Solving PDEs

**Task:** Solve Poisson equation, wave equation

**Results:**
- SIREN satisfies boundary conditions accurately
- Helmholtz equation: error < 0.5% (vs 5% for ReLU)

**Key advantage:** Accurate derivatives matter for physics!

### 4. Video Representation

**Task:** f(x, y, t) → (r, g, b)

**Results:**
- Smooth temporal interpolation
- Compact representation

---

## Critical Analysis

### Strengths

1. ✅ **Excellent quality:** State-of-the-art results on many tasks
2. ✅ **Smooth derivatives:** Essential for physics, normals, etc.
3. ✅ **Principled approach:** Grounded in theory
4. ✅ **Versatile:** Works for images, 3D, video, PDEs

### Weaknesses

1. ❌ **Sensitive to initialization:** Wrong init → training fails
2. ❌ **Hyperparameter tuning:** ω₀ needs tuning per task
3. ❌ **Slower convergence:** Sometimes needs more epochs than Fourier features
4. ❌ **Not always best:** Fourier features sometimes match or beat SIREN

### When SIREN Shines

- Smooth natural signals
- Need for accurate derivatives
- Physics simulations
- 3D shape representation

### When Fourier Features Might Be Better

- Very high-frequency signals (sharp edges, textures)
- Easier to tune (just one scale parameter)
- More stable training

---

## Implementation Gotchas

### 1. Initialization is Critical

**Wrong:**
```python
# This will fail!
self.linear = nn.Linear(in_features, out_features)
# Default initialization doesn't work for SIREN
```

**Right:**
```python
self.linear = nn.Linear(in_features, out_features)
with torch.no_grad():
    self.linear.weight.uniform_(-bound, bound)  # Use SIREN formula
```

### 2. Frequency Parameter

**ω₀ = 30.0** works for most tasks (paper default)

**Tuning:**
- Too low → blurry, underfits
- Too high → noisy, unstable

**Tip:** If training is unstable, lower ω₀.

### 3. Final Layer

**Don't** apply sine to the final layer:
```python
# Wrong: applies sine to output
output = sin(ω₀ * W_n(...))

# Right: final layer is linear
output = W_n(...)
```

---

## Comparisons

### SIREN vs Fourier Features

| Aspect | SIREN | Fourier Features |
|--------|-------|------------------|
| **Activation** | sin(ωx) | ReLU |
| **Input encoding** | None | Random Fourier |
| **Derivatives** | Excellent | Good |
| **Tuning difficulty** | Medium | Easy |
| **Stability** | Medium | High |
| **Best for** | Smooth signals, physics | General purpose |

**Recommendation:** 
- Start with Fourier Features
- Use SIREN if you need derivatives or have smooth signals

---

## Reproducibility Notes

### Paper Results

The paper provides:
- ✅ Code: https://github.com/vsitzmann/siren
- ✅ Trained models
- ✅ Hyperparameters

### Our Implementation

Differences from paper:
- Simplified for educational purposes
- Focuses on core concepts
- Matches paper results on standard benchmarks

---

## Impact

**Citations:** 2000+ (as of 2024)

**Influence:**
- Popularized periodic activations for INRs
- Enabled NeRF-style 3D representations
- Standard baseline for INR research

**Follow-up work:**
- Modulated SIREN (conditions on input)
- SIREN for inverse problems
- Applications to medical imaging, robotics

---

## Key Takeaways

1. **Use sine activations** for implicit neural representations
2. **Initialize properly** (this is non-negotiable!)
3. **SIREN excels at smooth signals** and derivatives
4. **Compare with Fourier features** for your specific task
5. **ω₀ = 30.0** is a good starting point

---

## Further Reading

- Paper: https://arxiv.org/abs/2006.09661
- Project page: https://www.vincentsitzmann.com/siren/
- Code: https://github.com/vsitzmann/siren
- [Our implementation](../../inr_toolkit/models/siren.py)
- [Tutorial 3: Comparing Architectures](../../tutorials/03_comparing_architectures.ipynb)
