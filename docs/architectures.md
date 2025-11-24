# INR Architectures: Deep Dive

A detailed look at each architecture in the toolkit.

---

## Table of Contents
1. [ReLU MLP - The Baseline](#relu-mlp)
2. [Fourier Features - The Workhorse](#fourier-features)
3. [SIREN - The Specialist](#siren)
4. [Architecture Comparison](#comparison)

---

## ReLU MLP

### Overview
Standard Multi-Layer Perceptron with ReLU activations.

```python
f(x) = W_n(...ReLU(W_2(ReLU(W_1(x)))))
```

### Why It Fails for INRs

**Spectral Bias:** Neural networks with ReLU activations learn low-frequency functions first and struggle with high frequencies.

**Theory:** The Neural Tangent Kernel (NTK) of ReLU networks has limited high-frequency components, making it hard to fit signals with fine details.

**Practical Impact:**
- Blurry reconstructions
- Poor PSNR (typically 24-26 dB on natural images)
- Can't capture sharp edges or textures

### When to Use
- As a baseline to compare against
- When you want to demonstrate the spectral bias problem
- Educational purposes

### Implementation Details
```python
class ReLUMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        # Simple: Linear → ReLU → ... → Linear
```

**Pros:**
- Simple to implement
- Fast to train

**Cons:**
- Poor quality for INRs
- Not suitable for production use

---

## Fourier Features

### Overview
Maps input coordinates through random Fourier features before passing to an MLP.

```python
γ(x) = [sin(2π B x), cos(2π B x)]
f(x) = MLP(γ(x))
```

Where B is a random Gaussian matrix: `B ~ N(0, σ²I)`

### Why It Works

**Key Insight:** The Fourier feature mapping changes the NTK spectrum to include higher frequencies.

**Intuition:** 
- Raw coordinates `(x, y)` are smooth
- Fourier features `[sin(Bx), cos(Bx)]` oscillate at various frequencies
- This gives the network "building blocks" for high-frequency patterns

### Parameters

**`fourier_dim`** (default: 256)
- Dimension of the Fourier feature space
- Higher = more expressive, but more parameters
- Typical range: 128-512

**`fourier_scale`** (σ in the paper, default: 10.0)
- Controls the frequency range of the encoding
- Higher scale → higher frequencies → finer details
- Typical range: 1.0-20.0
- **This is the most important hyperparameter!**

### Tuning Guide

For a signal with maximum frequency `f_max`:
- Set `fourier_scale ≈ f_max`
- For 256×256 images: start with 10.0
- For 1024×1024 images: try 20.0-40.0
- If too blurry → increase scale
- If noisy/overfitting → decrease scale

### Implementation Details
```python
# Fixed random matrix (not trainable)
self.B = torch.randn(in_dim, fourier_dim) * fourier_scale

def forward(self, x):
    # Apply Fourier features
    x_proj = 2 * π * (x @ self.B)
    features = torch.cat([sin(x_proj), cos(x_proj)], dim=-1)
    return self.mlp(features)
```

### Pros & Cons

**Pros:**
- Excellent quality (30-35 dB PSNR)
- Easy to tune (one main parameter: `fourier_scale`)
- Stable training
- Works for most tasks

**Cons:**
- More parameters than raw MLP
- Random features add some variance

### When to Use
- **Default choice for most INR tasks**
- Natural images
- Textures
- 3D volumes
- Any signal with high-frequency content

---

## SIREN

### Overview
Uses sine activations instead of ReLU, with special initialization.

```python
f(x) = W_n(...sin(ω W_2(sin(ω W_1(x)))))
```

Where ω (omega) is a frequency parameter.

### Why It Works

**Key Insight:** Sine activations have derivatives that are also periodic, creating a natural bias toward smooth, oscillating functions.

**Mathematical Property:**
- `d/dx sin(x) = cos(x)`
- `d²/dx² sin(x) = -sin(x)`

This makes SIREN excellent for tasks requiring derivatives (physics, PDEs).

### Special Initialization

SIREN requires careful weight initialization:

```python
# First layer
W ~ Uniform(-1/n, 1/n)

# Other layers
W ~ Uniform(-√(6/n)/ω, √(6/n)/ω)
```

This ensures activations stay in the linear regime of sine during initialization.

### Parameters

**`omega_0`** (ω, default: 30.0)
- Frequency of sine activations
- Higher → captures finer details
- Typical range: 1.0-30.0
- **Critical: affects both forward pass and initialization!**

### Tuning Guide

- Start with `omega_0=30.0` (paper default)
- For very smooth signals: try 10.0-20.0
- For high-frequency details: try 30.0-50.0
- If training is unstable: lower omega_0

### Implementation Details
```python
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0):
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        # Special initialization
        self.init_weights()
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
```

### Pros & Cons

**Pros:**
- Excellent quality (33-36 dB PSNR)
- Best for smooth signals
- Great derivatives (important for physics)
- No random components (deterministic)

**Cons:**
- Sensitive to initialization
- Harder to tune than Fourier Features
- Can be unstable with wrong omega_0

### When to Use
- Smooth natural signals
- Physics simulations (PDEs, fluid dynamics)
- Signed distance functions (SDFs)
- When you need high-quality derivatives
- 3D shape representation

---

## Comparison

### Quick Decision Guide

```
Need INR for:
├─ General images/signals?
│  └─ Use Fourier Features ✅
├─ Very smooth signals?
│  └─ Try SIREN ✅
├─ Physics/PDEs/derivatives?
│  └─ Use SIREN ✅
├─ Just a baseline?
│  └─ ReLU MLP (expect poor results)
└─ Unsure?
   └─ Start with Fourier Features ✅
```

### Quantitative Comparison

| Metric | ReLU MLP | Fourier Features | SIREN |
|--------|----------|------------------|-------|
| **PSNR (256×256 image)** | 24-26 dB | 33-35 dB | 34-36 dB |
| **Training Speed** | Fast | Medium | Medium |
| **Ease of Tuning** | Easy (doesn't help) | Easy | Medium |
| **Stability** | High | High | Medium |
| **Derivative Quality** | Poor | Good | Excellent |

### Implementation Complexity

| Aspect | ReLU MLP | Fourier Features | SIREN |
|--------|----------|------------------|-------|
| **Lines of Code** | ~50 | ~60 | ~80 |
| **Special Init** | No | No | Yes |
| **Hyperparameters** | Standard | 1 key (scale) | 1 key (omega) |

---

## Advanced Topics

### Combining Methods

Can you use both Fourier features AND sine activations?

**Yes!** Some researchers combine them:
```python
features = fourier_features(x)
output = siren_network(features)
```

Results are mixed - usually not better than either alone.

### Learned vs Random Fourier Features

Our implementation uses **random** Fourier features (fixed at initialization).

Alternative: **Learned** features (trainable B matrix)
- Pros: Potentially better fit
- Cons: More parameters, slower, can overfit

For most tasks, random features work great and are simpler.

### Other Architectures

Recent work includes:
- **BACON:** "Banf" activations
- **WIRE:** Complex Gabor wavelets
- **Multiplicative Filter Networks**
- **Modulated SIREN**

We focus on the foundational methods that work well in practice.

---

## References

1. **SIREN**  
   Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020  
   [Paper](https://arxiv.org/abs/2006.09661)

2. **Fourier Features**  
   Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020  
   [Paper](https://arxiv.org/abs/2006.10739)

3. **Spectral Bias**  
   Rahaman et al., "On the Spectral Bias of Neural Networks", ICML 2019  
   [Paper](https://arxiv.org/abs/1806.08734)

---

## Further Reading

- [When to Use](when_to_use.md) - Decision guide
- [Paper Notes](paper_notes/) - Detailed paper summaries
- [Tutorials](../tutorials/) - Hands-on learning
- [Benchmarks](../benchmarks/) - Quantitative comparisons
