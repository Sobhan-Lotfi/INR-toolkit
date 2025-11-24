# When to Use INRs - Decision Guide

Not sure if INRs are right for your task? This guide helps you decide.

---

## Quick Answer

**Use INRs when you want:**
- ✅ Resolution-independent representation
- ✅ Compact storage (small network, arbitrary detail)
- ✅ Continuous/differentiable representation
- ✅ To query at arbitrary coordinates

**Don't use INRs when you need:**
- ❌ Fast inference on fixed-resolution data
- ❌ To process many images (use CNNs instead)
- ❌ Real-time performance (INRs are slow)
- ❌ Simple storage (just save the pixels!)

---

## Decision Tree

```
What are you representing?
│
├─ Single image/signal
│  ├─ Need resolution independence? → INR ✅
│  ├─ Need compact storage? → INR ✅
│  └─ Just store/display? → Save as PNG ❌
│
├─ Dataset of images
│  ├─ Need to compress many images? → Consider INR-GAN
│  └─ Standard ML task? → Use CNNs ❌
│
├─ 3D shape/volume
│  ├─ Need continuous representation? → INR ✅
│  └─ Just rendering? → Mesh/voxels may be fine ❌
│
├─ Video
│  ├─ Need temporal interpolation? → INR ✅
│  └─ Standard playback? → Save as MP4 ❌
│
└─ Physical simulation
   └─ Need to solve PDEs? → INR (SIREN) ✅
```

---

## Use Cases

### ✅ Good Use Cases

#### 1. **Image Super-Resolution**
**Task:** Upscale images to arbitrary resolutions

**Why INRs:** Train on low-res, query at high-res
```python
# Train on 256×256
coords = get_coordinates((256, 256))
model.fit(coords, colors)

# Render at 1024×1024
high_res_coords = get_coordinates((1024, 1024))
high_res_image = model(high_res_coords)
```

**Alternative:** Bicubic interpolation, SR-CNNs (faster but fixed scale)

---

#### 2. **3D Shape Representation**
**Task:** Represent 3D objects as continuous functions

**Why INRs:** Query signed distance at any 3D point
```python
# SDF: f(x,y,z) → distance to surface
sdf = SIREN(in_dim=3, out_dim=1)
```

**Applications:**
- Signed Distance Functions (SDFs)
- Occupancy fields
- NeRF (Neural Radiance Fields)

**Alternative:** Meshes, voxel grids (less flexible)

---

#### 3. **Video Frame Interpolation**
**Task:** Generate frames between keyframes

**Why INRs:** Continuous in time
```python
# f(x, y, t) → (r, g, b)
model = SIREN(in_dim=3, out_dim=3)  # x, y, time

# Query at arbitrary time
frame_1_5 = model(coords, t=1.5)  # Smooth interpolation
```

**Alternative:** Optical flow methods (faster)

---

#### 4. **Compression**
**Task:** Store data compactly

**Why INRs:** Network weights << pixel values (for large enough data)

**Example:**
- 1024×1024 RGB image = 3MB raw
- SIREN with 200K params ≈ 0.8MB

**Alternative:** JPEG, WebP (much faster decode)

---

#### 5. **Physics Simulations**
**Task:** Solve PDEs (heat equation, fluid dynamics)

**Why INRs (SIREN):** Accurate derivatives, smooth solutions

```python
# Solve: ∇²u = 0 (Laplace equation)
u = SIREN(in_dim=2, out_dim=1)

# Loss includes derivatives
loss = boundary_loss + laplacian_loss
```

**Alternative:** Finite element methods (more established)

---

### ❌ Poor Use Cases

#### 1. **Image Classification**
**Don't:** Fit INR to each image, then classify

**Why not:** Slow, unnecessary  
**Use instead:** CNNs (ResNet, ViT)

---

#### 2. **Real-Time Video Processing**
**Don't:** Use INR for live video filters

**Why not:** Too slow (need forward pass for every pixel)  
**Use instead:** GPU shaders, traditional filters

---

#### 3. **Large-Scale Image Datasets**
**Don't:** Fit separate INR to each of 1M images

**Why not:** Training takes too long  
**Use instead:** Standard datasets + CNNs

Exception: INR-GAN (shared decoder)

---

#### 4. **Simple Viewing**
**Don't:** Fit INR just to display an image

**Why not:** Massive overkill  
**Use instead:** Save as PNG/JPEG

---

## Architecture Selection

Once you've decided to use INRs, which architecture?

### Fourier Features (Default)

**Use when:**
- General purpose tasks
- Natural images
- Not sure what to use

**Code:**
```python
model = FourierFeaturesMLP(
    in_dim=2, out_dim=3,
    fourier_scale=10.0  # Tune this
)
```

**Tuning:** Start with `fourier_scale=10.0`, increase for finer details

---

### SIREN (Specialist)

**Use when:**
- Need high-quality derivatives
- Very smooth signals
- Physics simulations
- SDFs

**Code:**
```python
model = SIREN(
    in_dim=3, out_dim=1,
    omega_0=30.0  # Tune this
)
```

**Tuning:** `omega_0=30.0` usually works

---

### ReLU MLP (Baseline)

**Use when:**
- Creating a baseline to compare against
- Demonstrating spectral bias
- Educational purposes

**Don't use for production!**

---

## Practical Considerations

### Training Time

| Task | Resolution | Training Time (GPU) |
|------|-----------|---------------------|
| Single 256² image | 256×256 | ~2 minutes |
| Single 512² image | 512×512 | ~5 minutes |
| Single 1024² image | 1024×1024 | ~15 minutes |
| 3D volume 128³ | 128×128×128 | ~30 minutes |

**Implication:** INRs are for tasks where training once is acceptable.

---

### Inference Time

**Slow:** Each pixel requires a forward pass

**Example (256×256 image on GPU):**
- Full render: ~50ms
- For comparison, PNG decode: <1ms

**When this matters:**
- Real-time applications ❌
- Batch processing many images ❌

**When it's okay:**
- One-time super-resolution ✅
- Interactive editing (render on demand) ✅

---

### Memory Usage

**Training:** 
- Small (just network parameters)
- Typical: 1-5MB for a 200K parameter model

**Inference:**
- Need to store network
- Need to compute forward pass

**Trade-off:** Compact storage but computational inference

---

## Examples from the Wild

### ✅ Successful Applications

1. **NeRF (Neural Radiance Fields)**
   - Represent 3D scenes as 5D functions: (x, y, z, θ, φ) → (r, g, b, density)
   - Achieved photorealistic novel view synthesis

2. **COIN (COmpression with Implicit Neural representations)**
   - Compress images/videos with INRs
   - Competitive with JPEG on large images

3. **Deep SDF**
   - Represent 3D shapes as signed distance functions
   - Enables shape interpolation, completion

4. **Neural Implicit Representations for Physics**
   - Solve PDEs with SIREN
   - Smooth, high-order derivatives

---

## Checklist

Before using INRs, ask yourself:

- [ ] Do I need continuous/resolution-independent representation?
- [ ] Is training time (minutes to hours) acceptable?
- [ ] Is slow inference acceptable?
- [ ] Am I representing a single signal (not a dataset)?
- [ ] Is this better than standard methods (PNG, mesh, etc.)?

If you answered **yes** to all → INRs are a good fit!

If you answered **no** to any → Consider alternatives

---

## Still Unsure?

**Try it!** The best way to know if INRs work for your task:

1. Start with [Tutorial 1](../tutorials/01_hello_inr.ipynb)
2. Use [fit_image.py](../examples/fit_image.py) on your data
3. Compare with baseline methods
4. Decide based on results

**Questions?** Open an issue on GitHub!

---

## Further Reading

- [Architectures Deep Dive](architectures.md)
- [Tutorials](../tutorials/)
- [Benchmarks](../benchmarks/)
- [Examples](../examples/)
