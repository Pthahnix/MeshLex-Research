<!-- markdownlint-disable -->
# MeshCraft: Exploring Efficient and Controllable Mesh Generation with Flow-based DiTs

## Core Architecture

**Type**: Flow-based Diffusion Transformer (DiT) with rectified flow

MeshCraft is a two-stage pipeline:

1. **Stage 1 -- Transformer-based VAE**: Encodes raw triangle meshes into continuous face-level latent tokens and decodes them back. Encoder: 12 transformer layers, hidden size 768. Decoder: 18 transformer layers, hidden size 384. A single GCN layer aggregates geometric features (normals, angles, areas, face adjacency) before the transformer encoder. Latent tokens are regularized with KL-divergence (not discrete VQ), producing 8-dimensional continuous tokens per face.

2. **Stage 2 -- Flow-based DiT**: A 24-layer transformer with hidden size 864 generates face-level continuous tokens using rectified flow (ODE-based, not SDE). Uses RoPE for positional encoding, SwiGLU activation, sandwich normalization, and QK-norm for training stability. Supports variable-length token sequences via padding + attention masking.

The architecture draws from SiT (Scalable Interpolant Transformers) and adapts it for mesh generation with variable-length sequences and multi-condition CFG.

## 3D Data Tokenization/Representation

Meshes are represented as **ordered face-level sequences**. Each mesh with n triangular faces is serialized as:
- Faces sorted by vertex indices (lowest to highest).
- Vertices within faces sorted by z-y-x coordinates.
- Each face = 3 vertices x 3 coordinates = 9 values.

The VAE compresses each face into an **8-dimensional continuous latent token** (not discrete). This achieves up to **9x token reduction** compared to prior auto-regressive methods that tokenize at the vertex or coordinate level:
- MeshGPT: 9 tokens per face (3 vertices x 3 quantized coords)
- MeshCraft: 1 token per face (8-dim continuous)

Coordinate space is quantized to 128 (ShapeNet) or 256 (Objaverse) resolution levels. The decoder outputs are reshaped into coordinates and trained with cross-entropy loss against the quantized input.

Key insight: Continuous tokens (KL-regularized) significantly outperform discrete tokens (RVQ) for reconstruction. On Objaverse, KL 8-dim achieves 99.66% triangle accuracy vs. RVQ's 65.12%.

## Conditioning Mechanism

MeshCraft supports two types of conditioning with **dual classifier-free guidance (CFG)**:

1. **Face number conditioning** (c_f): The number of target faces is embedded via a learnable embedding layer and added to timestep embeddings, injected through adaLN-Zero blocks. CFG weight w1 controls this condition.

2. **Image conditioning** (c_i): Image features from DINOv2 ViT-L/14 are injected via cross-attention modules in the DiT blocks. CFG weight w2 controls this condition.

The dual-CFG formulation:
```
v_t = v(z_t, empty, empty)
    + w1 * (v(z_t, c_f, empty) - v(z_t, empty, empty))
    + w2 * (v(z_t, c_f, c_i) - v(z_t, c_f, empty))
```

Optimal weights: w1=1.0, w2=5.0 for image-conditioned; w=8.0 for face-number-only. During training, conditions are randomly dropped for CFG. Variable-length generation is handled by padding + masking.

## Scale

- **VAE parameters**: ~comparable to baselines (12-layer encoder 768-dim + 18-layer decoder 384-dim).
- **DiT parameters**: 24-layer transformer, hidden 864 -- "similar number of parameters compared with baselines" (MeshGPT, PivotMesh, MeshXL-350M).
- **Training data**: ShapeNet (~10K meshes, 4 categories: chair, table, bench, lamp, meshes decimated to max 800 faces). Objaverse (~65K meshes with 1024-1536 faces).
- **Training compute**: 8x A100 80GB. VAE: ~2 days. DiT: ~3 days (ShapeNet), ~3 weeks (Objaverse). Uses bf16 mixed precision.
- **Inference**: 50-step Euler sampling. Generates 800-face mesh in 3.2 seconds (35x faster than MeshGPT).

## Key Innovation

MeshCraft is the first to apply continuous-space flow-based diffusion transformers to native mesh generation, compressing each triangle face into a single 8-dimensional continuous latent token (9x fewer tokens than AR methods) and enabling both controllable face count and 35x faster generation than auto-regressive baselines, while achieving state-of-the-art quality on ShapeNet.

## Main Limitation

1. **Limited face-count extrapolation**: The learnable face-number embedding cannot generalize to unseen face counts outside the training distribution.
2. **Domain gap sensitivity**: When input images or assigned face numbers diverge significantly from training data, generation fails to produce complete meshes.
3. **Scale validation**: While Objaverse results are shown, the method is not validated at the scale of Meshtron-style auto-regressive models (which handle much larger meshes). The paper explicitly notes it prioritizes efficiency and controllability over scalability.
4. **No text conditioning demonstrated**: Only image and face-number conditions are evaluated, though the architecture could support text via cross-attention.

## Quantitative Highlights

- **Reconstruction (ShapeNet)**: 99.42% triangle accuracy, L2 distance 0.06e-2 (competitive with MeshGPT's 99.99%/0.00).
- **Generation (ShapeNet Chair)**: COV 51.44, MMD 9.61, 1-NNA 54.31 -- beats MeshGPT (45.98/10.34/60.06), PivotMesh (47.99/10.00/60.06), and MeshXL (49.43/10.17/56.90).
- **Speed**: 800-face mesh in 3.2 sec = 35x faster than MeshGPT. Token count reduced 9x.
- **Ablation**: Continuous KL-8dim (99.66% accuracy on Objaverse) >> RVQ discrete (65.12%). CFG weight w=8.0 optimal for face-number conditioning.
