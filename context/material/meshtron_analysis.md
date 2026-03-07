<!-- markdownlint-disable -->
# Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale

**Paper**: Hao et al., 2024 (NVIDIA)
**arXiv**: 2412.09548

---

## Core Architecture Design

Meshtron is an autoregressive mesh generation model that directly produces triangle (and quad-dominant) meshes by predicting vertex coordinates sequentially. The backbone is an **Hourglass Transformer** with two shortening stages, each reducing the sequence by a factor of 3x. This creates a three-stage hierarchy:

1. **Coordinate level** (full resolution) — processes individual x/y/z coordinates
2. **Vertex level** (1/3 resolution) — groups of 3 coordinates compressed into vertex-level embeddings
3. **Face level** (1/9 resolution) — groups of 3 vertices compressed into face-level embeddings

The architecture uses linear shortening and linear upsampling layers with causality-preserving shifts. Deeper transformer blocks process the compressed face-level tokens, allocating more compute to the harder prediction steps (later tokens within each face/vertex group have higher perplexity due to vertex sharing patterns).

The full-scale model uses an HG-4-8-12 configuration: 4 blocks at full resolution, 8 blocks at 1/3 resolution, and 12 blocks at 1/9 resolution. It uses SwiGLU activations, 1536 channels, 96-dim head channels, and 4096-dim FFN hidden layers.

**Conditioning** is performed via cross-attention. A Perceiver encoder (12-layer transformer in the full-scale model) encodes 16,384 input points into 1024 embeddings. Face count and quad-face ratio are each encoded into 1 embedding via MLP and concatenated. Every 4th transformer layer is replaced with a cross-attention layer to enable interaction with conditioning signals.

---

## Tokenization / Representation Strategy

Meshtron uses **direct coordinate tokenization** without any VQ-VAE or learned codebook:

- Vertices are sorted in **yzx order** (y = vertical axis)
- Within each face, vertices are sorted lexicographically (lowest yzx-ordered vertex first)
- Faces are sorted in ascending yzx-order based on their sorted vertex values
- Each coordinate is **quantized into 1024 discrete bins** (the full-scale model)
- A triangle mesh with N faces becomes a sequence of **9N tokens** (3 vertices x 3 coordinates)
- Special tokens: start-of-sequence (S), end-of-sequence (E), padding (P), always in groups of 9 to preserve face/vertex structure

For a 32k-face mesh, this produces a sequence of ~288k tokens. The representation is lossless (up to quantization) — no learned compression is applied. This is a deliberate choice: the paper argues that architectural efficiency (Hourglass + truncated training) is a better path than representation compression.

---

## How Long Sequences Are Handled

This is the central contribution. Four mechanisms work together:

1. **Hourglass Transformer**: Reduces effective sequence length at deeper layers by 3x and 9x, cutting memory by 50%+ and improving throughput by 2.5x vs. plain transformers.

2. **Truncated sequence training**: Instead of training on full mesh sequences (which can be 288k+ tokens), the model trains on fixed-length truncated segments (8192 tokens in the full-scale model). This dramatically reduces training memory (e.g., from 35.2 GB to 12.8 GB per GPU).

3. **Sliding window inference**: During generation, a rolling KV-cache with buffer size equal to the attention window enables linear-time inference. Cached embeddings carry information from beyond the current window, effectively extending the receptive field beyond what was seen during training. Surprisingly, this train-test mismatch is beneficial — the truncated-trained model with SWA inference outperforms the full-sequence-trained model.

4. **Sequence ordering enforcement**: A constrained sampling strategy that prevents invalid token predictions by enforcing lexicographic ascending order within faces and across consecutive faces. This eliminates ~32% of invalid predictions at 1024-level quantization, narrowing the sampling space.

RoPE (Rotary Positional Embeddings) with theta=1M is used, which is natively compatible with the rolling KV-cache and supports length extrapolation.

---

## Conditioning Mechanisms

- **Primary**: Point cloud conditioning via Perceiver encoder (16,384 points → 1024 embeddings)
- **Auxiliary**: Face count (controls mesh density) and quad-face ratio (controls tessellation style), each encoded as 1 embedding via MLP
- **Integration**: Cross-attention layers inserted every 4th transformer block
- **Noise augmentation**: Gaussian noise on point positions (sigma=0.1) and normals (sigma=0.2) during training; adjustable at inference for creativity vs. faithfulness tradeoff
- Point clouds are sampled via multi-view rasterization (20 icosahedron views) to avoid sampling interior points

The cross-attention design is critical for the truncated training strategy: since each truncated segment independently attends to the full conditioning signal, global shape information is available regardless of segment position. This is a key advantage over prepending-based conditioning used by MeshXL/MeshAnything.

---

## Model Scale

| Aspect | Small Scale | Full Scale |
|--------|------------|------------|
| Parameters | 0.5B | **1.1B** |
| Layers | 24 | 24 |
| Channels | 1024 | 1536 |
| Coord quantization | 128 | 1024 |
| Max faces | 4,096 | **64,000** |
| Max sequence length | ~36,864 tokens | ~576,000 tokens |
| Training chunk size | full | 8,192 tokens |
| Point cloud size | 8,192 | 16,384 |
| Training data | - | **700k meshes** (curated, licensed) |
| Training setup | - | DDP (simple distributed data parallel) |
| Inference speed | - | ~140 tokens/sec |

The dataset is curated from a major 3D content provider, filtered to remove non-artist meshes (scanned, reconstructed, decimated, CAD-produced). This is proprietary data, not publicly available.

---

## Key Innovation

Meshtron demonstrates that scaling autoregressive mesh generation to 64k faces (40x more than prior art) and 1024-level coordinate resolution (8x higher) is achievable not through better tokenization or compression, but through **architectural efficiency** (Hourglass Transformer exploiting mesh structure) combined with **truncated training + sliding window inference**, which decouples training cost from generation length while maintaining or improving quality.

---

## Main Limitations

1. **Inference speed**: At 140 tokens/sec, generating a 32k-face mesh (~288k tokens) takes ~34 minutes. Large meshes remain slow.
2. **Low-level conditioning only**: Point cloud conditioning cannot add significant detail to degraded inputs (e.g., from marching-cube text-to-3D generators). No text or image conditioning.
3. **Robustness**: Occasional failures during long-sequence inference — missing parts, holes, or non-termination despite ordering enforcement.
4. **Data scarcity**: Relies on 700k proprietary artist-created meshes. The training data is not publicly available, hindering reproducibility.
5. **No end-to-end integration**: The point cloud encoder and mesh generator are trained jointly, but there is no integration with upstream 3D generation (e.g., text-to-3D pipelines).
6. **Sequential generation only**: No parallel token prediction — each of the 9N tokens is generated one at a time.
