<!-- markdownlint-disable -->
# OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation

**Paper**: Wang et al., 2025
**arXiv**: 2504.09975v2

---

## Core Architecture Design

OctGPT is a two-stage autoregressive model for 3D shape generation:

**Stage 1 — Octree-based VQVAE**: An asymmetric encoder-decoder that compresses 3D shapes into binary tokens at the finest octree level.
- **Encoder**: Octree-based CNNs (O-CNN) that compress the input octree into feature representations, reducing octree depth by 2. Features are quantized via Binary Spherical Quantization (BSQ) — no codebook needed.
- **Decoder**: Dual octree graph networks decode binary tokens into local SDFs, integrated via multi-level Partition-of-Unity into a global SDF. Final meshes extracted via marching cubes.

**Stage 2 — Autoregressive Transformer**: Predicts the serialized binary octree sequence token by token (or in parallel batches).
- Architecture: Stack of attention blocks with multi-head self-attention + FFN + LayerNorm + residual connections.
- Uses **octree-based window attention** (adapted from OctFormer) — tokens divided into fixed-size windows for local self-attention, achieving linear complexity.
- Alternates between **dilated octree attention** and **shifted window attention** to enable cross-window interactions.
- Unlike OctFormer (single-depth), OctGPT handles tokens from nodes at varying octree depths, enabling cross-scale interactions.

All predictions are binary classification tasks (split/no-split for octree nodes, 0/1 for quantized codes). Binary classifiers are attached to the transformer output.

---

## Tokenization / Representation Strategy

The representation is a **serialized multiscale binary sequence** with two components:

1. **Octree splitting signals**: For each octree node at depths 3 through (D-1), a binary value indicates whether the node splits (1) or not (0). Nodes are sorted in **z-order** (Morton code), which preserves spatial locality. The first 3 levels are fully populated (512 nodes at depth 3). Signals are concatenated across depths: O = (O^3, O^4, ..., O^{D-1}).

2. **VQVAE binary codes**: At the finest octree level (depth D), the VQVAE encoder produces feature vectors that are quantized into binary tokens via BSQ: q_i = sign(z_i / ||z_i||). These capture fine-grained geometric details that the octree structure alone cannot represent.

The final sequence concatenates octree signals + VQVAE codes: all binary (0/1) tokens. Total sequence length can exceed **50k tokens** (and up to 160k+ in some experiments).

Key design insight: Decomposing 3D shape prediction into binary classification tasks (inspired by chain-of-thought reasoning) dramatically improves convergence. The paper shows their method produces quality results after 10 epochs, while direct coordinate prediction fails even after 100 epochs.

---

## How Long Sequences Are Handled

Three mechanisms address the 50k+ token sequences:

1. **Octree-based window attention**: Tokens are divided into fixed-size windows along the z-order curve, and self-attention is computed only within each window. This reduces quadratic complexity to linear. Achieves **13x training speedup** over global attention at 160k+ token lengths.

2. **Dilated + shifted window attention**: Alternating between dilated octree attention (every k-th token from each window) and shifted window attention (overlapping windows) enables information flow across windows without global attention cost.

3. **Multi-token parallel generation (MAR-style)**: Adapted from Masked Autoregressive Models (Li et al., 2024). Multiple tokens are predicted in parallel within each depth level. A **depth-wise teacher-forcing mask** ensures hierarchical dependencies are preserved: tokens at deeper levels can see all shallower-level tokens, but not vice versa. During inference, tokens are generated depth-by-depth (depth 3 → depth D), with parallel prediction within each depth. This achieves **69x generation speedup** over single-token prediction at 40k+ token lengths.

The combination enables training on 4 NVIDIA 4090 GPUs (24GB each) and generating 1024^3 resolution shapes in under 30 seconds on a single 4090.

---

## Conditioning Mechanisms

OctGPT supports multiple conditioning modalities via cross-attention:

1. **Category conditioning**: Category labels for class-conditional generation (ShapeNet categories).

2. **Text conditioning**: CLIP text encoder extracts features from text prompts; integrated via cross-attention modules. Demonstrated on Text2Shape and Objaverse datasets.

3. **Sketch conditioning**: Pre-trained DINOv2 encodes sketch images; integrated via standard cross-attention (no view-aware attention needed). Simpler than competing approaches.

4. **Image conditioning**: Same architecture as sketch conditioning, using rendered images from ShapeNet during training.

5. **Scene-level generation**: Trained on the Synthetic Rooms dataset (5k scenes, 5 ShapeNet categories), generating multi-object scenes at 1024^3 resolution.

All conditioning signals are injected through cross-attention modules added to the transformer blocks.

---

## Model Scale

| Aspect | Details |
|--------|---------|
| Parameters | Not explicitly stated (described as "fewer parameters" than 3DILG) |
| Training hardware | **4 NVIDIA 4090 GPUs** (24GB each) |
| Training time | ~3 days for scene-level generation (200 epochs) |
| Max resolution | **1024^3** |
| Max token length | 50k-160k+ tokens |
| VQVAE binary code dimension | Not specified (BSQ-based, codebook-free) |
| Datasets | ShapeNet (5-13 categories), Objaverse, Synthetic Rooms |
| Training speedup | 13x over global attention |
| Generation speedup | 69x over single-token prediction |
| Generation time | <30 seconds for 1024^3 shapes on single 4090 |
| FID (chair, single-cat) | 31.05 (vs. MeshGPT 37.05, OctFusion 16.15) |
| FID (airplane, all-cat) | 29.27 (vs. 3DILG 54.38) |

The paper emphasizes accessibility: the entire system trains on consumer GPUs (4090s), making it practical for researchers without datacenter-scale compute.

---

## Key Innovation

OctGPT transforms 3D shape generation into a series of **binary classification tasks** over a serialized octree representation, where coarse geometry is captured by octree splitting signals and fine details by VQVAE binary codes — combined with octree-based efficient attention and parallel token prediction, this achieves 13x training and 69x generation speedups while producing quality that rivals or surpasses diffusion models, all trainable on 4 consumer GPUs.

---

## Main Limitations

1. **Two-stage pipeline**: VQVAE and autoregressive transformer are trained separately, not end-to-end. This may constrain overall performance since the tokenizer's errors propagate.

2. **Limited GPU resources in experiments**: While the 4090-accessibility is a feature, the paper acknowledges that scaling to more compute and larger datasets could further improve quality.

3. **Implicit field output**: Final output is SDF → marching cubes mesh, not direct mesh topology. This means the generated meshes have iso-surface-style tessellation, not artist-like topology (unlike Meshtron which produces artist-quality tessellation directly).

4. **No direct mesh generation**: The method generates occupancy/SDF fields, not meshes with controlled topology. Edge flow, quad dominance, and tessellation quality are not modeled.

5. **Binary-only tokens**: While binary classification simplifies training, it may limit the expressiveness of the representation compared to higher-cardinality vocabularies used in other approaches.

6. **Quality gap with best diffusion models**: While OctGPT surpasses diffusion models in some categories (Car, Rifle), it still trails OctFusion in others (Chair: 31.05 vs 16.15 FID). The gap is meaningful in categories with complex topology.

7. **Multi-modality not deeply explored**: Text/image/sketch conditioning is demonstrated but not systematically benchmarked against specialized conditional generation methods.
