<!-- markdownlint-disable -->
# Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer

## Core Architecture and Representation

Direct3D uses an **explicit triplane latent representation** with a two-component architecture:

- **D3D-VAE (Variational Auto-Encoder)**:
  - **Encoder**: Transformer-based point-to-latent encoder. Takes 81,920 surface points (position + normal) with Fourier positional features. Uses 1 cross-attention layer + 8 self-attention layers (12 heads, dim 64) to project point cloud features into learnable latent tokens.
  - **Latent space**: Explicit triplane z of shape (3 x r x r) x d_z, where r=32, d_z=16. Three planes concatenated vertically along height to prevent channel blending (inspired by RODIN).
  - **Decoder**: CNN-based, upsamples latent triplane from 32x32 to 256x256 resolution using 5 ResNet blocks. Channel dimension 32.
  - **Geometry mapping**: 5-layer MLP (hidden dim 64) predicts occupancy from triplane-interpolated features.
  - **Semi-continuous surface sampling**: Novel supervision strategy — when query points are near the surface (within s=1/512), occupancy transitions smoothly between 0 and 1 using SDF-based interpolation, rather than abrupt binary labels. This avoids gradient issues at surface boundaries.
  - **Loss**: BCE loss + KL regularization (weight 1e-6). Trained 100K steps on A100 80G.

- **D3D-DiT (Diffusion Transformer)**:
  - Based on DiT-XL/2 architecture: 28 DiT blocks, 16 attention heads, dim 72.
  - Operates on the flattened triplane latent tokens.
  - 1000 denoising steps (linear variance 1e-4 to 2e-2). Trained 800K steps with AdamW, lr=1e-4.
  - Inference: 50-step DDIM, guidance scale 7.5.

## Model Scale and Training Data

- **D3D-VAE**: Transformer encoder (1 cross-attn + 8 self-attn layers) + CNN decoder (5 ResNet blocks) + MLP geometry head. Exact parameter count not stated.
- **D3D-DiT**: DiT-XL/2 configuration (28 layers). Estimated ~675M parameters based on the standard DiT-XL/2 spec.
- **Training data**: Trained on large-scale 3D datasets (Objaverse-XL referenced as primary source with 10M shapes). Exact subset size not specified.
- **Training compute**: A100 80G GPUs; 100K steps for VAE, 800K steps for DiT.

## Conditioning Mechanism

Dual image conditioning at two semantic levels, injected into every DiT block:

1. **Pixel-level alignment**: Pretrained DINOv2 (ViT-L/14) extracts structural image tokens. Projected via 2 linear layers + GeLU, then concatenated with noisy latent tokens for joint self-attention in each DiT block. Image tokens are discarded after attention — only latent tokens proceed.

2. **Semantic-level alignment**: Pretrained CLIP (ViT-L/14) extracts semantic image tokens. Injected via cross-attention in each DiT block. The CLIP classification token is added to the timestep embedding (replacing class embedding from original DiT).

3. **Timestep conditioning**: adaLN-single (from PixArt-alpha) — predicts global shift/scale parameters from time embedding, with per-block trainable offsets.

4. **Classifier-free guidance**: 10% random zeroing of both pixel and semantic conditions during training.

## Key Innovation

Direct3D is (to its knowledge) the first native 3D generative model that directly generates 3D shapes from single images without multi-view diffusion or SDS optimization, using a triplane latent space with direct 3D geometric supervision (semi-continuous surface sampling) rather than indirect rendering-based losses, enabling scalable training on large 3D datasets with strong generalization to in-the-wild images.

## Main Limitations

- **Single objects only**: Cannot generate large-scale scenes; limited to individual or multiple discrete objects.
- **Geometry only**: Produces untextured meshes (occupancy-based). Requires separate texture synthesis (e.g., SyncMVD) for appearance.
- **Fixed triplane resolution**: The 32x32 latent triplane (upsampled to 256x256) imposes a resolution ceiling. Unlike sparse representations, the dense triplane cannot adaptively allocate capacity.
- **Image-conditioned only**: No native text-to-3D capability; relies on external text-to-image models (e.g., Hunyuan-DiT) for text-based generation.
- **Occupancy representation**: Binary/semi-continuous occupancy may lose fine surface details compared to SDF or other continuous representations.
- **No multi-format output**: Outputs only meshes via occupancy; no direct support for Gaussians or radiance fields.

## Quantitative Highlights

- User study (46 volunteers, 1-5 scale): Quality 4.41 vs next best 2.53 (InstantMesh); Consistency 4.35 vs 2.66.
- Qualitative comparisons show consistent superiority over Shap-E, Michelangelo, One-2-3-45, and InstantMesh on GSO dataset.
- Ablations confirm: (1) explicit triplane outperforms 1D implicit latent (Michelangelo-style), (2) semi-continuous sampling improves thin structure reconstruction, (3) D3D-DiT outperforms SD 1.5/2.1 U-Net architectures, (4) pixel-level alignment module improves detail consistency.
