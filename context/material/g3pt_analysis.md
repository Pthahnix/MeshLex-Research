<!-- markdownlint-disable -->
# G3PT: Cross-Scale Querying Transformer for Autoregressive 3D Generation

## Core Architecture

**Type: Pure Autoregressive (next-scale prediction)**

G3PT is a fully autoregressive model that uses **next-scale prediction** rather than next-token prediction. The key insight is that 3D data naturally exhibits level-of-detail characteristics, so instead of imposing an artificial sequential order on unordered 3D tokens, G3PT creates a sequential relationship across scales (coarse to fine).

The generation process starts from the coarsest scale (1 token) and the transformer predicts each subsequent finer-scale token map conditioned on all previously generated coarser scales. This follows the VAR (Visual AutoRegressive) paradigm but replaces the image-specific pooling/interpolation with cross-attention mechanisms suitable for unordered 3D data.

The AR transformer itself adopts a GPT-2-style decoder-only architecture with causal masking across scales. Queries and keys are normalized to unit vectors before attention for training stability.

## How 3D Data is Tokenized

**Tokenization: Cross-scale Vector Quantization (CVQ) with Lookup-Free Quantization (LFQ)**

The tokenizer is a **Cross-scale Querying Transformer (CQT)** that produces discrete tokens:

1. **Encoder**: Point cloud (N=16384 points, 3 position + 3 normal features) is encoded via cross-attention with learnable latent queries (L=2304, C=512), followed by 12 self-attention layers.
2. **Cross-scale quantization**: At each scale s, learnable "downsample" queries e_s compress the residual features Z_s into fewer tokens E_s via cross-attention, then LFQ quantizes them. "Upsample" queries reconstruct back to the original dimension via another cross-attention. The residual Z_{s+1} = Z_s - Z_tilde_s feeds into the next scale.
3. **Decoder**: 16 self-attention layers + cross-attention with query points to predict occupancy values.

The codebook size is 8192 (LFQ). The critical innovation over VAR is replacing average pooling / bilinear interpolation (which assume spatial order) with cross-attention-based downsampling/upsampling that handles unordered tokens globally.

## Multi-scale / Hierarchical Design

G3PT decomposes features into S scales with varying token counts (L^(1), L^(2), ..., L^(S)), where L^(1) = 1 at the coarsest level. The total token budget across all scales sums to 2408 in the best configuration.

The hierarchy operates via residual quantization: each scale captures the residual information not represented by coarser scales. The cross-attention mechanism globally "pools" and "upsamples" between scales without requiring token ordering.

During AR training, a progressive strategy is used: training begins with tokens up to scale S/2 and gradually includes finer scales, which improves convergence and stability.

The CQT tokenizer is trained in two phases: first the encoder-decoder without quantization (with layer normalization between), then fine-tuning with the quantization layers.

## Conditioning Mechanism

- **Image conditioning**: DINOv2 (ViT-L/14) extracts image tokens, which are projected to match the 3D token channel dimension and concatenated with cross-scale 3D tokens. An attention mask ensures only subsequent 3D tokens are predicted (causal structure).
- **Text conditioning**: CLIP text encoder extracts semantic tokens, injected via adaLN (adaptive layer normalization) to control the generation signal.
- The authors note this is a "basic attempt" and more sophisticated conditioning methods could be explored.

## Scale

- **Model sizes**: 0.1B, 0.5B, and 1.5B parameter variants tested.
- **Training**: 1.5B model trained for 2 weeks on 136 NVIDIA H20 GPUs (96GB each).
- **Tokenizer training**: 60,000 steps on 8 NVIDIA A100 GPUs (80GB).
- **Data**: Objaverse dataset (point clouds with 16384 points each).
- **Scaling law**: For the first time in 3D generation, G3PT demonstrates clear power-law scaling behavior -- test cross-entropy loss decreases predictably with increasing model parameters.

**Benchmark results (1.5B)**: IoU 87.6, Chamfer 0.013, F-score 83.0 -- substantially outperforming all baselines including CLAY (0.5B, diffusion), CRM, InstantMesh, and other LRM/diffusion methods.

## Key Innovation

G3PT is the first work to demonstrate scaling-law behavior in 3D generation, showing that autoregressive 3D models follow predictable power-law improvement with increasing parameters. The Cross-scale Querying Transformer (CQT) solves the fundamental incompatibility between next-scale autoregressive prediction and unordered 3D data by replacing spatial pooling/interpolation with cross-attention-based downsampling and upsampling, enabling global information integration without imposing token order.

## Main Limitation

- Requires substantial computational resources: the 1.5B model needs 136 H20 GPUs for 2 weeks of training, making reproduction difficult for most research groups.
- The conditioning mechanism is acknowledged as basic; the paper does not deeply explore multi-modal or complex conditioning strategies.
- The tokenizer uses discrete VQ (LFQ), which inherently introduces quantization loss compared to continuous latent approaches. The CQT mitigates this with high codebook utilization (97-99%) but the gap to continuous VAE reconstruction quality persists (Table 2).
- No texture generation -- only geometry (occupancy/SDF). Texturing requires a separate external method (SyncMVD).
- Scalability verification is limited to 1.5B parameters; behavior at larger scales remains unverified.
