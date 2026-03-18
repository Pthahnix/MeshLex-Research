<!-- markdownlint-disable -->
# SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-Scale 3D VQVAE

## Core Architecture

**Type: Pure Autoregressive (next-scale prediction)**

SAR3D follows the VAR (Visual AutoRegressive) paradigm, performing next-scale prediction on discrete triplane tokens. It uses a standard GPT-style transformer with AdaLN layers. Unlike G3PT which redesigns the pooling/upsampling for unordered data, SAR3D uses triplane representations which inherently have spatial structure compatible with VAR's scale interpolation.

The system has three components:
1. **Multi-scale 3D VQVAE**: Tokenizes 3D objects into multi-scale discrete triplane tokens.
2. **SAR3D Transformer**: Autoregressive next-scale prediction for 3D generation.
3. **SAR3D-LLM**: Fine-tuned LLaMA for 3D understanding using truncated-scale tokens.

## How 3D Data is Tokenized

**Tokenization: Multi-scale 3D VQVAE with triplane latent space**

The tokenization pipeline is:

1. **Input**: 6 multi-view RGB-D renderings with Plucker camera embeddings. Each view is represented as [RGB | Depth | Plucker] = H x W x 10.
2. **Encoder**: Multi-view convolutional encoder produces a latent triplane f in R^{3 x h x w x C}.
3. **Multi-scale quantization**: The triplane is interpolated to 10 different scales: 3 x (1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 8^2, 10^2, 13^2, 16^2). Each sub-plane is independently quantized using a shared codebook.
4. **Codebook**: Size V=16384, dimension C=8, with L2-normalization for stability and high utilization.
5. **Decoder**: Converts quantized triplane features back to triplane representation, then renders multi-view images via volume rendering.

The triplane representation provides spatial inductive bias that makes it naturally compatible with VAR's scale-based interpolation, avoiding the unordered-data problem that G3PT had to solve with cross-attention.

## Multi-scale / Hierarchical Design

10 scales from 1x1 to 16x16 per plane (3 planes), giving a total token sequence that grows from 3 tokens (scale 1) to 3x256=768 tokens (scale 10). The total across all scales is 3 x (1+4+9+16+25+36+64+100+169+256) = 3 x 680 = 2040 tokens.

Each scale shares the same codebook, and quantization follows VAR's residual approach with interpolation between scales.

A notable finding: **truncated scales** (first K-2 of K=10 scales, i.e., only 37.5% of tokens) contain sufficient information for 3D understanding tasks. This means coarser scales capture semantic content while finer scales primarily add geometric/textural detail.

## Conditioning Mechanism

- **Image conditioning**: DINOv2 ViT-L extracts local patch features, injected via pre-cross-attention blocks. Additionally, pooled CLIP_T / DINOv2 features serve as the start token of the sequence.
- **Text conditioning**: CLIP_T ViT-L text encoder, with text embeddings injected through cross-attention in the transformer blocks.
- **Classifier-free guidance**: 10% condition dropout during training. At inference, logit interpolation: r_g = r_u + s(r_c - r_u).
- **3D Understanding**: Truncated 3D tokens are projected via MLP into LLaMA's token space and concatenated with text instruction tokens. Two-stage fine-tuning (projector alignment, then joint fine-tuning).

## Scale

- **Training data**: ~176K high-quality 3D instances from G-Objaverse, each with 40 random views.
- **VQVAE training**: 7 NVIDIA A100 GPUs, batch size 28.
- **Image-conditioned transformer**: 24 transformer blocks, 16 heads. Batch size 63.
- **Text-conditioned transformer**: 16 transformer blocks, 16 heads. Batch size 52.
- **SAR3D-LLM**: Vicuna-7B backbone. Stage-1 batch 140, stage-2 batch 112.
- **Model parameters**: Not explicitly stated for the generation transformer, but the LLM component is 7B (Vicuna).
- **Generation speed**: 0.82 seconds on A6000 GPU (image to mesh), 1.46 seconds for Flexicubes variant.

## Key Innovation

SAR3D is the first framework to unify fast 3D generation and 3D understanding in a single VQVAE, demonstrating that multi-scale discrete 3D tokens serve dual purposes: full-scale tokens for high-quality generation and truncated-scale tokens for LLM-based 3D captioning. The sub-second generation time (0.82s) is achieved through next-scale prediction on triplane tokens, which is substantially faster than diffusion-based alternatives, while the truncated-token insight enables efficient 3D-language alignment without separate encoders.

## Main Limitation

- **Geometry quality limited by volume rendering**: The triplane + volume rendering pipeline constrains geometric fidelity compared to point-cloud-based approaches (MAR-3D, G3PT). Flexicubes fine-tuning partially addresses this but introduces additional training complexity.
- **Scaling laws unverified**: Unlike G3PT, SAR3D does not demonstrate scaling behavior due to resource constraints. The authors speculate it would scale based on 2D VAR results but provide no evidence.
- **Two separate models for generation and understanding**: Despite sharing the VQVAE, the generation transformer and SAR3D-LLM are separate models. A truly unified multimodal model remains future work.
- **Relatively small training set**: 176K instances is modest compared to the datasets used by competitors like CLAY or G3PT.
- **No demonstrated advantage in geometric metrics**: The paper evaluates primarily on rendering quality (FID, KID, MUSIQ) and coverage metrics rather than the geometric metrics (IoU, Chamfer, F-score) that G3PT and MAR-3D report, making direct geometric quality comparison difficult.
