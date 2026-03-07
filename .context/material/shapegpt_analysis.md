<!-- markdownlint-disable -->
# ShapeGPT: 3D Shape Generation with a Unified Multi-modal Language Model

## Core Architecture

**Type**: LLM-based (T5 encoder-decoder)

ShapeGPT is a unified multi-modal framework built on a pre-trained T5-base language model (220M parameters, 12-layer transformer encoder + 12-layer transformer decoder). It treats 3D shapes as a "foreign language" and integrates them into the LLM's token space alongside text and image tokens. The framework follows a sequence-to-sequence paradigm: multi-modal inputs (shapes, images, text) are encoded into a unified token sequence, and the LLM generates output sequences that may contain shape tokens, text tokens, or both.

The system has three main modules:
1. **3D VQ-VAE**: Encodes continuous SDF shape representations into discrete token sequences via vector quantization.
2. **Image Encoder**: CLIP ViT-Large-Patch14 with a 6-layer perceiver + linear layer for alignment.
3. **T5 Language Model**: Backbone for understanding instructions and generating multi-modal outputs.

## 3D Data Tokenization/Representation

Shapes are first converted to **Signed Distance Function (SDF)** representations at 64x64x64 resolution. The SDF is then discretized through a 3D VQ-VAE:

1. **Shape Encoder** compresses the SDF into a latent space, then unfolded into a 1D sequence along x-y-z axis order.
2. **Vector Quantization** maps latent vectors to the nearest entry in a learnable codebook of size 8192x64, producing discrete token indices.
3. **Shape Decoder** reconstructs the SDF from quantized tokens.

The resulting 512 shape tokens (from 8x8x8 latent grid) are then wrapped in a "word-sentence-paragraph" hierarchy:
- **Shape Words**: Each VQ token index j becomes a special token `<shape_id_j>`.
- **Shape Sentences**: Shape words are bracketed with start/end tokens `<shape_id_m>` and `<shape_id_m+1>`.
- **Multi-modal Paragraphs**: Shape sentences are embedded within natural language instructions alongside text and image features.

## Conditioning Mechanism

Conditioning is achieved through **natural language instructions** that mix modality placeholders:
- `<shape_place_holder>` is replaced with shape sentences.
- `<caption_place_holder>` is replaced with text descriptions.
- `<image_place_holder>` triggers concatenation with CLIP image embeddings.

The instruction template system supports arbitrary combinations of input modalities:
- Text-to-shape: Caption as input, shape as output.
- Image-to-shape: Image as input, shape as output.
- Multi-modal-to-shape: Caption + image as input, shape as output.
- Shape-to-text: Shape as input, caption as output.
- Shape editing: Shape + text instruction as input, modified shape as output.
- Shape completion: Partial shape as input, complete shape as output.

Hundreds of instruction templates were generated using ChatGPT for task diversity.

## Scale

- **Model size**: T5-base with 220M parameters. Also tested T5-small (60M).
- **Shape tokenizer**: Codebook size 8192x64, producing 512 tokens per shape.
- **Training data**: ShapeNet (16 categories, ~50K models). Text2Shape annotations for chairs/tables. Moat-generated tags for other categories.
- **Training compute**: 4x A100 GPUs. VQ-VAE: 315 epochs, lr=1e-4, batch 32. Multi-modal pretraining: 635 epochs, lr=4e-4, batch 24. Instruction fine-tuning: 315 epochs, lr=1e-4, batch 24.
- **Three-stage training**: (1) Shape representation (VQ-VAE), (2) Multi-modal alignment on chairs/tables with simple QA pairs, (3) Instruction-based generation on all categories with diverse prompts.

## Key Innovation

ShapeGPT unifies multiple 3D shape tasks (generation, captioning, editing, completion, reasoning) within a single LLM framework by discretizing shapes into VQ tokens that are treated as a "shape language" with explicit grammar (word-sentence-paragraph hierarchy), enabling instruction-driven multi-task shape manipulation through a pre-trained T5 model without task-specific architectures.

## Main Limitation

- **Low geometric resolution**: SDF representation at 64x64x64 with only 512 shape tokens severely limits geometric detail. The ablation shows 64 tokens (4x4x4) performs much worse, but higher resolutions are constrained by GPU memory.
- **Limited shape complexity**: Only validated on ShapeNet objects (simple CAD-like shapes). No evaluation on complex organic shapes or large-scale datasets like Objaverse.
- **No texture support**: Only generates geometry, not textured shapes.
- **Modest quantitative results**: On text-to-shape (ULIP 0.149 vs. Michelangelo's 0.165), ShapeGPT is competitive but not consistently SOTA. Its strength is versatility across tasks rather than excellence in any single one.

## Quantitative Highlights

- **Image-to-shape (chair)**: IoU 0.593, CD 1.221, F-score 0.424 -- competitive with SDFusion (0.595/1.323/0.412).
- **Text-to-shape**: ULIP 0.149 vs. SDFusion 0.105 and Michelangelo 0.165.
- **Multi-modal-to-shape**: IoU 0.587, ULIP 0.189 -- outperforms SDFusion on all metrics.
- **Shape caption**: CLIP similarity 0.812.
- **Ablation**: 512 tokens >> 64 tokens; T5-base >> T5-small; pre-training is essential for multi-modal alignment.
