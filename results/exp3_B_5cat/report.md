# Exp3: B-stage x 5-Category Results

> **Date**: 2026-03-09
> **Model**: MeshLex VQ-VAE with multi-token KV decoder (num_kv_tokens=4)
> **Training**: 200 epochs, resumed from A-stage checkpoint
> **Checkpoint**: `data/checkpoints/5cat_B/checkpoint_final.pt`

## Training Summary

| Parameter | Value |
|-----------|-------|
| Base checkpoint | `data/checkpoints/5cat_v2/checkpoint_final.pt` (A-stage) |
| New parameters | `decoder.kv_proj.weight`, `decoder.kv_proj.bias` |
| Total params | 1,127,043 (602,755 trainable) |
| Rotation trick | Disabled (caused collapse with SimVQ) |
| num_kv_tokens | 4 (vs 1 in A-stage) |
| Batch size | 256 |
| LR | 1e-4 (cosine annealing) |
| Training time | ~2.5 hours (RTX 4090) |

## Key Results

### A-stage vs B-stage Comparison

| Metric | A-stage (Exp1) | B-stage (Exp3) | Change |
|--------|---------------|---------------|--------|
| Same-cat CD | 238.3 | **223.5** | **-6.2%** |
| Cross-cat CD | 272.8 | **264.8** | **-3.0%** |
| CD Ratio | 1.14x | 1.18x | +0.04 |
| Train Utilization | 99.7% | 99.0% | -0.7% |
| Eval Utilization | 46.0% | 47.1% | +1.1% |
| Train Recon Loss | 0.228 | **0.209** | **-8.3%** |
| Decision | STRONG GO | **STRONG GO** | -- |

### Key Observations

1. **Multi-token KV decoder improves reconstruction quality**: CD dropped 6.2% for same-category and 3.0% for cross-category, confirming that expanding the single KV token to 4 tokens prevents cross-attention degeneracy.

2. **Rotation trick incompatible with SimVQ**: Two separate attempts with `use_rotation=True` both caused rapid collapse (util drops from 99% to <3% within 7 epochs). SimVQ's frozen-C mechanism already prevents collapse; the rotation trick's different gradient dynamics destabilize training.

3. **Cross-stage resume works well**: Loading A-stage weights into B-stage model (with `strict=False` for new `kv_proj` parameters) preserved the learned codebook and encoder, allowing the decoder to quickly adapt.

4. **Utilization stable at 99%**: The codebook remains highly utilized throughout B-stage training, confirming that the SimVQ fix is robust.

## Decision: STRONG GO

CD Ratio 1.18x (< 1.2 threshold) with utilization 47.1% (> 30% threshold).

**Next step**: Proceed to LVIS-Wide experiments (Exp2 A-stage + Exp4 B-stage) to validate universality at scale.

## Visualizations

- `training_curves.png`: Loss and utilization over 200 epochs
- `codebook_tsne.png`: t-SNE of effective codebook (CW space)
- `utilization_histogram.png`: Code usage distribution

## Failed Attempt: Rotation Trick

Two rotation trick attempts both failed:

**Attempt 1** (from scratch, no resume):
- Epochs 0-9: util ~0.8% (encoder warmup)
- Epoch 10+: K-means init → util still drops → total collapse at 0.2%

**Attempt 2** (resume from A-stage):
- Epoch 0: util 99.5% (inherited from A-stage)
- Epoch 5: util 64.2%
- Epoch 6: util 17.5%
- Epoch 7: util 2.2% → killed

**Root cause**: Rotation trick provides different gradient flow than STE. In SimVQ, C is frozen and only W learns. The rotation trick likely disrupts the W-learning dynamic that SimVQ relies on for stability.
