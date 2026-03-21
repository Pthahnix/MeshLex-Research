# tests/test_mdlm_model.py
"""Tests for src.mdlm_model -- full-scale MDLM."""
import torch
import pytest


def test_mdlm_forward_shape():
    """MDLM forward returns correct logit shape."""
    from src.mdlm_model import FullMDLM
    model = FullMDLM(vocab_size=1025, max_seq_len=240, d_model=128,
                     n_heads=4, n_layers=2)
    B, L = 2, 240
    x = torch.randint(0, 1025, (B, L))
    t = torch.rand(B)
    padding_mask = torch.ones(B, L, dtype=torch.bool)
    logits = model(x, t, padding_mask)
    assert logits.shape == (B, L, 1024)  # predicts real tokens only (not MASK)


def test_mdlm_masking():
    """MDLM masking produces correct mask counts."""
    from src.mdlm_model import apply_masking
    B, L = 4, 100
    tokens = torch.randint(0, 1024, (B, L))
    padding_mask = torch.ones(B, L, dtype=torch.bool)
    t = torch.full((B,), 0.5)  # 50% mask rate

    masked, mask_positions = apply_masking(tokens, t, padding_mask, mask_token=1024)
    # About 50% should be masked
    mask_rate = mask_positions.float().mean().item()
    assert 0.3 < mask_rate < 0.7, f"Mask rate {mask_rate} not near 0.5"
    # Masked positions should have mask_token
    assert (masked[mask_positions] == 1024).all()


def test_mdlm_generate_shape():
    """MDLM generate produces valid token sequences."""
    from src.mdlm_model import FullMDLM
    model = FullMDLM(vocab_size=1025, max_seq_len=60, d_model=64,
                     n_heads=2, n_layers=1)
    model.eval()
    seqs = model.generate(n_samples=2, seq_len=60, n_steps=10)
    assert seqs.shape == (2, 60)
    assert (seqs >= 0).all()
    assert (seqs < 1024).all()  # no MASK tokens in output
