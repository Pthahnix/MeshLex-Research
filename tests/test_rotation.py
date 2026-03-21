"""Tests for src/rotation.py — quaternion rotation encode/decode."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.rotation import dequantize_rotation, quantize_rotation

N_BINS = 64


def test_quantize_rotation_output_shape():
    """Output shape is (4,) with values in [0, n_bins)."""
    Vt = np.eye(3)
    bins = quantize_rotation(Vt, n_bins=N_BINS)
    assert bins.shape == (4,)
    assert np.all(bins >= 0)
    assert np.all(bins < N_BINS)


def test_dequantize_rotation_returns_valid_matrix():
    """Dequantized result is a 3x3 orthogonal matrix (det ~+1)."""
    bins = np.array([10, 20, 30, 50])
    mat = dequantize_rotation(bins, n_bins=N_BINS)
    assert mat.shape == (3, 3)
    # R^T R ≈ I
    np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-6)
    # det ≈ +1
    np.testing.assert_allclose(np.linalg.det(mat), 1.0, atol=1e-6)


def test_roundtrip_identity():
    """Identity matrix survives quantize → dequantize."""
    Vt = np.eye(3)
    bins = quantize_rotation(Vt, n_bins=N_BINS)
    recovered = dequantize_rotation(bins, n_bins=N_BINS)
    np.testing.assert_allclose(recovered, np.eye(3), atol=0.1)


def test_roundtrip_random_rotation():
    """20 random rotations have angular error < 5° after roundtrip."""
    rng = np.random.default_rng(42)
    max_err_deg = 0.0
    for _ in range(20):
        rot = Rotation.random(random_state=rng)
        Vt = rot.as_matrix()
        bins = quantize_rotation(Vt, n_bins=N_BINS)
        recovered = dequantize_rotation(bins, n_bins=N_BINS)
        # Angular error via relative rotation
        rel = Rotation.from_matrix(recovered @ Vt.T)
        angle_deg = np.degrees(rel.magnitude())
        max_err_deg = max(max_err_deg, angle_deg)
        assert angle_deg < 5.0, f"Angular error {angle_deg:.2f}° >= 5°"
    print(f"Max angular error across 20 rotations: {max_err_deg:.2f}°")


def test_canonical_form_w_positive():
    """180° rotation produces w bin >= n_bins/2 (canonical w >= 0)."""
    # 180° about z-axis: quat = [0, 0, 1, 0] or [0, 0, -1, 0]
    # Canonical: w >= 0, so w = 0 → bin = 32 for n_bins=64
    Vt = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
    bins = quantize_rotation(Vt, n_bins=N_BINS)
    # w is the 4th component (index 3)
    assert bins[3] >= N_BINS // 2, f"w bin {bins[3]} < {N_BINS // 2}"
