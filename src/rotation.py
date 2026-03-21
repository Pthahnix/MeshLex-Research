"""Quaternion rotation encode/decode utilities for MeshLex."""

import numpy as np
from scipy.spatial.transform import Rotation


def quantize_rotation(Vt: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Quantize a 3x3 rotation matrix to 4 integer bins.

    Args:
        Vt: (3, 3) rotation matrix (from SVD/PCA).
        n_bins: number of quantization bins per component.

    Returns:
        (4,) int array with values in [0, n_bins).
    """
    rot = Rotation.from_matrix(Vt)
    quat = rot.as_quat()  # [x, y, z, w]

    # Canonical form: ensure w >= 0
    if quat[3] < 0:
        quat = -quat

    bins = ((quat + 1.0) / 2.0 * n_bins).astype(int).clip(0, n_bins - 1)
    return bins


def dequantize_rotation(bins: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Dequantize 4 integer bins back to a 3x3 rotation matrix.

    Args:
        bins: (4,) int array with values in [0, n_bins).
        n_bins: number of quantization bins per component.

    Returns:
        (3, 3) rotation matrix.
    """
    quat = (bins + 0.5) / n_bins * 2.0 - 1.0

    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.eye(3)

    quat = quat / norm
    rot = Rotation.from_quat(quat)  # [x, y, z, w]
    return rot.as_matrix()
