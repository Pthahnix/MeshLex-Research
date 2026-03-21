"""Patch serialization to .npz and PyTorch Dataset for training."""
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset
from pathlib import Path

from torch_geometric.data import Data as _PyGData
from src.patch_segment import segment_mesh_to_patches


def compute_face_features(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute 15-dim face features: 9 vertex coords + 3 normal + 3 edge angles.

    Fully vectorized — no Python loops.

    Args:
        vertices: (V, 3) local normalized vertex coordinates
        faces: (F, 3) face vertex indices

    Returns:
        (F, 15) feature array
    """
    # Gather triangle vertices: (F, 3) each
    p0 = vertices[faces[:, 0]]  # (F, 3)
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]

    features = np.empty((faces.shape[0], 15), dtype=np.float32)

    # 9 vertex coordinates (flattened)
    features[:, 0:3] = p0
    features[:, 3:6] = p1
    features[:, 6:9] = p2

    # Face normals
    e1 = p1 - p0  # (F, 3)
    e2 = p2 - p0
    normals = np.cross(e1, e2)  # (F, 3)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)  # (F, 1)
    norm_len = np.maximum(norm_len, 1e-8)
    features[:, 9:12] = normals / norm_len

    # Edge vectors for angle computation
    # edges: e01 = p1-p0, e12 = p2-p1, e20 = p0-p2
    e01 = p1 - p0  # (F, 3)
    e12 = p2 - p1
    e20 = p0 - p2

    # Interior angles: angle at v0 = angle between -e20 and e01
    #                   angle at v1 = angle between -e01 and e12
    #                   angle at v2 = angle between -e12 and e20
    def _vec_angle(a, b):
        """Vectorized angle between pairs of vectors."""
        dot = np.sum(a * b, axis=1)  # (F,)
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        cos = dot / (na * nb + 1e-8)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    features[:, 12] = _vec_angle(-e20, e01)  # angle at v0
    features[:, 13] = _vec_angle(-e01, e12)  # angle at v1
    features[:, 14] = _vec_angle(-e12, e20)  # angle at v2

    return features


def build_face_edge_index(faces: np.ndarray) -> np.ndarray:
    """Build face adjacency edge_index (2, E) from face array.

    Two faces are adjacent if they share exactly 2 vertices (an edge).
    Vectorized using numpy sorting + groupby.
    """
    n_faces = faces.shape[0]
    if n_faces == 0:
        return np.zeros((2, 0), dtype=np.int64)

    # Sort vertex indices within each face
    sorted_faces = np.sort(faces, axis=1)  # (F, 3)

    # Extract all 3 edges per face: (v_lo, v_hi, face_idx)
    # Edge pairs: (0,1), (0,2), (1,2)
    face_ids = np.arange(n_faces)
    edges = np.empty((n_faces * 3, 3), dtype=np.int64)
    edges[0::3, 0] = sorted_faces[:, 0]
    edges[0::3, 1] = sorted_faces[:, 1]
    edges[0::3, 2] = face_ids
    edges[1::3, 0] = sorted_faces[:, 0]
    edges[1::3, 1] = sorted_faces[:, 2]
    edges[1::3, 2] = face_ids
    edges[2::3, 0] = sorted_faces[:, 1]
    edges[2::3, 1] = sorted_faces[:, 2]
    edges[2::3, 2] = face_ids

    # Sort by (v_lo, v_hi) to group shared edges
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]

    # Find consecutive rows with same (v0, v1) — these share an edge
    same = (edges[:-1, 0] == edges[1:, 0]) & (edges[:-1, 1] == edges[1:, 1])
    idx = np.where(same)[0]

    if len(idx) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    f0 = edges[idx, 2]
    f1 = edges[idx + 1, 2]

    # Bidirectional
    src = np.concatenate([f0, f1])
    dst = np.concatenate([f1, f0])
    return np.stack([src, dst], axis=0).astype(np.int64)


def process_and_save_patches(
    mesh_path: str,
    mesh_id: str,
    output_dir: str,
    target_patch_faces: int = 35,
) -> dict:
    """Segment a mesh and save each patch as .npz."""
    mesh = trimesh.load(mesh_path, force="mesh")
    patches = segment_mesh_to_patches(mesh, target_patch_faces=target_patch_faces)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, patch in enumerate(patches):
        np.savez_compressed(
            str(out / f"{mesh_id}_patch_{i:03d}.npz"),
            faces=patch.faces,
            vertices=patch.vertices,
            local_vertices=patch.local_vertices,
            centroid=patch.centroid,
            principal_axes=patch.principal_axes,
            scale=np.array([patch.scale]),
            boundary_vertices=np.array(patch.boundary_vertices),
            global_face_indices=patch.global_face_indices,
        )

    return {
        "mesh_id": mesh_id,
        "n_patches": len(patches),
        "face_counts": [p.faces.shape[0] for p in patches],
    }


class PatchDataset(Dataset):
    """PyTorch Dataset that loads .npz patch files."""

    MAX_FACES = 80
    MAX_VERTICES = 128

    def __init__(self, patch_dir: str):
        self.patch_dir = Path(patch_dir)
        self.files = sorted(self.patch_dir.glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        faces = data["faces"]
        local_verts = data["local_vertices"]

        # Face features (F, 15)
        face_feats = compute_face_features(local_verts, faces)

        # Face adjacency edge_index (2, E)
        edge_index = build_face_edge_index(faces)

        n_faces = faces.shape[0]
        n_verts = local_verts.shape[0]

        # Pad to fixed size for batching
        padded_feats = np.zeros((self.MAX_FACES, 15), dtype=np.float32)
        padded_feats[:n_faces] = face_feats

        padded_verts = np.zeros((self.MAX_VERTICES, 3), dtype=np.float32)
        padded_verts[:n_verts] = local_verts

        return {
            "face_features": torch.tensor(padded_feats),
            "edge_index": torch.tensor(edge_index),
            "local_vertices": torch.tensor(padded_verts),
            "n_faces": n_faces,
            "n_vertices": n_verts,
        }


class PatchGraphDataset(Dataset):
    """PyTorch Geometric compatible dataset. Returns Data objects
    with graph structure for SAGEConv + padded vertex targets for decoder.
    """

    MAX_VERTICES = 128

    def __init__(self, patch_dir: str, use_nopca: bool = False):
        self.patch_dir = Path(patch_dir)
        self.files = sorted(self.patch_dir.glob("*.npz"))
        self.use_nopca = use_nopca

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        faces = data["faces"]
        if self.use_nopca and "local_vertices_nopca" in data:
            local_verts = data["local_vertices_nopca"].astype(np.float32)
        else:
            local_verts = data["local_vertices"].astype(np.float32)

        # Face features (F, 15)
        face_feats = compute_face_features(local_verts, faces)

        # Face adjacency graph
        edge_index = build_face_edge_index(faces)

        n_verts = local_verts.shape[0]
        n_faces = faces.shape[0]

        # Pad vertices to max size
        padded_verts = np.zeros((self.MAX_VERTICES, 3), dtype=np.float32)
        padded_verts[:n_verts] = local_verts

        return PatchData(
            x=torch.tensor(face_feats, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            gt_vertices=torch.tensor(padded_verts, dtype=torch.float32),
            n_vertices=torch.tensor(n_verts, dtype=torch.long),
            n_faces=torch.tensor(n_faces, dtype=torch.long),
        )


class PatchData(_PyGData):
    """PyG Data subclass that stacks gt_vertices instead of concatenating."""

    def __cat_dim__(self, key, value, *args, **kw):
        if key in ("gt_vertices",):
            return None  # stack instead of cat
        return super().__cat_dim__(key, value, *args, **kw)


class ParquetPatchDataset(Dataset):
    """PyG-compatible dataset reading from Arrow datasets or Parquet files.

    Supports two modes:
    1. Pre-computed Arrow: `arrow_dir` points to a datasets.save_to_disk() output
    2. Raw Parquet: `parquet_dir` points to *.parquet files (slower first load)
    """

    MAX_VERTICES = 128

    def __init__(self, arrow_dir: str = None, parquet_dir: str = None,
                 use_nopca: bool = False, split_mesh_ids: set = None):
        """
        Args:
            arrow_dir: Pre-computed Arrow dataset directory (fast load).
            parquet_dir: Raw parquet files directory (needs filtering).
            use_nopca: If True, use local_vertices_nopca.
            split_mesh_ids: Filter by mesh IDs (only for parquet_dir mode).
        """
        from datasets import load_from_disk, load_dataset as hf_load
        self.use_nopca = use_nopca

        if arrow_dir and Path(arrow_dir).exists():
            self._ds = load_from_disk(arrow_dir)
        elif parquet_dir:
            parquet_files = sorted(str(f) for f in Path(parquet_dir).glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files in {parquet_dir}")
            self._ds = hf_load("parquet", data_files=parquet_files, split="train")
            if split_mesh_ids is not None:
                self._ds = self._ds.filter(
                    lambda batch: [mid in split_mesh_ids for mid in batch["mesh_id"]],
                    batched=True, batch_size=10000, num_proc=4,
                )
        else:
            raise ValueError("Must specify either arrow_dir or parquet_dir")

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        row = self._ds[idx]

        n_verts = row.get("n_verts", 30)
        n_faces = row.get("n_faces", 20)
        faces = np.array(row["faces"], dtype=np.int64).reshape(-1, 3)[:n_faces]

        if self.use_nopca and row.get("local_vertices_nopca"):
            local_verts = np.array(row["local_vertices_nopca"], dtype=np.float32).reshape(-1, 3)[:n_verts]
        else:
            local_verts = np.array(row["local_vertices"], dtype=np.float32).reshape(-1, 3)[:n_verts]

        face_feats = compute_face_features(local_verts, faces)
        edge_index = build_face_edge_index(faces)

        padded_verts = np.zeros((self.MAX_VERTICES, 3), dtype=np.float32)
        padded_verts[:n_verts] = local_verts

        return PatchData(
            x=torch.tensor(face_feats, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            gt_vertices=torch.tensor(padded_verts, dtype=torch.float32),
            n_vertices=torch.tensor(n_verts, dtype=torch.long),
            n_faces=torch.tensor(n_faces, dtype=torch.long),
        )


class MeshSequenceDataset(Dataset):
    """Dataset that returns full-mesh patch token sequences for AR training.

    Each item = one mesh's patch sequence (all patches in Z-order).
    Requires pre-computed codebook indices from a trained VQ-VAE.

    Expects a directory with files like: {mesh_id}_sequence.npz
    Each NPZ contains: centroids (M,3), scales (M,), tokens (M,) or (M,3)
    """

    def __init__(self, sequence_dir: str, mode: str = "rvq", max_seq_len: int = 1024,
                 use_rotation: bool = False):
        self.sequence_dir = Path(sequence_dir)
        self.files = sorted(self.sequence_dir.glob("*_sequence.npz"))
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.use_rotation = use_rotation

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(str(self.files[idx]))
        centroids = data["centroids"]
        scales = data["scales"]
        tokens = data["tokens"]

        if self.use_rotation and "principal_axes" in data:
            from src.patch_sequence import patches_to_token_sequence_rot
            rotations = data["principal_axes"]
            seq = patches_to_token_sequence_rot(centroids, scales, rotations, tokens)
        else:
            from src.patch_sequence import patches_to_token_sequence
            seq = patches_to_token_sequence(centroids, scales, tokens, mode=self.mode)

        # Build input (pad with 0) and target (pad with -100) separately
        # Input = seq[:-1], Target = seq[1:] (teacher forcing)
        seq_len = min(len(seq), self.max_seq_len + 1)  # +1 for shift
        seq = seq[:seq_len]

        input_ids = np.zeros(self.max_seq_len, dtype=np.int64)
        target_ids = np.full(self.max_seq_len, -100, dtype=np.int64)

        input_ids[:seq_len - 1] = seq[:-1]
        target_ids[:seq_len - 1] = seq[1:]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


class MmapPatchDataset(Dataset):
    """Ultra-fast dataset reading pre-computed features from memory-mapped numpy arrays.

    Created by scripts/precompute_features.py. Returns PatchData objects compatible
    with the PyG DataLoader and existing Trainer.
    """

    def __init__(self, feature_dir: str):
        self.feature_dir = Path(feature_dir)
        self._feats = np.load(str(self.feature_dir / "face_features.npy"), mmap_mode="r")
        self._edges = np.load(str(self.feature_dir / "edge_index.npy"), mmap_mode="r")
        self._verts = np.load(str(self.feature_dir / "gt_vertices.npy"), mmap_mode="r")
        self._n_verts = np.load(str(self.feature_dir / "n_vertices.npy"), mmap_mode="r")
        self._n_faces = np.load(str(self.feature_dir / "n_faces.npy"), mmap_mode="r")
        self._n_edges = np.load(str(self.feature_dir / "n_edges.npy"), mmap_mode="r")

    def __len__(self):
        return len(self._n_verts)

    def __getitem__(self, idx):
        n_faces = int(self._n_faces[idx])
        n_edges = int(self._n_edges[idx])

        # Read pre-computed arrays (already padded)
        x = torch.from_numpy(self._feats[idx, :n_faces].copy())
        edge_index = torch.from_numpy(self._edges[idx, :, :n_edges].copy())
        gt_vertices = torch.from_numpy(self._verts[idx].copy())
        n_vertices = torch.tensor(int(self._n_verts[idx]), dtype=torch.long)

        return PatchData(
            x=x.float(),
            edge_index=edge_index.long(),
            gt_vertices=gt_vertices.float(),
            n_vertices=n_vertices,
            n_faces=torch.tensor(n_faces, dtype=torch.long),
        )
