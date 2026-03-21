import pytest
import numpy as np
import trimesh
import json
from pathlib import Path

from src.patch_dataset import process_and_save_patches, PatchDataset, PatchGraphDataset


def test_process_and_save(tmp_path):
    """Process a mesh and save patches as .npz files."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh_path = tmp_path / "meshes"
    mesh_path.mkdir()
    obj_path = mesh_path / "test_sphere.obj"
    mesh.export(str(obj_path))

    patch_dir = tmp_path / "patches"
    meta = process_and_save_patches(
        mesh_path=str(obj_path),
        mesh_id="test_sphere",
        output_dir=str(patch_dir),
    )

    assert meta["n_patches"] > 0
    npz_files = list(patch_dir.glob("test_sphere_patch_*.npz"))
    assert len(npz_files) == meta["n_patches"]

    # Verify npz contents
    data = np.load(str(npz_files[0]))
    assert "faces" in data
    assert "local_vertices" in data
    assert "centroid" in data
    assert "principal_axes" in data
    assert "scale" in data


def test_patch_dataset_loads(tmp_path):
    """PatchDataset should load .npz files and return torch tensors."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    obj_path = tmp_path / "sphere.obj"
    mesh.export(str(obj_path))

    patch_dir = tmp_path / "patches"
    process_and_save_patches(str(obj_path), "sphere", str(patch_dir))

    ds = PatchDataset(str(patch_dir))
    assert len(ds) > 0

    sample = ds[0]
    assert "face_features" in sample   # (F, 15) input features
    assert "edge_index" in sample      # (2, E) face adjacency
    assert "local_vertices" in sample  # (V, 3) target
    assert "n_vertices" in sample      # int
    assert "n_faces" in sample         # int


def test_graph_dataset_and_loader(tmp_path):
    """PatchGraphDataset should work with PyG DataLoader."""
    from torch_geometric.loader import DataLoader

    mesh = trimesh.creation.icosphere(subdivisions=3)
    obj_path = tmp_path / "sphere.obj"
    mesh.export(str(obj_path))

    patch_dir = tmp_path / "patches"
    process_and_save_patches(str(obj_path), "sphere", str(patch_dir))

    ds = PatchGraphDataset(str(patch_dir))
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert hasattr(batch, "x")           # (N_total, 15) face features
    assert hasattr(batch, "edge_index")  # (2, E_total)
    assert hasattr(batch, "batch")       # (N_total,) batch vector
    assert hasattr(batch, "gt_vertices") # (B, max_V, 3)
    assert hasattr(batch, "n_vertices")  # (B,)
    assert batch.gt_vertices.shape[0] == 4


def test_patch_graph_dataset_nopca(tmp_path):
    """PatchGraphDataset with use_nopca=True loads local_vertices_nopca."""
    import numpy as np
    verts_pca = np.random.randn(30, 3).astype(np.float32)
    verts_nopca = np.random.randn(30, 3).astype(np.float32) * 2
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    np.savez(tmp_path / "test_patch_000.npz",
             local_vertices=verts_pca, local_vertices_nopca=verts_nopca,
             faces=faces, boundary_vertices=np.array([0]))
    from src.patch_dataset import PatchGraphDataset
    ds_pca = PatchGraphDataset(str(tmp_path), use_nopca=False)
    ds_nopca = PatchGraphDataset(str(tmp_path), use_nopca=True)
    assert len(ds_pca) == 1
    assert len(ds_nopca) == 1


def test_mesh_sequence_dataset_rotation(tmp_path):
    """MeshSequenceDataset with use_rotation=True produces 11-token sequences."""
    import numpy as np
    M = 5
    np.savez(tmp_path / "test_sequence.npz",
             centroids=np.random.randn(M, 3).astype(np.float32),
             scales=np.random.rand(M).astype(np.float32) + 0.1,
             tokens=np.random.randint(0, 1024, (M, 3)),
             principal_axes=np.tile(np.eye(3), (M, 1, 1)).astype(np.float32))
    from src.patch_dataset import MeshSequenceDataset
    ds = MeshSequenceDataset(str(tmp_path), mode="rvq", max_seq_len=1430, use_rotation=True)
    input_ids, target_ids = ds[0]
    assert input_ids.shape == (1430,)
    assert (input_ids[:54] != 0).any()
