"""Tests for split generation logic."""
import pytest


def _make_fake_metadata():
    meta = {}
    for cat_idx in range(10):
        cat_name = f"cat_{cat_idx:02d}"
        for mesh_idx in range(20):
            mesh_id = f"{cat_name}_mesh_{mesh_idx:03d}"
            meta[mesh_id] = {
                "category": cat_name,
                "source": "objaverse" if cat_idx < 5 else "shapenet",
                "n_patches": 30, "n_faces": 1000, "n_verts": 500,
            }
    return meta


def test_generate_splits_all_meshes_assigned():
    from scripts.generate_splits_daft import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    all_ids = set(splits["seen_train"] + splits["seen_test"] + splits["unseen"])
    assert all_ids == set(meta.keys())


def test_generate_splits_unseen_categories_excluded():
    from scripts.generate_splits_daft import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    unseen_cats = set(splits["unseen_categories"])
    for mesh_id in splits["seen_train"] + splits["seen_test"]:
        assert meta[mesh_id]["category"] not in unseen_cats


def test_generate_splits_test_ratio():
    from scripts.generate_splits_daft import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    n_seen = len(splits["seen_train"]) + len(splits["seen_test"])
    ratio = len(splits["seen_test"]) / n_seen
    assert 0.15 < ratio < 0.25


def test_generate_splits_holdout_count():
    from scripts.generate_splits_daft import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    assert len(splits["unseen_categories"]) == 2
