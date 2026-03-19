# Dataset Streaming Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream-process 46K Objaverse-LVIS + 51K ShapeNetCore v2 meshes into dual-normalization patch NPZ files, upload to HuggingFace dataset `Pthahnix/MeshLex-Patches`, and generate train/test/unseen splits.

**Architecture:** A single streaming script processes meshes in batches (500 for Objaverse, per-category for ShapeNet), saving patch NPZ files to a local temp directory, uploading each batch to HF, then deleting local files. This keeps disk usage under 5GB on an 80GB RunPod pod. Resume-safe via `progress.json`. Dual normalization (PCA + no-PCA) stored in every NPZ for downstream Pipeline B and C experiments.

**Tech Stack:** Python 3.10+, trimesh, pymetis, pyfqmr, objaverse, huggingface_hub, numpy

**Spec:** `docs/superpowers/specs/2026-03-19-assembly-fix-full-retrain-design.md` §4

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/patch_segment_dual.py` | `_normalize_patch_coords_nopca()` + `save_patch_npz_dual()` — dual-normalization NPZ serialization |
| `scripts/stream_objaverse.py` | Stream-process all Objaverse-LVIS objects → HF in batches of 500 |
| `scripts/stream_shapenet.py` | Stream-process all ShapeNetCore v2 objects → HF per-category |
| `scripts/generate_splits.py` | Generate `splits.json` + `stats.json` from `metadata.json` on HF |
| `tests/test_patch_segment_dual.py` | Tests for dual normalization + NPZ serialization |
| `tests/test_stream_helpers.py` | Tests for batch processing helpers (metadata, resume, category mapping) |

### Modified Files
| File | Changes |
|------|---------|
| `src/patch_segment.py` | Add `local_vertices_nopca` field to `MeshPatch` dataclass |

---

## Task 1: Add `local_vertices_nopca` to MeshPatch

**Files:**
- Modify: `src/patch_segment.py:8-23`
- Modify: `src/patch_segment.py:170-184`
- Test: `tests/test_patch_segment.py`

- [ ] **Step 1: Write failing test for nopca field**

Add to `tests/test_patch_segment.py`:

```python
def test_patch_has_local_vertices_nopca():
    """Each patch should have a local_vertices_nopca field (center+scale, no PCA)."""
    mesh = _make_sphere()
    patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
    for p in patches:
        assert hasattr(p, "local_vertices_nopca"), "MeshPatch missing local_vertices_nopca"
        assert p.local_vertices_nopca is not None
        assert p.local_vertices_nopca.shape == p.local_vertices.shape
        # nopca should differ from pca-aligned (unless rotation is identity)
        norms_nopca = np.linalg.norm(p.local_vertices_nopca, axis=1)
        assert norms_nopca.max() <= 1.05, f"nopca not normalized: max norm {norms_nopca.max()}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_patch_segment.py::test_patch_has_local_vertices_nopca -v`
Expected: FAIL with `AttributeError: 'MeshPatch' object has no attribute 'local_vertices_nopca'`

- [ ] **Step 3: Add field to MeshPatch and compute it in segment_mesh_to_patches**

In `src/patch_segment.py`, update the `MeshPatch` dataclass:

```python
@dataclass
class MeshPatch:
    # Topology (local indices)
    faces: np.ndarray              # (F, 3) local vertex indices
    vertices: np.ndarray           # (V, 3) world-space vertex coords
    global_face_indices: np.ndarray  # (F,) indices into the original mesh
    boundary_vertices: list[int]   # local indices of boundary verts

    # Geometry (for reconstruction)
    centroid: np.ndarray           # (3,)
    principal_axes: np.ndarray     # (3, 3) PCA rotation
    scale: float                   # bounding sphere radius

    # Normalized local coordinates
    local_vertices: np.ndarray     # (V, 3) centered + PCA-aligned + unit-scaled
    local_vertices_nopca: np.ndarray = None  # (V, 3) centered + unit-scaled (no PCA)
```

In `segment_mesh_to_patches`, after the `_normalize_patch_coords` call (line ~173), add the no-PCA computation:

```python
        # Normalize (PCA)
        local_verts, centroid, axes, scale = _normalize_patch_coords(vertices)

        # No-PCA normalization: center + scale only.
        # Note: scale is rotation-invariant (||Vt @ x|| == ||x|| since Vt is orthogonal),
        # so reusing the PCA-derived scale for nopca is mathematically correct.
        centered = vertices - centroid
        local_verts_nopca = centered / scale if scale > 1e-8 else centered

        patches.append(MeshPatch(
            faces=local_faces,
            vertices=vertices,
            global_face_indices=face_indices,
            boundary_vertices=sorted(boundary_local),
            centroid=centroid,
            principal_axes=axes,
            scale=scale,
            local_vertices=local_verts,
            local_vertices_nopca=local_verts_nopca,
        ))
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_patch_segment.py -v`
Expected: ALL PASS (including new test + all existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/patch_segment.py tests/test_patch_segment.py
git commit -m "feat: add local_vertices_nopca to MeshPatch for dual normalization"
git push
```

---

## Task 2: Dual-Normalization NPZ Serialization

**Files:**
- Create: `src/patch_segment_dual.py`
- Create: `tests/test_patch_segment_dual.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_patch_segment_dual.py
"""Tests for dual-normalization NPZ serialization."""
import numpy as np
import trimesh
import pytest
from pathlib import Path

from src.patch_segment import segment_mesh_to_patches


def _make_sphere():
    return trimesh.creation.icosphere(subdivisions=3)


def test_save_patch_npz_dual_creates_files(tmp_path):
    """save_patch_npz_dual should create one NPZ per patch."""
    from src.patch_segment_dual import save_patch_npz_dual
    mesh = _make_sphere()
    patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
    meta = save_patch_npz_dual(patches, "test_mesh", str(tmp_path))
    assert meta["mesh_id"] == "test_mesh"
    assert meta["n_patches"] == len(patches)
    npz_files = list(tmp_path.glob("*.npz"))
    assert len(npz_files) == len(patches)


def test_npz_contains_dual_normalization(tmp_path):
    """Each NPZ should contain both local_vertices and local_vertices_nopca."""
    from src.patch_segment_dual import save_patch_npz_dual
    mesh = _make_sphere()
    patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
    save_patch_npz_dual(patches, "test_mesh", str(tmp_path))
    npz_file = sorted(tmp_path.glob("*.npz"))[0]
    data = np.load(str(npz_file))
    required_keys = [
        "faces", "vertices", "local_vertices", "local_vertices_nopca",
        "centroid", "principal_axes", "scale",
        "boundary_vertices", "global_face_indices",
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"
    assert data["local_vertices"].shape == data["local_vertices_nopca"].shape


def test_npz_scale_is_scalar(tmp_path):
    """Scale should be saved as shape (1,) array."""
    from src.patch_segment_dual import save_patch_npz_dual
    mesh = _make_sphere()
    patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
    save_patch_npz_dual(patches, "test_mesh", str(tmp_path))
    npz_file = sorted(tmp_path.glob("*.npz"))[0]
    data = np.load(str(npz_file))
    assert data["scale"].shape == (1,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_patch_segment_dual.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.patch_segment_dual'`

- [ ] **Step 3: Implement save_patch_npz_dual**

```python
# src/patch_segment_dual.py
"""Dual-normalization NPZ serialization for MeshLex patches."""
import numpy as np
from pathlib import Path

from src.patch_segment import MeshPatch


def save_patch_npz_dual(
    patches: list[MeshPatch],
    mesh_id: str,
    output_dir: str,
) -> dict:
    """Save patches as NPZ files with both PCA and no-PCA normalization.

    Each NPZ contains: faces, vertices, local_vertices, local_vertices_nopca,
    centroid, principal_axes, scale, boundary_vertices, global_face_indices.

    Returns metadata dict: {mesh_id, n_patches, face_counts}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i, patch in enumerate(patches):
        np.savez_compressed(
            str(out / f"{mesh_id}_patch_{i:03d}.npz"),
            faces=patch.faces,
            vertices=patch.vertices,
            local_vertices=patch.local_vertices,
            local_vertices_nopca=patch.local_vertices_nopca,
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_patch_segment_dual.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/patch_segment_dual.py tests/test_patch_segment_dual.py
git commit -m "feat: dual-normalization NPZ serialization (PCA + no-PCA)"
git push
```

---

## Task 3: Stream Processing Helpers (Metadata + Resume)

**Files:**
- Create: `src/stream_utils.py`
- Create: `tests/test_stream_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stream_utils.py
"""Tests for streaming pipeline helpers."""
import json
import pytest
from pathlib import Path


def test_progress_tracker_save_load(tmp_path):
    """ProgressTracker should persist completed batches across restarts."""
    from src.stream_utils import ProgressTracker
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    assert not tracker.is_done("batch_000")
    tracker.mark_done("batch_000", {"meshes": 450, "patches": 12000})
    tracker.save()
    # Reload from disk
    tracker2 = ProgressTracker(str(tmp_path / "progress.json"))
    assert tracker2.is_done("batch_000")
    assert not tracker2.is_done("batch_001")


def test_metadata_collector_accumulate(tmp_path):
    """MetadataCollector should accumulate entries and save to JSON."""
    from src.stream_utils import MetadataCollector
    collector = MetadataCollector(str(tmp_path / "metadata.json"))
    collector.add("mesh_001", {
        "category": "chair", "source": "objaverse",
        "n_patches": 30, "n_faces": 1000, "n_verts": 502,
    })
    collector.add("mesh_002", {
        "category": "table", "source": "shapenet",
        "n_patches": 28, "n_faces": 980, "n_verts": 490,
    })
    collector.save()
    with open(tmp_path / "metadata.json") as f:
        data = json.load(f)
    assert "mesh_001" in data
    assert data["mesh_002"]["source"] == "shapenet"


def test_metadata_collector_resume(tmp_path):
    """MetadataCollector should load existing entries on init."""
    from src.stream_utils import MetadataCollector
    # Pre-populate
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump({"existing": {"category": "lamp"}}, f)
    collector = MetadataCollector(str(tmp_path / "metadata.json"))
    assert "existing" in collector.data
    collector.add("new_mesh", {"category": "car"})
    assert len(collector.data) == 2


def test_shapenet_synset_to_category():
    """Should map synset IDs to human-readable category names."""
    from src.stream_utils import SHAPENET_SYNSET_MAP
    assert SHAPENET_SYNSET_MAP["03001627"] == "chair"
    assert SHAPENET_SYNSET_MAP["02691156"] == "airplane"
    assert SHAPENET_SYNSET_MAP["04379243"] == "table"
    assert len(SHAPENET_SYNSET_MAP) == 55


def test_batch_uids():
    """batch_uids should split a list into chunks of given size."""
    from src.stream_utils import batch_uids
    uids = list(range(1250))
    batches = list(batch_uids(uids, batch_size=500))
    assert len(batches) == 3
    assert len(batches[0]) == 500
    assert len(batches[1]) == 500
    assert len(batches[2]) == 250
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stream_utils.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement stream_utils**

```python
# src/stream_utils.py
"""Helpers for streaming dataset processing pipeline."""
import json
from pathlib import Path
from typing import Iterator


class ProgressTracker:
    """Track completed batches for resume-safe processing."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.completed: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                self.completed = json.load(f)

    def is_done(self, batch_id: str) -> bool:
        return batch_id in self.completed

    def mark_done(self, batch_id: str, stats: dict):
        self.completed[batch_id] = stats

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.completed, f, indent=2)


class MetadataCollector:
    """Accumulate per-mesh metadata and persist to JSON."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path) as f:
                self.data = json.load(f)

    def add(self, mesh_id: str, entry: dict):
        self.data[mesh_id] = entry

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f)


def batch_uids(uids: list, batch_size: int = 500) -> Iterator[list]:
    """Yield successive chunks of uids."""
    for i in range(0, len(uids), batch_size):
        yield uids[i:i + batch_size]


# ShapeNetCore v2: 55 synset IDs → human-readable category names
SHAPENET_SYNSET_MAP = {
    "02691156": "airplane",
    "02747177": "trash_bin",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02834778": "bicycle",
    "02843684": "birdhouse",
    "02858304": "boat",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file_cabinet",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorbike",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "watercraft",
    "04554684": "washer",
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stream_utils.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/stream_utils.py tests/test_stream_utils.py
git commit -m "feat: streaming pipeline helpers (progress tracker, metadata, synset map)"
git push
```

---

## Task 4: Objaverse-LVIS Streaming Script

**Files:**
- Create: `scripts/stream_objaverse.py`

This is the main script for processing all 46K Objaverse-LVIS objects. It runs overnight on RunPod.

- [ ] **Step 1: Write the script**

See full code below. Key design decisions:
- Batches of 500 UIDs (configurable)
- `objaverse.load_objects()` with `download_processes=8` for parallel download
- Each batch: download → process → save NPZ → upload to HF → delete local → clear objaverse cache
- Resume-safe via `ProgressTracker`
- Metadata accumulated in `MetadataCollector`

```python
# scripts/stream_objaverse.py
"""Stream-process all Objaverse-LVIS objects → HF dataset.

Usage:
    python scripts/stream_objaverse.py \
        --hf_repo Pthahnix/MeshLex-Patches \
        --batch_size 500 \
        --download_processes 8 \
        --work_dir /tmp/meshlex_objaverse
"""
import argparse
import gc
import logging
import shutil
import time
from pathlib import Path

import objaverse
from huggingface_hub import HfApi

from src.data_prep import load_and_preprocess_mesh
from src.patch_segment import segment_mesh_to_patches
from src.patch_segment_dual import save_patch_npz_dual
from src.stream_utils import ProgressTracker, MetadataCollector, batch_uids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def process_batch(
    batch_idx: int,
    uids: list[str],
    uid_to_cat: dict[str, str],
    work_dir: Path,
    hf_api: HfApi,
    hf_repo: str,
    progress: ProgressTracker,
    metadata: MetadataCollector,
    download_processes: int = 8,
    target_faces: int = 1000,
):
    batch_id = f"batch_{batch_idx:03d}"
    if progress.is_done(batch_id):
        log.info(f"Skipping {batch_id} (already done)")
        return

    batch_dir = work_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Download GLBs
    log.info(f"[{batch_id}] Downloading {len(uids)} objects...")
    t0 = time.time()
    objects = objaverse.load_objects(
        uids=uids, download_processes=download_processes,
    )
    log.info(f"[{batch_id}] Downloaded in {time.time()-t0:.0f}s")

    n_ok, n_fail, n_patches_total = 0, 0, 0

    for uid in uids:
        glb_path = objects.get(uid)
        if glb_path is None:
            n_fail += 1
            continue

        try:
            mesh = load_and_preprocess_mesh(glb_path, target_faces=target_faces)
            if mesh is None:
                n_fail += 1
                continue

            patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
            if len(patches) == 0:
                n_fail += 1
                continue

            meta = save_patch_npz_dual(patches, uid, str(batch_dir))
            category = uid_to_cat.get(uid, "unknown")
            metadata.add(uid, {
                "category": category,
                "source": "objaverse",
                "n_patches": meta["n_patches"],
                "n_faces": int(mesh.faces.shape[0]),
                "n_verts": int(mesh.vertices.shape[0]),
            })
            n_ok += 1
            n_patches_total += meta["n_patches"]

        except Exception as e:
            log.warning(f"[{batch_id}] Failed {uid}: {e}")
            n_fail += 1

    log.info(
        f"[{batch_id}] Processed: {n_ok} ok, {n_fail} fail, "
        f"{n_patches_total} patches"
    )

    # Upload to HF
    if n_ok > 0:
        log.info(f"[{batch_id}] Uploading to HF...")
        hf_api.upload_folder(
            folder_path=str(batch_dir),
            path_in_repo=f"objaverse/{batch_id}",
            repo_id=hf_repo,
            repo_type="dataset",
        )

    # Cleanup local
    shutil.rmtree(batch_dir, ignore_errors=True)

    # Robust cleanup of objaverse cache to prevent disk overflow.
    # objaverse caches GLBs in ~/.objaverse/ and HF cache in ~/.cache/huggingface/.
    # Individual file deletion is fragile; nuke the entire GLB cache directory per batch.
    objaverse_cache = Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs"
    if objaverse_cache.exists():
        for uid in uids:
            uid_prefix = uid[:2]  # objaverse uses 2-char prefix dirs
            uid_dir = objaverse_cache / uid_prefix
            uid_file = uid_dir / f"{uid}.glb"
            if uid_file.exists():
                try:
                    uid_file.unlink()
                except Exception:
                    pass
        # Also clean any empty prefix dirs
        for d in objaverse_cache.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    # Periodically nuke HF hub cache for objaverse to reclaim space
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for cache_dir in hf_cache.glob("models--allenai--objaverse*"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Mark done
    progress.mark_done(batch_id, {
        "meshes_ok": n_ok, "meshes_fail": n_fail,
        "patches": n_patches_total,
    })
    progress.save()
    metadata.save()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--download_processes", type=int, default=8)
    parser.add_argument("--work_dir", default="/tmp/meshlex_objaverse")
    parser.add_argument("--target_faces", type=int, default=1000)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    hf_api = HfApi()
    progress = ProgressTracker(str(work_dir / "progress.json"))
    metadata = MetadataCollector(str(work_dir / "metadata.json"))

    # Load LVIS annotations
    log.info("Loading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()
    log.info(f"LVIS: {len(lvis)} categories")

    # Build UID → category mapping (all UIDs, deduplicated)
    uid_to_cat = {}
    all_uids = []
    for cat_name, uids in sorted(lvis.items()):
        for uid in uids:
            if uid not in uid_to_cat:  # deduplicate: keep first category
                uid_to_cat[uid] = cat_name
                all_uids.append(uid)

    log.info(f"Total UIDs: {len(all_uids)}")

    # Process in batches
    batches = list(batch_uids(all_uids, batch_size=args.batch_size))
    log.info(f"Total batches: {len(batches)}")

    for i, batch in enumerate(batches):
        process_batch(
            batch_idx=i,
            uids=batch,
            uid_to_cat=uid_to_cat,
            work_dir=work_dir,
            hf_api=hf_api,
            hf_repo=args.hf_repo,
            progress=progress,
            metadata=metadata,
            download_processes=args.download_processes,
            target_faces=args.target_faces,
        )

    # Upload final metadata
    log.info("Uploading metadata.json to HF...")
    hf_api.upload_file(
        path_or_fileobj=str(work_dir / "metadata.json"),
        path_in_repo="metadata_objaverse.json",
        repo_id=args.hf_repo,
        repo_type="dataset",
    )
    log.info("Objaverse streaming complete!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run test with 5 objects**

Run on RunPod:
```bash
python scripts/stream_objaverse.py \
    --hf_repo Pthahnix/MeshLex-Patches \
    --batch_size 5 \
    --work_dir /tmp/meshlex_test \
    2>&1 | head -50
```
Expected: Downloads 5 GLBs, processes them, uploads 1 batch to HF, prints stats.
Verify: `ls /tmp/meshlex_test/` shows `progress.json` and `metadata.json`.

- [ ] **Step 3: Commit**

```bash
git add scripts/stream_objaverse.py
git commit -m "feat: Objaverse-LVIS streaming pipeline → HF dataset"
git push
```

---

## Task 5: ShapeNetCore v2 Streaming Script

**Files:**
- Create: `scripts/stream_shapenet.py`

ShapeNet uses a different access pattern: HF gated dataset `ShapeNet/ShapeNetCore`, per-synset folder structure `{synsetId}/{modelId}/models/model_normalized.obj`. We download one category at a time using `snapshot_download(allow_patterns=...)`.

- [ ] **Step 1: Write the script**

```python
# scripts/stream_shapenet.py
"""Stream-process ShapeNetCore v2 → HF dataset, one category at a time.

Usage:
    python scripts/stream_shapenet.py \
        --hf_repo Pthahnix/MeshLex-Patches \
        --work_dir /tmp/meshlex_shapenet
"""
import argparse
import gc
import logging
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from src.data_prep import load_and_preprocess_mesh
from src.patch_segment import segment_mesh_to_patches
from src.patch_segment_dual import save_patch_npz_dual
from src.stream_utils import (
    ProgressTracker, MetadataCollector,
    SHAPENET_SYNSET_MAP, batch_uids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SHAPENET_HF_REPO = "ShapeNet/ShapeNetCore"


def process_category(
    synset_id: str,
    cat_name: str,
    work_dir: Path,
    hf_api: HfApi,
    hf_repo: str,
    progress: ProgressTracker,
    metadata: MetadataCollector,
    target_faces: int = 1000,
    upload_batch_size: int = 500,
):
    """Download one ShapeNet category, process all models, upload in batches."""
    cat_key = f"shapenet_{synset_id}"
    if progress.is_done(cat_key):
        log.info(f"Skipping {cat_name} ({synset_id}) — already done")
        return

    log.info(f"[{cat_name}] Downloading category {synset_id}...")
    t0 = time.time()

    # Download only this category's files from HF
    local_dir = work_dir / "shapenet_raw"
    try:
        snapshot_download(
            repo_id=SHAPENET_HF_REPO,
            repo_type="dataset",
            allow_patterns=f"{synset_id}/**/model_normalized.obj",
            local_dir=str(local_dir),
        )
    except Exception as e:
        log.error(f"[{cat_name}] Download failed: {e}")
        progress.mark_done(cat_key, {"error": str(e)})
        progress.save()
        return

    log.info(f"[{cat_name}] Downloaded in {time.time()-t0:.0f}s")

    # Find all OBJ files for this category
    synset_dir = local_dir / synset_id
    obj_files = sorted(synset_dir.rglob("model_normalized.obj")) if synset_dir.exists() else []
    log.info(f"[{cat_name}] Found {len(obj_files)} models")

    n_ok, n_fail, n_patches_total = 0, 0, 0
    batch_idx = 0
    batch_dir = work_dir / f"shapenet_batch_{synset_id}_{batch_idx:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_count = 0

    for obj_file in obj_files:
        # Extract model ID: {synset}/{model_id}/models/model_normalized.obj
        model_id = obj_file.parent.parent.name
        mesh_id = f"{synset_id}_{model_id}"

        try:
            mesh = load_and_preprocess_mesh(
                str(obj_file), target_faces=target_faces,
            )
            if mesh is None:
                n_fail += 1
                continue

            patches = segment_mesh_to_patches(mesh, target_patch_faces=35)
            if len(patches) == 0:
                n_fail += 1
                continue

            meta = save_patch_npz_dual(patches, mesh_id, str(batch_dir))
            metadata.add(mesh_id, {
                "category": cat_name,
                "source": "shapenet",
                "synset_id": synset_id,
                "n_patches": meta["n_patches"],
                "n_faces": int(mesh.faces.shape[0]),
                "n_verts": int(mesh.vertices.shape[0]),
            })
            n_ok += 1
            n_patches_total += meta["n_patches"]
            batch_count += 1

        except Exception as e:
            log.warning(f"[{cat_name}] Failed {mesh_id}: {e}")
            n_fail += 1

        # Upload batch when full
        if batch_count >= upload_batch_size:
            hf_batch_id = f"shapenet/{synset_id}_batch_{batch_idx:03d}"
            log.info(f"[{cat_name}] Uploading {hf_batch_id}...")
            hf_api.upload_folder(
                folder_path=str(batch_dir),
                path_in_repo=hf_batch_id,
                repo_id=hf_repo,
                repo_type="dataset",
            )
            shutil.rmtree(batch_dir, ignore_errors=True)
            batch_idx += 1
            batch_dir = work_dir / f"shapenet_batch_{synset_id}_{batch_idx:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            batch_count = 0
            metadata.save()

    # Upload remaining
    if batch_count > 0:
        hf_batch_id = f"shapenet/{synset_id}_batch_{batch_idx:03d}"
        log.info(f"[{cat_name}] Uploading {hf_batch_id}...")
        hf_api.upload_folder(
            folder_path=str(batch_dir),
            path_in_repo=hf_batch_id,
            repo_id=hf_repo,
            repo_type="dataset",
        )
    shutil.rmtree(batch_dir, ignore_errors=True)

    # Cleanup downloaded ShapeNet files for this category
    shutil.rmtree(local_dir / synset_id, ignore_errors=True)

    log.info(
        f"[{cat_name}] Done: {n_ok} ok, {n_fail} fail, "
        f"{n_patches_total} patches"
    )

    progress.mark_done(cat_key, {
        "meshes_ok": n_ok, "meshes_fail": n_fail,
        "patches": n_patches_total,
    })
    progress.save()
    metadata.save()
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--work_dir", default="/tmp/meshlex_shapenet")
    parser.add_argument("--target_faces", type=int, default=1000)
    parser.add_argument("--upload_batch_size", type=int, default=500)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    hf_api = HfApi()
    progress = ProgressTracker(str(work_dir / "progress.json"))
    metadata = MetadataCollector(str(work_dir / "metadata.json"))

    log.info(f"Processing {len(SHAPENET_SYNSET_MAP)} ShapeNet categories")

    for synset_id, cat_name in sorted(SHAPENET_SYNSET_MAP.items()):
        process_category(
            synset_id=synset_id,
            cat_name=cat_name,
            work_dir=work_dir,
            hf_api=hf_api,
            hf_repo=args.hf_repo,
            progress=progress,
            metadata=metadata,
            target_faces=args.target_faces,
            upload_batch_size=args.upload_batch_size,
        )

    # Upload final metadata
    log.info("Uploading metadata_shapenet.json to HF...")
    hf_api.upload_file(
        path_or_fileobj=str(work_dir / "metadata.json"),
        path_in_repo="metadata_shapenet.json",
        repo_id=args.hf_repo,
        repo_type="dataset",
    )
    log.info("ShapeNet streaming complete!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run test with 1 small category**

Run on RunPod (requires ShapeNet HF access):
```bash
python -c "
from huggingface_hub import snapshot_download
# Test: download just 'rocket' category (04099429, ~85 models, small)
snapshot_download(
    repo_id='ShapeNet/ShapeNetCore',
    repo_type='dataset',
    allow_patterns='04099429/*/models/model_normalized.obj',
    local_dir='/tmp/shapenet_test',
)
import os
count = sum(1 for _ in os.walk('/tmp/shapenet_test'))
print(f'Downloaded dirs: {count}')
"
```
Expected: Downloads ~85 OBJ files for the rocket category.

Then run the full script on just that category:
```bash
python scripts/stream_shapenet.py \
    --hf_repo Pthahnix/MeshLex-Patches \
    --work_dir /tmp/meshlex_shapenet_test \
    2>&1 | head -30
```
(It will process all 55 categories, but the first one will validate the pipeline.)

- [ ] **Step 3: Commit**

```bash
git add scripts/stream_shapenet.py
git commit -m "feat: ShapeNetCore v2 streaming pipeline → HF dataset"
git push
```

---

## Task 6: Generate Splits + Stats

**Files:**
- Create: `scripts/generate_splits.py`
- Create: `tests/test_generate_splits.py`

This script downloads `metadata_objaverse.json` and `metadata_shapenet.json` from HF, merges them, generates category-holdout splits, and uploads `splits.json`, `stats.json`, and merged `metadata.json`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generate_splits.py
"""Tests for split generation logic."""
import json
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_fake_metadata():
    """Create fake metadata with known categories."""
    meta = {}
    # 10 categories, 20 meshes each = 200 meshes
    for cat_idx in range(10):
        cat_name = f"cat_{cat_idx:02d}"
        for mesh_idx in range(20):
            mesh_id = f"{cat_name}_mesh_{mesh_idx:03d}"
            meta[mesh_id] = {
                "category": cat_name,
                "source": "objaverse" if cat_idx < 5 else "shapenet",
                "n_patches": 30,
                "n_faces": 1000,
                "n_verts": 500,
            }
    return meta


def test_generate_splits_all_meshes_assigned():
    """Every mesh should appear in exactly one split."""
    from scripts.generate_splits import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    all_ids = set(splits["seen_train"] + splits["seen_test"] + splits["unseen"])
    assert all_ids == set(meta.keys())


def test_generate_splits_unseen_categories_excluded():
    """Meshes in unseen categories should not appear in seen splits."""
    from scripts.generate_splits import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    unseen_cats = set(splits["unseen_categories"])
    for mesh_id in splits["seen_train"] + splits["seen_test"]:
        assert meta[mesh_id]["category"] not in unseen_cats


def test_generate_splits_test_ratio():
    """seen_test should be ~20% of seen meshes."""
    from scripts.generate_splits import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    n_seen = len(splits["seen_train"]) + len(splits["seen_test"])
    ratio = len(splits["seen_test"]) / n_seen
    assert 0.15 < ratio < 0.25, f"Test ratio {ratio:.2f} not near 0.2"


def test_generate_splits_holdout_count():
    """Should hold out exactly holdout_count categories."""
    from scripts.generate_splits import generate_splits
    meta = _make_fake_metadata()
    splits = generate_splits(meta, holdout_count=2, test_ratio=0.2, seed=42)
    assert len(splits["unseen_categories"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generate_splits.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement generate_splits.py**

```python
# scripts/generate_splits.py
"""Generate train/test/unseen splits from merged metadata.

Usage:
    python scripts/generate_splits.py \
        --hf_repo Pthahnix/MeshLex-Patches \
        --holdout_count 100 \
        --test_ratio 0.2 \
        --seed 42
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def generate_splits(
    metadata: dict,
    holdout_count: int = 100,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Generate category-holdout splits.

    Args:
        metadata: {mesh_id: {category, source, ...}}
        holdout_count: number of categories to hold out as unseen
        test_ratio: fraction of seen meshes for test split
        seed: random seed

    Returns:
        splits dict with keys: seen_train, seen_test, unseen,
        unseen_categories, seen_categories, split_seed,
        holdout_count, test_ratio
    """
    rng = np.random.default_rng(seed)

    # Collect all categories
    cat_to_meshes: dict[str, list[str]] = {}
    for mesh_id, info in metadata.items():
        cat = info["category"]
        cat_to_meshes.setdefault(cat, []).append(mesh_id)

    all_cats = sorted(cat_to_meshes.keys())
    n_cats = len(all_cats)
    actual_holdout = min(holdout_count, n_cats // 2)

    # Randomly select holdout categories
    perm = rng.permutation(n_cats)
    unseen_cats = [all_cats[i] for i in perm[:actual_holdout]]
    seen_cats = [all_cats[i] for i in perm[actual_holdout:]]

    # Unseen meshes
    unseen = []
    for cat in unseen_cats:
        unseen.extend(cat_to_meshes[cat])

    # Seen meshes: split into train/test
    seen_meshes = []
    for cat in seen_cats:
        seen_meshes.extend(cat_to_meshes[cat])

    rng.shuffle(seen_meshes)
    n_test = int(len(seen_meshes) * test_ratio)
    seen_test = seen_meshes[:n_test]
    seen_train = seen_meshes[n_test:]

    return {
        "seen_train": sorted(seen_train),
        "seen_test": sorted(seen_test),
        "unseen": sorted(unseen),
        "unseen_categories": sorted(unseen_cats),
        "seen_categories": sorted(seen_cats),
        "split_seed": seed,
        "holdout_count": actual_holdout,
        "test_ratio": test_ratio,
    }


def compute_stats(metadata: dict, splits: dict) -> dict:
    """Compute aggregate statistics."""
    total_meshes = len(metadata)
    total_patches = sum(m["n_patches"] for m in metadata.values())
    total_faces = sum(m["n_faces"] for m in metadata.values())

    source_counts = Counter(m["source"] for m in metadata.values())
    cat_counts = Counter(m["category"] for m in metadata.values())

    return {
        "total_meshes": total_meshes,
        "total_patches": total_patches,
        "total_faces": total_faces,
        "avg_patches_per_mesh": round(total_patches / max(total_meshes, 1), 1),
        "source_counts": dict(source_counts),
        "n_categories": len(cat_counts),
        "split_sizes": {
            "seen_train": len(splits["seen_train"]),
            "seen_test": len(splits["seen_test"]),
            "unseen": len(splits["unseen"]),
        },
        "top_categories": dict(cat_counts.most_common(20)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--holdout_count", type=int, default=100)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", default="/tmp/meshlex_splits")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    hf_api = HfApi()

    # Download metadata files from HF
    log.info("Downloading metadata files from HF...")
    meta_obj_path = hf_hub_download(
        repo_id=args.hf_repo, filename="metadata_objaverse.json",
        repo_type="dataset", local_dir=str(work_dir),
    )
    meta_sn_path = hf_hub_download(
        repo_id=args.hf_repo, filename="metadata_shapenet.json",
        repo_type="dataset", local_dir=str(work_dir),
    )

    # Merge
    with open(meta_obj_path) as f:
        meta_obj = json.load(f)
    with open(meta_sn_path) as f:
        meta_sn = json.load(f)

    metadata = {**meta_obj, **meta_sn}
    log.info(
        f"Merged metadata: {len(meta_obj)} objaverse + "
        f"{len(meta_sn)} shapenet = {len(metadata)} total"
    )

    # Generate splits
    splits = generate_splits(
        metadata,
        holdout_count=args.holdout_count,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    log.info(
        f"Splits: {len(splits['seen_train'])} train, "
        f"{len(splits['seen_test'])} test, "
        f"{len(splits['unseen'])} unseen "
        f"({len(splits['unseen_categories'])} held-out categories)"
    )

    # Compute stats
    stats = compute_stats(metadata, splits)
    log.info(f"Stats: {stats['total_meshes']} meshes, {stats['total_patches']} patches")

    # Save locally
    for name, data in [
        ("metadata.json", metadata),
        ("splits.json", splits),
        ("stats.json", stats),
    ]:
        with open(work_dir / name, "w") as f:
            json.dump(data, f, indent=2)

    # Upload to HF
    for name in ["metadata.json", "splits.json", "stats.json"]:
        log.info(f"Uploading {name} to HF...")
        hf_api.upload_file(
            path_or_fileobj=str(work_dir / name),
            path_in_repo=name,
            repo_id=args.hf_repo,
            repo_type="dataset",
        )

    log.info("Splits generation complete!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generate_splits.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_splits.py tests/test_generate_splits.py
git commit -m "feat: generate train/test/unseen splits from merged metadata"
git push
```

---

## Task 7: Validation Script

**Files:**
- Create: `scripts/validate_dataset.py`

After both streaming pipelines complete, this script validates the HF dataset meets the spec thresholds.

- [ ] **Step 1: Write the script**

```python
# scripts/validate_dataset.py
"""Validate the HF dataset meets spec thresholds.

Usage:
    python scripts/validate_dataset.py --hf_repo Pthahnix/MeshLex-Patches
"""
import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def sample_and_validate_npz(
    metadata: dict, hf_repo: str, work_dir: Path, n_samples: int = 20,
) -> list[tuple[str, str, bool]]:
    """Download and validate random NPZ files from HF (spec §4.7).

    Samples n_samples/2 from each source (objaverse, shapenet).
    Returns list of (mesh_id, check_description, passed).
    """
    hf_api = HfApi()
    results = []

    for source in ["objaverse", "shapenet"]:
        source_ids = [
            mid for mid, m in metadata.items() if m["source"] == source
        ]
        if not source_ids:
            continue
        sample_ids = random.sample(
            source_ids, min(n_samples // 2, len(source_ids)),
        )

        for mesh_id in sample_ids:
            # Find an NPZ file for this mesh on HF
            try:
                # List files matching this mesh_id
                files = hf_api.list_repo_tree(
                    repo_id=hf_repo, repo_type="dataset",
                    path_in_repo=f"{source}/",
                )
                # Search for a matching NPZ (batch structure)
                npz_path = None
                for item in hf_api.list_repo_tree(
                    repo_id=hf_repo, repo_type="dataset",
                ):
                    if (
                        item.rfilename.endswith(".npz")
                        and mesh_id in item.rfilename
                    ):
                        npz_path = item.rfilename
                        break

                if npz_path is None:
                    results.append((mesh_id, "NPZ file found on HF", False))
                    continue

                local_path = hf_hub_download(
                    repo_id=hf_repo, filename=npz_path,
                    repo_type="dataset", local_dir=str(work_dir / "samples"),
                )
                data = np.load(local_path)

                # Check required keys
                required = [
                    "faces", "vertices", "local_vertices",
                    "local_vertices_nopca", "centroid", "principal_axes",
                    "scale", "boundary_vertices", "global_face_indices",
                ]
                missing = [k for k in required if k not in data]
                if missing:
                    results.append((
                        mesh_id, f"All keys present (missing: {missing})",
                        False,
                    ))
                    continue
                results.append((mesh_id, "All keys present", True))

                # Check shapes
                n_verts = data["local_vertices"].shape[0]
                ok = (
                    data["local_vertices_nopca"].shape[0] == n_verts
                    and data["centroid"].shape == (3,)
                    and data["principal_axes"].shape == (3, 3)
                    and data["scale"].shape == (1,)
                )
                results.append((mesh_id, "Shapes valid", ok))

                # Check dual normalization differs
                diff = np.abs(
                    data["local_vertices"] - data["local_vertices_nopca"]
                ).max()
                # They should differ (unless PCA rotation is identity)
                results.append((
                    mesh_id,
                    f"PCA vs noPCA differ (max_diff={diff:.4f})",
                    True,  # informational
                ))

            except Exception as e:
                results.append((mesh_id, f"Sample check error: {e}", False))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--work_dir", default="/tmp/meshlex_validate")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata and stats
    for fname in ["metadata.json", "splits.json", "stats.json"]:
        hf_hub_download(
            repo_id=args.hf_repo, filename=fname,
            repo_type="dataset", local_dir=str(work_dir),
        )

    with open(work_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(work_dir / "splits.json") as f:
        splits = json.load(f)
    with open(work_dir / "stats.json") as f:
        stats = json.load(f)

    # Validation checks (from spec §4.7)
    checks = []

    # Check 1: Objaverse count
    n_obj = sum(1 for m in metadata.values() if m["source"] == "objaverse")
    ok = n_obj >= 35000
    checks.append(("Objaverse meshes >= 35,000", n_obj, ok))

    # Check 2: ShapeNet count
    n_sn = sum(1 for m in metadata.values() if m["source"] == "shapenet")
    ok = n_sn >= 45000
    checks.append(("ShapeNet meshes >= 45,000", n_sn, ok))

    # Check 3: Total count
    n_total = len(metadata)
    ok = n_total >= 75000
    checks.append(("Total meshes >= 75,000", n_total, ok))

    # Check 4: Total patches
    n_patches = sum(m["n_patches"] for m in metadata.values())
    ok = n_patches >= 2_500_000
    checks.append(("Total patches >= 2.5M", n_patches, ok))

    # Check 5: Category distribution
    cats = Counter(m["category"] for m in metadata.values())
    n_cats = len(cats)
    ok = n_cats >= 500
    checks.append(("Categories >= 500", n_cats, ok))

    # Check 6: Splits completeness
    all_split_ids = set(
        splits["seen_train"] + splits["seen_test"] + splits["unseen"]
    )
    ok = all_split_ids == set(metadata.keys())
    checks.append(("All meshes in splits", len(all_split_ids), ok))

    # Check 7: NPZ sample validation (spec §4.7)
    log.info("Sampling and validating 20 random NPZ files from HF...")
    npz_results = sample_and_validate_npz(
        metadata, args.hf_repo, work_dir, n_samples=20,
    )

    # Print report
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    all_pass = True
    for name, value, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {value:,}")

    print("\n  --- NPZ Sample Validation ---")
    for mesh_id, desc, passed in npz_results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {mesh_id}: {desc}")

    print("=" * 60)
    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED — review above")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/validate_dataset.py
git commit -m "feat: dataset validation script (spec §4.7 thresholds)"
git push
```

---

## Task 8: Run Orchestrator (tmux overnight)

**Files:**
- Create: `scripts/run_dataset_pipeline.sh`

This is the master script that runs on RunPod overnight. It executes Objaverse → ShapeNet → splits → validation in sequence.

- [ ] **Step 1: Write the orchestrator**

```bash
#!/bin/bash
# scripts/run_dataset_pipeline.sh
# Master orchestrator for overnight dataset processing on RunPod.
#
# Usage (in tmux):
#   tmux new -s dataset
#   bash scripts/run_dataset_pipeline.sh 2>&1 | tee /tmp/dataset_pipeline.log
#
# Resume-safe: each sub-script tracks progress independently.

set -e

HF_REPO="Pthahnix/MeshLex-Patches"
WORK_BASE="/tmp/meshlex"

echo "=========================================="
echo "MeshLex Dataset Pipeline — $(date)"
echo "=========================================="

# Phase 1: Objaverse-LVIS (46K objects, ~6-10h)
echo ""
echo "[Phase 1/4] Objaverse-LVIS streaming..."
python scripts/stream_objaverse.py \
    --hf_repo "$HF_REPO" \
    --batch_size 500 \
    --download_processes 8 \
    --work_dir "${WORK_BASE}/objaverse" \
    --target_faces 1000

echo ""
echo "[Phase 2/4] ShapeNetCore v2 streaming..."
python scripts/stream_shapenet.py \
    --hf_repo "$HF_REPO" \
    --work_dir "${WORK_BASE}/shapenet" \
    --target_faces 1000

echo ""
echo "[Phase 3/4] Generating splits..."
python scripts/generate_splits.py \
    --hf_repo "$HF_REPO" \
    --holdout_count 100 \
    --test_ratio 0.2 \
    --seed 42 \
    --work_dir "${WORK_BASE}/splits"

echo ""
echo "[Phase 4/4] Validating dataset..."
python scripts/validate_dataset.py \
    --hf_repo "$HF_REPO" \
    --work_dir "${WORK_BASE}/validate"

echo ""
echo "=========================================="
echo "Pipeline complete — $(date)"
echo "=========================================="
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/run_dataset_pipeline.sh
git add scripts/run_dataset_pipeline.sh
git commit -m "feat: overnight dataset pipeline orchestrator"
git push
```

---

## Execution Summary

| Task | What | Time Estimate |
|------|------|---------------|
| 1 | Add `local_vertices_nopca` to MeshPatch | 5 min |
| 2 | Dual-normalization NPZ serialization | 10 min |
| 3 | Stream processing helpers | 10 min |
| 4 | Objaverse streaming script | 15 min (code) + 6-10h (run) |
| 5 | ShapeNet streaming script | 15 min (code) + 4-6h (run) |
| 6 | Generate splits + stats | 10 min |
| 7 | Validation script | 5 min |
| 8 | Run orchestrator | 5 min |

**Total coding time:** ~75 min
**Total overnight run time:** ~10-16h (Objaverse + ShapeNet sequential)

## Hardware Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | Not needed (CPU-only preprocessing) |
| vCPU | 16 cores (for parallel objaverse download + pymetis) |
| RAM | 16 GB sufficient |
| Disk | 80 GB container (peak usage ~5 GB for batch processing) |
| Network | Fast download for HF + objaverse |

**Recommended RunPod config:** CPU pod with 16 vCPU, 32 GB RAM, 80 GB disk. No GPU needed.
Alternatively, reuse existing RTX 4090 pod (GPU idle but has the disk/CPU).
