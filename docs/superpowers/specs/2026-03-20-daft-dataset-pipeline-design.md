# Daft Dataset Pipeline — Design Spec

**Date:** 2026-03-20
**Status:** Draft
**Replaces:** `docs/superpowers/plans/2026-03-19-dataset-streaming-pipeline.md` (NPZ + huggingface_hub approach)

## 1. Goal

Stream-process ~46K Objaverse-LVIS + ~51K ShapeNetCore v2 meshes into dual-normalization patch data, stored as **Parquet** on HuggingFace dataset `Pthahnix/MeshLex-Patches` via the **Daft** dataframe engine. Generate train/test/unseen splits.

## 2. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage format | Parquet (via Daft) | Daft writes only Parquet to HF; columnar compression superior to NPZ |
| Row granularity | One row per patch | ~2.5M rows; finest granularity; no nested list-of-tensors complexity |
| Objaverse download | `objaverse` Python library | HF dataset is metadata-only; GLB files require objaverse API |
| ShapeNet download | `huggingface_hub.snapshot_download` | Gated dataset; OBJ files need local download for trimesh |
| Daft role | DataFrame construction + HF write + HF read for validation | Download and mesh processing remain trimesh/objaverse/numpy |
| NPZ serialization | Eliminated | Patches go directly from memory to Daft DataFrame; no disk IO |
| Downstream adaptation | Out of scope | `PatchDataset` etc. will be adapted in a separate task |

## 3. Parquet Schema

Each row represents one patch from one mesh.

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| `mesh_id` | string | — | Unique ID: Objaverse UID or `{synset}_{model_id}` |
| `patch_idx` | int32 | — | Patch index within this mesh |
| `category` | string | — | Category name |
| `source` | string | — | `"objaverse"` or `"shapenet"` |
| `n_faces` | int32 | — | Number of faces in patch |
| `n_verts` | int32 | — | Number of vertices in patch |
| `faces` | list(int32) | F×3 flattened | Local vertex indices; reshape with `(n_faces, 3)` |
| `vertices` | list(float32) | V×3 flattened | World-space vertex coords; reshape with `(n_verts, 3)` |
| `local_vertices` | list(float32) | V×3 flattened | PCA-aligned + unit-scaled coords; reshape with `(n_verts, 3)` |
| `local_vertices_nopca` | list(float32) | V×3 flattened | Centered + unit-scaled (no PCA); reshape with `(n_verts, 3)` |
| `centroid` | list(float32) | 3 | Patch centroid |
| `principal_axes` | list(float32) | 9 flattened | PCA rotation matrix; reshape with `(3, 3)` |
| `scale` | float32 | — | Bounding sphere radius |
| `boundary_vertices` | list(int32) | B | Boundary vertex local indices |
| `global_face_indices` | list(int32) | F | Original mesh face indices |

**Serialization:** All variable-length array columns use Parquet-native `list(float32)` or `list(int32)`. Arrays are flattened via `.flatten().tolist()` before insertion. Downstream reconstruction uses `n_faces` / `n_verts` columns for reshaping. Empty arrays (e.g., `boundary_vertices` for interior patches) produce empty lists `[]`.

**Estimated size:** ~2.5M rows, 5–15 GB Parquet (compressed). Exact size depends on vertex/face counts; validate after first batch.

## 4. Architecture

### 4.1 Processing Pipeline

```
Objaverse Pipeline (46K meshes, ~92 batches of 500):
  objaverse.load_lvis_annotations() → UID list
  for each batch of 500 UIDs:
    objaverse.load_objects(uids) → GLB files
    for each GLB:
      load_and_preprocess_mesh(glb, target_faces=1000) → trimesh
      segment_mesh_to_patches(mesh, target_patch_faces=35) → List[MeshPatch]
      patches_to_daft_rows(patches, mesh_id, category, "objaverse") → dict
      accumulate rows
    daft.from_pydict(accumulated_rows) → DataFrame
    df.write_huggingface("Pthahnix/MeshLex-Patches", io_config=config)
    cleanup GLB cache + gc.collect()
    ProgressTracker.mark_done(batch_id)

ShapeNet Pipeline (51K meshes, 55 synsets):
  for each synset_id in SHAPENET_SYNSET_MAP:
    snapshot_download(synset_id/**/*.obj) → local OBJs
    for each OBJ (in sub-batches of 500 meshes):
      load_and_preprocess_mesh(obj, target_faces=1000) → trimesh
      segment_mesh_to_patches(mesh, target_patch_faces=35) → List[MeshPatch]
      patches_to_daft_rows(patches, mesh_id, category, "shapenet") → dict
      accumulate rows
      if sub-batch full → write_huggingface() + clear
    write remaining rows
    cleanup synset directory
    ProgressTracker.mark_done(synset_id)

Post-processing:
  daft.read_parquet("hf://datasets/Pthahnix/MeshLex-Patches/**/*.parquet",
                    io_config=config) → lazy DataFrame
  df.select("mesh_id", "category", "source").distinct().collect() → unique meshes
  compute stats (total meshes, patches, category distribution)
  generate_splits(holdout_count=100, test_ratio=0.2, seed=42)
  upload splits.json + stats.json to HF via huggingface_hub
```

### 4.2 Core Conversion Function

```python
# src/daft_utils.py

def patches_to_daft_rows(
    patches: list[MeshPatch],
    mesh_id: str,
    category: str,
    source: str,
) -> dict[str, list]:
    """Convert one mesh's patches to column-oriented dict for Daft.

    Returns {col_name: [values...]}, one element per patch.
    Caller accumulates across meshes, then calls daft.from_pydict().
    """
    rows = {
        "mesh_id": [], "patch_idx": [], "category": [], "source": [],
        "n_faces": [], "n_verts": [],
        "faces": [], "vertices": [],
        "local_vertices": [], "local_vertices_nopca": [],
        "centroid": [], "principal_axes": [],
        "scale": [], "boundary_vertices": [], "global_face_indices": [],
    }
    for i, p in enumerate(patches):
        rows["mesh_id"].append(mesh_id)
        rows["patch_idx"].append(i)
        rows["category"].append(category)
        rows["source"].append(source)
        rows["n_faces"].append(p.faces.shape[0])
        rows["n_verts"].append(p.local_vertices.shape[0])
        # Flatten arrays to lists for Parquet-native list(int32)/list(float32) columns
        rows["faces"].append(p.faces.astype(np.int32).flatten().tolist())
        rows["vertices"].append(p.vertices.astype(np.float32).flatten().tolist())
        rows["local_vertices"].append(p.local_vertices.astype(np.float32).flatten().tolist())
        rows["local_vertices_nopca"].append(p.local_vertices_nopca.astype(np.float32).flatten().tolist())
        rows["centroid"].append(p.centroid.astype(np.float32).tolist())
        rows["principal_axes"].append(p.principal_axes.astype(np.float32).flatten().tolist())
        rows["scale"].append(float(p.scale))
        rows["boundary_vertices"].append(np.array(p.boundary_vertices, dtype=np.int32).tolist())
        rows["global_face_indices"].append(p.global_face_indices.astype(np.int32).tolist())
    return rows
```

### 4.3 Daft HF Write Configuration

```python
from daft.io import IOConfig, HuggingFaceConfig

def get_hf_io_config() -> IOConfig:
    return IOConfig(hf=HuggingFaceConfig(
        token=os.environ.get("HF_TOKEN"),
        target_filesize=128_000_000,       # 128 MB per Parquet file
        max_operations_per_commit=50,      # avoid large commit timeouts
    ))
```

### 4.4 Row Accumulation Pattern

```python
def accumulate_rows(target: dict, source: dict):
    """Merge source rows into target (in-place)."""
    for key in target:
        target[key].extend(source[key])

def make_empty_rows() -> dict[str, list]:
    """Create empty row accumulator."""
    return {col: [] for col in [
        "mesh_id", "patch_idx", "category", "source",
        "n_faces", "n_verts", "faces", "vertices",
        "local_vertices", "local_vertices_nopca",
        "centroid", "principal_axes", "scale",
        "boundary_vertices", "global_face_indices",
    ]}
```

## 5. File Changes

### New Files

| File | Responsibility |
|------|---------------|
| `src/daft_utils.py` | `patches_to_daft_rows()`, `get_hf_io_config()`, `accumulate_rows()`, `make_empty_rows()` |
| `src/stream_utils.py` | `ProgressTracker`, `MetadataCollector`, `batch_uids()`, `SHAPENET_SYNSET_MAP` |
| `scripts/stream_objaverse_daft.py` | Objaverse-LVIS streaming → Daft → HF |
| `scripts/stream_shapenet_daft.py` | ShapeNetCore v2 streaming → Daft → HF |
| `scripts/generate_splits_daft.py` | Read HF via Daft → generate splits → upload JSON |
| `scripts/validate_dataset_daft.py` | Read HF via Daft → validate thresholds |
| `scripts/run_dataset_pipeline.sh` | Overnight tmux orchestrator |
| `tests/test_daft_utils.py` | Unit tests for daft_utils |
| `tests/test_stream_utils.py` | Unit tests for stream helpers |
| `tests/test_generate_splits.py` | Unit tests for split logic |

### Modified Files

| File | Change |
|------|--------|
| `src/patch_segment.py` | Add `local_vertices_nopca: np.ndarray = None` to `MeshPatch` dataclass. In `segment_mesh_to_patches()`, after the PCA normalization call, compute: `centered = vertices - centroid; local_verts_nopca = centered / scale if scale > 1e-8 else centered`. Pass to `MeshPatch` constructor. |

### Not Created (vs original plan)

| Original Plan File | Reason |
|-------------------|--------|
| `src/patch_segment_dual.py` | NPZ serialization eliminated; data goes directly to Daft |
| `scripts/stream_objaverse.py` | Replaced by `_daft.py` version |
| `scripts/stream_shapenet.py` | Replaced by `_daft.py` version |

## 6. Disk Management

| Phase | Peak Disk | Strategy |
|-------|-----------|----------|
| Objaverse batch | ~2 GB | 500 GLBs downloaded → processed → cache cleared |
| ShapeNet synset | ~3 GB | One synset downloaded → processed → deleted |
| Daft DataFrame | In-memory only | `from_pydict()` → `write_huggingface()` → GC |
| **Total peak** | **< 5 GB** | Well within 80 GB container disk |

## 7. Resume Safety

- `ProgressTracker` persists completed batch/synset IDs to `progress.json`
- On restart, completed batches are skipped
- Daft `write_huggingface()` appends new Parquet files to the repo; each call creates new `.parquet` files without overwriting existing ones
- **Pre-implementation verification:** Before the full run, test append behavior with a 2-batch dry run: write batch A, then batch B, confirm both Parquet files exist on HF
- Metadata accumulated in `MetadataCollector`, saved after each batch
- If OOM occurs during batch accumulation, reduce batch size from 500 to 200

## 7.1 Cache Cleanup

After each Objaverse batch, clean:
- `~/.objaverse/hf-objaverse-v1/glbs/{uid_prefix}/{uid}.glb` (per-UID GLB files)
- `~/.cache/huggingface/hub/models--allenai--objaverse*` (HF hub cache)

After each ShapeNet synset, clean:
- `{work_dir}/shapenet_raw/{synset_id}/` (downloaded OBJ files)
- `~/.cache/huggingface/hub/datasets--ShapeNet--ShapeNetCore*/` (HF hub cache)

## 8. Validation Thresholds

| Check | Threshold |
|-------|-----------|
| Objaverse meshes | >= 35,000 |
| ShapeNet meshes | >= 45,000 |
| Total meshes | >= 75,000 |
| Total patches | >= 2,500,000 |
| Categories | >= 500 |
| All meshes in splits | 100% |
| Sample rows completeness | All columns non-null |

Validation script uses `daft.read_parquet("hf://datasets/Pthahnix/MeshLex-Patches/**/*.parquet", io_config=config)` to lazily read and check without downloading the full dataset. This works for any HF account tier (free/pro).

## 9. Dependencies

```
pip install "daft[huggingface]>=0.5.0" objaverse trimesh pyfqmr-fast pymetis numpy huggingface_hub
```

## 10. Execution Estimate

| Phase | Coding | Running |
|-------|--------|---------|
| MeshPatch nopca + daft_utils + stream_utils | ~30 min | — |
| Objaverse streaming script | ~15 min | 6–10h |
| ShapeNet streaming script | ~15 min | 4–6h |
| Splits + validation | ~15 min | ~5 min |
| **Total** | **~75 min** | **~10–16h overnight** |

## 11. Hardware

| Resource | Requirement |
|----------|-------------|
| GPU | Not needed (CPU-only) |
| vCPU | 16 cores |
| RAM | 32 GB sufficient |
| Disk | 80 GB (peak ~5 GB) |
| Network | Fast for HF + objaverse downloads |
