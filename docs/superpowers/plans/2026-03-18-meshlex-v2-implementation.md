# MeshLex v0.2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement MeshLex v0.2.0 — a 4-module pipeline (Partition → Tokenize → Generate → Stitch) with 2×2 ablation (METIS/BPE × SimVQ/RVQ) for compositional mesh generation via patch vocabulary.

**Architecture:** The pipeline decomposes meshes into ~130 patches (M1), encodes each patch as discrete tokens (M2), autoregressively generates patch token sequences (M3), and reassembles patches into watertight meshes (M4). Phase 0 validates BPE feasibility before committing to it. Phase 1 adds RVQ. Phase 3 adds AR generation. Phase 4 adds stitching.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric, trimesh, pymetis, numpy, scipy, matplotlib

**Spec:** `docs/superpowers/specs/2026-03-18-meshlex-v2-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `src/discretize.py` | Face feature discretization: icosphere binning, area binning, dihedral angle binning, MI computation |
| `src/dual_graph.py` | Dual graph construction from mesh: nodes=faces, edges=shared-edges, with discrete labels |
| `src/graph_bpe.py` | Graph BPE algorithm: bigram counting, greedy merge, vocabulary learning |
| `src/rvq.py` | 3-level Residual Vector Quantization using SimVQ per level |
| `src/model_rvq.py` | MeshLexRVQVAE: encoder → RVQ → decoder (same encoder/decoder as v1, new quantizer) |
| `src/patch_sequence.py` | Patch → token sequence: centroid quantization, scale quantization, Z-order (Morton code) |
| `src/ar_model.py` | GPT-2 style decoder-only Transformer for next-token prediction on patch sequences |
| `src/stitching.py` | Boundary alignment MLP + adjacency recovery + vertex merging |
| `src/metrics.py` | Extended metrics: Normal Consistency, F-Score, non-manifold edges, FID/COV/MMD |
| `scripts/run_phase0.py` | Phase 0: BPE feasibility validation (discretize → BPE → analysis → Go/No-Go) |
| `scripts/train_rvq.py` | Train MeshLexRVQVAE (extends train.py pattern) |
| `scripts/encode_sequences.py` | Encode all meshes through trained VQ-VAE → save per-mesh sequence NPZ files |
| `scripts/train_ar.py` | Train AR generation model |
| `scripts/generate.py` | Sample from AR model → decode patches → stitch → output mesh |
| `scripts/run_ablation.py` | 2×2 ablation matrix runner |
| `tests/test_discretize.py` | Tests for discretization |
| `tests/test_dual_graph.py` | Tests for dual graph construction |
| `tests/test_graph_bpe.py` | Tests for Graph BPE |
| `tests/test_rvq.py` | Tests for RVQ codebook |
| `tests/test_patch_sequence.py` | Tests for patch sequence encoding |
| `tests/test_ar_model.py` | Tests for AR Transformer |
| `tests/test_stitching.py` | Tests for stitching module |
| `tests/test_metrics.py` | Tests for extended evaluation metrics |

### Modified Files

| File | Changes |
|------|---------|
| `src/patch_segment.py` | Add `segment_mesh_bpe()` function that applies learned BPE vocabulary |
| `src/patch_dataset.py` | Add `MeshPatchSequenceDataset` for AR training (returns full-mesh patch sequences) |

---

## Phase 0: BPE Feasibility Validation (~2h CPU)

### Task 1: Face Feature Discretization

**Files:**
- Create: `src/discretize.py`
- Test: `tests/test_discretize.py`

- [ ] **Step 1: Write tests for icosphere binning**

```python
# tests/test_discretize.py
import numpy as np
import pytest
from src.discretize import (
    build_icosphere_bins,
    discretize_normal,
    discretize_area,
    discretize_dihedral,
    discretize_face_features,
    compute_discretization_mi,
)


def test_icosphere_bins_count():
    """Icosphere bins should have the requested number of directions."""
    bins = build_icosphere_bins(n_bins=64)
    assert bins.shape == (64, 3)
    # All should be unit vectors
    norms = np.linalg.norm(bins, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_discretize_normal_range():
    """Normal discretization should return indices in [0, n_bins)."""
    bins = build_icosphere_bins(64)
    normals = np.random.randn(100, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    indices = discretize_normal(normals, bins)
    assert indices.shape == (100,)
    assert indices.min() >= 0
    assert indices.max() < 64


def test_discretize_area_range():
    """Area discretization should return indices in [0, n_bins)."""
    areas = np.abs(np.random.randn(100)) * 0.1
    indices = discretize_area(areas, n_bins=8)
    assert indices.min() >= 0
    assert indices.max() < 8


def test_discretize_dihedral_range():
    """Dihedral discretization should return indices in [0, n_bins)."""
    angles = np.random.uniform(0, np.pi, 50)
    indices = discretize_dihedral(angles, n_bins=16)
    assert indices.min() >= 0
    assert indices.max() < 16


def test_discretize_face_features_combined():
    """Combined label = normal_bin * (area_bins * dihedral_bins) + area_bin * dihedral_bins + dihedral_bin."""
    normals = np.random.randn(10, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = np.abs(np.random.randn(10)) * 0.1
    labels = discretize_face_features(normals, areas, n_normal=64, n_area=8)
    assert labels.shape == (10,)
    assert labels.max() < 64 * 8


def test_mi_positive():
    """MI between discrete labels and continuous features should be non-negative."""
    np.random.seed(42)
    labels = np.random.randint(0, 10, 200)
    features = np.random.randn(200, 3)
    mi = compute_discretization_mi(labels, features)
    assert mi >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_discretize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.discretize'`

- [ ] **Step 3: Implement discretization module**

```python
# src/discretize.py
"""Face feature discretization for Graph BPE.

Converts continuous face features (normals, areas, dihedral angles)
into discrete labels for BPE bigram matching.
"""
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


def build_icosphere_bins(n_bins: int = 64) -> np.ndarray:
    """Build approximately uniform directions on the unit sphere.

    Uses Fibonacci lattice for near-uniform distribution.

    Returns:
        (n_bins, 3) unit vectors on the sphere.
    """
    indices = np.arange(n_bins, dtype=np.float64)
    golden_ratio = (1 + np.sqrt(5)) / 2

    theta = 2 * np.pi * indices / golden_ratio
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_bins)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    bins = np.stack([x, y, z], axis=1).astype(np.float32)
    bins /= np.linalg.norm(bins, axis=1, keepdims=True)
    return bins


def discretize_normal(normals: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Assign each normal to nearest bin direction.

    Args:
        normals: (N, 3) unit normal vectors
        bins: (B, 3) bin directions from build_icosphere_bins

    Returns:
        (N,) integer bin indices
    """
    # Use absolute dot product (normals and -normals are equivalent for faces)
    dots = np.abs(normals @ bins.T)  # (N, B)
    return dots.argmax(axis=1)


def discretize_area(areas: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Discretize face areas into log-scale bins.

    Args:
        areas: (N,) positive face areas
        n_bins: number of bins

    Returns:
        (N,) integer bin indices in [0, n_bins)
    """
    log_areas = np.log1p(areas)
    lo, hi = log_areas.min(), log_areas.max()
    if hi - lo < 1e-10:
        return np.zeros(len(areas), dtype=np.int64)
    normalized = (log_areas - lo) / (hi - lo)
    indices = np.clip((normalized * n_bins).astype(np.int64), 0, n_bins - 1)
    return indices


def discretize_dihedral(angles: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Discretize dihedral angles into uniform angular bins.

    Args:
        angles: (E,) angles in radians [0, pi]
        n_bins: number of bins

    Returns:
        (E,) integer bin indices in [0, n_bins)
    """
    normalized = angles / np.pi  # [0, 1]
    indices = np.clip((normalized * n_bins).astype(np.int64), 0, n_bins - 1)
    return indices


def discretize_face_features(
    normals: np.ndarray,
    areas: np.ndarray,
    n_normal: int = 64,
    n_area: int = 8,
) -> np.ndarray:
    """Combine normal and area discretization into single face (node) label.

    Note: Dihedral angles are edge-level features, discretized separately
    as edge labels in the dual graph (see dual_graph.py). The spec's
    "combined alphabet 64x8x16=8192" counts node labels (64*8=512) and
    edge labels (16) separately — the 8192 is the bigram space
    (node_label × edge_label × node_label), not the node label space.

    Args:
        normals: (N, 3) face normal vectors
        areas: (N,) face areas

    Returns:
        (N,) combined label indices in [0, n_normal * n_area)
    """
    bins = build_icosphere_bins(n_normal)
    normal_idx = discretize_normal(normals, bins)
    area_idx = discretize_area(areas, n_area)
    return normal_idx * n_area + area_idx


def compute_discretization_mi(
    labels: np.ndarray,
    continuous_features: np.ndarray,
    n_feature_bins: int = 20,
) -> float:
    """Compute mutual information between discrete labels and continuous features.

    Discretizes continuous features into bins, then computes MI.

    Args:
        labels: (N,) discrete labels
        continuous_features: (N, D) continuous feature matrix
        n_feature_bins: bins for discretizing continuous features

    Returns:
        Average MI across feature dimensions.
    """
    mi_total = 0.0
    n_dims = continuous_features.shape[1]

    for d in range(n_dims):
        col = continuous_features[:, d].reshape(-1, 1)
        kbd = KBinsDiscretizer(n_bins=n_feature_bins, encode="ordinal", strategy="quantile")
        binned = kbd.fit_transform(col).ravel().astype(int)
        mi_total += mutual_info_score(labels, binned)

    return mi_total / n_dims
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_discretize.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /g/MeshLex-Research
git add src/discretize.py tests/test_discretize.py
git commit -m "feat(phase0): add face feature discretization module"
git push
```

---

### Task 2: Dual Graph Construction

**Files:**
- Create: `src/dual_graph.py`
- Test: `tests/test_dual_graph.py`

- [ ] **Step 1: Write tests for dual graph**

```python
# tests/test_dual_graph.py
import numpy as np
import trimesh
import pytest
from src.dual_graph import build_labeled_dual_graph, DualGraph


def _make_simple_mesh():
    """4-triangle fan mesh (shared center vertex)."""
    vertices = np.array([
        [0, 0, 0],   # center
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, -1, 0],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_dual_graph_node_count():
    """Dual graph should have one node per face."""
    mesh = _make_simple_mesh()
    dg = build_labeled_dual_graph(mesh, n_normal_bins=32, n_area_bins=4, n_dihedral_bins=8)
    assert dg.n_nodes == 4


def test_dual_graph_edges_symmetric():
    """Each edge (u, v) should have a corresponding (v, u)."""
    mesh = _make_simple_mesh()
    dg = build_labeled_dual_graph(mesh, n_normal_bins=32, n_area_bins=4, n_dihedral_bins=8)
    edge_set = set()
    for u, v in zip(dg.edge_src, dg.edge_dst):
        edge_set.add((u, v))
    for u, v in zip(dg.edge_src, dg.edge_dst):
        assert (v, u) in edge_set, f"Missing reverse edge ({v}, {u})"


def test_dual_graph_labels_valid():
    """Node and edge labels should be non-negative integers."""
    mesh = _make_simple_mesh()
    dg = build_labeled_dual_graph(mesh, n_normal_bins=32, n_area_bins=4, n_dihedral_bins=8)
    assert all(l >= 0 for l in dg.node_labels)
    assert all(l >= 0 for l in dg.edge_labels)
    assert len(dg.node_labels) == dg.n_nodes
    assert len(dg.edge_labels) == len(dg.edge_src)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_dual_graph.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement dual graph module**

```python
# src/dual_graph.py
"""Dual graph construction from triangle mesh.

Each face becomes a node. Two nodes are connected if the corresponding
faces share an edge. Nodes carry discretized face labels; edges carry
discretized dihedral angle labels.
"""
from dataclasses import dataclass, field
import numpy as np
import trimesh

from src.discretize import (
    build_icosphere_bins,
    discretize_normal,
    discretize_area,
    discretize_dihedral,
)


@dataclass
class DualGraph:
    """Labeled dual graph of a triangle mesh."""
    n_nodes: int
    node_labels: np.ndarray      # (N,) int — combined normal+area label per face
    edge_src: np.ndarray          # (E,) int — source face index
    edge_dst: np.ndarray          # (E,) int — dest face index
    edge_labels: np.ndarray       # (E,) int — dihedral angle bin
    # Optional: keep continuous features for analysis
    face_normals: np.ndarray = field(default=None)  # (N, 3)
    face_areas: np.ndarray = field(default=None)     # (N,)


def build_labeled_dual_graph(
    mesh: trimesh.Trimesh,
    n_normal_bins: int = 64,
    n_area_bins: int = 8,
    n_dihedral_bins: int = 16,
) -> DualGraph:
    """Build labeled dual graph from a triangle mesh.

    Args:
        mesh: Triangle mesh (trimesh.Trimesh)
        n_normal_bins: Number of normal direction bins (icosphere)
        n_area_bins: Number of face area bins (log-scale)
        n_dihedral_bins: Number of dihedral angle bins (uniform 0-pi)

    Returns:
        DualGraph with node/edge labels
    """
    n_faces = len(mesh.faces)

    # Face normals and areas
    face_normals = mesh.face_normals.copy()  # (N, 3)
    face_areas = mesh.area_faces.copy()       # (N,)

    # Discretize node labels: combined normal + area
    ico_bins = build_icosphere_bins(n_normal_bins)
    normal_idx = discretize_normal(face_normals, ico_bins)
    area_idx = discretize_area(face_areas, n_area_bins)
    node_labels = normal_idx * n_area_bins + area_idx

    # Build edges from face adjacency
    face_adj = mesh.face_adjacency            # (E_undirected, 2)
    dihedral_angles = mesh.face_adjacency_angles  # (E_undirected,) radians

    # Make bidirectional
    src = np.concatenate([face_adj[:, 0], face_adj[:, 1]])
    dst = np.concatenate([face_adj[:, 1], face_adj[:, 0]])
    angles = np.concatenate([dihedral_angles, dihedral_angles])

    edge_labels = discretize_dihedral(angles, n_dihedral_bins)

    return DualGraph(
        n_nodes=n_faces,
        node_labels=node_labels,
        edge_src=src,
        edge_dst=dst,
        edge_labels=edge_labels,
        face_normals=face_normals,
        face_areas=face_areas,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_dual_graph.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /g/MeshLex-Research
git add src/dual_graph.py tests/test_dual_graph.py
git commit -m "feat(phase0): add dual graph construction from mesh"
git push
```

---

### Task 3: Graph BPE Algorithm

**Files:**
- Create: `src/graph_bpe.py`
- Test: `tests/test_graph_bpe.py`

- [ ] **Step 1: Write tests for Graph BPE**

```python
# tests/test_graph_bpe.py
import numpy as np
import pytest
from src.graph_bpe import GraphBPE, BPEVocabulary
from src.dual_graph import DualGraph


def _make_dual_graph(n_nodes=6, n_edges=10, max_label=4, seed=42):
    """Create a small synthetic dual graph."""
    rng = np.random.RandomState(seed)
    node_labels = rng.randint(0, max_label, n_nodes)
    # Random edges (undirected → bidirectional)
    pairs = set()
    while len(pairs) < n_edges:
        u, v = sorted(rng.randint(0, n_nodes, 2))
        if u != v:
            pairs.add((u, v))
    pairs = list(pairs)
    src = np.array([p[0] for p in pairs] + [p[1] for p in pairs])
    dst = np.array([p[1] for p in pairs] + [p[0] for p in pairs])
    edge_labels = rng.randint(0, 3, len(src))
    return DualGraph(
        n_nodes=n_nodes,
        node_labels=node_labels,
        edge_src=src,
        edge_dst=dst,
        edge_labels=edge_labels,
    )


def test_bpe_vocabulary_grows():
    """After training, vocabulary should be larger than base alphabet."""
    graphs = [_make_dual_graph(seed=i) for i in range(5)]
    bpe = GraphBPE(target_vocab_size=10)
    vocab = bpe.train(graphs)
    assert len(vocab.symbols) > 4  # base alphabet has 4 labels


def test_bpe_encode_produces_patches():
    """Encoding a graph should produce a list of face groups."""
    graphs = [_make_dual_graph(n_nodes=20, n_edges=30, seed=i) for i in range(10)]
    bpe = GraphBPE(target_vocab_size=15)
    vocab = bpe.train(graphs)
    patches = bpe.encode(graphs[0], vocab)
    # Each patch is a list of face indices
    assert len(patches) > 0
    # All faces should be covered
    all_faces = set()
    for p in patches:
        all_faces.update(p.face_indices)
    assert all_faces == set(range(graphs[0].n_nodes))


def test_bpe_deterministic():
    """Same input should produce same output."""
    graphs = [_make_dual_graph(seed=i) for i in range(5)]
    bpe1 = GraphBPE(target_vocab_size=10)
    vocab1 = bpe1.train(graphs)
    bpe2 = GraphBPE(target_vocab_size=10)
    vocab2 = bpe2.train(graphs)
    assert len(vocab1.symbols) == len(vocab2.symbols)


def test_bpe_merge_count():
    """Number of merges should equal vocab_size - base_alphabet_size."""
    graphs = [_make_dual_graph(seed=i) for i in range(5)]
    bpe = GraphBPE(target_vocab_size=10)
    vocab = bpe.train(graphs)
    assert len(vocab.merge_rules) == len(vocab.symbols) - vocab.base_alphabet_size
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_graph_bpe.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Graph BPE**

```python
# src/graph_bpe.py
"""Graph BPE: Byte-Pair Encoding on labeled dual graphs.

Extends standard BPE from sequences to graphs. A "bigram" is a pair of
adjacent nodes (u, v) characterized by the triple (label_u, edge_label, label_v).
Merge = contract the most frequent bigram across all training graphs.

Reference: Spec Section 3.2, Graph Tokenization (Guo et al., 2026).
"""
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from src.dual_graph import DualGraph


@dataclass
class BPEPatch:
    """A group of faces produced by BPE encoding."""
    face_indices: list[int]
    token_id: int


@dataclass
class BPEVocabulary:
    """Learned BPE vocabulary."""
    symbols: list[str]                     # All symbol names (base + merged)
    merge_rules: list[tuple[str, str, str]]  # (label_u, edge_label, label_v) merge history
    base_alphabet_size: int


class GraphBPE:
    """Graph BPE learner and encoder."""

    def __init__(self, target_vocab_size: int = 2000):
        self.target_vocab_size = target_vocab_size

    def train(self, graphs: list[DualGraph]) -> BPEVocabulary:
        """Learn BPE vocabulary from a list of labeled dual graphs.

        Algorithm:
        1. Initialize vocabulary = all unique node labels (base alphabet)
        2. Count bigram frequencies across all graphs
        3. Merge most frequent bigram (greedy, ID-ordered)
        4. Repeat until vocab reaches target size

        Args:
            graphs: List of DualGraph objects

        Returns:
            BPEVocabulary with symbols and merge rules
        """
        # Work on mutable copies
        work_graphs = [self._to_mutable(g) for g in graphs]

        # Base alphabet: all unique node labels
        all_labels = set()
        for g in work_graphs:
            all_labels.update(str(l) for l in g["node_labels"])
        symbols = sorted(all_labels)
        base_size = len(symbols)
        merge_rules = []

        n_merges = self.target_vocab_size - base_size
        for step in range(n_merges):
            # Count bigrams
            bigram_counts = self._count_bigrams(work_graphs)
            if not bigram_counts:
                break

            # Most frequent bigram
            best_bigram = max(bigram_counts, key=bigram_counts.get)
            lu, le, lv = best_bigram

            # New merged symbol
            new_symbol = f"M{base_size + step}_{lu}_{le}_{lv}"
            symbols.append(new_symbol)
            merge_rules.append((lu, le, lv))

            # Apply merge to all graphs
            for g in work_graphs:
                self._apply_merge(g, lu, le, lv, new_symbol)

        return BPEVocabulary(
            symbols=symbols,
            merge_rules=merge_rules,
            base_alphabet_size=base_size,
        )

    def encode(self, graph: DualGraph, vocab: BPEVocabulary) -> list[BPEPatch]:
        """Apply learned merges to a graph, returning BPE patches.

        Each resulting node (after all merges) = one patch = group of original faces.

        Args:
            graph: DualGraph to encode
            vocab: Learned BPEVocabulary

        Returns:
            List of BPEPatch, each containing face indices and token ID.
        """
        g = self._to_mutable(graph)

        # Apply merge rules in order
        for i, (lu, le, lv) in enumerate(vocab.merge_rules):
            new_symbol = vocab.symbols[vocab.base_alphabet_size + i]
            self._apply_merge(g, lu, le, lv, new_symbol)

        # Build patches from remaining nodes
        patches = []
        symbol_to_id = {s: i for i, s in enumerate(vocab.symbols)}
        for node_id in range(len(g["node_labels"])):
            if g["alive"][node_id]:
                label = g["node_labels"][node_id]
                token_id = symbol_to_id.get(label, -1)
                patches.append(BPEPatch(
                    face_indices=sorted(g["face_groups"][node_id]),
                    token_id=token_id,
                ))

        return patches

    def _to_mutable(self, graph: DualGraph) -> dict:
        """Convert DualGraph to mutable working representation."""
        n = graph.n_nodes
        node_labels = [str(l) for l in graph.node_labels]
        face_groups = [[i] for i in range(n)]
        alive = [True] * n

        # Adjacency as dict of dicts: adj[u][v] = edge_label
        adj = {i: {} for i in range(n)}
        for src, dst, el in zip(graph.edge_src, graph.edge_dst, graph.edge_labels):
            adj[int(src)][int(dst)] = str(el)

        return {
            "node_labels": node_labels,
            "face_groups": face_groups,
            "alive": alive,
            "adj": adj,
            "next_id": n,
        }

    def _count_bigrams(self, graphs: list[dict]) -> Counter:
        """Count (label_u, edge_label, label_v) bigram frequencies."""
        counts = Counter()
        for g in graphs:
            seen = set()
            for u in range(len(g["alive"])):
                if not g["alive"][u]:
                    continue
                for v, el in g["adj"][u].items():
                    if not g["alive"][v]:
                        continue
                    if u >= v:
                        continue  # count each undirected edge once
                    lu = g["node_labels"][u]
                    lv = g["node_labels"][v]
                    # Canonical ordering: (min_label, edge, max_label)
                    if lu <= lv:
                        bigram = (lu, el, lv)
                    else:
                        bigram = (lv, el, lu)
                    counts[bigram] += 1
        return counts

    def _apply_merge(self, g: dict, lu: str, le: str, lv: str, new_symbol: str):
        """Merge all matching bigram pairs in a graph (greedy, ID-ordered).

        For each matched (u, v): contract into u, mark v dead, inherit v's edges.
        """
        merged_this_round = set()

        for u in range(len(g["alive"])):
            if not g["alive"][u] or u in merged_this_round:
                continue
            for v in sorted(g["adj"][u].keys()):
                if not g["alive"][v] or v in merged_this_round:
                    continue
                el = g["adj"][u].get(v)
                if el is None:
                    continue

                label_u = g["node_labels"][u]
                label_v = g["node_labels"][v]

                # Check match (both orderings)
                match = False
                if label_u == lu and el == le and label_v == lv:
                    match = True
                elif label_v == lu and el == le and label_u == lv:
                    match = True

                if not match:
                    continue

                # Merge v into u
                g["node_labels"][u] = new_symbol
                g["face_groups"][u].extend(g["face_groups"][v])
                g["alive"][v] = False
                merged_this_round.add(u)
                merged_this_round.add(v)

                # Inherit v's edges (skip u-v edge)
                for w, w_el in g["adj"][v].items():
                    if w == u or not g["alive"][w]:
                        continue
                    # Add edge u-w (keep existing if present — multigraph via overwrite)
                    if w not in g["adj"][u]:
                        g["adj"][u][w] = w_el
                        g["adj"][w][u] = w_el

                # Remove v from all adjacency
                for w in list(g["adj"][v].keys()):
                    if w in g["adj"] and v in g["adj"][w]:
                        del g["adj"][w][v]
                if v in g["adj"][u]:
                    del g["adj"][u][v]
                g["adj"][v] = {}

                break  # u is done for this round
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_graph_bpe.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /g/MeshLex-Research
git add src/graph_bpe.py tests/test_graph_bpe.py
git commit -m "feat(phase0): add Graph BPE algorithm for mesh dual graphs"
git push
```

---

### Task 4: Phase 0 Feasibility Script

**Files:**
- Create: `scripts/run_phase0.py`
- Depends on: Tasks 1-3

This script orchestrates Phase 0: loads meshes, builds dual graphs, runs BPE, computes MI, analyzes normal variance, and produces Go/No-Go report.

- [ ] **Step 1: Write Phase 0 script**

```python
# scripts/run_phase0.py
"""Phase 0: BPE Feasibility Validation.

Loads preprocessed meshes, builds dual graphs, runs Graph BPE,
computes discretization MI, analyzes within-token normal variance,
and produces Go/No-Go decision.

Usage:
    python scripts/run_phase0.py --mesh_dir data/meshes/lvis_wide \
        --output_dir results/phase0 --n_meshes 200
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import trimesh
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dual_graph import build_labeled_dual_graph
from src.graph_bpe import GraphBPE
from src.discretize import compute_discretization_mi
from src.patch_segment import segment_mesh_to_patches


def load_meshes(mesh_dir: str, n_meshes: int):
    """Load OBJ meshes from directory."""
    mesh_dir = Path(mesh_dir)
    obj_files = sorted(mesh_dir.rglob("*.obj"))[:n_meshes]
    meshes = []
    for f in tqdm(obj_files, desc="Loading meshes"):
        try:
            m = trimesh.load(str(f), force="mesh")
            if len(m.faces) >= 20:
                meshes.append(m)
        except Exception:
            continue
    return meshes


def compute_metis_normal_variance(meshes, n_sample=50):
    """Compute METIS patch within-patch normal variance as baseline."""
    variances = []
    for mesh in meshes[:n_sample]:
        patches = segment_mesh_to_patches(mesh)
        for patch in patches:
            face_normals = []
            for f in patch.faces:
                v0, v1, v2 = patch.vertices[f[0]], patch.vertices[f[1]], patch.vertices[f[2]]
                n = np.cross(v1 - v0, v2 - v0)
                norm = np.linalg.norm(n)
                if norm > 1e-8:
                    face_normals.append(n / norm)
            if len(face_normals) >= 2:
                normals_arr = np.array(face_normals)
                var = np.var(normals_arr, axis=0).sum()
                variances.append(var)
    return np.median(variances) if variances else 1.0


def analyze_bpe_normal_variance(meshes, patches_per_mesh, metis_median):
    """Compute fraction of BPE tokens with normal variance < METIS median.

    Only considers tokens with >=10 faces (H1a criterion).
    """
    n_pass = 0
    n_total = 0
    all_variances = []
    patch_sizes = []

    for mesh, patches in zip(meshes, patches_per_mesh):
        for patch in patches:
            patch_sizes.append(len(patch.face_indices))
            if len(patch.face_indices) < 10:
                continue
            n_total += 1
            normals = mesh.face_normals[patch.face_indices]
            var = np.var(normals, axis=0).sum()
            all_variances.append(var)
            if var < metis_median:
                n_pass += 1

    ratio = n_pass / max(n_total, 1)
    return ratio, all_variances, patch_sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", required=True)
    parser.add_argument("--output_dir", default="results/phase0")
    parser.add_argument("--n_meshes", type=int, default=200)
    parser.add_argument("--target_vocab", type=int, default=2000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Load meshes ---
    meshes = load_meshes(args.mesh_dir, args.n_meshes)
    print(f"Loaded {len(meshes)} meshes")

    # --- Test 3 discretization granularities ---
    granularities = {
        "coarse": {"n_normal": 32, "n_area": 4, "n_dihedral": 8},
        "medium": {"n_normal": 64, "n_area": 8, "n_dihedral": 16},
        "fine":   {"n_normal": 128, "n_area": 16, "n_dihedral": 32},
    }

    mi_results = {}
    for name, params in granularities.items():
        print(f"\n--- Discretization: {name} ---")
        all_labels = []
        all_features = []
        for mesh in meshes[:100]:
            dg = build_labeled_dual_graph(mesh, **params)
            all_labels.append(dg.node_labels)
            all_features.append(dg.face_normals)
        labels = np.concatenate(all_labels)
        features = np.concatenate(all_features)
        mi = compute_discretization_mi(labels, features)
        mi_results[name] = mi
        print(f"  MI = {mi:.4f}")

    # --- Choose best granularity ---
    best_gran = max(mi_results, key=mi_results.get)
    best_params = granularities[best_gran]
    print(f"\nBest granularity: {best_gran} (MI={mi_results[best_gran]:.4f})")

    # --- Build dual graphs ---
    dual_graphs = []
    for mesh in tqdm(meshes, desc="Building dual graphs"):
        dg = build_labeled_dual_graph(mesh, **best_params)
        dual_graphs.append(dg)

    # --- Run Graph BPE ---
    t0 = time.time()
    bpe = GraphBPE(target_vocab_size=args.target_vocab)
    vocab = bpe.train(dual_graphs)
    bpe_time = time.time() - t0
    print(f"BPE training: {bpe_time:.1f}s, vocab size: {len(vocab.symbols)}")

    # --- Encode meshes and get patches ---
    all_patches = []
    for dg in tqdm(dual_graphs, desc="BPE encoding"):
        patches = bpe.encode(dg, vocab)
        all_patches.append(patches)

    # --- METIS baseline normal variance ---
    metis_median = compute_metis_normal_variance(meshes)
    print(f"METIS median within-patch normal variance: {metis_median:.6f}")

    # --- BPE normal variance analysis (H1a) ---
    h1a_ratio, bpe_variances, patch_sizes = analyze_bpe_normal_variance(
        meshes, all_patches, metis_median,
    )
    print(f"H1a: {h1a_ratio:.1%} of BPE tokens (>=10 faces) have var < METIS median")

    # --- H5: MI check ---
    h5_pass = any(mi > 0.5 for mi in mi_results.values())

    # --- Patch size distribution ---
    patch_sizes = np.array(patch_sizes)
    size_stats = {
        "mean": float(np.mean(patch_sizes)),
        "std": float(np.std(patch_sizes)),
        "min": int(np.min(patch_sizes)),
        "max": int(np.max(patch_sizes)),
        "p5": float(np.percentile(patch_sizes, 5)),
        "p25": float(np.percentile(patch_sizes, 25)),
        "p50": float(np.percentile(patch_sizes, 50)),
        "p75": float(np.percentile(patch_sizes, 75)),
        "p95": float(np.percentile(patch_sizes, 95)),
    }

    # --- Go/No-Go Decision ---
    h1a_go = h1a_ratio >= 0.60
    h5_go = h5_pass

    decision = "GO" if (h1a_go and h5_go) else "NO-GO"

    report = {
        "n_meshes": len(meshes),
        "granularities_mi": mi_results,
        "best_granularity": best_gran,
        "vocab_size": len(vocab.symbols),
        "base_alphabet_size": vocab.base_alphabet_size,
        "n_merges": len(vocab.merge_rules),
        "bpe_training_time_sec": bpe_time,
        "metis_normal_var_median": metis_median,
        "h1a_ratio": h1a_ratio,
        "h1a_go": h1a_go,
        "h5_any_mi_above_0.5": h5_pass,
        "h5_go": h5_go,
        "patch_size_stats": size_stats,
        "decision": decision,
    }

    # --- Save report ---
    with open(out / "phase0_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- Visualizations ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MI by granularity
    axes[0].bar(mi_results.keys(), mi_results.values())
    axes[0].axhline(0.5, color="r", linestyle="--", label="H5 threshold")
    axes[0].set_ylabel("MI")
    axes[0].set_title("Discretization MI")
    axes[0].legend()

    # Patch size distribution
    axes[1].hist(patch_sizes, bins=50, edgecolor="black")
    axes[1].set_xlabel("Faces per BPE token")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"BPE Patch Sizes (mean={size_stats['mean']:.1f})")

    # Normal variance comparison
    if bpe_variances:
        axes[2].hist(bpe_variances, bins=50, alpha=0.7, label="BPE tokens")
        axes[2].axvline(metis_median, color="r", linestyle="--", label="METIS median")
        axes[2].set_xlabel("Within-token normal variance")
        axes[2].set_title(f"H1a: {h1a_ratio:.0%} below threshold")
        axes[2].legend()

    plt.suptitle(f"Phase 0 BPE Feasibility — Decision: {decision}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "phase0_dashboard.png", dpi=150)
    plt.close()

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"Phase 0 BPE Feasibility Report")
    print(f"{'='*60}")
    print(f"H1a (normal variance): {h1a_ratio:.1%} >= 60%? {'YES' if h1a_go else 'NO'}")
    print(f"H5 (MI > 0.5):        {'YES' if h5_go else 'NO'}")
    print(f"Decision:              {decision}")
    print(f"Report saved to:       {out / 'phase0_report.json'}")
    print(f"Dashboard saved to:    {out / 'phase0_dashboard.png'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script runs with a small test**

Run: `cd /g/MeshLex-Research && python scripts/run_phase0.py --mesh_dir data/meshes/lvis_wide --output_dir results/phase0 --n_meshes 20 --target_vocab 50`
Expected: Produces `results/phase0/phase0_report.json` and `phase0_dashboard.png`

- [ ] **Step 3: Commit**

```bash
cd /g/MeshLex-Research
git add scripts/run_phase0.py
git commit -m "feat(phase0): add BPE feasibility validation script"
git push
```

---

## Phase 1: RVQ Tokenizer (~12h GPU)

### Task 5: RVQ Codebook Module

**Files:**
- Create: `src/rvq.py`
- Test: `tests/test_rvq.py`

- [ ] **Step 1: Write tests for RVQ**

```python
# tests/test_rvq.py
import torch
import pytest
from src.rvq import ResidualVQ


def test_rvq_output_shape():
    """RVQ should produce quantized embedding and multi-level indices."""
    rvq = ResidualVQ(n_levels=3, K=64, dim=128)
    z = torch.randn(8, 128)
    z_hat, indices = rvq(z)
    assert z_hat.shape == (8, 128)
    assert indices.shape == (8, 3)  # 3 levels
    assert indices.min() >= 0
    assert indices.max() < 64


def test_rvq_residual_reduction():
    """Each RVQ level should reduce the residual norm."""
    rvq = ResidualVQ(n_levels=3, K=256, dim=64)
    z = torch.randn(32, 64)
    z_hat, indices = rvq(z)
    # Reconstruction error should be less than original norm
    recon_error = (z - z_hat.detach()).norm(dim=1).mean()
    original_norm = z.norm(dim=1).mean()
    assert recon_error < original_norm


def test_rvq_gradient_flow():
    """Gradients should flow through RVQ via straight-through."""
    rvq = ResidualVQ(n_levels=3, K=64, dim=128)
    z = torch.randn(4, 128, requires_grad=True)
    z_hat, _ = rvq(z)
    loss = z_hat.sum()
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_rvq_compute_loss():
    """RVQ loss should include commit and embed losses from all levels."""
    rvq = ResidualVQ(n_levels=3, K=64, dim=128)
    z = torch.randn(8, 128, requires_grad=True)
    z_hat, indices = rvq(z)
    commit_loss, embed_loss = rvq.compute_loss(z, indices)
    assert commit_loss.item() >= 0
    assert embed_loss.item() >= 0


def test_rvq_utilization():
    """With diverse inputs, each level should use multiple codes."""
    rvq = ResidualVQ(n_levels=3, K=32, dim=16)
    z = torch.randn(256, 16)
    _, indices = rvq(z)
    for level in range(3):
        unique = indices[:, level].unique().numel()
        assert unique >= 2, f"Level {level}: only {unique}/32 codes used"


def test_rvq_decode_indices():
    """decode_indices should reconstruct the same z_hat as forward."""
    rvq = ResidualVQ(n_levels=3, K=64, dim=32)
    z = torch.randn(8, 32)
    z_hat, indices = rvq(z)
    z_hat_decoded = rvq.decode_indices(indices)
    assert torch.allclose(z_hat.detach(), z_hat_decoded, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_rvq.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement RVQ module**

```python
# src/rvq.py
"""Residual Vector Quantization (RVQ) using SimVQ per level.

Each level quantizes the residual from the previous level.
All levels use SimVQ (frozen C + learnable W) to prevent collapse.

Reference: Spec Section 4.2.
"""
import torch
import torch.nn as nn
from src.model import SimVQCodebook


class ResidualVQ(nn.Module):
    """3-level Residual Vector Quantization.

    Level 1: coarse shape     — quantize z
    Level 2: detail residual  — quantize (z - z1_hat)
    Level 3: fine residual    — quantize (z - z1_hat - z2_hat)

    Output: z_hat = z1_hat + z2_hat + z3_hat, indices = (idx_1, idx_2, idx_3)
    """

    def __init__(self, n_levels: int = 3, K: int = 1024, dim: int = 128):
        super().__init__()
        self.n_levels = n_levels
        self.K = K
        self.dim = dim
        self.codebooks = nn.ModuleList([
            SimVQCodebook(K=K, dim=dim, use_rotation=False)
            for _ in range(n_levels)
        ])

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, dim) encoder output

        Returns:
            z_hat: (B, dim) sum of all level quantizations (straight-through)
            indices: (B, n_levels) codebook indices per level
        """
        residual = z
        z_hat = torch.zeros_like(z)
        all_indices = []

        for codebook in self.codebooks:
            quantized_st, indices = codebook(residual)
            z_hat = z_hat + quantized_st
            all_indices.append(indices)
            # Next level quantizes the residual
            residual = residual - quantized_st.detach()

        return z_hat, torch.stack(all_indices, dim=1)  # (B, n_levels)

    def compute_loss(self, z: torch.Tensor, indices: torch.Tensor):
        """Compute commit + embed loss across all levels.

        Args:
            z: (B, dim) original encoder output
            indices: (B, n_levels) indices from forward pass

        Returns:
            commit_loss, embed_loss (summed across levels)
        """
        commit_total = 0.0
        embed_total = 0.0
        residual = z

        for level, codebook in enumerate(self.codebooks):
            level_indices = indices[:, level]
            # Re-quantize to get the quantized value at this level
            quant_codebook = codebook.linear(codebook.codebook.weight)
            quantized = quant_codebook[level_indices]

            commit_loss = torch.mean((residual - quantized.detach()) ** 2)
            embed_loss = torch.mean((residual.detach() - quantized) ** 2)

            commit_total = commit_total + commit_loss
            embed_total = embed_total + embed_loss

            residual = residual - quantized.detach()

        return commit_total, embed_total

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices back to quantized embedding.

        Args:
            indices: (B, n_levels) codebook indices per level

        Returns:
            z_hat: (B, dim) sum of quantized embeddings from all levels
        """
        z_hat = torch.zeros(indices.shape[0], self.dim, device=indices.device)
        for level, codebook in enumerate(self.codebooks):
            quant_codebook = codebook.linear(codebook.codebook.weight)
            z_hat = z_hat + quant_codebook[indices[:, level]]
        return z_hat

    @torch.no_grad()
    def get_utilization(self, indices: torch.Tensor) -> list[float]:
        """Fraction of codebook entries used per level."""
        utils = []
        for level in range(self.n_levels):
            utils.append(indices[:, level].unique().numel() / self.K)
        return utils

    @torch.no_grad()
    def init_from_z(self, all_z: torch.Tensor):
        """Initialize all codebook levels from encoder outputs.

        Level 1: K-means on z. Level 2: K-means on residual. Etc.
        """
        from sklearn.cluster import MiniBatchKMeans

        residual = all_z.clone()
        for level, codebook in enumerate(self.codebooks):
            n_samples = len(residual)
            effective_k = min(self.K, n_samples)
            kmeans = MiniBatchKMeans(
                n_clusters=effective_k,
                batch_size=min(4096, n_samples),
                max_iter=100,
                random_state=42 + level,
            )
            kmeans.fit(residual.numpy())
            centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

            if effective_k < self.K:
                extra = self.K - effective_k
                pad_idx = torch.randint(0, effective_k, (extra,))
                noise = torch.randn(extra, centroids.shape[1]) * 0.01
                centroids = torch.cat([centroids, centroids[pad_idx] + noise])

            codebook.init_from_z(centroids.to(codebook.codebook.weight.device))

            # Compute residual for next level
            quant_cb = codebook.get_quant_codebook()
            dists = torch.cdist(residual.unsqueeze(0), quant_cb.cpu().unsqueeze(0)).squeeze(0)
            indices = dists.argmin(dim=1)
            quantized = quant_cb.cpu()[indices]
            residual = residual - quantized
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_rvq.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /g/MeshLex-Research
git add src/rvq.py tests/test_rvq.py
git commit -m "feat(phase1): add 3-level Residual Vector Quantization module"
git push
```

---

### Task 6: MeshLex RVQ-VAE Model

**Files:**
- Create: `src/model_rvq.py`
- Modify: `scripts/train.py` (add `--quantizer` flag)
- Test: `tests/test_model_rvq.py`

- [ ] **Step 1: Write tests for RVQ-VAE**

```python
# tests/test_model_rvq.py
import torch
from torch_geometric.data import Data, Batch
from src.model_rvq import MeshLexRVQVAE


def test_rvqvae_forward():
    """Full forward: graph → RVQ quantize → reconstruct → losses."""
    max_verts = 60
    model = MeshLexRVQVAE(codebook_size=64, n_levels=3, embed_dim=128, max_vertices=max_verts)

    graphs = []
    for _ in range(4):
        nf = 30
        x = torch.randn(nf, 15)
        ei = torch.stack([torch.randint(0, nf, (60,)), torch.randint(0, nf, (60,))])
        graphs.append(Data(x=x, edge_index=ei))

    batch = Batch.from_data_list(graphs)
    n_vertices = torch.tensor([20, 25, 18, 30])
    gt_vertices = torch.randn(4, max_verts, 3)

    result = model(batch.x, batch.edge_index, batch.batch, n_vertices, gt_vertices)

    assert result["recon_vertices"].shape == (4, max_verts, 3)
    assert result["indices"].shape == (4, 3)  # 3 RVQ levels
    assert result["total_loss"].requires_grad


def test_rvqvae_encode_only():
    """encode_only should return encoder embeddings."""
    model = MeshLexRVQVAE(codebook_size=64, n_levels=3, embed_dim=128)
    nf = 20
    x = torch.randn(nf, 15)
    ei = torch.stack([torch.randint(0, nf, (40,)), torch.randint(0, nf, (40,))])
    data = Data(x=x, edge_index=ei)
    batch = Batch.from_data_list([data])
    z = model.encode_only(batch.x, batch.edge_index, batch.batch)
    assert z.shape == (1, 128)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_model_rvq.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement MeshLexRVQVAE**

```python
# src/model_rvq.py
"""MeshLex VQ-VAE with RVQ quantizer.

Same encoder and decoder as v1, but replaces SimVQ with 3-level RVQ.
"""
import torch
import torch.nn as nn

from src.model import PatchEncoder, PatchDecoder
from src.rvq import ResidualVQ
from src.losses import chamfer_distance


class MeshLexRVQVAE(nn.Module):
    """MeshLex VQ-VAE with Residual Vector Quantization.

    Encoder → RVQ (3-level) → Decoder.
    """

    def __init__(
        self,
        in_dim: int = 15,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        codebook_size: int = 1024,
        n_levels: int = 3,
        max_vertices: int = 128,
        lambda_commit: float = 1.0,
        lambda_embed: float = 1.0,
        num_kv_tokens: int = 4,
    ):
        super().__init__()
        self.encoder = PatchEncoder(in_dim, hidden_dim, embed_dim)
        self.rvq = ResidualVQ(n_levels=n_levels, K=codebook_size, dim=embed_dim)
        self.decoder = PatchDecoder(embed_dim, max_vertices, num_kv_tokens=num_kv_tokens)
        self.max_vertices = max_vertices
        self.lambda_commit = lambda_commit
        self.lambda_embed = lambda_embed

    def forward(self, x, edge_index, batch, n_vertices, gt_vertices):
        z = self.encoder(x, edge_index, batch)
        z_q, indices = self.rvq(z)
        recon = self.decoder(z_q, n_vertices)

        mask = torch.arange(self.max_vertices, device=x.device).unsqueeze(0) < n_vertices.unsqueeze(1)
        recon_loss = chamfer_distance(recon, gt_vertices, mask)
        commit_loss, embed_loss = self.rvq.compute_loss(z, indices)

        total_loss = recon_loss + self.lambda_commit * commit_loss + self.lambda_embed * embed_loss

        return {
            "recon_vertices": recon,
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "embed_loss": embed_loss,
            "indices": indices,
            "z": z,
        }

    def encode_only(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)

    @property
    def codebook(self):
        """Compatibility: return first level codebook for utilization tracking."""
        return self.rvq.codebooks[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_model_rvq.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Write RVQ training script**

```python
# scripts/train_rvq.py
"""Train MeshLex RVQ-VAE on preprocessed patches.

Usage:
    python scripts/train_rvq.py \
        --train_dirs data/patches/lvis_wide/seen_train \
        --val_dirs data/patches/lvis_wide/seen_test \
        --checkpoint_dir data/checkpoints/rvq_lvis \
        --epochs 200 --batch_size 256
"""
import argparse
import torch
from torch.utils.data import ConcatDataset

from src.model_rvq import MeshLexRVQVAE
from src.patch_dataset import PatchGraphDataset
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dirs", nargs="+", required=True)
    parser.add_argument("--val_dirs", nargs="+", default=None)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--n_levels", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_kv_tokens", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="data/checkpoints/rvq")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_datasets = [PatchGraphDataset(d) for d in args.train_dirs]
    train_dataset = ConcatDataset(train_datasets)
    print(f"Training patches: {len(train_dataset)}")

    val_dataset = None
    if args.val_dirs:
        val_datasets = [PatchGraphDataset(d) for d in args.val_dirs]
        val_dataset = ConcatDataset(val_datasets)
        print(f"Validation patches: {len(val_dataset)}")

    model = MeshLexRVQVAE(
        codebook_size=args.codebook_size,
        n_levels=args.n_levels,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_kv_tokens=args.num_kv_tokens,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"RVQ-VAE: {n_params:,} total, {n_trainable:,} trainable")
    print(f"Codebook: {args.n_levels} levels × K={args.codebook_size}")

    ckpt_data = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Resumed from {args.resume}")
        if not missing and not unexpected:
            ckpt_data = ckpt

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        resume_checkpoint=ckpt_data,
    )
    trainer.train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
cd /g/MeshLex-Research
git add src/model_rvq.py tests/test_model_rvq.py scripts/train_rvq.py
git commit -m "feat(phase1): add MeshLex RVQ-VAE model and training script"
git push
```

---

## Phase 3: AR Generation Model (~30h GPU)

### Task 7: Patch Sequence Encoding

**Files:**
- Create: `src/patch_sequence.py`
- Test: `tests/test_patch_sequence.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_patch_sequence.py
import numpy as np
import pytest
from src.patch_sequence import (
    quantize_position,
    quantize_scale,
    morton_code_3d,
    patches_to_token_sequence,
)


def test_quantize_position_range():
    """Quantized positions should be in [0, n_bins)."""
    positions = np.random.randn(50, 3)
    q = quantize_position(positions, n_bins=256)
    assert q.min() >= 0
    assert q.max() < 256
    assert q.shape == (50, 3)


def test_quantize_scale_range():
    scales = np.abs(np.random.randn(20)) * 0.5
    q = quantize_scale(scales, n_bins=64)
    assert q.min() >= 0
    assert q.max() < 64
    assert q.shape == (20,)


def test_morton_code_ordering():
    """Morton code should provide deterministic spatial ordering."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    codes = morton_code_3d(positions, n_bins=256)
    assert len(codes) == 4
    # Ordering should be deterministic
    order = np.argsort(codes)
    assert len(set(order)) == 4


def test_patches_to_sequence_simvq():
    """SimVQ mode: 5 tokens per patch (pos_x, pos_y, pos_z, scale, tok)."""
    centroids = np.random.randn(10, 3)
    scales = np.abs(np.random.randn(10)) * 0.5
    tokens = np.random.randint(0, 4096, 10)
    seq = patches_to_token_sequence(centroids, scales, tokens, mode="simvq")
    assert seq.shape == (10 * 5,)  # 5 tokens per patch


def test_patches_to_sequence_rvq():
    """RVQ mode: 7 tokens per patch (pos_x, pos_y, pos_z, scale, tok1, tok2, tok3)."""
    centroids = np.random.randn(10, 3)
    scales = np.abs(np.random.randn(10)) * 0.5
    tokens = np.random.randint(0, 1024, (10, 3))
    seq = patches_to_token_sequence(centroids, scales, tokens, mode="rvq")
    assert seq.shape == (10 * 7,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_patch_sequence.py -v`
Expected: FAIL

- [ ] **Step 3: Implement patch sequence module**

```python
# src/patch_sequence.py
"""Patch sequence encoding for AR generation.

Converts a mesh's patches into a flat token sequence suitable for
autoregressive next-token prediction.

Token layout per patch:
  SimVQ: (pos_x, pos_y, pos_z, scale, tok)          — 5 tokens
  RVQ:   (pos_x, pos_y, pos_z, scale, tok1, tok2, tok3) — 7 tokens

Ordering: Z-order (Morton code) on patch centroids.

Reference: Spec Section 5.
"""
import numpy as np


def quantize_position(positions: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Quantize 3D positions to integer bins.

    Normalizes to [0, 1] range per axis across the mesh, then bins.

    Args:
        positions: (M, 3) patch centroids
        n_bins: number of quantization levels per axis

    Returns:
        (M, 3) integer bin indices in [0, n_bins)
    """
    if len(positions) == 0:
        return np.empty((0, 3), dtype=np.int64)
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span = hi - lo
    span[span < 1e-10] = 1.0  # avoid division by zero
    normalized = (positions - lo) / span  # [0, 1]
    quantized = np.clip((normalized * n_bins).astype(np.int64), 0, n_bins - 1)
    return quantized


def quantize_scale(scales: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Quantize patch scales to integer bins.

    Args:
        scales: (M,) patch bounding sphere radii

    Returns:
        (M,) integer bin indices in [0, n_bins)
    """
    if len(scales) == 0:
        return np.empty(0, dtype=np.int64)
    lo, hi = scales.min(), scales.max()
    if hi - lo < 1e-10:
        return np.zeros(len(scales), dtype=np.int64)
    normalized = (scales - lo) / (hi - lo)
    return np.clip((normalized * n_bins).astype(np.int64), 0, n_bins - 1)


def morton_code_3d(positions: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Compute Morton (Z-order) codes for 3D positions.

    Args:
        positions: (M, 3) float positions (will be quantized internally)

    Returns:
        (M,) int64 Morton codes for spatial ordering
    """
    q = quantize_position(positions, n_bins)
    codes = np.zeros(len(q), dtype=np.int64)
    for bit in range(10):  # 10 bits per axis → 30-bit Morton code
        codes |= ((q[:, 0] >> bit) & 1).astype(np.int64) << (3 * bit)
        codes |= ((q[:, 1] >> bit) & 1).astype(np.int64) << (3 * bit + 1)
        codes |= ((q[:, 2] >> bit) & 1).astype(np.int64) << (3 * bit + 2)
    return codes


def patches_to_token_sequence(
    centroids: np.ndarray,
    scales: np.ndarray,
    codebook_tokens: np.ndarray,
    mode: str = "rvq",
    n_pos_bins: int = 256,
    n_scale_bins: int = 64,
) -> np.ndarray:
    """Convert mesh patches to flat token sequence.

    Patches are sorted by Morton code (Z-order) for spatial locality.

    Args:
        centroids: (M, 3) patch centroids
        scales: (M,) patch scales
        codebook_tokens: (M,) for SimVQ or (M, 3) for RVQ
        mode: "simvq" or "rvq"
        n_pos_bins: position quantization bins
        n_scale_bins: scale quantization bins

    Returns:
        (M * tokens_per_patch,) flat token sequence with offsets applied
    """
    M = len(centroids)
    order = np.argsort(morton_code_3d(centroids, n_pos_bins))

    pos_q = quantize_position(centroids, n_pos_bins)
    scale_q = quantize_scale(scales, n_scale_bins)

    # Token offsets to create non-overlapping vocabulary ranges
    # pos_x: [0, n_pos_bins)
    # pos_y: [n_pos_bins, 2*n_pos_bins)
    # pos_z: [2*n_pos_bins, 3*n_pos_bins)
    # scale: [3*n_pos_bins, 3*n_pos_bins + n_scale_bins)
    # codebook tokens: offset by (3*n_pos_bins + n_scale_bins)
    offset_y = n_pos_bins
    offset_z = 2 * n_pos_bins
    offset_scale = 3 * n_pos_bins
    offset_code = 3 * n_pos_bins + n_scale_bins

    tokens_per_patch = 5 if mode == "simvq" else 7
    sequence = np.zeros(M * tokens_per_patch, dtype=np.int64)

    for i, idx in enumerate(order):
        base = i * tokens_per_patch
        sequence[base + 0] = pos_q[idx, 0]
        sequence[base + 1] = pos_q[idx, 1] + offset_y
        sequence[base + 2] = pos_q[idx, 2] + offset_z
        sequence[base + 3] = scale_q[idx] + offset_scale

        if mode == "simvq":
            sequence[base + 4] = codebook_tokens[idx] + offset_code
        else:  # rvq
            sequence[base + 4] = codebook_tokens[idx, 0] + offset_code
            sequence[base + 5] = codebook_tokens[idx, 1] + offset_code
            sequence[base + 6] = codebook_tokens[idx, 2] + offset_code

    return sequence


def compute_vocab_size(
    n_pos_bins: int = 256,
    n_scale_bins: int = 64,
    codebook_K: int = 1024,
) -> int:
    """Compute total vocabulary size for AR model.

    Layout: pos_x[0..256) | pos_y[256..512) | pos_z[512..768) |
            scale[768..832) | codebook[832..832+K)
    """
    return 3 * n_pos_bins + n_scale_bins + codebook_K
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_patch_sequence.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /g/MeshLex-Research
git add src/patch_sequence.py tests/test_patch_sequence.py
git commit -m "feat(phase3): add patch sequence encoding with Morton ordering"
git push
```

---

### Task 8: AR Transformer Model

**Files:**
- Create: `src/ar_model.py`
- Test: `tests/test_ar_model.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_ar_model.py
import torch
import pytest
from src.ar_model import PatchGPT


def test_patchgpt_forward_shape():
    """Forward pass should return logits with correct vocab size."""
    model = PatchGPT(vocab_size=2000, d_model=256, n_heads=4, n_layers=4, max_seq_len=512)
    tokens = torch.randint(0, 2000, (2, 100))
    logits = model(tokens)
    assert logits.shape == (2, 100, 2000)


def test_patchgpt_causal():
    """Output at position i should not depend on position j > i."""
    model = PatchGPT(vocab_size=100, d_model=128, n_heads=4, n_layers=2, max_seq_len=64)
    model.eval()
    tokens = torch.randint(0, 100, (1, 20))
    with torch.no_grad():
        logits_full = model(tokens)
        logits_prefix = model(tokens[:, :10])
    # First 10 positions should match
    assert torch.allclose(logits_full[:, :10], logits_prefix, atol=1e-5)


def test_patchgpt_loss():
    """Training loss should be computable."""
    model = PatchGPT(vocab_size=500, d_model=128, n_heads=4, n_layers=2, max_seq_len=128)
    tokens = torch.randint(0, 500, (4, 50))
    loss = model.compute_loss(tokens)
    assert loss.requires_grad
    assert loss.item() > 0


def test_patchgpt_generate():
    """Generate should produce tokens without errors."""
    model = PatchGPT(vocab_size=500, d_model=128, n_heads=4, n_layers=2, max_seq_len=128)
    model.eval()
    generated = model.generate(max_len=30, temperature=1.0)
    assert generated.shape[0] <= 30
    assert generated.min() >= 0
    assert generated.max() < 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_ar_model.py -v`
Expected: FAIL

- [ ] **Step 3: Implement AR Transformer**

```python
# src/ar_model.py
"""GPT-2 style decoder-only Transformer for patch sequence generation.

Predicts next token in a sequence of patch tokens. Each patch is
(pos_x, pos_y, pos_z, scale, tok_L1, [tok_L2, tok_L3]).

Reference: Spec Section 5.3.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGPT(nn.Module):
    """GPT-2 style Transformer for patch token sequence generation.

    Args:
        vocab_size: Total vocabulary size (positions + scales + codebook tokens)
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of Transformer layers
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, T) token indices

        Returns:
            (B, T, vocab_size) logits
        """
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"

        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.drop(self.token_emb(tokens) + self.pos_emb(pos))

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)

        for block in self.blocks:
            x = block(x, src_mask=causal_mask, is_causal=True)

        x = self.ln_f(x)
        return self.head(x)

    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute next-token prediction loss.

        Args:
            tokens: (B, T) full sequences. Predicts tokens[1:] from tokens[:-1].

        Returns:
            Cross-entropy loss scalar.
        """
        logits = self.forward(tokens[:, :-1])  # (B, T-1, V)
        targets = tokens[:, 1:]                 # (B, T-1)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size), targets.reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def generate(
        self,
        max_len: int = 910,
        temperature: float = 1.0,
        top_k: int = 50,
        prompt: torch.Tensor = None,
    ) -> torch.Tensor:
        """Autoregressively generate a token sequence.

        Args:
            max_len: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            prompt: Optional (T,) prompt tokens

        Returns:
            (L,) generated token sequence
        """
        self.eval()
        device = next(self.parameters()).device

        if prompt is not None:
            tokens = prompt.unsqueeze(0).to(device)
        else:
            # Start with a random position token
            tokens = torch.randint(0, 256, (1, 1), device=device)

        for _ in range(max_len - tokens.shape[1]):
            logits = self.forward(tokens[:, -self.max_seq_len:])
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens.squeeze(0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_ar_model.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /g/MeshLex-Research
git add src/ar_model.py tests/test_ar_model.py
git commit -m "feat(phase3): add GPT-2 style AR Transformer for patch generation"
git push
```

---

### Task 9: AR Training Script + Mesh Sequence Dataset

**Files:**
- Modify: `src/patch_dataset.py` — add `MeshSequenceDataset`
- Create: `scripts/train_ar.py`

- [ ] **Step 1: Add MeshSequenceDataset to patch_dataset.py**

Add at end of `src/patch_dataset.py`:

```python
class MeshSequenceDataset(Dataset):
    """Dataset that returns full-mesh patch token sequences for AR training.

    Each item = one mesh's patch sequence (all patches in Z-order).
    Requires pre-computed codebook indices from a trained VQ-VAE.

    Expects a directory with files like: {mesh_id}_sequence.npz
    Each NPZ contains: centroids (M,3), scales (M,), tokens (M,) or (M,3)
    """

    def __init__(self, sequence_dir: str, mode: str = "rvq", max_seq_len: int = 1024):
        self.sequence_dir = Path(sequence_dir)
        self.files = sorted(self.sequence_dir.glob("*_sequence.npz"))
        self.mode = mode
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from src.patch_sequence import patches_to_token_sequence

        data = np.load(str(self.files[idx]))
        centroids = data["centroids"]
        scales = data["scales"]
        tokens = data["tokens"]

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
```

- [ ] **Step 2: Write AR training script**

```python
# scripts/train_ar.py
"""Train AR generation model on patch token sequences.

Usage:
    python scripts/train_ar.py \
        --sequence_dir data/sequences/lvis_wide \
        --checkpoint_dir data/checkpoints/ar_rvq \
        --mode rvq --epochs 100 --batch_size 32
"""
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from src.ar_model import PatchGPT
from src.patch_dataset import MeshSequenceDataset
from src.patch_sequence import compute_vocab_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--checkpoint_dir", default="data/checkpoints/ar")
    parser.add_argument("--codebook_size", type=int, default=1024,
                        help="Codebook K (1024 for RVQ, 4096 for SimVQ)")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = MeshSequenceDataset(args.sequence_dir, mode=args.mode, max_seq_len=args.max_seq_len)
    print(f"Sequences: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Compute vocab size dynamically
    vocab_size = compute_vocab_size(codebook_K=args.codebook_size)
    print(f"Vocab size: {vocab_size} (codebook K={args.codebook_size})")

    model = PatchGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PatchGPT: {n_params / 1e6:.1f}M params")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    history = []
    config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_seq_len": args.max_seq_len,
    }

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)  # (B, seq_len, vocab_size)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        torch.cuda.empty_cache()

        metrics = {"epoch": epoch, "loss": avg_loss, "time_sec": elapsed}
        history.append(metrics)
        print(f"Epoch {epoch:03d} | loss {avg_loss:.4f} | {elapsed:.1f}s")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": config,
            }, ckpt_dir / f"checkpoint_epoch{epoch:03d}.pt")

    # Final checkpoint
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": config,
    }, ckpt_dir / "checkpoint_final.pt")

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete. Final checkpoint saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
cd /g/MeshLex-Research
git add src/patch_dataset.py scripts/train_ar.py
git commit -m "feat(phase3): add AR training script and mesh sequence dataset"
git push
```

---

### Task 9.5: Encode Meshes to Sequences (Bridge Task)

**Files:**
- Create: `scripts/encode_sequences.py`

This task bridges VQ-VAE training (Phase 1/2) and AR training (Phase 3). It encodes all meshes through a trained VQ-VAE and saves per-mesh sequence NPZ files containing centroids, scales, and codebook indices.

- [ ] **Step 1: Write encode_sequences script**

```python
# scripts/encode_sequences.py
"""Encode all meshes through trained VQ-VAE → save per-mesh sequence NPZ.

This bridges the VQ-VAE (Phase 1/2) and AR (Phase 3) training stages.
For each mesh, loads its patches, runs them through the trained VQ-VAE encoder
+ quantizer, then saves centroids, scales, and codebook indices as a single
NPZ file.

Usage:
    python scripts/encode_sequences.py \
        --patch_dirs data/patches/lvis_wide/chair data/patches/lvis_wide/table \
        --checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
        --output_dir data/sequences/lvis_wide \
        --mode rvq
"""
import argparse
import gc
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from src.patch_dataset import PatchGraphDataset
from src.model import MeshLexVQVAE
from src.model_rvq import MeshLexRVQVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dirs", nargs="+", required=True,
                        help="Directories with patch NPZ files (one per category)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained VQ-VAE checkpoint")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save per-mesh sequence NPZs")
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load VQ-VAE
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.mode == "rvq":
        model = MeshLexRVQVAE()
    else:
        model = MeshLexVQVAE()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # Group patches by mesh_id
    for patch_dir in args.patch_dirs:
        patch_dir = Path(patch_dir)
        npz_files = sorted(patch_dir.glob("*.npz"))
        print(f"Processing {len(npz_files)} patches from {patch_dir.name}...")

        # Group files by mesh_id (filename format: {mesh_id}_patch{N}.npz)
        mesh_groups = {}
        for f in npz_files:
            stem = f.stem
            if "_patch" in stem:
                mesh_id = stem.rsplit("_patch", 1)[0]
            else:
                mesh_id = stem
            mesh_groups.setdefault(mesh_id, []).append(f)

        # Load full dataset once for this directory
        dataset = PatchGraphDataset(str(patch_dir))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Encode all patches in one pass
        all_tokens = []
        all_centroids = []
        all_scales = []
        all_mesh_ids = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                z = model.encoder(batch.x, batch.edge_index, batch.batch)
                if args.mode == "rvq":
                    _, indices = model.rvq(z)  # (B, n_levels)
                else:
                    _, indices = model.codebook(z)  # (B,)

                all_tokens.append(indices.cpu().numpy())

                # Centroid and scale from patch metadata
                # These are stored in the NPZ and loaded as graph-level attrs
                if hasattr(batch, 'centroid'):
                    all_centroids.append(batch.centroid.cpu().numpy())
                if hasattr(batch, 'scale'):
                    all_scales.append(batch.scale.cpu().numpy())

        tokens = np.concatenate(all_tokens, axis=0)
        centroids = np.concatenate(all_centroids, axis=0) if all_centroids else None
        scales = np.concatenate(all_scales, axis=0) if all_scales else None

        # Map dataset indices back to mesh_ids and save per-mesh
        # This requires PatchGraphDataset to expose file paths or mesh_ids
        # Fallback: save all patches for this category as one sequence file
        # (AR model treats each mesh independently via mesh_id grouping)
        idx = 0
        for mesh_id, patch_files in mesh_groups.items():
            out_path = out / f"{mesh_id}_sequence.npz"
            if out_path.exists():
                idx += len(patch_files)
                continue

            n = len(patch_files)
            mesh_tokens = tokens[idx:idx + n]
            mesh_centroids = centroids[idx:idx + n] if centroids is not None else np.zeros((n, 3))
            mesh_scales = scales[idx:idx + n] if scales is not None else np.ones(n)
            idx += n

            np.savez(out_path, centroids=mesh_centroids, scales=mesh_scales, tokens=mesh_tokens)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"All sequences saved to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /g/MeshLex-Research
git add scripts/encode_sequences.py
git commit -m "feat(bridge): add script to encode meshes to AR-ready sequences"
git push
```

**Note**: Before running this script, `PatchGraphDataset` must expose centroid and scale as graph-level attributes. Add the following to `PatchGraphDataset.__getitem__` in `src/patch_dataset.py` during implementation:

```python
# In PatchGraphDataset.__getitem__, after creating the Data object:
data.centroid = torch.tensor(npz["centroid"], dtype=torch.float32)  # (3,)
data.scale = torch.tensor([npz["scale"]], dtype=torch.float32)      # (1,)
```

This requires that `run_preprocessing.py` saves `centroid` and `scale` fields in each patch NPZ file (already available from `MeshPatch.centroid` and `MeshPatch.scale` during preprocessing). If these fields are missing from existing NPZ files, re-run preprocessing or add a fallback that computes centroid from `gt_vertices`.

---

## Phase 2: BPE Partitioning (Conditional on Phase 0 Go)

> **This phase only executes if Phase 0 Go criteria are met (H1a + H5).**
> If No-Go → skip entirely, proceed with METIS-only configs (C1, C2).

### Task 9.6: BPE Mesh Partitioning Function

**Files:**
- Modify: `src/patch_segment.py` — add `segment_mesh_bpe()`
- Test: `tests/test_patch_segment.py` — add BPE partitioning test

- [ ] **Step 1: Write test for BPE partitioning**

Add to `tests/test_patch_segment.py`:

```python
def test_segment_mesh_bpe_returns_patches():
    """BPE partitioning should return a list of MeshPatch objects."""
    from src.patch_segment import segment_mesh_bpe, MeshPatch
    from src.graph_bpe import GraphBPE, BPEVocabulary

    # Create a simple mesh (icosphere)
    mesh = trimesh.creation.icosphere(subdivisions=3)

    # Create a minimal BPE vocabulary (just base symbols, no merges)
    # In practice this comes from a trained GraphBPE
    vocab = BPEVocabulary(symbols=[], merge_rules=[], base_alphabet_size=512)

    patches = segment_mesh_bpe(mesh, vocab)
    assert len(patches) > 0
    assert all(isinstance(p, MeshPatch) for p in patches)
    # All faces should be covered
    total_faces = sum(len(p.faces) for p in patches)
    assert total_faces == len(mesh.faces)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_patch_segment.py::test_segment_mesh_bpe_returns_patches -v`
Expected: FAIL

- [ ] **Step 3: Implement `segment_mesh_bpe`**

Add to `src/patch_segment.py`:

```python
def segment_mesh_bpe(
    mesh: trimesh.Trimesh,
    bpe_vocab,
    n_normal_bins: int = 64,
    n_area_bins: int = 8,
    n_dihedral_bins: int = 16,
) -> list[MeshPatch]:
    """Segment mesh into patches using a trained BPE vocabulary.

    Each BPE token corresponds to a group of merged faces = one patch.

    Args:
        mesh: Input mesh
        bpe_vocab: Trained BPEVocabulary from GraphBPE
        n_normal_bins: Number of icosphere bins for normal discretization
        n_area_bins: Number of log-scale bins for area discretization
        n_dihedral_bins: Number of angular bins for dihedral angles

    Returns:
        List of MeshPatch objects
    """
    from src.discretize import discretize_face_features, discretize_dihedral
    from src.dual_graph import build_labeled_dual_graph
    from src.graph_bpe import GraphBPE

    # Build labeled dual graph
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces

    node_labels = discretize_face_features(face_normals, face_areas, n_normal_bins, n_area_bins)
    dual = build_labeled_dual_graph(mesh, node_labels, n_dihedral_bins)

    # Apply BPE encoding to get face groups
    bpe = GraphBPE()
    face_groups = bpe.encode(dual, bpe_vocab)

    # Build MeshPatch objects (same logic as METIS post-processing)
    adj_list = _build_face_adjacency(mesh)
    patches = []

    for face_indices in face_groups:
        face_indices = np.array(face_indices)
        patch_faces_global = mesh.faces[face_indices]

        unique_verts = np.unique(patch_faces_global.flatten())
        vert_map = {int(g): l for l, g in enumerate(unique_verts)}
        local_faces = np.vectorize(vert_map.get)(patch_faces_global)
        vertices = mesh.vertices[unique_verts]

        # Find boundary vertices
        face_set = set(face_indices.tolist())
        boundary_local = set()
        for fi in face_indices:
            for nf in adj_list[int(fi)]:
                if nf not in face_set:
                    shared = set(mesh.faces[int(fi)].tolist()) & set(mesh.faces[nf].tolist())
                    for v in shared:
                        if v in vert_map:
                            boundary_local.add(vert_map[v])

        local_verts, centroid, axes, scale = _normalize_patch_coords(vertices)

        patches.append(MeshPatch(
            faces=local_faces,
            vertices=vertices,
            global_face_indices=face_indices,
            boundary_vertices=sorted(boundary_local),
            centroid=centroid,
            principal_axes=axes,
            scale=scale,
            local_vertices=local_verts,
        ))

    return patches
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_patch_segment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /g/MeshLex-Research
git add src/patch_segment.py tests/test_patch_segment.py
git commit -m "feat(phase2): add BPE mesh partitioning function"
git push
```

---

Phase 2 execution (after Task 9.6):
1. Re-run preprocessing with BPE partitioning on all training meshes
2. Train C3 (BPE+SimVQ) using `scripts/train.py` with BPE-partitioned data
3. Train C4 (BPE+RVQ) using `scripts/train_rvq.py` with BPE-partitioned data
4. Encode BPE-partitioned meshes to AR sequences using `scripts/encode_sequences.py`

**Estimated GPU time**: ~20h (retrain 2 VQ-VAE models + re-encode sequences)

---

## Phase 4: Stitching & Full Pipeline (~15h GPU)

### Task 10: Stitching Module

**Files:**
- Create: `src/stitching.py`
- Test: `tests/test_stitching.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_stitching.py
import numpy as np
import torch
import pytest
from src.stitching import (
    infer_adjacency,
    StitchingMLP,
    merge_boundary_vertices,
)


def test_infer_adjacency():
    """Nearby patches should be marked as adjacent."""
    # 3 patches: 0 and 1 close, 2 far
    boundary_verts = [
        np.array([[0, 0, 0], [0.1, 0, 0]]),   # patch 0
        np.array([[0.05, 0, 0], [0.15, 0, 0]]),  # patch 1 (close to 0)
        np.array([[10, 10, 10], [10.1, 10, 10]]),  # patch 2 (far)
    ]
    adj = infer_adjacency(boundary_verts, threshold=0.2)
    assert (0, 1) in adj or (1, 0) in adj
    assert (0, 2) not in adj and (2, 0) not in adj


def test_stitching_mlp_shape():
    """MLP should output merged positions."""
    mlp = StitchingMLP(input_dim=9, hidden_dim=128)
    # 5 boundary vertex pairs
    x = torch.randn(5, 9)
    out = mlp(x)
    assert out.shape == (5, 3)


def test_merge_boundary_vertices():
    """Merge should reduce vertex count and update face indices."""
    verts_a = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0, 0]])
    verts_b = np.array([[0.51, 0, 0], [2, 0, 0]])  # verts_b[0] close to verts_a[2]
    faces_a = np.array([[0, 1, 2]])
    faces_b = np.array([[0, 1, 1]])  # dummy

    merged_v, merged_f = merge_boundary_vertices(
        verts_a, faces_a, verts_b, faces_b, threshold=0.1,
    )
    # Should have fewer total vertices than sum
    assert len(merged_v) < len(verts_a) + len(verts_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_stitching.py -v`
Expected: FAIL

- [ ] **Step 3: Implement stitching module**

```python
# src/stitching.py
"""Boundary stitching for patch-level mesh assembly.

Handles:
1. Adjacency recovery (geometric proximity)
2. Boundary vertex alignment (learned MLP or nearest-neighbor)
3. Inter-patch face connectivity (vertex merging)

Reference: Spec Section 6.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree


def infer_adjacency(
    boundary_vertices: list[np.ndarray],
    threshold: float = 0.05,
) -> set[tuple[int, int]]:
    """Infer patch adjacency from geometric proximity of boundary vertices.

    Args:
        boundary_vertices: List of (N_i, 3) arrays, one per patch
        threshold: Distance threshold for adjacency

    Returns:
        Set of (i, j) pairs where i < j
    """
    n_patches = len(boundary_vertices)
    adjacency = set()

    for i in range(n_patches):
        if len(boundary_vertices[i]) == 0:
            continue
        tree_i = cKDTree(boundary_vertices[i])
        for j in range(i + 1, n_patches):
            if len(boundary_vertices[j]) == 0:
                continue
            # Min distance between boundary vertex sets
            dists, _ = tree_i.query(boundary_vertices[j], k=1)
            if dists.min() < threshold:
                adjacency.add((i, j))

    return adjacency


class StitchingMLP(nn.Module):
    """Small MLP to predict merged boundary vertex positions.

    Input: concat(vert_i, vert_j, relative_offset) = 9 dims
    Output: merged position = 3 dims
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 9) concatenated boundary vertex pairs + relative offset

        Returns:
            (N, 3) predicted merged positions
        """
        return self.net(x)


def merge_boundary_vertices(
    verts_a: np.ndarray,
    faces_a: np.ndarray,
    verts_b: np.ndarray,
    faces_b: np.ndarray,
    threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two patches by matching and fusing nearby boundary vertices.

    Fallback method: nearest-neighbor matching without learned MLP.

    Args:
        verts_a: (Va, 3) vertices of patch A
        faces_a: (Fa, 3) faces of patch A (local indices)
        verts_b: (Vb, 3) vertices of patch B
        faces_b: (Fb, 3) faces of patch B (local indices)
        threshold: max distance for vertex matching

    Returns:
        merged_verts: (V_merged, 3)
        merged_faces: (F_merged, 3)
    """
    tree_a = cKDTree(verts_a)
    dists, match_a = tree_a.query(verts_b, k=1)

    # Build vertex mapping: b_idx -> a_idx (for matched) or new_idx
    offset_b = len(verts_a)
    b_to_merged = {}

    for b_idx in range(len(verts_b)):
        if dists[b_idx] < threshold:
            b_to_merged[b_idx] = match_a[b_idx]  # map to A's vertex
        else:
            b_to_merged[b_idx] = offset_b
            offset_b += 1

    # Merged vertices: A's vertices + unmatched B vertices
    unmatched_b = [i for i in range(len(verts_b)) if dists[i] >= threshold]
    if unmatched_b:
        merged_verts = np.concatenate([verts_a, verts_b[unmatched_b]])
    else:
        merged_verts = verts_a.copy()

    # Remap face B indices
    remapped_faces_b = np.array([
        [b_to_merged[v] for v in face]
        for face in faces_b
    ])

    merged_faces = np.concatenate([faces_a, remapped_faces_b])

    return merged_verts, merged_faces
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_stitching.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /g/MeshLex-Research
git add src/stitching.py tests/test_stitching.py
git commit -m "feat(phase4): add boundary stitching module for patch assembly"
git push
```

---

### Task 11: Generation Pipeline Script

**Files:**
- Create: `scripts/generate.py`

- [ ] **Step 1: Write generation script**

```python
# scripts/generate.py
"""Generate meshes from trained AR model + VQ-VAE decoder.

Usage:
    python scripts/generate.py \
        --ar_checkpoint data/checkpoints/ar/checkpoint_final.pt \
        --vqvae_checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
        --output_dir results/generated --n_meshes 10 --mode rvq
"""
import argparse
import torch
import numpy as np
import trimesh
from pathlib import Path

from src.ar_model import PatchGPT
from src.model_rvq import MeshLexRVQVAE
from src.model import MeshLexVQVAE
from src.patch_sequence import quantize_position, quantize_scale
from src.stitching import infer_adjacency, merge_boundary_vertices


def decode_token_sequence(
    sequence: np.ndarray,
    mode: str,
    n_pos_bins: int = 256,
    n_scale_bins: int = 64,
):
    """Decode flat token sequence back to patch parameters.

    Returns: list of dicts with pos, scale, codebook_tokens
    """
    tokens_per_patch = 5 if mode == "simvq" else 7
    offset_y = n_pos_bins
    offset_z = 2 * n_pos_bins
    offset_scale = 3 * n_pos_bins
    offset_code = 3 * n_pos_bins + n_scale_bins

    n_patches = len(sequence) // tokens_per_patch
    patches = []

    for i in range(n_patches):
        base = i * tokens_per_patch
        pos_x = sequence[base + 0]
        pos_y = sequence[base + 1] - offset_y
        pos_z = sequence[base + 2] - offset_z
        scale = sequence[base + 3] - offset_scale

        if mode == "simvq":
            tok = sequence[base + 4] - offset_code
            patches.append({"pos": [pos_x, pos_y, pos_z], "scale": scale, "tokens": tok})
        else:
            tok1 = sequence[base + 4] - offset_code
            tok2 = sequence[base + 5] - offset_code
            tok3 = sequence[base + 6] - offset_code
            patches.append({"pos": [pos_x, pos_y, pos_z], "scale": scale, "tokens": [tok1, tok2, tok3]})

    return patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar_checkpoint", required=True)
    parser.add_argument("--vqvae_checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/generated")
    parser.add_argument("--n_meshes", type=int, default=10)
    parser.add_argument("--mode", choices=["simvq", "rvq"], default="rvq")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load AR model
    ar_ckpt = torch.load(args.ar_checkpoint, map_location=device, weights_only=False)
    ar_model = PatchGPT(**ar_ckpt.get("config", {})).to(device)
    ar_model.load_state_dict(ar_ckpt["model_state_dict"])
    ar_model.eval()

    # Load VQ-VAE
    vq_ckpt = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
    if args.mode == "rvq":
        vqvae = MeshLexRVQVAE().to(device)
    else:
        vqvae = MeshLexVQVAE().to(device)
    vqvae.load_state_dict(vq_ckpt["model_state_dict"], strict=False)
    vqvae.eval()

    for mesh_idx in range(args.n_meshes):
        print(f"Generating mesh {mesh_idx + 1}/{args.n_meshes}...")

        # Generate token sequence
        with torch.no_grad():
            seq = ar_model.generate(
                max_len=910 if args.mode == "rvq" else 650,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        seq_np = seq.cpu().numpy()
        patch_params = decode_token_sequence(seq_np, mode=args.mode)
        print(f"  Generated {len(patch_params)} patches")

        # Decode each patch through VQ-VAE decoder
        all_verts = []
        all_centroids = []
        all_scales = []
        boundary_verts_list = []

        for p in patch_params:
            with torch.no_grad():
                # Look up codebook embeddings
                if args.mode == "rvq":
                    tok_indices = torch.tensor(p["tokens"], dtype=torch.long, device=device)
                    z_hat = vqvae.rvq.decode_indices(tok_indices.unsqueeze(0))  # (1, dim)
                else:
                    tok_idx = torch.tensor([p["tokens"]], dtype=torch.long, device=device)
                    cw = vqvae.codebook.get_quant_codebook()
                    z_hat = cw[tok_idx]  # (1, dim)

                # Decode to local vertices (use max_vertices=60, estimate n_verts=30)
                n_verts = torch.tensor([30], device=device)
                local_verts = vqvae.decoder(z_hat, n_verts)  # (1, 60, 3)
                local_verts = local_verts[0, :30].cpu().numpy()

            # Inverse transform: scale and translate back to world space
            pos = np.array(p["pos"]) / 255.0  # De-quantize position (256 bins → [0,1])
            scale = p["scale"] / 63.0  # De-quantize scale (64 bins → [0,1])
            world_verts = local_verts * scale + pos

            all_verts.append(world_verts)
            all_centroids.append(pos)
            all_scales.append(scale)
            # Boundary verts = vertices on edges of each patch (last few vertices heuristic)
            boundary_verts_list.append(world_verts)

        # Stitch patches together
        if len(all_verts) > 1:
            adj = infer_adjacency(boundary_verts_list, threshold=0.05)
            # Simple assembly: concatenate and merge nearby boundary vertices
            combined_verts = np.concatenate(all_verts, axis=0)
            # For now, save as point cloud; full stitching requires face connectivity
            mesh = trimesh.PointCloud(combined_verts)
        else:
            mesh = trimesh.PointCloud(all_verts[0])

        mesh.export(str(out / f"mesh_{mesh_idx:03d}.ply"))
        # Also save raw sequence for analysis
        np.savez(
            out / f"mesh_{mesh_idx:03d}_sequence.npz",
            sequence=seq_np,
            n_patches=len(patch_params),
        )

    print(f"Generated {args.n_meshes} meshes to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /g/MeshLex-Research
git add scripts/generate.py
git commit -m "feat(phase4): add mesh generation pipeline script"
git push
```

---

## Phase 5: Extended Metrics & Ablation

### Task 12: Extended Evaluation Metrics

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write tests for metrics**

```python
# tests/test_metrics.py
import numpy as np
import pytest
from src.metrics import normal_consistency, f_score, count_non_manifold_edges


def test_normal_consistency_identical():
    """Identical meshes should have NC = 1.0."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    nc = normal_consistency(verts, faces, verts, faces)
    assert nc > 0.99


def test_normal_consistency_range():
    """NC should be in [0, 1]."""
    pred_v = np.random.randn(20, 3)
    gt_v = np.random.randn(20, 3)
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    nc = normal_consistency(pred_v, faces, gt_v, faces)
    assert 0.0 <= nc <= 1.0


def test_f_score_identical():
    """Identical point clouds should have F-Score = 1.0."""
    verts = np.random.randn(100, 3)
    fs = f_score(verts, verts, threshold=0.01)
    assert fs > 0.99


def test_f_score_distant():
    """Distant point clouds should have F-Score ~ 0."""
    pred = np.random.randn(50, 3)
    gt = np.random.randn(50, 3) + 1000
    fs = f_score(pred, gt, threshold=0.01)
    assert fs < 0.01


def test_non_manifold_watertight():
    """A watertight tetrahedron has 0 non-manifold edges."""
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    n_nm, n_total = count_non_manifold_edges(faces)
    assert n_nm == 0
    assert n_total == 6


def test_non_manifold_open():
    """A single triangle has 3 boundary (non-manifold) edges."""
    faces = np.array([[0, 1, 2]])
    n_nm, n_total = count_non_manifold_edges(faces)
    assert n_nm == 3  # Each edge shared by only 1 face, not 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement extended metrics**

```python
# src/metrics.py
"""Extended evaluation metrics for MeshLex v0.2.0.

Includes Normal Consistency, F-Score, non-manifold detection,
and generation metrics (FID, COV, MMD).

Reference: Spec Section 8.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def normal_consistency(pred_verts, pred_faces, gt_verts, gt_faces):
    """Compute normal consistency between predicted and GT mesh.

    NC = average |dot(n_pred, n_gt_nearest)| for corresponding points.

    Returns: float in [0, 1], higher = better.
    """
    # Sample points and normals from both meshes
    # Simplified: compare face normals of nearest faces
    pred_centroids = pred_verts[pred_faces].mean(axis=1)  # (F, 3)
    gt_centroids = gt_verts[gt_faces].mean(axis=1)

    # Compute face normals
    def face_normals(verts, faces):
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return n / norms

    pred_n = face_normals(pred_verts, pred_faces)
    gt_n = face_normals(gt_verts, gt_faces)

    # Match pred faces to nearest GT faces
    tree = cKDTree(gt_centroids)
    _, indices = tree.query(pred_centroids)

    dots = np.abs(np.sum(pred_n * gt_n[indices], axis=1))
    return float(dots.mean())


def f_score(pred_verts, gt_verts, threshold=0.01):
    """Compute F-Score at given distance threshold.

    F = 2 * precision * recall / (precision + recall)
    precision = fraction of pred points within threshold of any GT point
    recall = fraction of GT points within threshold of any pred point

    Returns: float in [0, 1]
    """
    tree_gt = cKDTree(gt_verts)
    tree_pred = cKDTree(pred_verts)

    d_pred_to_gt, _ = tree_gt.query(pred_verts)
    d_gt_to_pred, _ = tree_pred.query(gt_verts)

    precision = (d_pred_to_gt < threshold).mean()
    recall = (d_gt_to_pred < threshold).mean()

    if precision + recall < 1e-10:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def count_non_manifold_edges(faces: np.ndarray) -> tuple[int, int]:
    """Count non-manifold edges (shared by != 2 faces).

    Returns: (n_non_manifold, n_total_edges)
    """
    edge_count = {}
    for face in faces:
        for i in range(3):
            e = tuple(sorted([face[i], face[(i + 1) % 3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    n_total = len(edge_count)
    n_non_manifold = sum(1 for c in edge_count.values() if c != 2)
    return n_non_manifold, n_total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /g/MeshLex-Research && python -m pytest tests/test_metrics.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
cd /g/MeshLex-Research
git add src/metrics.py tests/test_metrics.py
git commit -m "feat(phase5): add extended metrics (NC, F-Score, non-manifold) with tests"
git push
```

---

### Task 13: 2×2 Ablation Runner

**Files:**
- Create: `scripts/run_ablation.py`

- [ ] **Step 1: Write ablation runner**

```python
# scripts/run_ablation.py
"""Run 2×2 ablation matrix: Partition(METIS/BPE) × Tokenizer(SimVQ/RVQ).

Evaluates reconstruction quality for all 4 configs using the same
dataset and metrics.

Usage:
    python scripts/run_ablation.py --data_dir data/patches/lvis_wide \
        --output_dir results/ablation
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from src.model import MeshLexVQVAE
from src.model_rvq import MeshLexRVQVAE
from src.patch_dataset import PatchGraphDataset
from src.evaluate import evaluate_reconstruction
from src.metrics import f_score, count_non_manifold_edges


CONFIGS = {
    "C1_METIS_SimVQ": {
        "model_class": "MeshLexVQVAE",
        "partition": "METIS",
        "tokenizer": "SimVQ",
    },
    "C2_METIS_RVQ": {
        "model_class": "MeshLexRVQVAE",
        "partition": "METIS",
        "tokenizer": "RVQ",
    },
    "C3_BPE_SimVQ": {
        "model_class": "MeshLexVQVAE",
        "partition": "BPE",
        "tokenizer": "SimVQ",
    },
    "C4_BPE_RVQ": {
        "model_class": "MeshLexRVQVAE",
        "partition": "BPE",
        "tokenizer": "RVQ",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs=4, required=True,
                        help="Checkpoint paths for C1 C2 C3 C4")
    parser.add_argument("--data_dirs", nargs="+", required=True,
                        help="Evaluation data directories (METIS-partitioned)")
    parser.add_argument("--bpe_data_dirs", nargs="*", default=None,
                        help="BPE-partitioned data directories (for C3, C4)")
    parser.add_argument("--output_dir", default="results/ablation")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    config_names = list(CONFIGS.keys())

    for i, (name, config) in enumerate(CONFIGS.items()):
        print(f"\n{'='*60}")
        print(f"Evaluating {name}")
        print(f"{'='*60}")

        # Select data dirs based on partition type
        if config["partition"] == "BPE" and args.bpe_data_dirs:
            data_dirs = args.bpe_data_dirs
        else:
            data_dirs = args.data_dirs

        datasets = [PatchGraphDataset(d) for d in data_dirs]
        dataset = ConcatDataset(datasets)

        # Load model
        if config["model_class"] == "MeshLexRVQVAE":
            model = MeshLexRVQVAE()
        else:
            model = MeshLexVQVAE(num_kv_tokens=4)

        ckpt = torch.load(args.checkpoints[i], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model = model.to(device)

        # Evaluate
        metrics = evaluate_reconstruction(model, dataset, device=device)

        # Fix utilization for RVQ models: evaluate_reconstruction computes
        # utilization assuming flat indices. For RVQ, compute per-level util.
        if config["model_class"] == "MeshLexRVQVAE":
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
            all_indices = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    z = model.encoder(batch.x, batch.edge_index, batch.batch)
                    _, indices = model.rvq(z)  # (B, n_levels)
                    all_indices.append(indices.cpu())
            all_idx = torch.cat(all_indices, dim=0)
            per_level_util = []
            for lvl in range(all_idx.shape[1]):
                per_level_util.append(all_idx[:, lvl].unique().numel() / model.rvq.K)
            metrics["utilization"] = sum(per_level_util) / len(per_level_util)
            metrics["per_level_utilization"] = per_level_util

        results[name] = metrics
        print(f"  CD: {metrics['mean_cd']:.1f}, Util: {metrics['utilization']:.1%}")

    # Save results
    # Convert non-serializable items
    for name in results:
        results[name]["code_histogram"] = dict(results[name]["code_histogram"])
    with open(out / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"2×2 Ablation Results")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'CD':>8} {'Util':>8}")
    print(f"{'-'*36}")
    for name, m in results.items():
        print(f"{name:<20} {m['mean_cd']:8.1f} {m['utilization']:8.1%}")

    print(f"\nResults saved to {out / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd /g/MeshLex-Research
git add scripts/run_ablation.py
git commit -m "feat(phase5): add 2x2 ablation runner for METIS/BPE x SimVQ/RVQ"
git push
```

---

## Execution Dependency Graph

```
Task 1 (discretize)     Task 5 (RVQ)
    ↓                       ↓
Task 2 (dual graph)     Task 6 (RVQ-VAE + train script)
    ↓                       ↓
Task 3 (Graph BPE)      [Train RVQ-VAE on GPU ~12h]
    ↓                       ↓
Task 4 (Phase 0 script)  Task 7 (patch sequence)
    ↓                       ↓
[Run Phase 0 ~2h CPU]   Task 8 (AR Transformer)
    ↓                       ↓
 Go/No-Go              Task 9 (AR train script)
    ↓                       ↓
[If Go: Phase 2]       Task 9.5 (encode_sequences.py) ← needs trained VQ-VAE
    ↓                       ↓
[BPE partition +       [Encode meshes → sequences]
 retrain C3/C4 ~20h]       ↓
    ↓                  [Train AR model ~30h GPU]
    ↓                       ↓
    └───────────────→  Task 10 (stitching)
                            ↓
                        Task 11 (generation script)
                            ↓
                        Task 12 (extended metrics + tests)
                            ↓
                        Task 13 (ablation runner)
```

**Parallelizable:**
- Tasks 1-3 (Phase 0 code) can run in parallel with Tasks 5-6 (Phase 1 code)
- Phase 0 (CPU) can run simultaneously with RVQ training (GPU)
- Tasks 7-9 can be implemented while waiting for Phase 0/1 results
- Task 9.5 (encoding) must wait for at least one trained VQ-VAE checkpoint

**Go/No-Go gates:**
- After Phase 0: If BPE No-Go → skip Phase 2 (C3/C4), proceed with METIS only (C1, C2)
- After Phase 1: If RVQ worse than SimVQ → investigate before proceeding
- After Phase 3: If AR generations not plausible → simplify/tune before Phase 4

**Note on point-cloud conditioning:** The spec mentions point-cloud conditioned generation (cross-attention from PC encoder) as a comparison mode. This is deferred to after the unconditional pipeline is working end-to-end. If needed, it can be added as a follow-up task after Task 13.
