"""Pre-compute face features and edge indices for all patches.

Reads from Arrow splits using sequential iteration (much faster than random access),
computes features using vectorized numpy, and saves as memory-mapped numpy arrays.

Usage:
    PYTHONPATH=. python scripts/precompute_features.py \
        --split seen_train --output_dir /data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/features
"""
import argparse
import numpy as np
import time
from pathlib import Path
from datasets import load_from_disk

from src.patch_dataset import compute_face_features, build_face_edge_index

MAX_FACES = 80
MAX_VERTICES = 128
MAX_EDGES = 240  # max edges in face adjacency graph


def process_batch(batch, use_nopca: bool):
    """Process a batch of rows from HF datasets (columnar format)."""
    batch_size = len(batch["n_faces"])
    all_feats = np.zeros((batch_size, MAX_FACES, 15), dtype=np.float32)
    all_edges = np.zeros((batch_size, 2, MAX_EDGES), dtype=np.int64)
    all_verts = np.zeros((batch_size, MAX_VERTICES, 3), dtype=np.float32)
    all_n_verts = np.zeros(batch_size, dtype=np.int32)
    all_n_faces = np.zeros(batch_size, dtype=np.int32)
    all_n_edges = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        n_verts = batch["n_verts"][i]
        n_faces = batch["n_faces"][i]

        faces = np.array(batch["faces"][i], dtype=np.int64).reshape(-1, 3)[:n_faces]

        if use_nopca and batch.get("local_vertices_nopca") and batch["local_vertices_nopca"][i]:
            local_verts = np.array(batch["local_vertices_nopca"][i], dtype=np.float32).reshape(-1, 3)[:n_verts]
        else:
            local_verts = np.array(batch["local_vertices"][i], dtype=np.float32).reshape(-1, 3)[:n_verts]

        face_feats = compute_face_features(local_verts, faces)
        edge_index = build_face_edge_index(faces)
        n_edges = edge_index.shape[1] if edge_index.ndim == 2 and edge_index.shape[1] > 0 else 0

        all_feats[i, :n_faces] = face_feats
        if n_edges > 0:
            ne = min(n_edges, MAX_EDGES)
            all_edges[i, :, :ne] = edge_index[:, :ne]
            n_edges = ne
        all_verts[i, :n_verts] = local_verts
        all_n_verts[i] = n_verts
        all_n_faces[i] = n_faces
        all_n_edges[i] = n_edges

    return all_feats, all_edges, all_verts, all_n_verts, all_n_faces, all_n_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True,
                        help="Split name (seen_train, seen_test, unseen)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--nopca", action="store_true")
    parser.add_argument("--batch_size", type=int, default=5000)
    args = parser.parse_args()

    arrow_dir = f"/data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/splits/{args.split}"
    ds = load_from_disk(arrow_dir)
    N = len(ds)

    out_dir = Path(args.output_dir) / args.split
    if args.nopca:
        out_dir = Path(args.output_dir) / f"{args.split}_nopca"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already computed
    if (out_dir / "face_features.npy").exists():
        print(f"Already computed: {out_dir}")
        return

    print(f"Pre-computing features for {N} patches ({args.split})")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {out_dir}")

    # Create memory-mapped output arrays
    feats_mmap = np.lib.format.open_memmap(
        str(out_dir / "face_features.npy"), mode="w+",
        dtype=np.float32, shape=(N, MAX_FACES, 15))
    edges_mmap = np.lib.format.open_memmap(
        str(out_dir / "edge_index.npy"), mode="w+",
        dtype=np.int64, shape=(N, 2, MAX_EDGES))
    verts_mmap = np.lib.format.open_memmap(
        str(out_dir / "gt_vertices.npy"), mode="w+",
        dtype=np.float32, shape=(N, MAX_VERTICES, 3))
    nverts_mmap = np.lib.format.open_memmap(
        str(out_dir / "n_vertices.npy"), mode="w+",
        dtype=np.int32, shape=(N,))
    nfaces_mmap = np.lib.format.open_memmap(
        str(out_dir / "n_faces.npy"), mode="w+",
        dtype=np.int32, shape=(N,))
    nedges_mmap = np.lib.format.open_memmap(
        str(out_dir / "n_edges.npy"), mode="w+",
        dtype=np.int32, shape=(N,))

    t0 = time.time()
    done = 0
    bs = args.batch_size

    # Sequential batch iteration (much faster than random access)
    for start in range(0, N, bs):
        end = min(start + bs, N)
        batch = ds[start:end]  # HF datasets batch access returns columnar dict

        feats, edges, verts, nv, nf, ne = process_batch(batch, args.nopca)
        actual = end - start

        feats_mmap[done:done+actual] = feats[:actual]
        edges_mmap[done:done+actual] = edges[:actual]
        verts_mmap[done:done+actual] = verts[:actual]
        nverts_mmap[done:done+actual] = nv[:actual]
        nfaces_mmap[done:done+actual] = nf[:actual]
        nedges_mmap[done:done+actual] = ne[:actual]
        done += actual

        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (N - done) / rate if rate > 0 else 0
        if done % (bs * 10) == 0 or done >= N:
            print(f"  {done}/{N} ({done/N*100:.1f}%) | {rate:.0f} samples/sec | ETA {eta/60:.1f}min")

    # Flush
    del feats_mmap, edges_mmap, verts_mmap, nverts_mmap, nfaces_mmap, nedges_mmap

    elapsed = time.time() - t0
    print(f"\nDone! {N} patches in {elapsed/60:.1f}min ({N/elapsed:.0f} samples/sec)")
    print(f"Output: {out_dir}")

    for f in sorted(out_dir.glob("*.npy")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
