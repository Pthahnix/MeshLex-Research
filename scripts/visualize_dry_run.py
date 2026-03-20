"""Visualize dry-run dataset pipeline results.

Reads Parquet data from HF, reconstructs patches, and generates visualizations:
1. Per-mesh overview: all patches as colored point clouds in world space
2. Per-patch detail: PCA vs no-PCA normalization comparison
3. Dataset statistics summary

Usage:
    PYTHONPATH=/workspace/MeshLex-Research HF_TOKEN=... \
    python scripts/visualize_dry_run.py --output_dir results/dry-run-dataset-pipeline
"""
import argparse
import logging
import os
from pathlib import Path

import daft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from src.daft_utils import get_hf_io_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def reconstruct_patch(row):
    """Reconstruct patch arrays from a Parquet row (pandas Series)."""
    n_faces = int(row["n_faces"])
    n_verts = int(row["n_verts"])
    faces = np.array(row["faces"], dtype=np.int32).reshape(n_faces, 3)
    vertices = np.array(row["vertices"], dtype=np.float32).reshape(n_verts, 3)
    local_vertices = np.array(row["local_vertices"], dtype=np.float32).reshape(n_verts, 3)
    local_vertices_nopca = np.array(row["local_vertices_nopca"], dtype=np.float32).reshape(n_verts, 3)
    centroid = np.array(row["centroid"], dtype=np.float32)
    principal_axes = np.array(row["principal_axes"], dtype=np.float32).reshape(3, 3)
    scale = float(row["scale"])
    return {
        "faces": faces, "vertices": vertices,
        "local_vertices": local_vertices,
        "local_vertices_nopca": local_vertices_nopca,
        "centroid": centroid, "principal_axes": principal_axes,
        "scale": scale, "n_faces": n_faces, "n_verts": n_verts,
    }


def plot_mesh_patches_world(patches_data, mesh_id, category, source, output_path):
    """Plot all patches of a mesh in world space with different colors."""
    fig = plt.figure(figsize=(16, 6))

    # Plot 1: World-space point cloud with patch colors
    ax1 = fig.add_subplot(131, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, len(patches_data)))
    all_verts = []
    for i, pd in enumerate(patches_data):
        verts = pd["vertices"]
        all_verts.append(verts)
        ax1.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                    c=[colors[i % 20]], s=3, alpha=0.7)
    ax1.set_title(f"World Space\n{len(patches_data)} patches")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    _set_equal_axes(ax1, np.vstack(all_verts))

    # Plot 2: World-space wireframe
    ax2 = fig.add_subplot(132, projection="3d")
    for i, pd in enumerate(patches_data):
        verts = pd["vertices"]
        faces = pd["faces"]
        color = colors[i % 20]
        polys = [[verts[f[0]], verts[f[1]], verts[f[2]]] for f in faces]
        mesh_coll = Poly3DCollection(polys, alpha=0.3, facecolors=[color],
                                      edgecolors=[(0.2, 0.2, 0.2, 0.3)], linewidths=0.3)
        ax2.add_collection3d(mesh_coll)
    ax2.set_title("World Space (Wireframe)")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    _set_equal_axes(ax2, np.vstack(all_verts))

    # Plot 3: PCA-normalized patches (local space, overlaid)
    ax3 = fig.add_subplot(133, projection="3d")
    for i, pd in enumerate(patches_data[:6]):  # Show first 6 for clarity
        lv = pd["local_vertices"]
        ax3.scatter(lv[:, 0], lv[:, 1], lv[:, 2],
                    c=[colors[i % 20]], s=5, alpha=0.6, label=f"P{i}")
    ax3.set_title("PCA-Normalized (first 6)")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.legend(fontsize=6, loc="upper right")

    fig.suptitle(f"{source} / {category} / {mesh_id[:16]}...\n"
                 f"{sum(p['n_faces'] for p in patches_data)} faces, "
                 f"{sum(p['n_verts'] for p in patches_data)} verts, "
                 f"{len(patches_data)} patches", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


def plot_patch_normalization_comparison(pd, patch_idx, mesh_id, output_path):
    """Compare PCA vs no-PCA normalization for a single patch."""
    fig = plt.figure(figsize=(18, 5))

    # Original world-space
    ax1 = fig.add_subplot(141, projection="3d")
    v = pd["vertices"]
    f = pd["faces"]
    polys = [[v[fi[0]], v[fi[1]], v[fi[2]]] for fi in f]
    mesh_coll = Poly3DCollection(polys, alpha=0.5, facecolors=[(0.4, 0.6, 0.9)],
                                  edgecolors=[(0.2, 0.2, 0.2, 0.5)], linewidths=0.5)
    ax1.add_collection3d(mesh_coll)
    ax1.set_title(f"Original (world)\n{pd['n_faces']}F, {pd['n_verts']}V")
    _set_equal_axes(ax1, v)

    # PCA-normalized
    ax2 = fig.add_subplot(142, projection="3d")
    lv = pd["local_vertices"]
    polys2 = [[lv[fi[0]], lv[fi[1]], lv[fi[2]]] for fi in f]
    mesh_coll2 = Poly3DCollection(polys2, alpha=0.5, facecolors=[(0.3, 0.8, 0.4)],
                                   edgecolors=[(0.2, 0.2, 0.2, 0.5)], linewidths=0.5)
    ax2.add_collection3d(mesh_coll2)
    ax2.set_title(f"PCA Normalized\nscale={pd['scale']:.3f}")
    _set_equal_axes(ax2, lv)

    # No-PCA normalized
    ax3 = fig.add_subplot(143, projection="3d")
    lv_nopca = pd["local_vertices_nopca"]
    polys3 = [[lv_nopca[fi[0]], lv_nopca[fi[1]], lv_nopca[fi[2]]] for fi in f]
    mesh_coll3 = Poly3DCollection(polys3, alpha=0.5, facecolors=[(0.9, 0.5, 0.3)],
                                   edgecolors=[(0.2, 0.2, 0.2, 0.5)], linewidths=0.5)
    ax3.add_collection3d(mesh_coll3)
    ax3.set_title("No-PCA Normalized\n(center+scale only)")
    _set_equal_axes(ax3, lv_nopca)

    # Overlay: PCA vs no-PCA
    ax4 = fig.add_subplot(144, projection="3d")
    ax4.scatter(lv[:, 0], lv[:, 1], lv[:, 2], c="green", s=8, alpha=0.6, label="PCA")
    ax4.scatter(lv_nopca[:, 0], lv_nopca[:, 1], lv_nopca[:, 2], c="orange", s=8, alpha=0.6, label="no-PCA")
    ax4.set_title("PCA vs No-PCA Overlay")
    ax4.legend(fontsize=8)
    combined = np.vstack([lv, lv_nopca])
    _set_equal_axes(ax4, combined)

    fig.suptitle(f"Patch {patch_idx} from {mesh_id[:16]}...", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


def plot_processing_pipeline(pd, patch_idx, mesh_id, output_path):
    """Show the complete processing pipeline for a single patch."""
    fig = plt.figure(figsize=(20, 4))

    # Step 1: Original mesh in world space
    ax1 = fig.add_subplot(151, projection="3d")
    v = pd["vertices"]
    f = pd["faces"]
    polys = [[v[fi[0]], v[fi[1]], v[fi[2]]] for fi in f]
    mc = Poly3DCollection(polys, alpha=0.5, facecolors=[(0.7, 0.7, 0.7)],
                           edgecolors=[(0.3, 0.3, 0.3, 0.5)], linewidths=0.5)
    ax1.add_collection3d(mc)
    ax1.set_title("1. World Space\n(after decimate)")
    _set_equal_axes(ax1, v)

    # Step 2: Center at origin
    centroid = pd["centroid"]
    centered = v - centroid
    ax2 = fig.add_subplot(152, projection="3d")
    polys2 = [[centered[fi[0]], centered[fi[1]], centered[fi[2]]] for fi in f]
    mc2 = Poly3DCollection(polys2, alpha=0.5, facecolors=[(0.8, 0.7, 0.5)],
                            edgecolors=[(0.3, 0.3, 0.3, 0.5)], linewidths=0.5)
    ax2.add_collection3d(mc2)
    ax2.scatter([0], [0], [0], c="red", s=50, marker="+", label="origin")
    ax2.set_title(f"2. Centered\nc=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f})")
    _set_equal_axes(ax2, centered)

    # Step 3: PCA rotation
    Vt = pd["principal_axes"]
    rotated = centered @ Vt.T
    ax3 = fig.add_subplot(153, projection="3d")
    polys3 = [[rotated[fi[0]], rotated[fi[1]], rotated[fi[2]]] for fi in f]
    mc3 = Poly3DCollection(polys3, alpha=0.5, facecolors=[(0.4, 0.7, 0.9)],
                            edgecolors=[(0.3, 0.3, 0.3, 0.5)], linewidths=0.5)
    ax3.add_collection3d(mc3)
    ax3.set_title("3. PCA Rotated")
    _set_equal_axes(ax3, rotated)

    # Step 4: Scaled to unit sphere (PCA)
    ax4 = fig.add_subplot(154, projection="3d")
    lv = pd["local_vertices"]
    polys4 = [[lv[fi[0]], lv[fi[1]], lv[fi[2]]] for fi in f]
    mc4 = Poly3DCollection(polys4, alpha=0.5, facecolors=[(0.3, 0.8, 0.4)],
                            edgecolors=[(0.3, 0.3, 0.3, 0.5)], linewidths=0.5)
    ax4.add_collection3d(mc4)
    ax4.set_title(f"4. PCA + Scale\nscale={pd['scale']:.3f}")
    _set_equal_axes(ax4, lv)

    # Step 5: No-PCA (center + scale only)
    ax5 = fig.add_subplot(155, projection="3d")
    lv_nopca = pd["local_vertices_nopca"]
    polys5 = [[lv_nopca[fi[0]], lv_nopca[fi[1]], lv_nopca[fi[2]]] for fi in f]
    mc5 = Poly3DCollection(polys5, alpha=0.5, facecolors=[(0.9, 0.5, 0.3)],
                            edgecolors=[(0.3, 0.3, 0.3, 0.5)], linewidths=0.5)
    ax5.add_collection3d(mc5)
    ax5.set_title("5. No-PCA + Scale")
    _set_equal_axes(ax5, lv_nopca)

    fig.suptitle(f"Processing Pipeline: Patch {patch_idx} from {mesh_id[:16]}...", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


def _set_equal_axes(ax, points):
    """Set equal aspect ratio for 3D axes."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = max(maxs - mins) / 2 * 1.1
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def plot_stats_summary(df_pandas, output_path):
    """Plot dataset statistics from the dry-run data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Patch count distribution per mesh
    mesh_patch_counts = df_pandas.groupby("mesh_id").size()
    axes[0, 0].hist(mesh_patch_counts.values, bins=20, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Patches per Mesh")
    axes[0, 0].set_xlabel("# Patches")
    axes[0, 0].set_ylabel("Count")

    # Face count distribution per patch
    axes[0, 1].hist(df_pandas["n_faces"].values, bins=20, color="salmon", edgecolor="black")
    axes[0, 1].set_title("Faces per Patch")
    axes[0, 1].set_xlabel("# Faces")
    axes[0, 1].set_ylabel("Count")

    # Vertex count distribution per patch
    axes[1, 0].hist(df_pandas["n_verts"].values, bins=20, color="lightgreen", edgecolor="black")
    axes[1, 0].set_title("Vertices per Patch")
    axes[1, 0].set_xlabel("# Vertices")
    axes[1, 0].set_ylabel("Count")

    # Source distribution
    source_counts = df_pandas["source"].value_counts()
    axes[1, 1].bar(source_counts.index, source_counts.values, color=["steelblue", "salmon"])
    axes[1, 1].set_title("Patches by Source")
    axes[1, 1].set_ylabel("# Patches")

    # Summary text
    total_meshes = df_pandas["mesh_id"].nunique()
    total_patches = len(df_pandas)
    categories = df_pandas["category"].unique()
    fig.suptitle(
        f"Dry-Run Dataset Summary: {total_meshes} meshes, {total_patches} patches, "
        f"{len(categories)} categories\n"
        f"Sources: {dict(source_counts)}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="Pthahnix/MeshLex-Patches")
    parser.add_argument("--output_dir", default="results/dry-run-dataset-pipeline")
    parser.add_argument("--n_objaverse", type=int, default=10)
    parser.add_argument("--n_shapenet", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "objaverse").mkdir(exist_ok=True)
    (output_dir / "shapenet").mkdir(exist_ok=True)
    (output_dir / "pipeline").mkdir(exist_ok=True)

    io_config = get_hf_io_config()
    log.info("Reading Parquet from HF...")
    df = daft.read_parquet(
        f"hf://datasets/{args.hf_repo}/**/*.parquet", io_config=io_config,
    )
    pdf = df.collect().to_pandas()
    log.info(f"Total rows: {len(pdf)}, Unique meshes: {pdf['mesh_id'].nunique()}")

    # Stats summary
    plot_stats_summary(pdf, output_dir / "stats_summary.png")

    # Objaverse visualizations
    obj_meshes = pdf[pdf["source"] == "objaverse"]["mesh_id"].unique()[:args.n_objaverse]
    log.info(f"Visualizing {len(obj_meshes)} Objaverse meshes")
    for mesh_id in obj_meshes:
        mesh_df = pdf[pdf["mesh_id"] == mesh_id].sort_values("patch_idx")
        patches_data = [reconstruct_patch(row) for _, row in mesh_df.iterrows()]
        category = mesh_df.iloc[0]["category"]

        # Overview
        plot_mesh_patches_world(
            patches_data, mesh_id, category, "objaverse",
            output_dir / "objaverse" / f"{mesh_id[:16]}_overview.png",
        )
        # First patch pipeline visualization
        if patches_data:
            plot_processing_pipeline(
                patches_data[0], 0, mesh_id,
                output_dir / "pipeline" / f"objaverse_{mesh_id[:16]}_pipeline.png",
            )
            plot_patch_normalization_comparison(
                patches_data[0], 0, mesh_id,
                output_dir / "objaverse" / f"{mesh_id[:16]}_norm_compare.png",
            )

    # ShapeNet visualizations
    sn_meshes = pdf[pdf["source"] == "shapenet"]["mesh_id"].unique()[:args.n_shapenet]
    log.info(f"Visualizing {len(sn_meshes)} ShapeNet meshes")
    for mesh_id in sn_meshes:
        mesh_df = pdf[pdf["mesh_id"] == mesh_id].sort_values("patch_idx")
        patches_data = [reconstruct_patch(row) for _, row in mesh_df.iterrows()]
        category = mesh_df.iloc[0]["category"]

        plot_mesh_patches_world(
            patches_data, mesh_id, category, "shapenet",
            output_dir / "shapenet" / f"{mesh_id[:20]}_overview.png",
        )
        if patches_data:
            plot_processing_pipeline(
                patches_data[0], 0, mesh_id,
                output_dir / "pipeline" / f"shapenet_{mesh_id[:20]}_pipeline.png",
            )
            plot_patch_normalization_comparison(
                patches_data[0], 0, mesh_id,
                output_dir / "shapenet" / f"{mesh_id[:20]}_norm_compare.png",
            )

    # Write summary markdown
    with open(output_dir / "README.md", "w") as f:
        f.write("# Dry-Run Dataset Pipeline Visualization\n\n")
        f.write(f"**Date:** 2026-03-20\n\n")
        f.write(f"## Dataset Summary\n\n")
        f.write(f"- Total meshes: {pdf['mesh_id'].nunique()}\n")
        f.write(f"- Total patches: {len(pdf)}\n")
        f.write(f"- Sources: {dict(pdf['source'].value_counts())}\n")
        f.write(f"- Categories: {list(pdf['category'].unique())}\n\n")
        f.write(f"## Visualizations\n\n")
        f.write(f"### Statistics\n![Stats](stats_summary.png)\n\n")
        f.write(f"### Objaverse Samples\n")
        for mesh_id in obj_meshes:
            f.write(f"\n#### {mesh_id[:16]}...\n")
            f.write(f"![Overview](objaverse/{mesh_id[:16]}_overview.png)\n")
            f.write(f"![Normalization](objaverse/{mesh_id[:16]}_norm_compare.png)\n")
        f.write(f"\n### ShapeNet Samples\n")
        for mesh_id in sn_meshes:
            f.write(f"\n#### {mesh_id[:20]}...\n")
            f.write(f"![Overview](shapenet/{mesh_id[:20]}_overview.png)\n")
            f.write(f"![Normalization](shapenet/{mesh_id[:20]}_norm_compare.png)\n")
        f.write(f"\n### Processing Pipeline\n")
        f.write(f"Shows: World Space → Centered → PCA Rotated → PCA+Scale → No-PCA+Scale\n\n")
        for mesh_id in list(obj_meshes)[:3] + list(sn_meshes)[:3]:
            prefix = "objaverse" if mesh_id in obj_meshes else "shapenet"
            short = mesh_id[:16] if prefix == "objaverse" else mesh_id[:20]
            f.write(f"![Pipeline](pipeline/{prefix}_{short}_pipeline.png)\n")

    log.info(f"All visualizations saved to {output_dir}")
    log.info("Done!")


if __name__ == "__main__":
    main()
