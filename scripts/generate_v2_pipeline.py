"""Enhanced MeshLex v2 Generation Pipeline with Full Visualization.

Generates meshes using AR v2 model + RVQ decoder, saving comprehensive
intermediate visualizations at every pipeline stage for human inspection.

Pipeline stages visualized:
  1. Raw token sequence → heatmap
  2. Decoded patch parameters → position/scale distributions
  3. Per-patch decoded geometry → individual patch point clouds
  4. Assembled point cloud → full mesh visualization
  5. Summary dashboard per mesh

Usage:
    python scripts/generate_v2_pipeline.py \
        --ar_checkpoint data/checkpoints/ar_v2/checkpoint_final.pt \
        --vqvae_checkpoint data/checkpoints/rvq_lvis/checkpoint_final.pt \
        --output_dir results/generation_v2_pipeline \
        --n_meshes 10 --temperatures 0.7 0.8 0.9 1.0
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ar_model import PatchGPT
from src.model_rvq import MeshLexRVQVAE
from src.patch_sequence import compute_vocab_size


def decode_token_sequence(sequence, n_pos_bins=256, n_scale_bins=64):
    """Decode flat token sequence back to patch parameters (RVQ mode, 7 tokens/patch)."""
    tokens_per_patch = 7
    offset_y = n_pos_bins
    offset_z = 2 * n_pos_bins
    offset_scale = 3 * n_pos_bins
    offset_code = 3 * n_pos_bins + n_scale_bins

    n_patches = len(sequence) // tokens_per_patch
    patches = []

    for i in range(n_patches):
        base = i * tokens_per_patch
        pos_x = int(sequence[base + 0])
        pos_y = int(sequence[base + 1]) - offset_y
        pos_z = int(sequence[base + 2]) - offset_z
        scale_tok = int(sequence[base + 3]) - offset_scale
        tok1 = int(sequence[base + 4]) - offset_code
        tok2 = int(sequence[base + 5]) - offset_code
        tok3 = int(sequence[base + 6]) - offset_code
        patches.append({
            "pos": [pos_x, pos_y, pos_z],
            "scale": scale_tok,
            "tokens": [tok1, tok2, tok3],
            "raw_tokens": [int(sequence[base + j]) for j in range(7)],
        })

    return patches


def viz_token_sequence(seq_np, mesh_dir, mesh_idx):
    """Stage 1: Visualize raw token sequence as heatmap."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 6))

    # Token value distribution
    axes[0].plot(seq_np, linewidth=0.5, alpha=0.8)
    axes[0].set_xlabel("Token position")
    axes[0].set_ylabel("Token value")
    axes[0].set_title(f"Mesh {mesh_idx:03d} — Raw Token Sequence ({len(seq_np)} tokens)")
    axes[0].axhline(y=768, color='r', linestyle='--', alpha=0.5, label='pos/scale boundary')
    axes[0].axhline(y=832, color='g', linestyle='--', alpha=0.5, label='scale/code boundary')
    axes[0].legend(fontsize=8)

    # Token sequence as 2D heatmap (reshape to 7-column for RVQ)
    n_patches = len(seq_np) // 7
    if n_patches > 0:
        reshaped = seq_np[:n_patches * 7].reshape(n_patches, 7)
        im = axes[1].imshow(reshaped.T, aspect='auto', cmap='viridis')
        axes[1].set_xlabel("Patch index")
        axes[1].set_ylabel("Token slot")
        axes[1].set_yticks(range(7))
        axes[1].set_yticklabels(['pos_x', 'pos_y', 'pos_z', 'scale', 'cb_L1', 'cb_L2', 'cb_L3'])
        axes[1].set_title(f"Token Heatmap ({n_patches} patches × 7 tokens)")
        plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig(mesh_dir / "stage1_token_sequence.png", dpi=150)
    plt.close()


def viz_patch_positions(patch_params, mesh_dir, mesh_idx):
    """Stage 2: Visualize decoded patch positions and scales."""
    if not patch_params:
        return

    positions = np.array([p["pos"] for p in patch_params], dtype=np.float32) / 255.0
    scales = np.array([max(p["scale"] / 63.0, 0.01) for p in patch_params])
    codebook_tokens = np.array([p["tokens"] for p in patch_params])

    fig = plt.figure(figsize=(18, 12))

    # 3D scatter of patch positions
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    sc = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                     c=scales, cmap='plasma', s=scales * 500 + 10, alpha=0.7)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Patch Positions (n={len(patch_params)})')
    plt.colorbar(sc, ax=ax1, label='Scale', shrink=0.6)

    # Position histograms
    for i, (ax_idx, label) in enumerate(zip([2, 3, 4], ['X', 'Y', 'Z'])):
        ax = fig.add_subplot(2, 3, ax_idx)
        ax.hist(positions[:, i], bins=30, alpha=0.7, color='steelblue')
        ax.set_xlabel(f'Position {label}')
        ax.set_ylabel('Count')
        ax.set_title(f'{label} Distribution')

    # Scale distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(scales, bins=30, alpha=0.7, color='coral')
    ax5.set_xlabel('Scale')
    ax5.set_ylabel('Count')
    ax5.set_title('Scale Distribution')

    # Codebook token usage per level
    ax6 = fig.add_subplot(2, 3, 6)
    for lvl in range(3):
        tokens_lvl = codebook_tokens[:, lvl]
        ax6.hist(tokens_lvl, bins=50, alpha=0.5, label=f'Level {lvl}')
    ax6.set_xlabel('Codebook Index')
    ax6.set_ylabel('Count')
    ax6.set_title('Codebook Token Usage')
    ax6.legend()

    plt.suptitle(f'Mesh {mesh_idx:03d} — Patch Parameters', fontsize=14)
    plt.tight_layout()
    plt.savefig(mesh_dir / "stage2_patch_positions.png", dpi=150)
    plt.close()


def decode_patches_through_vqvae(patch_params, vqvae, device):
    """Stage 3: Decode each patch through VQ-VAE decoder, return local + world vertices."""
    all_local = []
    all_world = []
    all_z_hat = []

    for p in patch_params:
        with torch.no_grad():
            tok_indices = torch.tensor(p["tokens"], dtype=torch.long, device=device)
            z_hat = vqvae.rvq.decode_indices(tok_indices.unsqueeze(0))  # (1, 128)
            all_z_hat.append(z_hat.cpu().numpy())

            n_verts = torch.tensor([30], device=device)
            local_verts = vqvae.decoder(z_hat, n_verts)  # (1, max_V, 3)
            local_verts = local_verts[0, :30].cpu().numpy()

        pos = np.array(p["pos"], dtype=np.float32) / 255.0
        scale = max(p["scale"] / 63.0, 0.01)
        world_verts = local_verts * scale + pos

        all_local.append(local_verts)
        all_world.append(world_verts)

    return all_local, all_world, np.concatenate(all_z_hat, axis=0)


def viz_individual_patches(all_local, all_world, mesh_dir, mesh_idx, max_show=16):
    """Stage 3 viz: Show individual decoded patches (local + world space)."""
    n_patches = len(all_local)
    n_show = min(n_patches, max_show)

    if n_show == 0:
        return

    cols = 4
    rows = (n_show + cols - 1) // cols

    # Local space patches
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(n_show):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        v = all_local[i]
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=8, alpha=0.7)
        ax.set_title(f'Patch {i} (local)', fontsize=8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    plt.suptitle(f'Mesh {mesh_idx:03d} — Individual Patches (Local Space)', fontsize=12)
    plt.tight_layout()
    plt.savefig(mesh_dir / "stage3_patches_local.png", dpi=120)
    plt.close()

    # World space patches
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(n_show):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        v = all_world[i]
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=8, alpha=0.7, c='coral')
        ax.set_title(f'Patch {i} (world)', fontsize=8)
    plt.suptitle(f'Mesh {mesh_idx:03d} — Individual Patches (World Space)', fontsize=12)
    plt.tight_layout()
    plt.savefig(mesh_dir / "stage3_patches_world.png", dpi=120)
    plt.close()


def viz_assembled_pointcloud(all_world, patch_params, mesh_dir, mesh_idx):
    """Stage 4: Visualize assembled point cloud from all patches."""
    if not all_world:
        return

    combined = np.concatenate(all_world, axis=0)

    # Color by patch index
    colors = []
    cmap = plt.colormaps.get_cmap('tab20')
    for i, v in enumerate(all_world):
        c = cmap(i % 20)
        colors.extend([c] * len(v))
    colors = np.array(colors)

    fig = plt.figure(figsize=(18, 6))

    for view_idx, (elev, azim, title) in enumerate([
        (30, 45, 'View 1 (30°, 45°)'),
        (30, 135, 'View 2 (30°, 135°)'),
        (90, 0, 'Top View'),
    ]):
        ax = fig.add_subplot(1, 3, view_idx + 1, projection='3d')
        ax.scatter(combined[:, 0], combined[:, 1], combined[:, 2],
                   c=colors, s=1, alpha=0.6)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

    plt.suptitle(
        f'Mesh {mesh_idx:03d} — Assembled Point Cloud '
        f'({len(patch_params)} patches, {len(combined)} points)',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(mesh_dir / "stage4_assembled_pointcloud.png", dpi=150)
    plt.close()

    # Also save colored-by-patch version with larger points for detail
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(combined[:, 0], combined[:, 1], combined[:, 2],
               c=colors, s=3, alpha=0.7)
    ax.view_init(elev=20, azim=60)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Mesh {mesh_idx:03d} — Colored by Patch ({len(patch_params)} patches)')
    plt.tight_layout()
    plt.savefig(mesh_dir / "stage4_assembled_detail.png", dpi=200)
    plt.close()


def viz_embedding_space(all_z_hat, mesh_dir, mesh_idx):
    """Bonus: Visualize patch embeddings via PCA/t-SNE."""
    if len(all_z_hat) < 3:
        return

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(all_z_hat)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1],
                    c=np.arange(len(z_2d)), cmap='viridis', s=30, alpha=0.8)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'Mesh {mesh_idx:03d} — Patch Embeddings (PCA)')
    plt.colorbar(sc, ax=ax, label='Patch index (Z-order)')
    plt.tight_layout()
    plt.savefig(mesh_dir / "bonus_embedding_pca.png", dpi=150)
    plt.close()


def create_summary_dashboard(mesh_dir, mesh_idx, patch_params, all_world, seq_np, gen_time):
    """Create a summary dashboard combining key stats."""
    n_patches = len(patch_params)
    n_points = sum(len(v) for v in all_world) if all_world else 0

    positions = np.array([p["pos"] for p in patch_params], dtype=np.float32) / 255.0 if patch_params else np.zeros((0, 3))
    scales = np.array([max(p["scale"] / 63.0, 0.01) for p in patch_params]) if patch_params else np.zeros(0)

    # Compute token statistics
    unique_tokens = len(np.unique(seq_np))
    vocab_size = compute_vocab_size()

    stats = {
        "mesh_idx": mesh_idx,
        "n_tokens": len(seq_np),
        "n_patches": n_patches,
        "n_points": n_points,
        "unique_tokens": unique_tokens,
        "vocab_coverage": unique_tokens / vocab_size,
        "position_range": {
            "x": [float(positions[:, 0].min()), float(positions[:, 0].max())] if len(positions) > 0 else [0, 0],
            "y": [float(positions[:, 1].min()), float(positions[:, 1].max())] if len(positions) > 0 else [0, 0],
            "z": [float(positions[:, 2].min()), float(positions[:, 2].max())] if len(positions) > 0 else [0, 0],
        },
        "scale_stats": {
            "mean": float(scales.mean()) if len(scales) > 0 else 0,
            "std": float(scales.std()) if len(scales) > 0 else 0,
            "min": float(scales.min()) if len(scales) > 0 else 0,
            "max": float(scales.max()) if len(scales) > 0 else 0,
        },
        "generation_time_sec": gen_time,
    }

    with open(mesh_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def save_pointcloud_ply(all_world, mesh_dir, mesh_idx):
    """Save assembled point cloud as PLY."""
    if not all_world:
        return
    combined = np.concatenate(all_world, axis=0)
    try:
        import trimesh
        pc = trimesh.PointCloud(combined)
        pc.export(str(mesh_dir / f"mesh_{mesh_idx:03d}.ply"))
    except Exception:
        # Fallback: save as numpy
        np.save(mesh_dir / f"mesh_{mesh_idx:03d}_points.npy", combined)


def main():
    parser = argparse.ArgumentParser(description="MeshLex v2 Generation Pipeline with Visualization")
    parser.add_argument("--ar_checkpoint", required=True)
    parser.add_argument("--vqvae_checkpoint", required=True)
    parser.add_argument("--output_dir", default="results/generation_v2_pipeline")
    parser.add_argument("--n_meshes", type=int, default=10)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 0.9, 1.0])
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=910)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {out}")
    print(f"Temperatures: {args.temperatures}")
    print(f"Meshes per temperature: {args.n_meshes}")

    # Load AR model
    print("Loading AR v2 model...")
    ar_ckpt = torch.load(args.ar_checkpoint, map_location=device, weights_only=False)
    ar_config = ar_ckpt.get("config", {})
    ar_model = PatchGPT(**ar_config).to(device)
    ar_model.load_state_dict(ar_ckpt["model_state_dict"])
    ar_model.eval()
    n_params = sum(p.numel() for p in ar_model.parameters())
    print(f"  AR model: {n_params/1e6:.1f}M params, config={ar_config}")

    # Load VQ-VAE
    print("Loading RVQ VQ-VAE...")
    vq_ckpt = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
    vqvae = MeshLexRVQVAE().to(device)
    vqvae.load_state_dict(vq_ckpt["model_state_dict"], strict=False)
    vqvae.eval()
    print("  VQ-VAE loaded")

    all_stats = []

    for temp in args.temperatures:
        temp_dir = out / f"temp_{temp:.1f}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Temperature: {temp}")
        print(f"{'='*60}")

        for mesh_idx in range(args.n_meshes):
            mesh_dir = temp_dir / f"mesh_{mesh_idx:03d}"
            mesh_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Generating mesh {mesh_idx + 1}/{args.n_meshes} (temp={temp})...")
            t0 = time.time()

            # Stage 1: Generate token sequence
            with torch.no_grad():
                seq = ar_model.generate(
                    max_len=args.max_len,
                    temperature=temp,
                    top_k=args.top_k,
                )
            seq_np = seq[0].cpu().numpy()
            gen_time = time.time() - t0
            print(f"    Generated {len(seq_np)} tokens in {gen_time:.1f}s")

            # Save raw sequence
            np.savez(mesh_dir / "raw_sequence.npz", sequence=seq_np)

            # Visualize Stage 1
            viz_token_sequence(seq_np, mesh_dir, mesh_idx)

            # Stage 2: Decode token sequence to patch parameters
            patch_params = decode_token_sequence(seq_np)
            print(f"    Decoded {len(patch_params)} patches")

            # Visualize Stage 2
            viz_patch_positions(patch_params, mesh_dir, mesh_idx)

            # Stage 3: Decode patches through VQ-VAE
            if patch_params:
                all_local, all_world, all_z_hat = decode_patches_through_vqvae(
                    patch_params, vqvae, device
                )
                print(f"    Decoded {len(all_world)} patches → {sum(len(v) for v in all_world)} points")

                # Visualize Stage 3
                viz_individual_patches(all_local, all_world, mesh_dir, mesh_idx)

                # Visualize Stage 4
                viz_assembled_pointcloud(all_world, patch_params, mesh_dir, mesh_idx)

                # Bonus: embedding space
                viz_embedding_space(all_z_hat, mesh_dir, mesh_idx)

                # Save PLY
                save_pointcloud_ply(all_world, mesh_dir, mesh_idx)
            else:
                all_world = []

            # Summary
            stats = create_summary_dashboard(
                mesh_dir, mesh_idx, patch_params, all_world, seq_np, gen_time
            )
            stats["temperature"] = temp
            all_stats.append(stats)

    # Save global summary
    with open(out / "generation_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # Create comparison dashboard across temperatures
    create_temperature_comparison(all_stats, out, args.temperatures)

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Total meshes: {len(all_stats)}")
    print(f"  Output: {out}")
    print(f"{'='*60}")


def create_temperature_comparison(all_stats, out, temperatures):
    """Create comparison plots across temperatures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for temp in temperatures:
        temp_stats = [s for s in all_stats if s["temperature"] == temp]
        n_patches = [s["n_patches"] for s in temp_stats]
        n_points = [s["n_points"] for s in temp_stats]
        gen_times = [s["generation_time_sec"] for s in temp_stats]
        vocab_cov = [s["vocab_coverage"] for s in temp_stats]

        axes[0, 0].bar(
            [f"T={temp}" for _ in range(len(n_patches))],
            n_patches, alpha=0.6, label=f"T={temp}"
        )
        axes[0, 1].bar(
            [f"T={temp}" for _ in range(len(n_points))],
            n_points, alpha=0.6
        )

    # Aggregate by temperature
    for ax_idx, (metric, ylabel, title) in enumerate([
        ("n_patches", "# Patches", "Patches per Mesh"),
        ("n_points", "# Points", "Points per Mesh"),
        ("generation_time_sec", "Time (s)", "Generation Time"),
        ("vocab_coverage", "Coverage", "Vocab Coverage"),
    ]):
        ax = axes[ax_idx // 2, ax_idx % 2]
        for temp in temperatures:
            temp_stats = [s for s in all_stats if s["temperature"] == temp]
            values = [s[metric] for s in temp_stats]
            ax.plot(range(len(values)), values, 'o-', label=f'T={temp}', alpha=0.7)
        ax.set_xlabel("Mesh index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle("Generation Pipeline — Temperature Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "temperature_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
