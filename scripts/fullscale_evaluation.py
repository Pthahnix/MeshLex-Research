"""Phase 4: Unified evaluation — reconstruction + generation + ablation dashboard.

Usage:
    # Evaluate reconstruction quality for a single VQ-VAE:
    PYTHONPATH=. python scripts/fullscale_evaluation.py --action recon \
        --checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
        --patch_dir data/patches_full/seen_test \
        --output_dir results/fullscale_eval/recon_rvq_full_pca

    # Evaluate with noPCA mode:
    PYTHONPATH=. python scripts/fullscale_evaluation.py --action recon \
        --checkpoint data/checkpoints/rvq_full_nopca/checkpoint_final.pt \
        --patch_dir data/patches_full/seen_test \
        --output_dir results/fullscale_eval/recon_rvq_full_nopca \
        --nopca

    # Evaluate with mmap features (full-scale, no NPZ needed):
    PYTHONPATH=. python scripts/fullscale_evaluation.py --action recon_mmap \
        --checkpoint data/checkpoints/rvq_full_pca/checkpoint_final.pt \
        --feature_dir /data/pthahnix/MeshLex-Research/datasets/MeshLex-Patches/features/seen_test \
        --output_dir results/fullscale_eval/recon_rvq_full_pca

    # Build unified dashboard from all results:
    PYTHONPATH=. python scripts/fullscale_evaluation.py --action dashboard \
        --output_dir results/fullscale_eval
"""
import argparse
import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model_rvq import MeshLexRVQVAE


def chamfer_distance_np(pts_a, pts_b):
    """Compute symmetric Chamfer Distance between two point sets."""
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    return (dist_a.mean() + dist_b.mean()) / 2.0


def evaluate_reconstruction_mmap(checkpoint, feature_dir, output_dir,
                                  n_samples=2000, batch_size=4096,
                                  chunk_size=200000, device_id=0):
    """Evaluate VQ-VAE reconstruction using mmap features (full-scale path)."""
    from src.patch_dataset import MmapPatchDataset
    from src.stream_utils import ChunkBatchIterator

    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    codebook_size = config.get("codebook_size", 1024)
    n_levels = config.get("n_levels", 3)
    embed_dim = config.get("embed_dim", 128)
    hidden_dim = config.get("hidden_dim", 256)

    model = MeshLexRVQVAE(
        codebook_size=codebook_size,
        n_levels=n_levels,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Load mmap dataset
    dataset = MmapPatchDataset(feature_dir, chunk_size=chunk_size)
    print(f"Evaluating on {len(dataset)} patches (sampling {n_samples} for CD)")

    cds = []
    all_indices = []
    n_processed = 0

    with torch.no_grad():
        for chunk_start in range(0, min(len(dataset), n_samples * 2), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(dataset))
            dataset.load_chunk(chunk_start)

            iterator = ChunkBatchIterator(dataset, batch_size=batch_size)
            for batch in iterator:
                x = batch["x"].to(device)
                n_vertices = batch["n_vertices"]
                gt_vertices = batch["gt_vertices"].to(device)

                # Encode + quantize
                z = model.encoder.forward_flat(x)
                z_hat, result = model.rvq(z)
                indices = result["indices"]  # (B, n_levels)
                all_indices.append(indices.cpu())

                # Reconstruct
                recon = model.decoder(z_hat, n_vertices)

                # CD on sampled patches
                for i in range(len(n_vertices)):
                    if len(cds) >= n_samples:
                        break
                    n_v = n_vertices[i]
                    gt = gt_vertices[i, :n_v].cpu().numpy()
                    pred = recon[i, :n_v].cpu().numpy()
                    if len(gt) > 2 and len(pred) > 2:
                        cd = chamfer_distance_np(gt, pred)
                        cds.append(cd)

                n_processed += len(n_vertices)

            if len(cds) >= n_samples:
                break

    # Compute utilization
    all_idx = torch.cat(all_indices, dim=0)  # (N, n_levels)
    utilization = float(all_idx[:, 0].unique().numel() / codebook_size)

    results = {
        "n_patches_evaluated": n_processed,
        "n_samples_cd": len(cds),
        "mean_cd": float(np.mean(cds)) if cds else 0.0,
        "std_cd": float(np.std(cds)) if cds else 0.0,
        "median_cd": float(np.median(cds)) if cds else 0.0,
        "p95_cd": float(np.percentile(cds, 95)) if cds else 0.0,
        "codebook_utilization": utilization,
        "checkpoint": str(checkpoint),
        "feature_dir": str(feature_dir),
    }

    with open(out / "reconstruction_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # CD histogram
    if cds:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(cds, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(results["mean_cd"], color="red", linestyle="--",
                   label=f"Mean={results['mean_cd']:.6f}")
        ax.axvline(results["median_cd"], color="blue", linestyle="--",
                   label=f"Median={results['median_cd']:.6f}")
        ax.set_xlabel("Chamfer Distance")
        ax.set_ylabel("Count")
        ax.set_title("Reconstruction CD Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out / "cd_histogram.png", dpi=150)
        plt.close()

    print(f"Reconstruction: CD={results['mean_cd']:.6f} ± {results['std_cd']:.6f}, "
          f"util={utilization:.3f}, n={len(cds)}")
    return results


def evaluate_reconstruction(checkpoint, patch_dir, output_dir,
                             n_samples=500, use_nopca=False):
    """Evaluate VQ-VAE reconstruction quality on NPZ patches."""
    from src.patch_dataset import PatchGraphDataset
    from torch_geometric.loader import DataLoader as PyGDataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})
    model = MeshLexRVQVAE(
        codebook_size=ckpt_config.get("codebook_size", 1024),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    dataset = PatchGraphDataset(patch_dir, use_nopca=use_nopca)
    loader = PyGDataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    cds = []
    utilization_counts = np.zeros(
        ckpt_config.get("codebook_size", 1024), dtype=int)
    total_patches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z = model.encoder(batch.x, batch.edge_index, batch.batch)
            z_hat, indices = model.rvq(z)

            # Track utilization
            for level in range(3):
                for idx in indices[:, level].cpu().numpy():
                    utilization_counts[idx] += 1

            # Reconstruct and compute CD (sample)
            if total_patches < n_samples:
                recon = model.decoder(z_hat, batch.n_vertices)
                for i in range(min(len(batch.n_vertices),
                                   n_samples - total_patches)):
                    n_v = batch.n_vertices[i].item()
                    gt = batch.gt_vertices[i, :n_v].cpu().numpy()
                    pred = recon[i, :n_v].cpu().numpy()
                    cd = chamfer_distance_np(gt, pred)
                    cds.append(cd)

            total_patches += len(batch.n_vertices)

    K = ckpt_config.get("codebook_size", 1024)
    utilization = float(np.sum(utilization_counts > 0) / K)
    results = {
        "n_patches": total_patches,
        "n_samples_cd": len(cds),
        "mean_cd": float(np.mean(cds)),
        "std_cd": float(np.std(cds)),
        "median_cd": float(np.median(cds)),
        "codebook_utilization": utilization,
    }

    with open(out / "reconstruction_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Reconstruction: CD={results['mean_cd']:.6f}, util={utilization:.3f}")
    return results


def build_dashboard(output_dir):
    """Build unified DASHBOARD.md from all evaluation results."""
    out = Path(output_dir)
    lines = ["# Full-Scale Evaluation Dashboard\n"]

    # VQ-VAE results
    lines.append("## VQ-VAE Reconstruction\n")
    lines.append("| Config | Mean CD | Std CD | Median CD | Utilization |")
    lines.append("|--------|---------|--------|-----------|-------------|")
    for name, label in [
        ("rvq_full_pca", "PCA K=1024"),
        ("rvq_full_nopca", "noPCA K=1024"),
        ("rvq_full_pca_k512", "PCA K=512"),
        ("rvq_full_pca_k2048", "PCA K=2048"),
    ]:
        p = out / f"recon_{name}" / "reconstruction_results.json"
        if p.exists():
            r = json.loads(p.read_text())
            lines.append(
                f"| {label} | {r['mean_cd']:.6f} | {r.get('std_cd', 0):.6f} "
                f"| {r.get('median_cd', 0):.6f} | {r['codebook_utilization']:.3f} |"
            )

    # AR results
    lines.append("\n## AR Generation\n")
    for name, label in [
        ("fullscale_gen_pca", "PCA AR"),
        ("fullscale_gen_nopca", "noPCA AR"),
    ]:
        p = Path(f"results/{name}/evaluation_results.json")
        if p.exists():
            r = json.loads(p.read_text())
            summary = r.get("summary", {})
            lines.append(f"**{label}**: {json.dumps(summary, indent=2)}\n")

    # Theory-driven results
    lines.append("\n## Theory-Driven Findings\n")

    # K ablation
    k_path = Path("results/fullscale_theory/k_ablation/k_ablation_results.json")
    if k_path.exists():
        r = json.loads(k_path.read_text())
        lines.append("### K Ablation\n")
        lines.append("| K | L1 σ | L2 σ | L3 σ | L1 Util |")
        lines.append("|---|------|------|------|---------|")
        for k in ["512", "1024", "2048"]:
            if k in r:
                d = r[k]
                lines.append(
                    f"| {k} | {d['L1']['lognormal_sigma']:.3f} "
                    f"| {d['L2']['lognormal_sigma']:.3f} "
                    f"| {d['L3']['lognormal_sigma']:.3f} "
                    f"| {d['L1']['utilization']:.3f} |"
                )

    # VQ comparison
    vq_path = Path(
        "results/fullscale_theory/vq_comparison/vq_comparison_results.json")
    if vq_path.exists():
        r = json.loads(vq_path.read_text())
        lines.append("\n### VQ Method Comparison (FM1 Test)\n")
        lines.append("| Method | Avg σ | Avg α | Verdict |")
        lines.append("|--------|-------|-------|---------|")
        for method in ["simvq", "vanilla", "ema"]:
            if method in r:
                avg_sigma = np.mean([
                    r[method][f"L{l}"]["lognormal_sigma"] for l in [1, 2, 3]
                ])
                avg_alpha = np.mean([
                    r[method][f"L{l}"]["zipf_alpha"] for l in [1, 2, 3]
                ])
                lines.append(f"| {method} | {avg_sigma:.3f} | {avg_alpha:.3f} | — |")

    # Curvature
    curv_path = Path(
        "results/fullscale_theory/curvature/curvature_results.json")
    if curv_path.exists():
        r = json.loads(curv_path.read_text())
        lines.append(
            f"\n### Curvature-Frequency Correlation\n"
            f"Spearman ρ = {r['spearman_rho']:.3f} "
            f"(p = {r['p_value']:.2e})\n"
        )

    # MDLM results
    mdlm_path = Path("results/fullscale_mdlm/summary.json")
    if mdlm_path.exists():
        r = json.loads(mdlm_path.read_text())
        lines.append(
            f"\n## MDLM Feasibility\n"
            f"PPL={r['final_val_ppl']:.1f}, "
            f"Acc={r['final_val_acc']:.4f}, "
            f"**{r['verdict']}**\n"
        )

    # Ablation
    abl_path = out / "ablation_results.json"
    if abl_path.exists():
        r = json.loads(abl_path.read_text())
        lines.append("\n## PCA vs noPCA Ablation\n")
        pca_nopca = r.get("PCA_vs_noPCA", {})
        if pca_nopca:
            lines.append("| Metric | PCA | noPCA |")
            lines.append("|--------|-----|-------|")
            pca = pca_nopca.get("PCA", {})
            nopca = pca_nopca.get("noPCA", {})
            for metric in ["mean_cd", "codebook_utilization"]:
                pca_val = pca.get(metric, "N/A")
                nopca_val = nopca.get(metric, "N/A")
                if isinstance(pca_val, float):
                    pca_val = f"{pca_val:.6f}"
                if isinstance(nopca_val, float):
                    nopca_val = f"{nopca_val:.6f}"
                lines.append(f"| {metric} | {pca_val} | {nopca_val} |")

    dashboard = "\n".join(lines)
    (out / "DASHBOARD.md").write_text(dashboard)
    print(f"Dashboard written to {out / 'DASHBOARD.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action",
                        choices=["recon", "recon_mmap", "dashboard"],
                        required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--patch_dir", default=None)
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--output_dir", default="results/fullscale_eval")
    parser.add_argument("--nopca", action="store_true")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    if args.action == "recon":
        evaluate_reconstruction(args.checkpoint, args.patch_dir,
                                args.output_dir, args.n_samples, args.nopca)
    elif args.action == "recon_mmap":
        evaluate_reconstruction_mmap(
            args.checkpoint, args.feature_dir, args.output_dir,
            args.n_samples, args.batch_size, device_id=args.device_id)
    elif args.action == "dashboard":
        build_dashboard(args.output_dir)
