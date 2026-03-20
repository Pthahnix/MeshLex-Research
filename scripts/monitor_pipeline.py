"""Monitor dataset pipeline progress and generate reports.

Parses pipeline log, computes stats, samples a few patches from HF for
visualization, and writes a report to results/dataset-pipeline/.

Usage:
    PYTHONPATH=/workspace/MeshLex-Research HF_TOKEN=... \
    python scripts/monitor_pipeline.py
"""
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "/tmp/dataset_pipeline.log"
OUTPUT_DIR = Path("results/dataset-pipeline")


def parse_log(log_path=LOG_PATH):
    """Parse pipeline log and extract batch stats."""
    batches = []
    current_phase = None
    with open(log_path) as f:
        for line in f:
            # Phase detection
            m = re.search(r"\[Phase (\d)/4\] (.+)\.\.\.", line)
            if m:
                current_phase = int(m.group(1))

            # Batch processed line
            m = re.search(
                r"\[(\w+)\] Processed: (\d+) ok, (\d+) fail, (\d+) patches", line
            )
            if m:
                ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                ts = ts_match.group(1) if ts_match else ""
                batches.append({
                    "batch_id": m.group(1),
                    "ok": int(m.group(2)),
                    "fail": int(m.group(3)),
                    "patches": int(m.group(4)),
                    "phase": current_phase,
                    "timestamp": ts,
                })

            # ShapeNet category line
            m = re.search(
                r"\[(\w+)\] Writing sub-batch \((\d+) patches\)", line
            )
            if m and current_phase == 2:
                pass  # ShapeNet uses different logging

    return batches


def compute_summary(batches):
    """Compute summary statistics from parsed batches."""
    if not batches:
        return {}

    phase1 = [b for b in batches if b["phase"] == 1]
    phase2 = [b for b in batches if b["phase"] == 2]

    def phase_stats(bs, total_batches=None):
        if not bs:
            return {"completed": 0, "total_ok": 0, "total_fail": 0,
                    "total_patches": 0, "success_rate": 0}
        total_ok = sum(b["ok"] for b in bs)
        total_fail = sum(b["fail"] for b in bs)
        total_patches = sum(b["patches"] for b in bs)
        total_meshes = total_ok + total_fail
        return {
            "completed": len(bs),
            "total_batches": total_batches,
            "total_ok": total_ok,
            "total_fail": total_fail,
            "total_patches": total_patches,
            "success_rate": round(total_ok / max(total_meshes, 1) * 100, 1),
            "avg_patches_per_mesh": round(total_patches / max(total_ok, 1), 1),
        }

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "objaverse": phase_stats(phase1, 93),
        "shapenet": phase_stats(phase2, None),
        "total_batches_done": len(batches),
        "total_patches": sum(b["patches"] for b in batches),
        "total_ok": sum(b["ok"] for b in batches),
        "total_fail": sum(b["fail"] for b in batches),
    }


def plot_progress(batches, summary, output_dir):
    """Generate progress visualization plots."""
    if not batches:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    phase1 = [b for b in batches if b["phase"] == 1]

    # 1. Cumulative patches over batches
    ax = axes[0, 0]
    if phase1:
        cum_patches = np.cumsum([b["patches"] for b in phase1])
        cum_ok = np.cumsum([b["ok"] for b in phase1])
        ax.plot(range(1, len(phase1) + 1), cum_patches, "b-o", markersize=3,
                label="Cumulative patches")
        ax.set_xlabel("Batch #")
        ax.set_ylabel("Patches", color="b")
        ax2 = ax.twinx()
        ax2.plot(range(1, len(phase1) + 1), cum_ok, "g-s", markersize=3,
                 label="Cumulative meshes OK")
        ax2.set_ylabel("Meshes OK", color="g")
        ax.set_title("Objaverse: Cumulative Progress")
        ax.axhline(y=cum_patches[-1], color="b", linestyle="--", alpha=0.3)

    # 2. Per-batch success rate
    ax = axes[0, 1]
    if phase1:
        rates = [b["ok"] / max(b["ok"] + b["fail"], 1) * 100 for b in phase1]
        ax.bar(range(len(phase1)), rates, color="steelblue", alpha=0.7)
        avg_rate = summary["objaverse"]["success_rate"]
        ax.axhline(y=avg_rate, color="red", linestyle="--",
                   label=f"Avg: {avg_rate:.1f}%")
        ax.set_xlabel("Batch #")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Objaverse: Per-Batch Success Rate")
        ax.legend()
        ax.set_ylim(0, 100)

    # 3. Patches per batch
    ax = axes[1, 0]
    if phase1:
        patches_per = [b["patches"] for b in phase1]
        ax.bar(range(len(phase1)), patches_per, color="salmon", alpha=0.7)
        ax.set_xlabel("Batch #")
        ax.set_ylabel("Patches")
        ax.set_title("Objaverse: Patches per Batch")

    # 4. Overall progress pie
    ax = axes[1, 1]
    obj_done = summary["objaverse"]["completed"]
    obj_total = summary["objaverse"].get("total_batches", 93) or 93
    obj_remain = obj_total - obj_done
    sn_info = summary.get("shapenet", {})
    sn_done = sn_info.get("completed", 0)

    labels = [f"Objaverse done\n({obj_done} batches)",
              f"Objaverse remaining\n({obj_remain} batches)"]
    sizes = [obj_done, obj_remain]
    colors = ["#4CAF50", "#E0E0E0"]
    if sn_done > 0:
        labels.append(f"ShapeNet done\n({sn_done} batches)")
        sizes.append(sn_done)
        colors.append("#2196F3")
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90)
    ax.set_title("Overall Progress")

    total_ok = summary["total_ok"]
    total_fail = summary["total_fail"]
    total_patches = summary["total_patches"]
    fig.suptitle(
        f"Pipeline Progress — {summary['timestamp']}\n"
        f"Meshes: {total_ok} OK / {total_fail} fail | "
        f"Patches: {total_patches:,} | "
        f"Objaverse: {obj_done}/{obj_total} batches",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "progress.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sample_patches(output_dir):
    """Sample 3 meshes from HF and visualize patches."""
    try:
        import daft
        from src.daft_utils import get_hf_io_config
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        io_config = get_hf_io_config()
        df = daft.read_parquet(
            "hf://datasets/Pthahnix/MeshLex-Patches/**/*.parquet",
            io_config=io_config,
        )
        # Get 3 distinct mesh_ids
        sample = df.select("mesh_id").distinct().limit(30).collect().to_pandas()
        if len(sample) == 0:
            return
        mesh_ids = sample["mesh_id"].unique()[:3]

        for idx, mesh_id in enumerate(mesh_ids):
            mesh_df = (
                df.where(df["mesh_id"] == mesh_id)
                .collect()
                .to_pandas()
                .sort_values("patch_idx")
            )
            if len(mesh_df) == 0:
                continue

            fig = plt.figure(figsize=(16, 5))
            n_patches = len(mesh_df)
            source = mesh_df.iloc[0]["source"]
            category = mesh_df.iloc[0]["category"]

            # Plot 1: World-space point cloud
            ax1 = fig.add_subplot(131, projection="3d")
            colors = plt.cm.tab20(np.linspace(0, 1, min(n_patches, 20)))
            all_verts = []
            for i, (_, row) in enumerate(mesh_df.iterrows()):
                nv = int(row["n_verts"])
                verts = np.array(row["vertices"], dtype=np.float32).reshape(nv, 3)
                all_verts.append(verts)
                ax1.scatter(verts[:, 0], verts[:, 1], verts[:, 2],
                            c=[colors[i % 20]], s=3, alpha=0.7)
            ax1.set_title(f"World Space\n{n_patches} patches")
            all_v = np.vstack(all_verts)
            _set_equal_axes(ax1, all_v)

            # Plot 2: World-space wireframe
            ax2 = fig.add_subplot(132, projection="3d")
            for i, (_, row) in enumerate(mesh_df.iterrows()):
                nv = int(row["n_verts"])
                nf = int(row["n_faces"])
                verts = np.array(row["vertices"], dtype=np.float32).reshape(nv, 3)
                faces = np.array(row["faces"], dtype=np.int32).reshape(nf, 3)
                color = colors[i % 20]
                polys = [[verts[f[0]], verts[f[1]], verts[f[2]]] for f in faces]
                mc = Poly3DCollection(polys, alpha=0.3, facecolors=[color],
                                      edgecolors=[(0.2, 0.2, 0.2, 0.3)],
                                      linewidths=0.3)
                ax2.add_collection3d(mc)
            ax2.set_title("Wireframe")
            _set_equal_axes(ax2, all_v)

            # Plot 3: PCA-normalized first 6 patches
            ax3 = fig.add_subplot(133, projection="3d")
            for i, (_, row) in enumerate(mesh_df.head(6).iterrows()):
                nv = int(row["n_verts"])
                lv = np.array(row["local_vertices"], dtype=np.float32).reshape(nv, 3)
                ax3.scatter(lv[:, 0], lv[:, 1], lv[:, 2],
                            c=[colors[i % 20]], s=5, alpha=0.6, label=f"P{i}")
            ax3.set_title("PCA-Normalized (first 6)")
            ax3.legend(fontsize=6)

            total_faces = sum(int(r["n_faces"]) for _, r in mesh_df.iterrows())
            total_verts = sum(int(r["n_verts"]) for _, r in mesh_df.iterrows())
            short_id = mesh_id[:16] if source == "objaverse" else mesh_id[:20]
            fig.suptitle(
                f"{source} / {category} / {short_id}...\n"
                f"{total_faces} faces, {total_verts} verts, {n_patches} patches",
                fontsize=11,
            )
            plt.tight_layout()
            plt.savefig(output_dir / f"sample_{idx}_{short_id}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

    except Exception as e:
        print(f"Sample visualization failed: {e}")


def _set_equal_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = max(maxs - mins) / 2 * 1.1
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batches = parse_log()
    summary = compute_summary(batches)

    if not batches:
        print("No batches completed yet.")
        return

    # Save JSON summary
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Progress plots
    plot_progress(batches, summary, OUTPUT_DIR)

    # Sample visualizations
    plot_sample_patches(OUTPUT_DIR)

    # Text report
    obj = summary["objaverse"]
    sn = summary.get("shapenet", {})
    report = f"""# Dataset Pipeline Progress Report

**Generated:** {summary['timestamp']}

## Overall
- Total meshes processed: {summary['total_ok']} OK / {summary['total_fail']} fail
- Total patches: {summary['total_patches']:,}

## Objaverse-LVIS (Phase 1)
- Batches: {obj['completed']} / {obj.get('total_batches', 93)} completed
- Meshes: {obj['total_ok']} OK / {obj['total_fail']} fail
- Success rate: {obj['success_rate']}%
- Patches: {obj['total_patches']:,}
- Avg patches/mesh: {obj.get('avg_patches_per_mesh', 0)}

## ShapeNet (Phase 2)
- Batches: {sn.get('completed', 0)} completed
- Meshes: {sn.get('total_ok', 0)} OK / {sn.get('total_fail', 0)} fail
- Success rate: {sn.get('success_rate', 0)}%
- Patches: {sn.get('total_patches', 0):,}

## Disk Usage
"""
    disk = shutil.disk_usage("/")
    report += f"- Total: {disk.total // (1024**3)}GB\n"
    report += f"- Used: {disk.used // (1024**3)}GB\n"
    report += f"- Free: {disk.free // (1024**3)}GB\n"

    with open(OUTPUT_DIR / "REPORT.md", "w") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    main()
