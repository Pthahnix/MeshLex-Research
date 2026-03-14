"""Generate final comparison visualizations across all 4 experiments."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────
EXPERIMENTS = {
    "Exp1\nA×5cat": "results/exp1_A_5cat/eval_results.json",
    "Exp2\nA×LVIS": "results/exp2_A_lvis_wide/eval_results.json",
    "Exp3\nB×5cat": "results/exp3_B_5cat/eval_results.json",
    "Exp4\nB×LVIS": "results/exp4_B_lvis_wide/eval_results.json",
}

HISTORY_FILES = {
    "Exp1 A×5cat": "data/checkpoints/5cat_v2/training_history.json",
    "Exp2 A×LVIS": "data/checkpoints/lvis_wide_A/training_history.json",
    "Exp3 B×5cat": "data/checkpoints/5cat_B/training_history.json",
    "Exp4 B×LVIS": "data/checkpoints/lvis_wide_B/training_history.json",
}

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
OUT_DIR = Path("results/final_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_evals():
    data = {}
    for name, path in EXPERIMENTS.items():
        with open(path) as f:
            data[name] = json.load(f)
    return data


def load_histories():
    data = {}
    for name, path in HISTORY_FILES.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data[name] = json.load(f)
    return data


# ── Plot 1: CD Comparison Bar Chart ──────────────────────────────────
def plot_cd_comparison(evals):
    names = list(evals.keys())
    same_cd = [evals[n]["same_category"]["mean_cd"] for n in names]
    cross_cd = [evals[n]["cross_category"]["mean_cd"] for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, same_cd, w, label="Same-category CD", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + w/2, cross_cd, w, label="Cross-category CD", color="#C44E52", edgecolor="white")

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Chamfer Distance (×10⁴)", fontsize=12)
    ax.set_title("Reconstruction Quality: Same-cat vs Cross-cat CD", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(same_cd), max(cross_cd)) * 1.15)

    plt.tight_layout()
    path = str(OUT_DIR / "cd_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 2: CD Ratio (Generalization Gap) ────────────────────────────
def plot_ratio_comparison(evals):
    names = list(evals.keys())
    ratios = [evals[n]["go_nogo"]["ratio"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, ratios, color=COLORS, edgecolor="white", width=0.5)

    # Threshold lines
    ax.axhline(y=1.0, color="green", linestyle="-", alpha=0.5, label="Perfect (1.0x)")
    ax.axhline(y=1.2, color="orange", linestyle="--", alpha=0.5, label="STRONG GO threshold (1.2x)")
    ax.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="Fail threshold (3.0x)")

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{ratio:.3f}x", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Cross-cat / Same-cat CD Ratio", fontsize=12)
    ax.set_title("Generalization Gap: Cross-category vs Same-category", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0.8, 1.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = str(OUT_DIR / "ratio_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 3: Utilization Comparison ───────────────────────────────────
def plot_utilization_comparison(evals):
    names = list(evals.keys())
    same_util = [evals[n]["same_category"]["utilization"] * 100 for n in names]
    cross_util = [evals[n]["cross_category"]["utilization"] * 100 for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w/2, same_util, w, label="Same-cat utilization", color="#55A868", edgecolor="white")
    bars2 = ax.bar(x + w/2, cross_util, w, label="Cross-cat utilization", color="#8172B2", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.4, label="50% threshold")
    ax.set_ylabel("Codebook Utilization (%)", fontsize=12)
    ax.set_title("Codebook Utilization: Same-cat vs Cross-cat", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    path = str(OUT_DIR / "utilization_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 4: Training Curves Overlay ──────────────────────────────────
def plot_training_overlay(histories):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, (name, hist) in enumerate(histories.items()):
        epochs = [h["epoch"] for h in hist]
        recon = [h["recon_loss"] for h in hist]
        util = [h["codebook_utilization"] for h in hist]
        val_recon = [h.get("val_recon_loss", h["recon_loss"]) for h in hist]

        axes[0].plot(epochs, recon, label=name, color=COLORS[i], alpha=0.8)
        axes[1].plot(epochs, util, label=name, color=COLORS[i], alpha=0.8)
        axes[2].plot(epochs, val_recon, label=name, color=COLORS[i], alpha=0.8)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train Reconstruction Loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Utilization")
    axes[1].set_title("Codebook Utilization (Train)")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Validation Reconstruction Loss")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(OUT_DIR / "training_overlay.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Plot 5: Summary Dashboard ────────────────────────────────────────
def plot_summary_dashboard(evals):
    names = list(evals.keys())
    short_names = ["Exp1\nA×5cat", "Exp2\nA×LVIS", "Exp3\nB×5cat", "Exp4\nB×LVIS"]

    ratios = [evals[n]["go_nogo"]["ratio"] for n in names]
    same_cd = [evals[n]["same_category"]["mean_cd"] for n in names]
    util = [evals[n]["same_category"]["utilization"] * 100 for n in names]
    n_codes = [evals[n]["same_category"]["n_unique_codes"] for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MeshLex Feasibility Validation — Final Results Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    # (0,0) CD Ratio
    ax = axes[0, 0]
    bars = ax.bar(short_names, ratios, color=COLORS, edgecolor="white")
    ax.axhline(y=1.2, color="orange", linestyle="--", alpha=0.6, label="GO threshold")
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{r:.3f}x", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("CD Ratio")
    ax.set_title("Generalization Gap (lower = better)")
    ax.set_ylim(0.9, 1.3)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (0,1) Same-cat CD
    ax = axes[0, 1]
    bars = ax.bar(short_names, same_cd, color=COLORS, edgecolor="white")
    for bar, cd in zip(bars, same_cd):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{cd:.1f}", ha="center", fontsize=11)
    ax.set_ylabel("Chamfer Distance")
    ax.set_title("Reconstruction Quality (lower = better)")
    ax.grid(axis="y", alpha=0.3)

    # (1,0) Utilization
    ax = axes[1, 0]
    bars = ax.bar(short_names, util, color=COLORS, edgecolor="white")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.4)
    for bar, u in zip(bars, util):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{u:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Utilization (%)")
    ax.set_title("Codebook Utilization (higher = better)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # (1,1) Active Codes
    ax = axes[1, 1]
    bars = ax.bar(short_names, n_codes, color=COLORS, edgecolor="white")
    ax.axhline(y=4096, color="gray", linestyle="--", alpha=0.4, label="Max (4096)")
    for bar, n in zip(bars, n_codes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{n}", ha="center", fontsize=11)
    ax.set_ylabel("Active Codes")
    ax.set_title("Unique Codebook Entries Used")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = str(OUT_DIR / "summary_dashboard.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    evals = load_evals()
    histories = load_histories()

    plot_cd_comparison(evals)
    plot_ratio_comparison(evals)
    plot_utilization_comparison(evals)
    plot_training_overlay(histories)
    plot_summary_dashboard(evals)

    print("\nAll final comparison plots generated.")
