# Objaverse Migration & Dual-Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace ShapeNet with Objaverse-LVIS as data source, then run two experiments (5-Category + LVIS-Wide) to validate MeshLex's core hypothesis that mesh local topology has a universal, finite vocabulary.

**Architecture:** Objaverse Python API downloads GLB files → trimesh loads GLB (already supported) → existing pipeline (pyfqmr decimation → METIS patch segmentation → NPZ serialization → staged VQ-VAE training). Two independent experiment runs share the same codebase, differing only in data selection.

**Tech Stack:** Python 3.11, objaverse 0.1.7+, PyTorch 2.4.1+cu124, torch-geometric, trimesh (GLB support built-in), pyfqmr, pymetis, sklearn, matplotlib.

**Important context:**
- All `src/` modules are already implemented and tested (17 tests passing)
- `src/data_prep.py:load_and_preprocess_mesh` uses `trimesh.load(path, force="mesh")` which handles GLB natively — no code change needed
- `scripts/run_preprocessing.py` currently expects ShapeNet directory structure (`{synset_id}/{model_id}/models/model_normalized.obj`). Needs modification to accept a manifest JSON listing `{uid, glb_path, category}`
- Disk budget: 25GB total for both experiments (60GB available)
- CLAUDE.md rules: commit per functional unit, push after every commit, real data validation with visible outputs saved to `results/`

---

## Task 1: Create Objaverse Download Script

**Files:**
- Create: `scripts/download_objaverse.py`

**Step 1: Write the download script**

支持两种模式：
- `--mode 5cat`: 下载 5 个精确 LVIS 类别（chair, table, airplane, car_automobile, lamp）
- `--mode lvis_wide`: 从 LVIS 中筛选 ≥10 objects 的类别，每类最多采样 10 个

```python
"""Download Objaverse-LVIS 3D objects for MeshLex experiments."""
import argparse
import json
import random
from pathlib import Path
import objaverse

# 实验 1：5-Category 精确匹配
FIVE_CAT = {
    "chair":          "chair",
    "table":          "table",
    "airplane":       "airplane",
    "car":            "car_(automobile)",
    "lamp":           "lamp",
}

def select_5cat(lvis):
    """Select UIDs for 5-category experiment."""
    selected = {}
    for our_name, lvis_tag in FIVE_CAT.items():
        uids = lvis.get(lvis_tag, [])
        selected[our_name] = uids
        print(f"  {our_name} ({lvis_tag}): {len(uids)} objects")
    return selected

def select_lvis_wide(lvis, min_per_cat=10, max_per_cat=10, seed=42):
    """Select UIDs for LVIS-wide experiment: sample from all large-enough categories."""
    rng = random.Random(seed)
    selected = {}
    for cat_name, uids in sorted(lvis.items()):
        if len(uids) >= min_per_cat:
            sampled = rng.sample(uids, min(max_per_cat, len(uids)))
            selected[cat_name] = sampled
    print(f"  {len(selected)} categories selected, "
          f"{sum(len(v) for v in selected.values())} total objects")
    return selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["5cat", "lvis_wide"], required=True)
    parser.add_argument("--output_dir", type=str, default="data/objaverse")
    parser.add_argument("--max_per_cat", type=int, default=10,
                        help="Max objects per category (lvis_wide mode)")
    parser.add_argument("--min_per_cat", type=int, default=10,
                        help="Min objects to include a category (lvis_wide mode)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir) / args.mode
    out.mkdir(parents=True, exist_ok=True)

    print("Loading LVIS annotations...")
    lvis = objaverse.load_lvis_annotations()
    print(f"LVIS: {len(lvis)} categories, {sum(len(v) for v in lvis.values())} objects")

    if args.mode == "5cat":
        selected = select_5cat(lvis)
    else:
        selected = select_lvis_wide(
            lvis, min_per_cat=args.min_per_cat,
            max_per_cat=args.max_per_cat, seed=args.seed,
        )

    # Collect all UIDs
    all_uids = []
    uid_to_cat = {}
    for cat_name, uids in selected.items():
        for uid in uids:
            all_uids.append(uid)
            uid_to_cat[uid] = cat_name

    print(f"\nDownloading {len(all_uids)} objects...")
    objects = objaverse.load_objects(uids=all_uids)

    # Build manifest
    manifest = []
    for uid, glb_path in objects.items():
        manifest.append({
            "uid": uid,
            "category": uid_to_cat[uid],
            "glb_path": str(glb_path),
        })

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path} ({len(manifest)} objects)")

    # Summary
    from collections import Counter
    cat_counts = Counter(m["category"] for m in manifest)
    print(f"\nCategory summary ({len(cat_counts)} categories):")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {cat}: {count}")
    if len(cat_counts) > 20:
        print(f"  ... and {len(cat_counts) - 20} more")

if __name__ == "__main__":
    main()
```

**Step 2: Run 5cat download**

```bash
cd /home/cc/Meshlex-Research
python scripts/download_objaverse.py --mode 5cat --output_dir data/objaverse
```

预期产出：`data/objaverse/5cat/manifest.json`，包含 ~847 objects（chair 453 + table 101 + airplane 112 + car 102 + lamp 79）。

**Step 3: Verify with a few GLB files**

用 trimesh 加载 3 个 GLB 验证格式兼容性：

```bash
python -c "
import json, trimesh
manifest = json.load(open('data/objaverse/5cat/manifest.json'))
for m in manifest[:3]:
    mesh = trimesh.load(m['glb_path'], force='mesh')
    print(f'{m[\"category\"]} {m[\"uid\"][:8]}: {mesh.faces.shape[0]} faces, {mesh.vertices.shape[0]} verts')
"
```

**Step 4: Commit**

```bash
git add scripts/download_objaverse.py
git commit -m "feat: add Objaverse-LVIS download script (5cat + lvis_wide modes)"
git push
```

---

## Task 2: Modify Preprocessing Script to Support Manifest Input

**Files:**
- Modify: `scripts/run_preprocessing.py`

**目的：** 现有脚本按 ShapeNet 目录结构查找 OBJ 文件。需要添加 `--input_manifest` 参数，读取 Task 1 生成的 manifest.json，直接使用 GLB 路径。

**Step 1: Add manifest mode to `run_preprocessing.py`**

在 `main()` 中添加分支：如果提供了 `--input_manifest`，则从 manifest 读取文件列表，跳过 ShapeNet 目录扫描。

关键改动：
- 新增参数 `--input_manifest`：JSON 文件路径，格式 `[{uid, category, glb_path}, ...]`
- 新增参数 `--experiment_name`：实验名称（如 `5cat`, `lvis_wide`），用于组织输出目录
- 当使用 manifest 时，`mesh_id` 取自 `uid`，`category` 取自 manifest entry
- `load_and_preprocess_mesh` 已支持 GLB（trimesh 内部处理），无需改动 `src/data_prep.py`

Train/test split 逻辑：
- 5cat 模式：与现有逻辑一致，`TRAIN_CATEGORIES = {"chair", "table", "airplane"}`，按 mesh_id 80/20 split
- lvis_wide 模式：新增 `--split_mode category_holdout` 参数
  - 随机留出 50 个类别作为 unseen test
  - 其余类别中按 mesh_id 80/20 split 为 seen_train / seen_test

**Step 2: Test with 3 GLB files from manifest**

```bash
python scripts/run_preprocessing.py \
    --input_manifest data/objaverse/5cat/manifest.json \
    --experiment_name 5cat \
    --output_root data \
    --target_faces 1000 \
    --max_per_category 3
```

预期：`data/meshes/5cat/{category}/` 产出 OBJ，`data/patches/5cat/{category}/` 产出 NPZ。

**Step 3: Commit**

```bash
git add scripts/run_preprocessing.py
git commit -m "feat: support manifest-based preprocessing for Objaverse GLB files"
git push
```

---

## Task 3: Experiment 1 — 5-Category Data Preparation (Phase A1)

**Files:**
- No code changes — running existing scripts

**Step 1: Run full 5cat preprocessing**

```bash
python scripts/run_preprocessing.py \
    --input_manifest data/objaverse/5cat/manifest.json \
    --experiment_name 5cat \
    --output_root data \
    --target_faces 1000
```

不设 `max_per_category`，取全部可用数据。预计耗时 ~30min-1h。

产出：
- `data/meshes/5cat/{chair,table,airplane,car,lamp}/` — 预处理后 OBJ
- `data/patches/5cat/{chair,table,airplane}_train/` — 训练 patches
- `data/patches/5cat/{chair,table,airplane}_test/` — 测试 patches
- `data/patches/5cat/{car,lamp}/` — 跨类别测试 patches
- `data/patch_metadata_5cat.json`

**Step 2: Verify patch statistics**

```bash
python -c "
import json, numpy as np
meta = json.load(open('data/patch_metadata_5cat.json'))
cats = {}
for m in meta:
    c = m['category']
    cats.setdefault(c, []).append(m['n_patches'])
for c, patches in sorted(cats.items()):
    p = np.array(patches)
    print(f'{c}: {len(patches)} meshes, {p.sum()} patches, median {np.median(p):.0f} patches/mesh')
print(f'Total: {len(meta)} meshes, {sum(sum(v) for v in cats.values())} patches')
"
```

预期：chair ~400+ meshes, table ~90+, airplane ~100+, car ~90+, lamp ~70+。

**Step 3: Save validation report + visualizations to `results/exp1_phase_a/`**

包含：patch 数量统计、patch size 分布 histogram、3 个 sample mesh 的 patch 着色预览图。

**Step 4: Commit**

```bash
git add results/exp1_phase_a/ data/patch_metadata_5cat.json
git commit -m "data: Experiment 1 Phase A complete — 5-cat Objaverse preprocessed"
git push
```

---

## Task 4: Experiment 1 — Encoder-Only Training (Phase B1)

**Files:**
- No code changes — using existing `scripts/train.py`

**Step 1: Verify .gitignore**

确认 `data/` 已在 `.gitignore` 中（已确认）。

**Step 2: Run encoder-only training (20 epochs, VQ disabled)**

```bash
python scripts/train.py \
    --train_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --val_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 999 \
    --checkpoint_dir data/checkpoints/5cat
```

预期：`recon_loss` 持续下降，产出 `data/checkpoints/5cat/checkpoint_epoch019.pt`。

**Step 3: Verify**

```bash
python -c "
import torch
ckpt = torch.load('data/checkpoints/5cat/checkpoint_epoch019.pt', map_location='cpu', weights_only=False)
print(f'Epoch: {ckpt[\"epoch\"]}')
history = ckpt['history']
print(f'Loss: {history[0][\"recon_loss\"]:.6f} -> {history[-1][\"recon_loss\"]:.6f}')
print(f'Decreased: {history[0][\"recon_loss\"] > history[-1][\"recon_loss\"]}')
"
```

---

## Task 5: Experiment 1 — K-means Init + Quick VQ-VAE (Phase C1+D1)

**Files:**
- No code changes

**Step 1: K-means codebook initialization**

```bash
python scripts/init_codebook.py \
    --checkpoint data/checkpoints/5cat/checkpoint_epoch019.pt \
    --patch_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --codebook_size 4096 \
    --output data/checkpoints/5cat/checkpoint_kmeans_init.pt
```

**Step 2: Quick VQ-VAE training (20 epochs)**

```bash
python scripts/train.py \
    --train_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --val_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --resume data/checkpoints/5cat/checkpoint_kmeans_init.pt \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 0 \
    --checkpoint_dir data/checkpoints/5cat_vq
```

监控：`codebook_utilization` > 30%，`recon_loss` 收敛。

---

## Task 6: Experiment 1 — Evaluate + Visualize (Phase E1)

**Files:**
- No code changes

**Step 1: Evaluate**

```bash
python scripts/evaluate.py \
    --checkpoint data/checkpoints/5cat_vq/checkpoint_final.pt \
    --same_cat_dirs data/patches/5cat/chair_test data/patches/5cat/table_test data/patches/5cat/airplane_test \
    --cross_cat_dirs data/patches/5cat/car data/patches/5cat/lamp \
    --output results/exp1_eval.json
```

**Step 2: Visualize**

```bash
python scripts/visualize.py \
    --checkpoint data/checkpoints/5cat_vq/checkpoint_final.pt \
    --history data/checkpoints/5cat_vq/training_history.json \
    --patch_dirs data/patches/5cat/chair_train data/patches/5cat/table_train data/patches/5cat/airplane_train \
    --output_dir results/exp1_plots
```

**Step 3: Write validation report to `results/exp1_plots/report.md`**

包含：training curves、codebook utilization、same-cat CD、cross-cat CD、cross/same ratio、preliminary Go/No-Go。

**Step 4: Commit**

```bash
git add results/exp1_eval.json results/exp1_plots/
git commit -m "eval: Experiment 1 (5-cat) quick validation complete"
git push
```

---

## 决策点 1

审查实验 1 结果：
- utilization > 30% + loss 收敛 → 继续实验 2
- codebook collapse 或 loss 不收敛 → debug，不进入实验 2

---

## Task 7: LVIS-Wide Download (Phase A2)

**Files:**
- No code changes — using `scripts/download_objaverse.py`

**Step 1: Download LVIS-wide data**

```bash
python scripts/download_objaverse.py \
    --mode lvis_wide \
    --output_dir data/objaverse \
    --min_per_cat 10 \
    --max_per_cat 10 \
    --seed 42
```

预期：~500+ 类别，~3000-5000 objects。产出 `data/objaverse/lvis_wide/manifest.json`。

**Step 2: Verify disk usage**

```bash
du -sh data/objaverse/lvis_wide/ ~/.objaverse/
```

如果 `~/.objaverse/` 缓存过大，可选择清理。

---

## Task 8: Experiment 2 — LVIS-Wide Preprocessing (Phase A2)

**Files:**
- No code changes

**Step 1: Run preprocessing with category-holdout split**

```bash
python scripts/run_preprocessing.py \
    --input_manifest data/objaverse/lvis_wide/manifest.json \
    --experiment_name lvis_wide \
    --output_root data \
    --target_faces 1000 \
    --split_mode category_holdout \
    --holdout_categories 50 \
    --seed 42
```

产出：
- `data/patches/lvis_wide/seen_train/` — 训练 patches（~450 类）
- `data/patches/lvis_wide/seen_test/` — seen 类别测试 patches
- `data/patches/lvis_wide/unseen/` — 50 个 unseen 类别 patches
- `data/patch_metadata_lvis_wide.json`

**Step 2: Verify**

```bash
python -c "
import json
meta = json.load(open('data/patch_metadata_lvis_wide.json'))
print(f'Total: {len(meta)} meshes')
cats = set(m['category'] for m in meta)
print(f'Categories: {len(cats)}')
"
```

**Step 3: Save validation report to `results/exp2_phase_a/`**

**Step 4: Commit**

```bash
git add results/exp2_phase_a/ data/patch_metadata_lvis_wide.json
git commit -m "data: Experiment 2 Phase A complete — LVIS-wide preprocessed"
git push
```

---

## Task 9: Experiment 2 — Training Pipeline (Phase B2+C2+D2)

**Files:**
- No code changes

**Step 1: Encoder-Only 20 epochs**

```bash
python scripts/train.py \
    --train_dirs data/patches/lvis_wide/seen_train \
    --val_dirs data/patches/lvis_wide/seen_test \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 999 \
    --checkpoint_dir data/checkpoints/lvis_wide
```

**Step 2: K-means init**

```bash
python scripts/init_codebook.py \
    --checkpoint data/checkpoints/lvis_wide/checkpoint_epoch019.pt \
    --patch_dirs data/patches/lvis_wide/seen_train \
    --codebook_size 4096 \
    --output data/checkpoints/lvis_wide/checkpoint_kmeans_init.pt
```

**Step 3: Quick VQ-VAE 20 epochs**

```bash
python scripts/train.py \
    --train_dirs data/patches/lvis_wide/seen_train \
    --val_dirs data/patches/lvis_wide/seen_test \
    --resume data/checkpoints/lvis_wide/checkpoint_kmeans_init.pt \
    --epochs 20 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 0 \
    --checkpoint_dir data/checkpoints/lvis_wide_vq
```

---

## Task 10: Experiment 2 — Evaluate + Visualize (Phase E2)

**Files:**
- No code changes

**Step 1: Evaluate**

```bash
python scripts/evaluate.py \
    --checkpoint data/checkpoints/lvis_wide_vq/checkpoint_final.pt \
    --same_cat_dirs data/patches/lvis_wide/seen_test \
    --cross_cat_dirs data/patches/lvis_wide/unseen \
    --output results/exp2_eval.json
```

**Step 2: Visualize**

```bash
python scripts/visualize.py \
    --checkpoint data/checkpoints/lvis_wide_vq/checkpoint_final.pt \
    --history data/checkpoints/lvis_wide_vq/training_history.json \
    --patch_dirs data/patches/lvis_wide/seen_train \
    --output_dir results/exp2_plots
```

**Step 3: Write validation report to `results/exp2_plots/report.md`**

重点关注：unseen 50 categories 的 CD 与 seen categories 的比较。

**Step 4: Commit**

```bash
git add results/exp2_eval.json results/exp2_plots/
git commit -m "eval: Experiment 2 (LVIS-wide) quick validation complete"
git push
```

---

## 决策点 2：综合评估

对比两组实验结果：

| 指标 | 实验 1 (5-cat) | 实验 2 (LVIS-wide) |
|------|----------------|-------------------|
| Same-cat CD | ? | ? |
| Cross-cat CD | ? | ? |
| Cross/Same ratio | ? | ? |
| Utilization | ? | ? |

按 Go/No-Go 矩阵决策：

| Cross/Same CD Ratio | Utilization | Decision |
|---------------------|-------------|----------|
| < 1.2x | > 50% | **STRONG GO** |
| 1.2x - 2.0x | > 50% | **WEAK GO** |
| < 2.0x | 30% - 50% | **CONDITIONAL GO** |
| 2.0x - 3.0x | any | **HOLD** |
| > 3.0x | any | **NO-GO** |

- 两组都 GO → 选表现更好的那组进入 Phase F（200 epoch 全量训练）
- 一组 GO 一组 HOLD → 分析差异原因，用 GO 的那组继续
- 两组都 NO-GO → 核心假设被推翻，pivot

---

## Task 11: Full Training (Phase F) — 仅在 Go 后执行

**选表现好的实验组（或两组都跑），训练 200 epochs：**

```bash
python scripts/train.py \
    --train_dirs <best_experiment_train_dirs> \
    --val_dirs <best_experiment_val_dirs> \
    --resume <best_experiment_kmeans_checkpoint> \
    --epochs 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --vq_start_epoch 0 \
    --checkpoint_dir data/checkpoints/<experiment>_full
```

## Task 12: Final Evaluation + Go/No-Go (Phase G)

完整评估 + 可视化 + 最终 Go/No-Go 决策。产出保存到 `results/final_evaluation/`。
