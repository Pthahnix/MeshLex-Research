# MeshLex — Session Handoff TODO

> **最后更新**: 2026-03-12 11:00
> **当前状态**: LVIS-Wide 数据已就绪，Exp2 A-stage 训练中（7/200 epochs，预计 ~36h）。

## 实验进度总览

| # | 实验 | 状态 | 结果 | HF Checkpoint |
|---|------|------|------|---------------|
| 1 | A-stage × 5-Category | **完成** | STRONG GO (ratio 1.14x, util 46%) | `checkpoints/exp1_A_5cat/` |
| 2 | A-stage × LVIS-Wide | **训练中** (7/200 ep) | 上次 STRONG GO (ratio 1.07x, util 67.8%) | `checkpoints/exp2_A_lvis_wide/` (上次) |
| 3 | B-stage × 5-Category | **完成** | STRONG GO (ratio 1.18x, util 47%) | `checkpoints/exp3_B_5cat/` |
| 4 | B-stage × LVIS-Wide | **待执行** | — | — |

**HF Repo**: `Pthahnix/Meshlex-Research` (model repo)

## Pod 重置说明 (2026-03-12)

Pod 被重置，所有本地数据和 checkpoint 丢失。重建步骤：
1. Exp1 A-stage 5cat — **已重训** (checkpoint 已上传 HF)
2. Exp3 B-stage 5cat — **已重训** (checkpoint 已上传 HF)
3. LVIS-Wide 数据集 — **已完成** (188,696 train / 45,441 val / 12,655 unseen patches)
4. Exp2 A-stage LVIS-Wide — **训练中** (7/200 epochs, ~11min/epoch, 预计 ~36h)
5. Exp4 B-stage LVIS-Wide — 待 Exp2 完成后执行

注意：Exp2 上次的 checkpoint 已在 HF 上备份，但本地需要重新训练以获得 LVIS-Wide 数据集的 A-stage checkpoint 供 Exp4 resume。

## 当前本地状态

### 5-Category 数据 (已就绪)
- patches: `data/patches/` (chair, table, airplane, car, lamp — train/test splits)
- 可直接用于训练

### LVIS-Wide 数据 (已就绪)
- 下载 + 预处理 + category_holdout split 全部完成
- patches 分布：
  - `data/patches/lvis_wide/seen_train/` — 188,696 patches
  - `data/patches/lvis_wide/seen_test/` — 45,441 patches
  - `data/patches/lvis_wide/unseen/` — 12,655 patches

### Checkpoints (本地)
- Exp1: `data/checkpoints/5cat_v2/checkpoint_final.pt`
- Exp3: `data/checkpoints/5cat_B/checkpoint_final.pt`

## 待执行步骤

### Step 1: ~~完成 LVIS-Wide 数据下载 + split~~ ✅ 已完成

### Step 2: ~~训练 Exp2 A-stage LVIS-Wide~~ 🔄 训练中
- PID: 212028, 已完成 7/200 epochs, ~11min/epoch
- 当前在 encoder warmup 阶段 (epoch < 10)
- Loss: 0.61 → 0.27 (稳步下降)
- 预计完成时间：~36h (约 2026-03-13 晚间)
```bash
PYTHONPATH=. python scripts/train.py \
  --train_dirs data/patches/lvis_wide/seen_train \
  --val_dirs data/patches/lvis_wide/seen_test \
  --epochs 200 --batch_size 256 --lr 1e-4 \
  --warmup_epochs 5 --dead_code_interval 10 --encoder_warmup_epochs 10 \
  --checkpoint_dir data/checkpoints/lvis_wide_A
```

### Step 3: 评估 Exp2
```bash
PYTHONPATH=. python scripts/evaluate.py \
  --checkpoint data/checkpoints/lvis_wide_A/checkpoint_final.pt \
  --same_cat_dirs data/patches/lvis_wide/seen_test \
  --cross_cat_dirs data/patches/lvis_wide/unseen \
  --output results/exp2_A_lvis_wide/eval_results.json
```

### Step 4: 训练 Exp4 B-stage LVIS-Wide
```bash
PYTHONPATH=. python scripts/train.py \
  --train_dirs data/patches/lvis_wide/seen_train \
  --val_dirs data/patches/lvis_wide/seen_test \
  --epochs 200 --batch_size 256 --lr 1e-4 \
  --warmup_epochs 5 --dead_code_interval 10 --encoder_warmup_epochs 0 \
  --num_kv_tokens 4 \
  --resume data/checkpoints/lvis_wide_A/checkpoint_final.pt \
  --checkpoint_dir data/checkpoints/lvis_wide_B
```
**注意**: 不要用 `--use_rotation`，rotation trick 与 SimVQ 不兼容

### Step 5: 评估 Exp4
```bash
PYTHONPATH=. python scripts/evaluate.py \
  --checkpoint data/checkpoints/lvis_wide_B/checkpoint_final.pt \
  --same_cat_dirs data/patches/lvis_wide/seen_test \
  --cross_cat_dirs data/patches/lvis_wide/unseen \
  --output results/exp4_B_lvis_wide/eval_results.json
```

### Step 6: 可视化 + 最终报告
- 对比 4 组实验，写 final report
- 更新 TODO.md, CLAUDE.md, README.md

## 已完成实验详情

### Exp1: A-stage × 5cat — STRONG GO
- HF Checkpoint: `checkpoints/exp1_A_5cat/`
- CD Ratio: 1.14x, Utilization: 46% (eval) / 99.7% (train)
- Recon loss: 0.241 (final epoch, 重训结果)

### Exp2: A-stage × LVIS-Wide — STRONG GO (上次结果，需重训)
- HF Checkpoint: `checkpoints/exp2_A_lvis_wide/` (上次训练的)
- CD Ratio: 1.07x, Same-cat CD: 217.0, Cross-cat CD: 232.3
- Eval Util: 67.8%, Train Util: 74.7%
- **关键发现**: 更多类别 = 更好泛化

### Exp3: B-stage × 5cat — STRONG GO
- HF Checkpoint: `checkpoints/exp3_B_5cat/`
- CD Ratio: 1.18x, Same-cat CD: 223.5 (vs A-stage 238.3, -6.2%)
- Recon loss: 0.229 (final epoch, 重训结果)
- **关键发现**: rotation trick 与 SimVQ 不兼容，仅用 num_kv_tokens=4

## 重要注意事项
- **磁盘限制**: 80GB 总量，LVIS 下载必须用 batched 方式
- **Checkpoint 备份**: 训练完成后立即上传 HF，不得跳过
- **恢复训练**: B-stage resume 需 strict=False
- **监控**: 训练时注意 disk 使用，达到 88% 停止下载
