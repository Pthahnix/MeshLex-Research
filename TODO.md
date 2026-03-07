# MeshLex TODO

## 当前进度

### 已完成

- [x] 研究方向确定：MeshLex — topology-aware patch vocabulary
- [x] 可行性验证实验设计（`context/06_plan_meshlex_validation.md`）
- [x] 14-Task 代码实现（`context/07_impl_plan_meshlex_validation.md`）
  - src/: data_prep, patch_segment, patch_dataset, model, losses, trainer, evaluate
  - scripts/: train, evaluate, visualize, init_codebook, run_preprocessing, download_shapenet
  - tests/: 17 unit tests 全部通过
  - results/: task1-13 验证产出
- [x] Phase A+B ShapeNet 实验执行计划（`context/08`, `09`）
- [x] HuggingFace 登录（用户 Pthahnix）
- [x] 数据源切换决策：ShapeNet → Objaverse-LVIS（ShapeNet 审批未通过）
- [x] Objaverse 迁移设计（`context/10_objaverse_migration_design.md`）
- [x] 双实验实施计划（`context/11_objaverse_experiment_plan.md`，12 Tasks）

### 已废弃

- ~~ShapeNet/ShapeNetCore 数据集审批~~ — 已切换到 Objaverse，不再需要

---

## 接下来要做的（按顺序）

详细计划见 `context/11_objaverse_experiment_plan.md`。

### 代码改动（Task 1-2）

1. [ ] **Task 1: 创建 `scripts/download_objaverse.py`**
   - 支持 `--mode 5cat` / `--mode lvis_wide`
   - 产出 manifest.json

2. [ ] **Task 2: 修改 `scripts/run_preprocessing.py`**
   - 支持 `--input_manifest` 读取 GLB 文件列表
   - 支持 `--split_mode category_holdout`（LVIS-wide 用）

### 实验 1：5-Category（Task 3-6）

3. [ ] **Task 3: Phase A1** — 预处理 5 类 Objaverse 数据
4. [ ] **Task 4: Phase B1** — Encoder-Only 20 epochs
5. [ ] **Task 5: Phase C1+D1** — K-means init + VQ-VAE 20 epochs
6. [ ] **Task 6: Phase E1** — 评估 + 可视化 + 快速 Go/No-Go

### 决策点 1

- utilization > 30% + loss 收敛 → 继续实验 2
- codebook collapse 或 loss 不收敛 → debug

### 实验 2：LVIS-Wide（Task 7-10）

7. [ ] **Task 7: Phase A2** — 下载 LVIS 广采样数据（500+ 类别）
8. [ ] **Task 8: Phase A2** — 预处理 + category-holdout split
9. [ ] **Task 9: Phase B2+C2+D2** — Encoder-Only → K-means → VQ-VAE
10. [ ] **Task 10: Phase E2** — 评估 + 可视化

### 决策点 2：综合评估

对比两组实验 → Go/No-Go 矩阵决策（见 `RUN_GUIDE.md`）

### 全量训练（Task 11-12，仅在 Go 后执行）

11. [ ] **Task 11: Phase F** — 200 epochs 全量训练
12. [ ] **Task 12: Phase G** — 最终评估 + Go/No-Go 决策

---

## 关键文件速查

| 用途 | 文件 |
|------|------|
| 完整运行指南 | `RUN_GUIDE.md` |
| 迁移设计 | `context/10_objaverse_migration_design.md` |
| 实施计划 | `context/11_objaverse_experiment_plan.md` |
| 下载脚本 | `scripts/download_objaverse.py`（待创建） |
| 预处理脚本 | `scripts/run_preprocessing.py`（待修改） |
| 训练脚本 | `scripts/train.py` |
| 评估脚本 | `scripts/evaluate.py` |

## 环境信息

- GPU: RTX 4090 24GB
- RAM: 503GB
- Disk: ~60GB available
- Python 3.11, PyTorch 2.4.1+cu124, torch-geometric 2.7.0
- objaverse: installed
