# MeshLex — Session Handoff TODO

> **最后更新**: 2026-03-08 15:30
> **上次完成**: Exp1 A-stage × 5cat 评估完成 — STRONG GO

## 实验进度总览

| # | 实验 | 状态 | 结果 |
|---|------|------|------|
| 1 | A-stage × 5-Category | **完成** | STRONG GO (ratio 1.14x, util 46%) |
| 2 | A-stage × LVIS-Wide | 待执行 | — |
| 3 | B-stage × 5-Category | 待执行 | — |
| 4 | B-stage × LVIS-Wide | 待执行 | — |

## 下一步执行 — 实验 2: A-stage × LVIS-Wide

### Step 1: 下载 LVIS-Wide 数据
```bash
PYTHONPATH=. python scripts/download_objaverse.py --mode lvis_wide --output_dir data/objaverse --min_per_cat 10 --max_per_cat 10
```

### Step 2: 预处理
```bash
PYTHONPATH=. python scripts/run_preprocessing.py \
  --input_manifest data/objaverse/lvis_wide/manifest.json \
  --experiment_name lvis_wide --output_root data --target_faces 1000
```

### Step 3: 训练 200 epochs (A-stage)
```bash
PYTHONPATH=. python scripts/train.py \
  --train_dirs <lvis_wide 训练目录> \
  --val_dirs <lvis_wide 验证目录> \
  --epochs 200 --batch_size 256 --lr 1e-4 \
  --warmup_epochs 5 --dead_code_interval 10 \
  --checkpoint_dir data/checkpoints/lvis_wide_A
```

### Step 4: 评估 + 可视化 + 报告
保存到 results/exp2_A_lvis_wide/

## 后续实验

### 实验 3: B-stage × 5-Category
1. 实现 B-stage 代码修改（rotation trick + multi-token KV decoder）
2. 训练 200 epochs on 5cat
3. 评估 + 报告 → results/exp3_B_5cat/

### 实验 4: B-stage × LVIS-Wide
1. 训练 200 epochs on LVIS-Wide
2. 评估 + 报告 → results/exp4_B_lvis_wide/

### 全部完成后
1. 写综合对比报告 results/final_comparison_report.md
2. 更新 CLAUDE.md, README.md
3. 删除 TODO.md
4. commit + push

## 重要文件
- Exp1 结果: results/exp1_A_5cat/
- A-stage checkpoint: data/checkpoints/5cat_v2/checkpoint_final.pt
- B-stage 设计: context/19_codebook_collapse_fix_design.md 第四节
- 实验计划: context/11_objaverse_experiment_plan.md
