# MeshLex — Session Handoff TODO

> **最后更新**: 2026-03-08 15:10
> **上次完成**: Exp1 v2 (Collapse Fix) 训练 200 epochs 完成

## 立即执行 — Task 11: 评估 + Go/No-Go 决策

### 前置条件
- Final checkpoint: `data/checkpoints/5cat_v2/checkpoint_final.pt`
- 训练数据: `data/objaverse_patches/` (5 categories: chair/table/airplane/car/lamp)
- 训练类别: chair, table, airplane
- 测试类别 (跨类别): car, lamp

### 执行步骤

1. **运行评估脚本**:
   ```bash
   python scripts/evaluate.py \
     --checkpoint data/checkpoints/5cat_v2/checkpoint_final.pt \
     --data_dir data/objaverse_patches
   ```

2. **评估内容** (参见 `src/evaluate.py`):
   - Same-category Chamfer Distance (chair/table/airplane 上重建)
   - Cross-category Chamfer Distance (car/lamp 上重建)
   - Codebook utilization (已确认 99.7%)
   - Cross-cat/Same-cat CD ratio → Go/No-Go 决策

3. **Go/No-Go 标准** (参见 `context/10_objaverse_migration_design.md`):
   - Cross-cat CD / Same-cat CD < 1.2× → **强成功**
   - 1.2× ~ 3.0× → 部分成功，需分析
   - > 3.0× → 失败止损

4. **结果保存**: 保存到 `results/exp1_v2_collapse_fix/` 目录

### 注意事项
- 评估前先检查 GPU 显存: `nvidia-smi`
- 评估脚本使用 eval 模式，显存需求更低

## 后续任务

### 如果 Go/No-Go = Go
- 进入 Exp2: LVIS-Wide (500+ 类别广覆盖测试)
- 参见 `context/11_objaverse_experiment_plan.md` Task 12

### 如果 Go/No-Go = No-Go
- 分析失败原因
- 可能需要调整 codebook size / patch size / model architecture

## 训练关键结果摘要 (供参考)

| 指标 | v1 (Collapse) | v2 (Fixed) |
|------|--------------|-----------|
| Utilization | 0.46% | **99.7%** |
| Recon loss | 0.326 | **0.228** |
| Active codes | 19/4096 | 4084/4096 |

## 重要文件路径

- 实验计划: `context/11_objaverse_experiment_plan.md`
- Collapse 诊断: `context/12_codebook_collapse_diagnosis.md`
- Collapse Fix 设计: `context/20_collapse_fix_implementation_plan.md`
- 训练报告: `results/exp1_v2_collapse_fix/`
- 模型代码: `src/model.py`, `src/trainer.py`
- 评估代码: `src/evaluate.py`, `scripts/evaluate.py`

## Checkpoint 管理

按 CLAUDE.md 规范，只保留最新 3 个 checkpoint。当前状态:
```
data/checkpoints/5cat_v2/
├── checkpoint_epoch179.pt
├── checkpoint_epoch199.pt
├── checkpoint_final.pt        ← 使用这个
└── training_history.json
```
