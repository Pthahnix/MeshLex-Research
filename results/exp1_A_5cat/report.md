# Exp1: A-stage × 5-Category — 评估报告

> **时间**: 2026-03-08
> **实验**: A-stage 代码 × 5-Category 数据 (Objaverse-LVIS)
> **Checkpoint**: `data/checkpoints/5cat_v2/checkpoint_final.pt` (200 epochs)

## Go/No-Go 决策

### **STRONG GO**

| 指标 | 值 | 标准 | 判定 |
|------|------|------|------|
| CD Ratio (cross/same) | **1.14x** | < 1.2x = STRONG GO | PASS |
| Codebook Utilization (eval) | **46.0%** (1884/4096) | > 30% | PASS |
| Codebook Utilization (train final) | **99.7%** (4084/4096) | > 30% | PASS |

## 评估详情

### Same-Category (chair/table/airplane test set)
- Test patches: 3,354
- Mean CD: **238.30** (×10³)
- Std CD: 17.11
- Utilization: 46.0% (1,884/4,096 codes)

### Cross-Category (car/lamp — unseen during training)
- Test patches: 4,065
- Mean CD: **272.82** (×10³)
- Std CD: 28.43
- Utilization: 47.0% (1,927/4,096 codes)

## 关键分析

### 1. CD Ratio 1.14x — 核心假设验证成功
- Cross-cat CD 仅比 same-cat 高 14%
- 这意味着：在 car/lamp 这两个**训练时完全未见过**的类别上，codebook 的重建质量仅略低于训练类别
- **核心假设成立**：mesh local topology 确实存在 universal vocabulary，跨类别可迁移

### 2. Eval Utilization 46% vs Train Utilization 99.7%
- 训练集上 99.7% 的 codes 被使用，说明 codebook 充分利用
- 测试集上 46% 的 codes 被使用，说明约半数 codes 是类别特异的
- 有趣的是 cross-cat utilization (47%) 略高于 same-cat (46%)，说明 car/lamp 使用了一些 chair/table/airplane 未高频使用的 codes

### 3. CD 绝对值分析
- Same-cat CD 238 (×10³) 表示平均每个 patch 的 Chamfer Distance
- Cross-cat CD 273 稍高但量级一致
- 后续可通过 B-stage（rotation trick + multi-token KV decoder）进一步降低

## 训练摘要（200 epochs）

```
训练数据: 12,854 patches (chair/table/airplane train)
验证数据: 3,354 patches (chair/table/airplane test)
模型: 1.06M params (0.54M trainable)
训练时间: ~2.5 hours (RTX 4090)
最终 train loss: 0.253, recon: 0.228
Utilization 轨迹: 1.2% → 62.3% → 36.5% (trough) → 99.7% (V-shaped recovery)
```

## 可视化文件

- `training_curves.png` — Loss + utilization + recon 曲线
- `codebook_tsne.png` — CW 空间 t-SNE 可视化
- `utilization_histogram.png` — Code 使用频率直方图
- `eval_results.json` — 完整评估数据

## 结论与下一步

**STRONG GO** — 核心假设验证成功。接下来：
1. 实验 2 (A-stage × LVIS-Wide)：500+ 类别广覆盖测试
2. 实验 3 (B-stage × 5cat)：rotation trick + multi-token KV decoder 增强
3. 实验 4 (B-stage × LVIS-Wide)：增强版广覆盖测试
