# Exp1 v2 (Collapse Fix) — 最终训练报告

> **时间**: 2026-03-08 15:10
> **状态**: 训练完成 (200/200 epochs)
> **Checkpoint**: `data/checkpoints/5cat_v2/checkpoint_final.pt`

## 最终指标 (Epoch 199)

| 指标 | 值 |
|------|------|
| Total loss | 0.2530 |
| Recon loss | 0.2280 |
| Commit loss | 0.0125 |
| Embed loss | 0.0125 |
| **Codebook Utilization** | **99.7%** (4084/4096) |
| Dead codes | **12** (仅 0.3%) |
| Val recon loss | 0.2383 |
| Val utilization | 46.0% |
| LR (final) | 0.0 (cosine 已归零) |
| Speed | ~45s/epoch |
| 总训练时间 | ~2.5 小时 |

## 完整训练轨迹

```
Epoch   0 [warmup]: util  1.2%, recon 0.6649, total 0.6649, lr 2.08e-05
Epoch   9 [warmup]: util  1.8%, recon 0.3183, total 0.3183, lr 9.98e-05
--- K-means Init (Epoch 10) ---
Epoch  19: util 55.0%, recon 0.2853, total 0.3059, lr 9.85e-05
Epoch  29: util 46.9%, recon 0.2719, total 0.2947, lr 9.60e-05
Epoch  39: util 41.8%, recon 0.2634, total 0.2906, lr 9.23e-05
Epoch  49: util 38.5%, recon 0.2586, total 0.2945, lr 8.74e-05
Epoch  59: util 37.1%, recon 0.2557, total 0.2970, lr 8.16e-05  ← 最低点
Epoch  69: util 39.2%, recon 0.2520, total 0.2938, lr 7.50e-05  ← 开始恢复
Epoch  79: util 42.9%, recon 0.2476, total 0.2884, lr 6.77e-05
Epoch  89: util 48.2%, recon 0.2451, total 0.2864, lr 6.00e-05
Epoch  99: util 57.3%, recon 0.2438, total 0.2846, lr 5.20e-05
Epoch 109: util 70.3%, recon 0.2416, total 0.2797, lr 4.40e-05
Epoch 119: util 83.2%, recon 0.2384, total 0.2705, lr 3.61e-05
Epoch 129: util 87.0%, recon 0.2343, total 0.2614, lr 2.86e-05
Epoch 139: util 92.6%, recon 0.2321, total 0.2578, lr 2.16e-05
Epoch 149: util 95.7%, recon 0.2302, total 0.2553, lr 1.54e-05
Epoch 159: util 96.5%, recon 0.2294, total 0.2547, lr 1.00e-05
Epoch 169: util 97.9%, recon 0.2282, total 0.2531, lr 5.73e-06
Epoch 179: util 98.8%, recon 0.2277, total 0.2528, lr 2.57e-06
Epoch 189: util 99.2%, recon 0.2277, total 0.2527, lr 6.47e-07
Epoch 199: util 99.7%, recon 0.2280, total 0.2530, lr 0.00e+00
```

## Utilization 四阶段分析

### Phase 1: Encoder Warmup (Epoch 0-9)
- Codebook C frozen, 随机初始化, util ~1-2%
- Recon loss 快速从 0.66 降到 0.32
- 仅训练 encoder weights，建立合理的 z-space

### Phase 2: K-means Init → 逐渐下降 (Epoch 10-59)
- K-means 将 C 初始化到 z-space 聚类中心
- Post-init util 瞬间跳到 55%（epoch 19）
- 随后 W 快速更新导致 CW 空间变形，边缘 codes 失活
- Util 从 55% 逐渐降到 37.1%（epoch 59，最低点）
- Dead codes 从 1845 增到 2578（peak）

### Phase 3: V 型恢复 (Epoch 60-110)
- Cosine LR 降低 → W 更新减缓 → CW 空间稳定
- Dead code revival 持续注入新 codes
- 正循环：更多活跃 codes → 更好梯度 → 更稳定 CW → 更少 dead codes
- Util 从 37.1% 恢复到 70.3%

### Phase 4: 饱和 (Epoch 110-199)
- LR 持续降低到接近零
- Util 持续攀升：70% → 83% → 93% → 97% → 99.7%
- Dead codes 从 ~1200 降到 12
- Codebook 几乎完全活跃

## Dead Code Revival 完整历史

| Epoch | Dead codes replaced | Util (post) | 趋势 |
|-------|-------------------|-------------|------|
| 19 | 1845/4096 | 55.0% | 初始 |
| 29 | 2173/4096 | 46.9% | ↑ |
| 39 | 2383/4096 | 41.8% | ↑ |
| 49 | 2520/4096 | 38.5% | ↑ peak |
| 59 | 2578/4096 | 37.1% | ↑ peak |
| 69 | 2490/4096 | 39.2% | ↓ 开始下降 |
| 79 | 2340/4096 | 42.9% | ↓ |
| 89 | 2123/4096 | 48.2% | ↓ |
| 99 | 1750/4096 | 57.3% | ↓↓ |
| 109 | 1218/4096 | 70.3% | ↓↓↓ |
| 119 | ~700 (est.) | 83.2% | ↓↓↓ |
| 189 | ~30 (est.) | 99.2% | 几乎为零 |
| 199 | 12 | 99.7% | 接近消失 |

## vs v1 (Collapse 版) 对比

| 指标 | v1 (Collapse) | v2 (Fixed) | 改善 |
|------|--------------|-----------|------|
| Utilization | 0.46% | **99.7%** | **217x** |
| Recon loss | 0.326 | **0.228** | **30%** |
| Active codes | 19/4096 | **4084/4096** | **215x** |
| Dead codes | 4077 | **12** | -99.7% |

## 关键 Collapse Fix 技术

1. **SimVQ 正确实现**：Frozen C + Learnable W，CW = W(C) 为 effective codebook
2. **Encoder Warmup**：10 epochs recon-only，建立合理 z-space 再 K-means init
3. **K-means Init**：C = W^T(z-centroids)，保证 CW ≈ z-centroids
4. **CW-aligned Dead Code Revival**：C[dead] = (z_sample + noise) @ W，确保 CW[dead] ≈ z
5. **Cosine Annealing LR**：从 1e-4 降到 0，稳定后期 CW 空间

## 下一步

- **Task 11**: 运行评估脚本，计算 same-cat CD / cross-cat CD / Go/No-Go 决策
- Checkpoint: `data/checkpoints/5cat_v2/checkpoint_final.pt`
- 评估命令: `python scripts/evaluate.py --checkpoint data/checkpoints/5cat_v2/checkpoint_final.pt --data_dir data/objaverse_patches`
