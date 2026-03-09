# Exp2: A-stage × LVIS-Wide — 完整报告

> **实验**: Exp2 (A-stage × LVIS-Wide)
> **日期**: 2026-03-09
> **状态**: **STRONG GO**

## 实验配置

| 参数 | 值 |
|------|------|
| 数据集 | Objaverse-LVIS Wide (844 categories) |
| 训练集 | seen_train: 53,424 patches (794 categories) |
| 测试集 (seen) | seen_test: 13,315 patches |
| 测试集 (unseen) | unseen: 4,167 patches (50 categories) |
| Codebook size | 4096 |
| Embed dim | 128 |
| Hidden dim | 256 |
| Batch size | 256 |
| Epochs | 200 |
| LR | 1e-4 (cosine, 5-epoch warmup) |
| Dead code interval | 10 epochs |
| Encoder warmup | 10 epochs |
| Checkpoint | `data/checkpoints/lvis_wide_A/checkpoint_final.pt` |

## Go/No-Go 结果

| 指标 | 值 | 判定标准 |
|------|------|---------|
| Same-cat CD | 217.0 ± 20.9 | — |
| Cross-cat CD | 232.3 ± 23.2 | — |
| **CD Ratio** | **1.07x** | < 1.2x → STRONG GO |
| Same-cat Utilization | **67.8%** (2779/4096) | > 30% → STRONG GO |
| Cross-cat Utilization | 48.2% (1976/4096) | — |
| **Decision** | **STRONG GO** | — |

## 与 5-Category 实验对比

| 指标 | Exp1 (5-cat A) | **Exp2 (LVIS-Wide A)** | 变化 |
|------|----------------|----------------------|------|
| CD Ratio | 1.14x | **1.07x** | **-6.1%** (更好) |
| Same-cat CD | 228.3 | 217.0 | -4.9% (更好) |
| Cross-cat CD | 261.1 | 232.3 | -11.0% (更好) |
| Eval Utilization | 46.0% | **67.8%** | **+47%** (大幅提升) |
| Train Util | 99.7% | 74.7% | -25% (更真实) |

**关键发现**:
- **更多类别 = 更好的泛化**: LVIS-Wide (844 cat) 的 CD ratio 1.07x 优于 5-cat 的 1.14x
- **Eval utilization 大幅提升**: 46% → 67.8%，说明更丰富的训练分布让更多 codebook entries 被有效利用
- **绝对 CD 也更好**: 217.0 vs 228.3，说明多样数据不损害重建质量，反而提升
- **跨类别 CD 改善更大**: -11% vs -4.9%，证实 universal vocabulary 假设——更多类别训练后，unseen 类别的重建更好

## 训练趋势

```
Epoch   0: recon=0.675  val_recon=0.655  util=1.0%   val_util=0.9%
Epoch  10: recon=0.338  val_recon=0.273  util=79.8%  val_util=52.8%  (encoder warmup end)
Epoch  50: recon=0.231  val_recon=0.225  util=76.5%  val_util=66.8%
Epoch 100: recon=0.223  val_recon=0.219  util=75.3%  val_util=67.5%
Epoch 159: recon=0.220  val_recon=0.217  util=74.8%  val_util=67.7%
Epoch 199: recon=0.219  val_recon=0.217  util=74.7%  val_util=67.8%
```

- 训练 ~10h（200 epochs × ~186s/epoch）
- 无 codebook collapse，无过拟合
- Val recon 在 epoch ~100 后基本收敛

## 可视化产出

- `training_curves.png` — Loss + utilization 训练曲线
- `codebook_tsne.png` — CW space t-SNE
- `utilization_histogram.png` — Code usage 分布

## 下一步

1. **Exp4: B-stage LVIS-Wide** — 在此 checkpoint 基础上训练 multi-token KV decoder (num_kv_tokens=4)
   - 预期: CD 进一步改善 5-10%（参考 5-cat B-stage 的 -6.2%）
   - 不使用 rotation trick（已验证与 SimVQ 不兼容）
2. 完成 Exp4 后进行最终 4 实验综合评估
