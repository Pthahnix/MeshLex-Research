# Exp2 A-stage LVIS-Wide 进度 #10 — FINAL — Epoch 199/200

> **时间**: 2026-03-09 16:10
> **状态**: **训练完成 + 评估完成 → STRONG GO**

## 最终指标 (Epoch 199)

| 指标 | 值 |
|------|------|
| Train Recon loss | 0.2193 |
| Val Recon loss | 0.2170 |
| Train Utilization | **74.7%** |
| Val Utilization | **67.8%** |

## 评估结果

| 指标 | 值 |
|------|------|
| Same-cat CD | 217.0 ± 20.9 |
| Cross-cat CD | 232.3 ± 23.2 |
| **CD Ratio** | **1.07x** |
| Eval Utilization | **67.8%** (2779/4096) |
| **Decision** | **STRONG GO** |

## 关键发现

- CD ratio 1.07x 优于 5-cat 的 1.14x — 更多类别训练 = 更好泛化
- Eval utilization 67.8% vs 5-cat 的 46% — 大幅提升
- 绝对 CD 也更好 (217.0 vs 228.3) — 多样数据不损害反而提升重建质量

详见 `report.md` 完整分析。
