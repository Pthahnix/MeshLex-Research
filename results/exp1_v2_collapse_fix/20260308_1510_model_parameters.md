# 模型参数规模与显存分析

> **时间**: 2026-03-08
> **模型**: MeshLexVQVAE (SimVQ variant)

## 参数总览

| 组件 | 参数量 | 可训练 | 说明 |
|------|--------|--------|------|
| PatchEncoder (SAGEConv GNN) | 338,048 | 338,048 | 2-layer SAGEConv + projection |
| SimVQCodebook | 541,696 | 16,512 | C frozen (524,288), W trainable (16,384+128) |
| PatchDecoder (Cross-Attn MLP) | 181,251 | 181,251 | Cross-attention + 3-layer MLP |
| **Total** | **1,060,995** | **536,707 (50.6%)** | |

## 各组件详细

### PatchEncoder (338,048 params)
- Input: PyG graph (15-dim face features)
- SAGEConv Layer 1: 15 → 64 (2,112 params)
- SAGEConv Layer 2: 64 → 128 (16,512 params)
- Batch normalization × 2
- Projection head: 128 → 128
- Output: 128-dim patch embedding (z)

### SimVQCodebook (541,696 params)
- **C (Frozen)**: 4096 × 128 = 524,288 params — 不参与梯度更新
- **W (Trainable)**: 128 × 128 = 16,384 params — 线性映射 C → CW
- **W bias**: 128 params
- Effective codebook CW = W(C), shape [4096, 128]
- SimVQ 核心：通过学习 W 间接更新 codebook，避免 straight-through gradient 问题

### PatchDecoder (181,251 params)
- Input: quantized embedding (128-dim) + face queries
- Cross-attention: query (face positions) attend to code embedding
- MLP: 128 → 256 → 128 → 3 (vertex positions)
- Output: per-face 3 vertex positions (9-dim per face)

## GPU 显存使用

| 项目 | 估算 |
|------|------|
| 模型参数 (FP32) | ~4 MB |
| 优化器状态 (Adam, 2× momentum) | ~4 MB |
| 梯度 | ~2 MB |
| 激活值 (batch=32) | ~50-100 MB |
| PyG 图数据 | ~100-200 MB |
| CUDA 开销 | ~200-300 MB |
| **总计** | **~500-600 MB** |
| **实测 (nvidia-smi)** | **~1058 MiB** |

- RTX 4090 显存: 24 GB
- 占用比例: **~4.3%** — 极轻量级模型
- 即使 batch_size 翻倍，显存仍远未饱和

## 模型规模评价

- **1M 参数级别**：属于轻量级模型，远小于 ViT-Base (86M) 或 ResNet-50 (25M)
- 这符合 VQ-VAE codebook 学习的定位 — 重点不在模型大小，而在 codebook 的表示能力
- 后续如需扩展：可增大 embedding dim (128→256) 或增加 encoder 层数
