# GPU 利用率最大化指南

> **重要**：本文档是活文档（living document），在后续实践过程中应不断迭代升级——包括修复原始设计不合理的地方、添加新发现的好用的 GPU 利用率技巧等。遇到更好的实践经验时，直接更新本文件。

## 1. 硬件基准

| 项目 | 规格 |
|------|------|
| GPU | RTX 5090 × 3 (32 GB VRAM each) |
| 架构 | Blackwell, sm_120 |
| bf16 算力 | ~1.8 PFLOPS |
| 内存带宽 | 1792 GB/s |
| 系统内存 | 251 GB |
| CPU | 128 vCPU |

## 2. 总则

- 所有 GPU 任务（训练、编码、生成、分析）均适用本指南
- **GPU 0 永不使用**，仅使用 GPU 1 / GPU 2
- 默认单卡，除非模型大到单卡放不下才考虑双卡
- 目标：**VRAM 利用率 ≥ 80%，GPU 算力利用率 ≥ 60%**

## 3. 启动前 Checklist

每次启动 GPU 任务前，必须按顺序执行：

1. `nvidia-smi` 确认目标卡空闲
2. 选定 `CUDA_VISIBLE_DEVICES=1` 或 `=2`
3. 设置进程名 `setproctitle("Pthahnix-<Task>")`
4. 确认启用 bf16 混合精度
5. 用试探协议（见 §5）确定最大 batch size
6. 确认 DataLoader 配置最优（见 §6）
7. 跑 10 step 后检查 VRAM 利用率（见 §7）

## 4. 混合精度

所有任务默认启用 bf16（RTX 5090 原生支持）：

| 场景 | 配置 |
|------|------|
| 训练 | `torch.amp.autocast('cuda', dtype=torch.bfloat16)` + GradScaler |
| 推理/编码 | `torch.amp.autocast('cuda', dtype=torch.bfloat16)` |
| torch.compile | 条件启用（PyTorch ≥ 2.1 且模型支持） |

**禁用场景：**
- 数值精度敏感的 metric 计算（如 Chamfer Distance 最终评估）
- 已知 bf16 导致 NaN 的特殊层

## 5. Batch Size 试探协议

**目标**：找到不 OOM 的最大 batch size。

**步骤：**

1. 从「推荐值 × 2」开始尝试
2. 跑 3 个 step（forward + backward）
3. OOM → batch size 减半，回到步骤 2
4. 成功 → 检查 VRAM 占用：
   - **< 70%**：batch size × 1.5，回到步骤 2
   - **70-90%**：确定为最终值
   - **> 90%**：batch size × 0.85（留安全余量）

**推荐起始值（RTX 5090, 32GB, bf16）：**

| 任务类型 | 模型参数量 | 推荐 batch size |
|---------|-----------|----------------|
| VQ-VAE (~2M) | < 5M | 2048–4096 |
| AR Transformer (~20M) | 20–60M | 16–32 |
| AR Transformer (~57M) | 50–100M | 8–16 |
| MDLM (~40M) | 20–60M | 32–64 |
| Sequence 编码 (inference) | any | 4096–8192 |
| 生成 (inference) | any | 尽量大 |

## 6. DataLoader 标准配置

**标准配置（128 vCPU 服务器）：**

```python
DataLoader(
    dataset,
    batch_size=bs,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
```

**调优规则：**

| 现象 | 诊断 | 动作 |
|------|------|------|
| GPU 利用率 < 50% 且 VRAM 未满 | 数据加载瓶颈 | `num_workers` 翻倍（最多 16） |
| 系统内存压力大 | worker 吃内存过多 | `num_workers` 降到 4 |
| Parquet 数据源 | per-batch I/O 开销 | 用 `datasets.load_dataset()` 预加载到 Arrow |

## 7. 多任务并发

当 GPU 1 和 GPU 2 都空闲，且有 **≥ 2 个独立任务**排队时：

- 两个任务分别用 tmux session 启动在 GPU 1 和 GPU 2
- 命名：`tmux new -s meshlex-gpu1` / `tmux new -s meshlex-gpu2`
- 每个任务独立设置 `CUDA_VISIBLE_DEVICES`
- **内存约束**：两个任务的 DataLoader 共享系统内存，总 `num_workers ≤ 16`

单任务不做 DDP 多卡（除非模型 > 32GB 单卡放不下）。

## 8. 运行时监控

**训练开始 10 step 后**，执行一次 VRAM 检查：

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

**判断标准：**

| VRAM 利用率 | 动作 |
|------------|------|
| < 70% | 警告：「VRAM 利用率偏低，建议调大 batch size」 |
| 70–90% | 正常，继续 |
| > 90% | 注意 OOM 风险，必要时 batch size × 0.85 |

| GPU 算力利用率 | 动作 |
|---------------|------|
| < 30% | 警告：「GPU 算力利用率偏低，检查数据加载瓶颈」 |
| ≥ 60% | 正常 |

**长时间训练额外要求：**

- 每 10 epoch 记录一次 VRAM 和 GPU 利用率到 `training_history.json`
- 训练结束后在报告中注明峰值 VRAM 和平均 GPU 利用率
