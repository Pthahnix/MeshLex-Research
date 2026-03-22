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
- **共享服务器**：GPU 0/1/2 均可能有其他用户在使用。启动前必须 `nvidia-smi` 确认目标卡空闲（VRAM 和利用率均低）
- 默认单卡，除非模型大到单卡放不下才考虑双卡
- 目标：**VRAM 利用率 ≥ 80%，GPU 算力利用率 ≥ 60%**
- 若某卡被他人长期占用，使用其他空闲卡，**不要抢占他人资源**

## 3. 启动前 Checklist

每次启动 GPU 任务前，必须按顺序执行：

1. `nvidia-smi` 确认目标卡空闲
2. 选定 `CUDA_VISIBLE_DEVICES=<N>`（0、1 或 2，哪个空闲用哪个）
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

当有 **≥ 2 个独立任务**排队，且有多张空闲 GPU 时：

- 两个任务分别用 tmux session 启动在不同 GPU
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

## 9. 断点续训（Resume）

**所有训练脚本均支持 `--resume`**，OOM 或中断后从最近 checkpoint 恢复，**禁止从头重跑**。

### 9.1 Checkpoint 保存规范

| 脚本 | 保存频率 | 文件命名 |
|------|---------|---------|
| `train_rvq.py` | 每 3 epochs | `checkpoint_epoch{N:03d}.pt` |
| `train_ar.py` | 每 10 epochs | `checkpoint_epoch{N:03d}.pt` |
| `train_mdlm.py` | 每 10 epochs | `checkpoint_epoch{N+1}.pt` |

磁盘管理：**只保留最新 3 个 checkpoint**，旧的自动删除（各脚本已内置）。

### 9.2 Resume 方法

```bash
# 找最新 checkpoint
latest=$(ls -t /data/pthahnix/MeshLex-Research/checkpoints/<exp>/checkpoint_epoch*.pt | head -1)
echo "Resuming from: $latest"

# 恢复训练（以 train_ar.py 为例）
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/train_ar.py \
    --sequence_dir /data/.../sequences/rvq_full_pca \
    --checkpoint_dir /data/.../checkpoints/ar_full_pca \
    --resume "$latest" \
    --stop_flag_file /tmp/stop_gpu1_ar_pca \
    ... \
    >> results/fullscale_eval/train_ar_pca.log 2>&1 &
```

## 10. GPU 让出机制（共享服务器礼让）

本服务器为多人共享，**当他人需要使用某块 GPU 时，我们的任务应主动让出**。

### 10.1 工作原理

所有训练脚本支持 `--stop_flag_file <path>` 参数。训练循环在每个 epoch 结束后检查该文件是否存在：
- **存在** → 保存当前 epoch checkpoint + training_history.json，然后优雅退出
- **不存在** → 继续训练

### 10.2 启动时必须指定 stop_flag_file

**每次启动训练任务时，必须加上 `--stop_flag_file`**，按 GPU 编号 + job 名命名：

```bash
# GPU 0 上的任务
--stop_flag_file /tmp/stop_gpu0_<job_name>

# GPU 1 上的任务
--stop_flag_file /tmp/stop_gpu1_<job_name>

# GPU 2 上的任务
--stop_flag_file /tmp/stop_gpu2_<job_name>
```

示例：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/train_ar.py \
    --sequence_dir /data/.../sequences/rvq_full_pca \
    --checkpoint_dir /data/.../checkpoints/ar_full_pca \
    --stop_flag_file /tmp/stop_gpu1_ar_pca \
    --epochs 200 ... \
    >> results/fullscale_eval/train_ar_pca.log 2>&1 &
```

### 10.3 让出操作

```bash
# 创建 stop flag，训练在当前 epoch 结束后自动保存并退出
touch /tmp/stop_gpu1_ar_pca

# 观察日志，出现 "Stop flag detected" 表示已安全退出
tail -f results/fullscale_eval/train_ar_pca.log
```

### 10.4 恢复操作

```bash
# 1. 删除 stop flag
rm -f /tmp/stop_gpu1_ar_pca

# 2. 找最新 checkpoint
latest=$(ls -t /data/pthahnix/MeshLex-Research/checkpoints/ar_full_pca/checkpoint_epoch*.pt | head -1)

# 3. 恢复训练
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/train_ar.py \
    --resume "$latest" \
    --stop_flag_file /tmp/stop_gpu1_ar_pca \
    ... \
    >> results/fullscale_eval/train_ar_pca.log 2>&1 &
```

### 10.5 支持 stop_flag_file 的脚本

| 脚本 | 触发位置 |
|------|---------|
| `scripts/train_rvq.py` | 每 epoch 末尾（via `src/trainer.py`） |
| `scripts/train_ar.py` | 每 epoch 末尾（`gc.collect()` 之后） |
| `scripts/train_mdlm.py` | 每 epoch 末尾（`cuda.empty_cache()` 之后） |
