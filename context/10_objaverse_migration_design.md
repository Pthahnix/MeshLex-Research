# MeshLex: Objaverse Migration & Dual-Experiment Design

> 数据源从 ShapeNet 切换到 Objaverse-LVIS，设计两组实验验证 MeshLex 核心假设。

---

## 背景

ShapeNet/ShapeNetCore 在 HuggingFace 上需要审批，长期未通过。Objaverse（Allen AI）完全开放，且其 LVIS 子集（46207 objects, 1156 categories）提供了精确类别标注。

切换理由：
1. **无需审批**，`pip install objaverse` 即可下载
2. LVIS 子集标签质量高（LVIS 是 COCO 体系的大规模词汇标注）
3. 1156 个类别 → 可做远超 ShapeNet 5 类的跨类别 universality 验证

---

## 环境约束

| 资源 | 可用 |
|------|------|
| Disk | 60GB（需留 30GB+ 余量） |
| RAM | 503GB（可用 ~425GB） |
| GPU | RTX 4090 24GB |

**磁盘预算**：两组实验合计 ≤ 25GB。

---

## 实验设计

### 实验 1：5-Category（对标原 ShapeNet plan）

与 `09_phase_ab_execution_plan.md` 的实验设计保持一致，仅数据源替换。

| 角色 | 类别 | LVIS tag | 可用数量 |
|------|------|----------|---------|
| Train | Chair | `chair` | 453 |
| Train | Table | `table` | 101 |
| Train | Airplane | `airplane` | 112 |
| Cross-cat Test | Car | `car_(automobile)` | 102 |
| Cross-cat Test | Lamp | `lamp` | 79 |

- 训练类 80/20 split（按 mesh_id）
- max_per_category: 取全部可用数据
- 磁盘预算：~5GB

**目的**：验证 pipeline 端到端可用，与原 plan 的 Go/No-Go 标准对齐。

### 实验 2：LVIS-Wide（广覆盖 universality 验证）

| 参数 | 值 |
|------|-----|
| 类别筛选 | LVIS 中 ≥ 10 objects 的类别 |
| 每类采样 | 最多 10 个 objects |
| 预计总量 | ~3000-5000 meshes |
| Train 类别 | 随机选 80% 的类别 |
| Test 类别（seen） | Train 类别中 20% mesh 留出 |
| Test 类别（unseen） | 随机留出 50 个从未见过的类别 |

- 磁盘预算：~15-20GB

**目的**：验证 topology vocabulary 在 500+ 类别上的 universality。这是 MeshLex 论文的核心卖点——如果 4096 个 patch prototype 能覆盖 500+ 类别的 mesh，说服力远强于 5 类。

### 评估对比

| 指标 | 实验 1 | 实验 2 |
|------|--------|--------|
| Same-cat CD | chair/table/airplane test split | seen categories test split |
| Cross-cat CD | car + lamp | 50 unseen categories |
| Cross/Same ratio | 原 plan 的 Go/No-Go 标准 | 同标准 |
| Codebook utilization | > 30% | > 30% |
| 类别覆盖可视化 | 5 类 t-SNE | 500+ 类 t-SNE（核心 figure） |

---

## 执行顺序

```
实验 1（5-Category）              实验 2（LVIS-Wide）
─────────────────              ────────────────────
Phase A1: 下载 5 类 GLB          Phase A2: 下载 LVIS 广采样 GLB
Phase A1: GLB→OBJ→patches       Phase A2: GLB→OBJ→patches
Phase B1: Encoder-Only 20ep     Phase B2: Encoder-Only 20ep
Phase C1: K-means init          Phase C2: K-means init
Phase D1: VQ-VAE 20ep           Phase D2: VQ-VAE 20ep
Phase E1: 评估 + 可视化          Phase E2: 评估 + 可视化
         ↓                               ↓
    Quick Go/No-Go               Wide-Coverage Go/No-Go
         ↓                               ↓
         └──────── 综合决策 ────────────────┘
                    ↓
          Phase F: 全量 200ep（选表现好的那组或两组都跑）
          Phase G: 最终 Go/No-Go
```

实验 1 先跑，验证 pipeline；实验 2 复用同一套代码。

---

## 代码改动清单

### 新增

| 文件 | 说明 |
|------|------|
| `scripts/download_objaverse.py` | Objaverse 下载脚本，两种模式：`--mode 5cat` / `--mode lvis_wide` |

### 修改

| 文件 | 改动 |
|------|------|
| `scripts/run_preprocessing.py` | 支持 GLB 输入（除 ShapeNet OBJ 目录外），接受 `--input_manifest` JSON 文件指定 {uid, path, category} |
| `src/data_prep.py` | `load_and_preprocess_mesh` 已通过 trimesh 支持 GLB，无需改动（验证即可） |

### 不动

| 文件 | 原因 |
|------|------|
| `src/patch_segment.py` | mesh 格式无关 |
| `src/patch_dataset.py` | NPZ 格式不变 |
| `src/model.py` | 模型架构不变 |
| `src/losses.py` | loss 不变 |
| `src/trainer.py` | 训练逻辑不变 |
| `src/evaluate.py` | 评估逻辑不变 |
| `scripts/train.py` | 训练入口不变 |
| `scripts/evaluate.py` | 评估入口不变 |
| `scripts/visualize.py` | 可视化入口不变 |
| `scripts/init_codebook.py` | K-means 不变 |

---

## 数据流

### 下载阶段

```
objaverse.load_lvis_annotations()
    → 筛选类别 + 采样 UIDs
    → objaverse.load_objects(uids)
    → GLB files 下载到 ~/.objaverse/
    → 生成 manifest.json: [{uid, glb_path, category}, ...]
```

### 预处理阶段（复用现有 pipeline）

```
manifest.json
    → trimesh.load(glb_path)  # trimesh 直接支持 GLB
    → pyfqmr 降面 → 归一化
    → METIS patch 分割
    → NPZ 序列化到 data/patches/{experiment}/{category}/
```

---

## Go/No-Go 标准（沿用原 plan）

| Cross/Same CD Ratio | Utilization | Decision |
|---------------------|-------------|----------|
| < 1.2x | > 50% | **STRONG GO** |
| 1.2x - 2.0x | > 50% | **WEAK GO** |
| < 2.0x | 30% - 50% | **CONDITIONAL GO** |
| 2.0x - 3.0x | any | **HOLD** |
| > 3.0x | any | **NO-GO** |

实验 2 额外关注：unseen 50 categories 的 CD 是否和 seen categories 接近。

---

## 磁盘估算

| 阶段 | 实验 1 | 实验 2 | 合计 |
|------|--------|--------|------|
| GLB 下载 | ~2GB | ~10GB | ~12GB |
| 预处理 OBJ | ~500MB | ~2GB | ~2.5GB |
| Patches NPZ | ~500MB | ~2GB | ~2.5GB |
| Checkpoints | ~500MB | ~500MB | ~1GB |
| Results | ~100MB | ~100MB | ~200MB |
| **Total** | **~3.5GB** | **~14.5GB** | **~18GB** |

安全余量：60GB - 18GB = 42GB。

注意：objaverse 默认缓存在 `~/.objaverse/`，下载完 GLB 后可清理缓存释放空间。
