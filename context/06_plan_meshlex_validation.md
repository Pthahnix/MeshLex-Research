<!-- markdownlint-disable -->
# MeshLex 可行性验证实验计划

- [MeshLex 可行性验证实验计划](#meshlex-可行性验证实验计划)
  - [执行摘要](#执行摘要)
  - [核心假设与验证逻辑](#核心假设与验证逻辑)
  - [实验数据](#实验数据)
  - [Step 1: Patch 分割 (Day 0.5)](#step-1-patch-分割)
  - [Step 2: Patch Codebook 学习 (Day 1-1.5)](#step-2-patch-codebook-学习)
  - [Step 3: 评估与决策 (Day 0.5)](#step-3-评估与决策)
  - [Step 4: Bonus 实验 (Day 0.5, 可选)](#step-4-bonus-实验)
  - [技术实现细节](#技术实现细节)
  - [时间与资源估算](#时间与资源估算)
  - [风险与应对](#风险与应对)
  - [与现有工作的精确差异化](#与现有工作的精确差异化)
  - [实验后的决策路径](#实验后的决策路径)

---

## 执行摘要

- **实验目的**: 验证 MeshLex 的核心假设——mesh 的局部拓扑结构是否存在 universal vocabulary
- **预计耗时**: 2-3 天（单 GPU）
- **预计成本**: < $50（单张 A100/4090 租用）
- **关键产出**: Go/No-Go 决策，决定是否投入数月时间做完整论文
- **成功标准**: 跨类别重建 CD 与同类别 CD 比值 < 2.0×
- **硬性止损**: 比值 > 3.0× 则放弃 MeshLex 方向

---

## 核心假设与验证逻辑

### 假设

**Mesh 的局部拓扑结构存在有限且可复用的 "vocabulary"。** 具体地：

1. **低熵假设**: 三角 mesh 的局部拓扑模式（20-50 face 的 patch）数量有限，可被 ~4096 个 codebook entry 覆盖
2. **通用性假设**: 这些拓扑模式跨物体类别共享（椅子上的局部 patch 和汽车上的局部 patch 存在共性）
3. **可重建假设**: 从 codebook 中选取 patch prototype + 仿射变换 + 残差变形，可以高保真重建原始 mesh

### 先行证据

| 证据来源 | 关键发现 | 与本假设的关系 |
|----------|---------|--------------|
| PatchNets (ECCV 2020) | 仅在 Cabinet 训练的 implicit patch，跨类别 F-score 仅降 <1%（94.8→93.9），甚至能重建人体 | 直接支持通用性假设 |
| PatchComplete (NeurIPS 2022) | 学习的 multi-resolution patch priors 跨类别形状补全 CD 降低 19.3% | 支持 patch prior 的跨类别迁移性 |
| 三角 mesh 顶点 valence 统计 | 绝大多数 mesh 的 valence 集中在 5-6-7，跨类别分布高度一致 (Alliez & Desbrun, 2001) | 直接支持低熵假设 |
| VQGraph (ICLR 2024) | GNN encoder + VQ 可有效学习图的局部子结构 discrete codebook | 支持技术路线可行性 |
| MeshGPT (CVPR 2024) | Graph conv encoder 可提取 mesh face 的局部几何/拓扑特征，RVQ codebook 保留足够信息 | 支持 per-face VQ 可行，我们扩展到 per-patch |

### 验证逻辑

```
假设成立的充分条件:
  (1) Codebook utilization > 50% (低熵假设)
  AND (2) 同类别重建 CD 在合理范围 (可重建假设)
  AND (3) 跨类别/同类别 CD 比值 < 2.0× (通用性假设)

任一条件不满足 → 需要调整策略或止损
```

### 这个实验不回答的问题

- Patch 序列的**生成**质量（本实验只验证 codebook 的**表达力**，不训练生成模型）
- Boundary stitching 的质量（留给后续完整实验）
- 与 FACE/BPT 等方法的**生成**指标对比（本实验仅对比重建质量）

---

## 实验数据

### 数据来源

**ShapeNet Core v2** — 选择 5 个拓扑多样性较大的类别：

| 类别 | ShapeNet ID | 预计数量 | 选择理由 |
|------|-----------|---------|---------|
| Chair | 03001627 | 500 | 拓扑最多样（有/无扶手、各种腿部结构） |
| Table | 04379243 | 500 | 与椅子共享桌腿等局部结构，验证类内共享 |
| Airplane | 02691156 | 500 | 拓扑与家具完全不同（机翼、机身），验证跨域泛化 |
| Car | 02958343 | 500 | 曲面为主，局部拓扑模式不同于棱角分明的家具 |
| Lamp | 03636649 | 500 | 形态多样（台灯、吊灯、落地灯），测试 codebook 覆盖极端情况 |

**训练/测试划分**:
- **训练集** (codebook 学习): Chair + Table + Airplane，各 400 个 = 1200 mesh
- **同类别测试集**: Chair + Table + Airplane，各 100 个 = 300 mesh
- **跨类别测试集**: Car + Lamp，各 500 个 = 1000 mesh（从未参与训练）

### 数据预处理

1. **Watertight 化**: 使用 ManifoldPlus 或 trimesh 的 `repair` 功能确保所有 mesh 为闭合流形
2. **Decimation**: 使用 PyMeshLab 的 Quadric Edge Collapse 将每个 mesh 简化到 **800-1200 faces**
   - 目标：每个 mesh 产生 ~25-35 个 patches（按平均 35 faces/patch）
   - 这个范围与 ShapeNet 默认精度匹配，不需要过度简化
3. **归一化**: 平移到原点，缩放到 [-1, 1] 的单位立方体
4. **质量过滤**: 移除面数 < 200 或非流形的 mesh

**预期数据量**: 2500 mesh × ~30 patches/mesh ≈ **75,000 个 patches**

---

## Step 1: Patch 分割 (Day 0.5)

### 目标

将每个训练/测试 mesh 分割为大小大致均匀的 patches，每 patch 约 20-50 faces。

### 分割策略：Graph-based K-way Partitioning

不采用语义分割（如 SDF），因为我们的 patch 是**拓扑原语**而非语义部件。采用纯拓扑+几何的分割：

**算法**:
1. 构建 face adjacency graph（dual graph）：每个 face 是一个节点，共享边的 face 之间有边
2. 边权重 = 两个 face 法向量的余弦相似度（鼓励沿曲率变化大的位置切割）
3. 使用 METIS 做 k-way graph partitioning，k = ⌈N_faces / 35⌉
4. 后处理：
   - 合并过小的 partition（< 15 faces）到最大邻居
   - 二分过大的 partition（> 60 faces）

**为什么选 METIS 而非 Spectral Clustering**:
- METIS 对大图更高效（O(N) vs O(N log N)），且天然支持 balanced partition
- METIS 的 edge-cut 最小化目标天然倾向于沿几何特征线切割（法向量变化大 → 权重低 → 被切割）
- PyMETIS 提供现成的 Python binding

**为什么不用 BFS/Shell-based（Nautilus 风格）**:
- Nautilus 的 shell 是以单顶点为中心的 fan，粒度太细（~6 faces/shell）
- 我们需要 20-50 face 的粗粒度 patch，METIS 分割更灵活

### Patch 表示

每个 patch 存储以下信息：

```python
@dataclass
class MeshPatch:
    # 拓扑
    faces: np.ndarray          # (F, 3) — patch 内的 face indices (局部编号)
    vertices: np.ndarray       # (V, 3) — patch 内的顶点坐标
    boundary_vertices: list    # 边界顶点的局部 index 列表（有序，沿边界环）
    boundary_edges: list       # 边界边列表

    # 几何 (相对于 patch centroid 和 principal axes)
    centroid: np.ndarray       # (3,) — patch 质心
    principal_axes: np.ndarray # (3, 3) — PCA 主轴（用于 affine 对齐）
    scale: float               # patch 的 bounding sphere 半径

    # 归一化后的局部坐标
    local_vertices: np.ndarray # (V, 3) — 去中心化 + PCA 对齐 + 归一化后的坐标
```

**归一化流程**: 对每个 patch：
1. 减去质心 → 平移到原点
2. PCA 对齐 → 消除旋转差异（使拓扑相似的 patch 在坐标空间也相似）
3. 除以 bounding sphere 半径 → 归一化到单位球

这保证了 **拓扑相同但位置/朝向/大小不同的 patch 在归一化后坐标接近**，有利于 codebook 学习。

### 输出格式

```
data/
├── patches/
│   ├── chair_0001_patch_00.npz   # 每个 .npz 包含一个 MeshPatch
│   ├── chair_0001_patch_01.npz
│   └── ...
├── meshes/                        # 预处理后的原始 mesh
└── metadata.json                  # mesh → patch 映射关系
```

### 验收标准

- 所有 mesh 被成功分割，无空 patch
- Patch 面数分布的中位数在 30-40，95% 在 [15, 60] 范围内
- 可视化抽检 20 个 mesh：切割位置是否在几何特征线上

---

## Step 2: Patch Codebook 学习 (Day 1-1.5)

### 整体架构

```
Input Patch (faces + vertices)
        │
        ▼
  GNN Encoder (SAGEConv, 4 layers)
        │
        ▼
  Patch Embedding (128-dim)
        │
        ▼
  SimVQ Layer (K=4096 codebook)
        │
        ▼
  Quantized Embedding (128-dim)
        │
        ▼
  MLP Decoder → Reconstructed Vertices
```

### GNN Encoder 架构

参考 MeshGPT 的 graph convolution encoder（SAGEConv 效果最优，比 EdgeConv/GAT 高 7-12%），但适配到 patch 级别：

**输入**: Face adjacency graph within a patch
- 每个 face 节点的 input feature (15-dim):
  - 9 个相对顶点坐标（3 vertices × 3 coords，相对于 patch centroid）
  - 3 个 face normal 分量
  - 3 个 edge angle（face 三条边的夹角）

**架构**:
```python
class PatchEncoder(nn.Module):
    def __init__(self, in_dim=15, hidden_dim=256, out_dim=128):
        # 4 层 SAGEConv，每层后接 LayerNorm + GELU
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.conv4 = SAGEConv(hidden_dim, out_dim)
        # Global mean pooling: (F, out_dim) → (out_dim,)
        # 得到整个 patch 的单一 embedding
```

**关键设计决策**:
- 用 **面（face）为节点**而非顶点，因为 face 包含更丰富的局部拓扑信息
- 4 层 SAGEConv 的感受野约 4-hop，对 30 face 的 patch 足以覆盖全局
- 最终用 **global mean pooling** 将所有 face embedding 聚合为一个 patch embedding
- MeshGPT 用 SAGEConv 在 face-level 达到 98.49% triangle accuracy，验证了此架构选择

### SimVQ Codebook

采用 SimVQ (ICCV 2025) 策略解决 codebook collapse：

```python
class SimVQCodebook(nn.Module):
    def __init__(self, K=4096, dim=128):
        self.codebook = nn.Embedding(K, dim)        # K=4096, dim=128
        self.linear = nn.Linear(dim, dim, bias=False) # SimVQ 的关键：可学习线性变换

    def forward(self, z):
        # z: (B, dim) — encoder output
        z_proj = self.linear(z)                       # 线性重参数化
        distances = torch.cdist(z_proj, self.codebook.weight)
        indices = distances.argmin(dim=-1)            # 最近邻查找
        quantized = self.codebook(indices)
        # Straight-through estimator: 前向用 quantized，反向梯度流过 z
        return quantized + (z - z.detach()), indices
```

**为什么选 SimVQ 而非标准 VQ-VAE**:
- 标准 VQ-VAE 在 K=16384 时 utilization 仅 11.2%（VQGAN-FC 数据）
- SimVQ 通过 learnable linear layer 让整个 codebook 空间参与梯度更新，utilization 接近 100%
- 实现极简——仅多一个线性层

**Codebook 初始化**:
1. 先跑一遍所有训练 patch 的 GNN encoder forward pass（无梯度）
2. 收集 ~50,000 个 128 维 embeddings
3. 用 K-means (K=4096) 聚类，用聚类中心初始化 codebook
4. 这参考 VQGAN-LC 的策略，从 11.2% utilization 提升到 99.4%

### Decoder 架构

Decoder 将 quantized embedding 解码回 patch 内每个顶点的坐标。

**挑战**: patch 内顶点数不固定（15-40 个内部顶点）。

**方案**: Set-based decoder with masking

```python
class PatchDecoder(nn.Module):
    def __init__(self, embed_dim=128, max_vertices=50):
        # 将 codebook embedding 扩展为每个顶点的初始 feature
        self.vertex_queries = nn.Parameter(torch.randn(max_vertices, embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 3)  # 输出 xyz 坐标
        )

    def forward(self, z_quantized, n_vertices, vertex_mask):
        # z_quantized: (B, 128)
        # 用 cross-attention: vertex queries attend to codebook embedding
        # 然后 MLP 解码每个顶点坐标
        # vertex_mask 屏蔽超出实际顶点数的位置
```

**备选方案**（如果 cross-attention decoder 效果不好）:
- 简单 MLP: (128) → (256) → (256) → (max_V × 3)，mask 掉多余位置
- 参考 MeshGPT 的 1D ResNet-34 decoder（但那个是 per-face 而非 per-patch）

### 训练策略

**Loss 函数**:
```
L_total = L_recon + λ_commit * L_commit + λ_embed * L_embed

其中:
  L_recon = Chamfer Distance(decoded_vertices, GT_vertices)  # 主要重建损失
  L_commit = ||z - sg(quantized)||²                          # Commitment loss
  L_embed = ||sg(z) - quantized||²                           # Codebook embedding loss
  λ_commit = 0.25, λ_embed = 1.0 (标准 VQ-VAE 设置)
```

**优化器**: AdamW, lr=1e-4, weight_decay=1e-5, cosine annealing

**训练参数**:
- Batch size: 256 patches
- Epochs: 200（~50K patches × 200 / 256 ≈ 39K steps）
- 单 GPU (A100 80GB 或 RTX 4090 24GB)
- 预估训练时间: 4-8 小时

**训练流程**:
1. 前 20 epochs: 仅训练 encoder + decoder（无 VQ），验证重建能力
2. 20 epochs 后: 引入 SimVQ codebook，联合训练
3. 每 20 epochs: 记录 codebook utilization，如 < 30% 则调参

**超参搜索**（小规模，在 Chair 类别上）:
- Codebook size K: [1024, 2048, 4096]
- Embedding dim: [64, 128, 256]
- GNN layers: [3, 4, 5]
- 选择 codebook utilization 最高且 CD 最低的组合

---

## Step 3: 评估与决策 (Day 0.5)

### Metric 1: Codebook Utilization

**定义**: 4096 个 code 中在测试集上被至少使用一次的比例。

**计算**: 对所有测试集 patches，统计每个 code 被选为最近邻的次数。

**参考标准**:

| Utilization | 等级 | 含义 |
|-------------|------|------|
| ≥ 80% | 优秀 | SimVQ 级别，codebook 高效利用 |
| 50%-80% | 可接受 | 核心假设成立但有改进空间 |
| 30%-50% | 勉强 | 部分 collapse，需要增大 K 或调整 encoder |
| < 30% | 严重 collapse | Encoder/SimVQ 实现有问题，或拓扑多样性超出预期 |

**额外统计**:
- 每个 code 的使用频率直方图（是否为长尾分布？）
- Top-20 最常用 code 覆盖了多少比例的 patches

### Metric 2: 同类别重建 Chamfer Distance

**定义**: 用 codebook patch 重建同类别测试 mesh 的 Chamfer Distance。

**计算流程**:
1. 对测试 mesh 做 patch 分割（与训练集相同策略）
2. 每个 patch → encoder → SimVQ → decoder → 重建 patch 顶点
3. 将所有重建 patch 的顶点拼回完整 mesh（暂不做 boundary stitching，允许接缝间隙）
4. 在 GT mesh 和重建 mesh 表面各均匀采样 10K 点
5. 计算 L2 Chamfer Distance (× 10³ 报告)

**参考基线**: MeshGPT 的 encoder-decoder 在 ShapeNet Chair 上的 face-level 重建 CD 约为 0.05-0.10（×10³）。由于我们是 patch-level 重建（更粗粒度），CD 会更高。**预期合理范围: 0.1-0.5 (×10³)**。

### Metric 3: 跨类别重建 Chamfer Distance

**定义**: 在 Chair+Table+Airplane 上训练的 codebook，用来重建 Car+Lamp 的 CD。

**这是整个实验最关键的指标**——直接回答 "universal vocabulary" 假设是否成立。

**计算**: 与 Metric 2 相同流程，但测试 mesh 来自 Car 和 Lamp 类别。

**参考**: PatchNets (ECCV 2020) 在 implicit patch 上的跨类别 F-score 仅降 < 1%（94.8 → 93.9）。但 PatchNets 用的是连续 implicit representation（SDF），我们用的是离散 codebook + 显式 mesh，信息瓶颈更大，预计降幅会更大。

### Metric 4: Codebook 可视化

**定性评估**，不设数值标准，但对论文非常重要：

1. **t-SNE / UMAP 可视化**: 将 4096 个 codebook embedding 降维到 2D，着色按使用频率或拓扑特征（面数/boundary 顶点数/Euler characteristic）
2. **Top-20 Prototype 渲染**: 用 Open3D 或 PyVista 渲染使用频率最高的 20 个 patch prototype 的 3D 形状
3. **预期可以看到的模式**:
   - 平面 patch 簇（规则三角网格，低曲率区域）
   - 棱线 patch 簇（两个平面相交处）
   - 弯曲面 patch 簇（高斯曲率非零）
   - 尖端/角点 patch 簇（多棱线汇聚）
4. **跨类别 patch 分配**: 可视化 Car/Lamp 上每个 patch 被映射到哪个 code，与 Chair 上的分配对比

### Go/No-Go 决策矩阵

| 跨类别 CD / 同类别 CD | Codebook Util. | 决策 | 下一步 |
|----------------------|----------------|------|--------|
| < 1.2× | > 50% | **强 Go** | 直接进入完整 MeshLex 实验设计 |
| 1.2× - 2.0× | > 50% | **弱 Go** | 调整 story 为 "transferable vocabulary"，继续 |
| < 2.0× | 30%-50% | **条件 Go** | 增大 K 或改用 RQ-VAE 再试一轮 |
| 2.0× - 3.0× | 任意 | **暂缓** | 分析失败原因，考虑仅做类别内 codebook |
| > 3.0× | 任意 | **No-Go** | 核心假设不成立，止损转方向 |

**决策后的 story 调整**:
- **强 Go** → "Mesh 具有 universal topology vocabulary"（最强 claim）
- **弱 Go** → "Mesh 的 topology patches 具有跨类别迁移性"（弱化但仍有价值）
- **条件 Go** → 需要更大 codebook 或 hierarchical codebook，增加实验复杂度

---

## Step 4: Bonus 实验 (Day 0.5, 可选)

仅在 Step 3 结果为 Go 时执行。

### Bonus 1: Mesh Patch Arithmetic

类似 word2vec 的 king - man + woman = queen，验证 codebook embedding 空间是否具有语义结构：

1. 选取一个"平面 patch"的 embedding **e_flat**
2. 选取一个"弯曲 patch"的 embedding **e_curved**
3. 选取一个"平面 + 锐利边缘 patch"的 embedding **e_edge**
4. 计算 **e_edge - e_flat + e_curved**，找最近邻 code
5. 预期结果：得到 "弯曲 + 锐利边缘 patch"

如果 work，这是 "vocabulary structure" 假设的强定性验证，也是论文中极具 visual impact 的结果。

### Bonus 2: Codebook Size Ablation

快速测试不同 K 值的 trade-off：

| K | 预期 Utilization | 预期 CD | 含义 |
|---|-----------------|---------|------|
| 512 | 高 | 较高 | codebook 太小，表达力不足 |
| 1024 | 高 | 中 | 可能是最优平衡点 |
| 2048 | 中-高 | 较低 | 较好的精度-效率权衡 |
| 4096 | 中 | 最低 | 默认设置 |

每个 K 训练 50 epochs（~2 小时），快速扫描最优 K。

### Bonus 3: Affine-only vs Affine+Residual 重建对比

测量两种重建模式的 CD 差异：
- **(a) Affine-only**: codebook patch prototype + 仿射变换（12 参数），不做残差变形
- **(b) Affine + Residual (16-dim)**: codebook + affine + 16 维 residual latent
- **(c) Affine + Residual (32-dim)**: codebook + affine + 32 维 residual latent

如果 (a) 已经很好，说明 codebook 本身表达力强，residual 的要求更低。如果 (a) 和 (b) 差异巨大，需要增大 residual 维度或改用 RQ-VAE。

---

## 技术实现细节

### 依赖库

```
# 核心
torch >= 2.0
torch-geometric >= 2.4      # SAGEConv, graph utils
trimesh >= 4.0               # mesh I/O, processing
numpy, scipy

# Mesh 处理
pymeshlab                    # decimation, watertight repair
pymetis                      # METIS graph partitioning

# 可视化
open3d 或 pyvista            # 3D patch 渲染
matplotlib                   # 2D plots (t-SNE, histograms)
umap-learn                   # UMAP 降维

# 工具
scikit-learn                 # K-means 初始化
tqdm                         # 进度条
```

### 关键实现注意事项

**1. Face Adjacency Graph 构建**

```python
import trimesh

mesh = trimesh.load('model.obj')
# trimesh 自带 face adjacency
face_adj = mesh.face_adjacency  # (E, 2) — 共享边的 face pair
# 边权重 = face normal 相似度
normals = mesh.face_normals
w = np.abs(np.sum(normals[face_adj[:, 0]] * normals[face_adj[:, 1]], axis=1))
# w 高 → 法向量相似 → 同一平面 → 不想切
# w 低 → 法向量差异大 → 棱线 → 想切
```

**2. METIS 分割**

```python
import pymetis

# 构建 adjacency list
adj_list = [[] for _ in range(n_faces)]
for i, (f1, f2) in enumerate(face_adj):
    adj_list[f1].append(f2)
    adj_list[f2].append(f1)

k = max(2, round(n_faces / 35))
_, partition = pymetis.part_graph(k, adjacency=adj_list)
# partition[i] = patch_id for face i
```

**3. Patch 坐标归一化**

```python
def normalize_patch(vertices):
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    # PCA 对齐
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    aligned = centered @ Vt.T
    # 归一化到单位球
    scale = np.max(np.linalg.norm(aligned, axis=1))
    normalized = aligned / (scale + 1e-8)
    return normalized, centroid, Vt, scale
```

**4. SimVQ 的 Straight-Through Estimator**

```python
def forward(self, z):
    z_proj = self.linear(z)
    distances = torch.cdist(z_proj.unsqueeze(0),
                           self.codebook.weight.unsqueeze(0)).squeeze(0)
    indices = distances.argmin(dim=-1)
    quantized = self.codebook(indices)
    # Straight-through: gradient flows through z, not through argmin
    quantized_st = z + (quantized - z).detach()
    return quantized_st, indices
```

---

## 时间与资源估算

### 时间线

| 阶段 | 耗时 | 产出 |
|------|------|------|
| 数据下载 + 预处理 | 2-3h | 2500 个标准化 mesh |
| Patch 分割 + 质检 | 3-4h | ~75K patches + 可视化 |
| Codebook 训练（主实验） | 4-8h | 训练好的 codebook 模型 |
| 超参搜索（可选） | 4-6h | 最优 K/dim 组合 |
| 评估 + 可视化 | 2-3h | 4 个 metrics + plots |
| Bonus 实验（可选） | 3-4h | Arithmetic + ablation 结果 |
| **总计** | **~2-3 天** | **Go/No-Go 决策** |

### 硬件需求

- **最低**: 单张 RTX 4090 (24GB) — GNN + SimVQ 模型很小（< 50M 参数），VRAM 不是瓶颈
- **推荐**: 单张 A100 (80GB) — 训练更快，batch size 可以更大
- **存储**: ~5GB（ShapeNet 子集 + patches + checkpoints）

### 成本估算

| 方案 | GPU | 耗时 | 成本 |
|------|-----|------|------|
| RunPod A100 80GB | $1.64/hr | 24h | ~$40 |
| RunPod RTX 4090 | $0.44/hr | 36h | ~$16 |
| 本地 4090 | 自有 | 36h | 电费 |

---

## 风险与应对

### 风险 1: Patch 大小不均匀导致 codebook 学习困难

**原因**: METIS 分割可能产生面数差异大的 patches（如 15 vs 55）
**影响**: Set-based decoder 需要处理 variable size，增加实现复杂度
**应对**:
- 严格的后处理约束（合并 < 15，二分 > 60）
- Decoder 使用 masking 处理 variable size
- 备选: 将所有 patch remesh 到固定面数（如 32 faces），牺牲少量精度换实现简洁

### 风险 2: Codebook Collapse

**原因**: 标准 VQ 在 K 较大时极易 collapse
**影响**: 大量 code 不被使用，有效 codebook 远小于 4096
**应对**:
- SimVQ 是第一道防线（linear reparameterization）
- K-means 初始化是第二道防线
- 如果仍然 collapse: 尝试 EMA 更新 + codebook reset（每 50 epochs 重置使用率 < 1% 的 code）

### 风险 3: 拓扑模式比预期更多

**原因**: 某些物体（镂空结构、有机形态）的局部拓扑可能高度独特
**影响**: 4096 个 prototype 不够，quantization error 过高
**应对**:
- 统计 quantization error 的 per-patch 分布，如 > 5% 的 patch error > 3× 均值 → 增大 K
- 尝试 Hierarchical codebook: coarse code (K=512) + fine residual code (K=512)，总表达力 512² = 262K
- 尝试 RQ-VAE: 2 层 residual quantization，每层 K=512

### 风险 4: PCA 归一化不足以消除 patch 变形差异

**原因**: 两个拓扑相同但几何变形差异大的 patch，PCA 对齐后可能仍然很不同
**影响**: 相同拓扑被映射到不同 code，浪费 codebook 容量
**应对**:
- 增大 residual deformation 的维度（16 → 32 → 64）
- 在 encoder 输入中加入拓扑特征（valence 分布、Euler 特征），引导 encoder 更关注拓扑而非几何

### 风险 5: ShapeNet 数据质量问题

**原因**: ShapeNet 的部分 mesh 质量较低（non-manifold、self-intersections）
**影响**: Patch 分割失败或质量差
**应对**:
- 预处理阶段强制 watertight + manifold check
- 移除无法修复的 mesh（预计 5-10%）
- 如果过滤后数据量不足，补充 Objaverse 子集

---

## 与现有工作的精确差异化

| 维度 | MeshGPT | FACE | Nautilus | FreeMesh | MeshMosaic | **MeshLex (Ours)** |
|------|---------|------|---------|----------|------------|-------------------|
| Token 粒度 | per-face (RVQ) | per-face (MLP) | per-face (shell) | per-coordinate (BPE merge) | per-face within large patch | **per-topology-patch (codebook)** |
| 有无 Codebook | 有 (face-level, K=16384) | 无 (端到端 MLP) | 无 (直接坐标量化) | 有 (coordinate-level BPE) | 无 | **有 (patch-level, K=4096)** |
| 压缩比 (4K face) | 0.67 | 0.11 | 0.275 | ~0.3 | N/A (逐 face) | **~0.03** (~130 tokens) |
| 生成单元 | 单个 face | 单个 face | 单个 face (shell 内) | merged coordinates | 单个 face | **整个 patch (30 faces)** |
| 核心创新 | face embedding | one-face-one-token | locality-aware shell | BPE on coordinates | divide-and-conquer | **topology vocabulary** |

**关键差异化论点**:
1. MeshMosaic 的 patch 内部仍然逐 face 生成（仍是序列化范式），MeshLex 从 codebook 选取（跳出序列化）
2. FreeMesh 的 BPE 在坐标值层面合并（数字频率），MeshLex 在拓扑结构层面学习（几何-拓扑模式）
3. FACE 的极限是 1 face = 1 token，无法更细粒度；MeshLex 反方向走——30 faces = 1 token

---

## 实验后的决策路径

### 如果 Strong Go (跨类别 CD ratio < 1.2×)

```
验证实验成功
    │
    ▼
编写完整 MeshLex 实验设计 (07_experiment_design_meshlex.md)
    │
    ├── Component 1: Patch Codebook (本实验已验证)
    ├── Component 2: Patch Sequence Generation (AR Transformer on patch tokens)
    ├── Component 3: Boundary Stitching (constrained decoding + vertex welding)
    ├── Component 4: Multi-modal Conditioning (image/text → patch sequence)
    │
    ▼
Objaverse-scale 训练 (200K mesh)
    │
    ▼
论文撰写，目标 CCF-A
```

### 如果 Weak Go (CD ratio 1.2×-2.0×)

```
调整 story: "transferable but not universal"
    │
    ▼
尝试增大训练类别数 (10 → 20 categories)
    │
    ├── 如果 ratio 降到 < 1.2× → 回到 Strong Go 路径
    └── 如果 ratio 仍 > 1.5× → 考虑 category-adaptive codebook
```

### 如果 No-Go (CD ratio > 3.0×)

```
分析失败原因
    │
    ├── 如果是 codebook size 不够 → 尝试 Hierarchical VQ (K=512², 总 262K)
    ├── 如果是 patch 分割质量差 → 改用其他分割方法 (SDF, spectral)
    ├── 如果是拓扑多样性根本太高 → 放弃 MeshLex
    │
    ▼ (如果放弃 MeshLex)
回到 MeshFoundation v2 方向
    │
    ├── 用 FACE tokenizer 替代 BPT (更高压缩比)
    ├── 保留多模态条件 + Scaling Law 贡献
    └── 重新差异化 vs FACE
```

---

*本文档基于 05_cc_pplx_debate.md 中 CC 与 Perplexity 的共识编写。实验设计参考了 MeshGPT (CVPR 2024), SimVQ (ICCV 2025), PatchNets (ECCV 2020), VQGraph (ICLR 2024) 的技术细节。*
