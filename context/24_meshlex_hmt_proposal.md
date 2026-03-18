<!-- markdownlint-disable -->
# MeshLex-HMT: Hierarchical Mesh Tokenization with Structure-Geometry Decoupling

> **Date**: 2026-03-17
> **Status**: Proposal — 经 2 轮 NeurIPS 级别审稿迭代，评分 5/10 → 7/10 (Almost Ready)
> **前置文档**: context/23_gap_analysis_graph_tokenization.md (Gap Analysis), context/AUTO_REVIEW.md (审稿记录)
> **基于**: MeshLex v1 (4/4 STRONG GO, CD ratio 1.019×, util 95.3%) + Graph Tokenization (Guo et al., 2026)

---

## 1. 动机与问题

MeshLex v1 验证了一个核心假设：**mesh 的局部面拓扑结构存在 universal vocabulary**，4000-face mesh 可用 ~130 patch tokens 表示（~277× 压缩 vs MeshGPT），跨 1156 类别泛化损失仅 1.9%。

但 v1 存在一个根本局限：**分区（partitioning）和编码（encoding）是割裂的两步**。

```
Step 1: Mesh → Patches    via METIS     [固定，不可微，启发式]
Step 2: Patches → Tokens   via GCN+SimVQ  [可学习]
```

METIS 是一个通用图分区算法，对 mesh 语义毫无感知。如果 METIS 切割不当（跨语义边界切割、patch 大小不均），后续的 VQ 编码无论多好都无法弥补。

**核心问题**: 能否用数据驱动的方式学习 mesh 应该怎么切？

---

## 2. 关键洞察：Structure-Geometry Decoupling

Graph Tokenization (Guo et al., 2026, arXiv:2603.11099) 证明了 BPE 可以在图上发现 recurring substructure patterns，在 14 个图基准上达到 SOTA。但它有一个致命局限：**仅支持离散标签图**，而 mesh 面携带连续特征（法向量、面积、曲率）。

我们的解决方案：**将 mesh tokenization 解耦为两个独立通道**。

| 通道 | 输入 | 方法 | 输出 |
|------|------|------|------|
| **结构通道** (Structural) | 面连接模式 (离散化后的法向/面积/二面角) | Graph BPE | struct-token IDs |
| **几何通道** (Geometry) | 精确顶点坐标 (连续) | GCN + SimVQ | geo-token IDs |

离散化**仅影响结构通道**（用于 BPE 的 pattern matching），几何通道保持完全连续。这从根本上绕过了 Graph Tokenization 的离散标签限制。

> **术语说明**: "Structure" 指面邻接对偶图上的连接模式 (connectivity patterns)，由离散化的局部微分几何特征标注。这不是拓扑不变量（亏格、边界环等），而是 **recurring connectivity-geometry motifs** — 哪些面以什么样的几何关系反复相连。

---

## 3. 方法

### 3.1 Graph BPE 算法（核心贡献）

标准 BPE 作用于线性序列；我们将其扩展到图结构。

**输入**: 训练集中所有 mesh 的 labeled dual graphs $\{G_i = (V_i, E_i, l_V, l_E)\}$，其中 $l_V$: 面 → 离散标签，$l_E$: 共边 → 离散标签（二面角 bin）。

**定义**:

- **Bigram**: 对偶图中相邻节点对 $(u, v)$，表示为 triple $(l_V(u),\; l_E(u,v),\; l_V(v))$
- **Merge**: 将匹配某 bigram pattern 的 $(u, v)$ 合并为单节点 $w$：
  - $w$ 继承 $u, v$ 的所有外部边（不含 $u$-$v$ 边）
  - $w$ 获得新标签 $l_V(w) = \text{merged}\_{l_V(u)}\_{l_E}\_{l_V(v)}$
  - 若合并产生多重边，保留所有边（multigraph）
- **频率统计**: 遍历所有训练图，统计每种 bigram pattern 的全局出现次数

**算法**:

```
1. 初始词汇表 V_struct = 所有离散面标签 (base alphabet)
2. 统计所有训练图中的 bigram 频率
3. 选择最高频 bigram (l_a, l_e, l_b)
4. 在所有训练图中执行 merge:
   - 贪心策略: 按节点 ID 排序，先匹配先合并，已合并节点不再参与本轮
5. 将新符号加入 V_struct
6. 重复 2-5 直到 |V_struct| 达到目标大小 (e.g., 2000)
```

**已知 complications**:
- 图中 merge 顺序不唯一（vs 线性 BPE 的左→右确定性）→ 贪心 ID 排序提供确定性
- 合并可能产生 multigraph → 后续统计需考虑多重边
- 不同 mesh 中结构相同但 ID 不同的子图 → 通过 label-based pattern matching（非 subgraph isomorphism）规避，复杂度 $O(|E|)$

### 3.2 面特征离散化

每个面 $f$ 携带连续特征 $(\mathbf{n}_f, a_f)$（法向量、面积），每条共边携带二面角 $\theta_e$。

| 特征 | 离散化方式 | 候选粒度 (ablation) |
|------|-----------|-------------------|
| 法向量 $\mathbf{n}_f$ | 球面均匀分桶 (icosphere) | {32, 64, 128} bins |
| 面积 $a_f$ | 对数分桶 | {4, 8, 16} bins |
| 二面角 $\theta_e$ | 均匀角度分桶 (0°-180°) | {8, 16, 32} bins |

组合 alphabet size: $32 \times 4 \times 8 = 1{,}024$ (coarse) 到 $128 \times 16 \times 32 = 65{,}536$ (fine)。

**信息损失量化**: 计算离散化互信息 $\text{MI}(\text{discrete\_label}, \text{continuous\_features})$，确保离散化保留了足够的几何区分度。

### 3.3 完整 Pipeline

```
=== 离线阶段 (一次性) ===
1. 面特征离散化 → labeled dual graph
2. Graph BPE → structural vocabulary V_struct (size ~2000)
3. 每个 BPE token = 一组面 = "struct-patch"
4. 统计 patch 大小分布 (mean, std, min, max, percentiles)

=== 训练阶段 ===
5. 对每个 mesh:
   a) BPE encode → struct-patch 划分 + struct-token IDs
   b) 对每个 struct-patch: GCN encode patch geometry → z_geo ∈ R^d
   c) z_geo → SimVQ → geo-token IDs

6. Dual-Token Representation:
   - Struct-token: 面连接模式 (connectivity-geometry motifs)
   - Geo-token: 精确几何 (局部坐标系下的顶点坐标)

7. Fusion → h_patch (见 3.4 节)
8. Sequence model → contextualized patch embeddings (见 3.5 节)
9. Decoder: cross-attention → reconstruct vertices

=== 生成阶段 ===
10. Dual-token 序列 → AR 模型 (GPT-2 scale, ~50M params) → NTP
11. 采样 → decode → 生成新 mesh
```

### 3.4 Fusion 策略（ablation）

| 策略 | 公式 | 参数量 |
|------|------|--------|
| Additive | $h = e_s(s) + e_g(g)$ | 0 |
| Concatenation | $h = W_p[e_s(s); e_g(g)]$ | $2d \times d$ |
| Gated | $\gamma = \sigma(W_\gamma[e_s; e_g])$, $h = \gamma \odot e_s + (1-\gamma) \odot e_g$ | $2d \times d$ |
| Cross-attention | $h = \text{CrossAttn}(Q{=}e_s, K{=}e_g, V{=}e_g)$ | $3d^2$ |

### 3.5 Sequence Model（ablation）

| 选项 | 描述 |
|------|------|
| Transformer | Self-attention across patches（捕捉全局 patch 关系） |
| GCN | 在 patch-level coarsened graph 上做 message passing（MeshLex v1 baseline） |

---

## 4. 实验设计

### 4.1 五阶段实验

| 阶段 | 目标 | 内容 | 计算量 |
|------|------|------|--------|
| **Phase 0** | BPE 可行性 | 离散化 MI、BPE 词汇分布、patch 大小统计、法向方差分析 | ~2h CPU |
| **Phase 1** | 分区对比 | BPE vs METIS vs spectral clustering vs MinCutPool | ~8h |
| **Phase 2** | Dual-token 训练 | Fusion ablation (4-way) + Decoupling ablation (3-way) | ~20h GPU |
| **Phase 3** | 完整对比 | MeshLex-HMT vs MeshLex v1 vs MeshLLM decomposition | ~30h GPU |
| **Phase 4** | 生成 | AR model on dual-tokens → FID/COV/MMD vs PolyGen/MeshGPT/FACE | ~40h GPU |
| **Buffer** | 重跑/补充 | 失败实验重跑 + 额外 ablation | ~20h |
| **Total** | | | **~120h GPU** |

### 4.2 评估指标

**重建** (Phase 1-3):
- Chamfer Distance (CD)
- Normal Consistency (NC)
- F-Score @ {0.01, 0.02, 0.05}
- 非流形边比例、自交面比例

**生成** (Phase 4):
- FID, COV, MMD on ShapeNet Chair/Table
- 生成基线: PolyGen, MeshGPT, FACE（使用已发表数值；若指标协议不完全一致则注明差异）

**分析** (Phase 0-1):
- 离散化 MI
- BPE patch 大小分布
- Token 内法向方差（对比 METIS patch 中位数 $\tau_n$）
- PartNet semantic purity（若有标注）

### 4.3 Ablation 矩阵

**Primary Ablations (Main Paper)**:
| 实验 | 变量 | 固定条件 | Runs |
|------|------|---------|------|
| Decoupling | Struct-only / Geo-only / Struct+Geo | Best fusion, BPE partition | 3 |
| Fusion | add / concat / gated / cross-attn | Struct+Geo, BPE partition | 4 |
| Partition | METIS / spectral / MinCutPool / BPE | 独立对比表，不与其他因子交叉 | 4 |

**Secondary Ablations (Appendix)**:
| 实验 | 变量 | Runs |
|------|------|------|
| Discretization | coarse / medium / fine | 3 |
| Sequence model | Transformer / GCN | 2 |

**Total**: 11 primary + 5 secondary + 1 generation = **17 runs**

### 4.4 Go/No-Go 假设

| 假设 | 验证方法 | Go 标准 | No-Go 标准 |
|------|---------|---------|-----------|
| **H1a** (几何一致性) | Token 内法向方差 < $\tau_n$ 的 token 比例 | ≥60% | <30% |
| **H1b** (语义一致性) | PartNet semantic purity | ≥50% | <25% |
| **H1 综合** | H1a 为必要条件; H1b 为加分项 | H1a Go → 继续 | H1a No-Go → 停止 |
| **H2** (分区质量) | BPE CD vs best baseline (METIS/spectral/MinCutPool) | ≤ best × 1.2 | > best × 2 |
| **H3** (双通道增益) | Wilcoxon signed-rank test, Struct+Geo vs Geo-only | p < 0.05 且 median CD ↓ ≥ 3% | p > 0.1 或无改善 |
| **H4** (融合增益) | Best fusion vs additive | ≥1 种 fusion > add, p < 0.05 | 所有 fusion 无显著差异 |
| **H5** (离散化可控) | 3-level MI ablation | 存在 sweet spot, MI > 0.5 | 所有 MI < 0.3 |

---

## 5. 论文定位

### 5.1 贡献列表

1. **首次在 3D mesh 对偶图上应用 Graph BPE tokenization**，发现 mesh 面连接模式存在 NLP-like 的频率分布规律，并量化离散化信息损失
2. **Structure-Geometry decoupled tokenization**: 结构通道 BPE + 几何通道 VQ 的双通道表示，含 4 种融合策略对比
3. **Data-driven vs heuristic partition**: BPE patches 与 METIS / spectral / learned pooling 多基线对比
4. **Dual-token mesh generation**: 基于 struct+geo dual tokens 的 AR 生成模型，报告标准生成指标
5. **Comprehensive evaluation**: CD + NC + F-Score + 流形质量 + 语义纯度的多维评估体系

### 5.2 双轨论文策略

| 路径 | 条件 | 定位 | 标题侧重 |
|------|------|------|---------|
| **主路径** | Phase 4 生成质量可接受 | Mesh Generation | "... for Compositional Mesh Generation" |
| **备用路径** | Phase 4 生成质量不足 | Mesh Representation | "... a Study on Data-Driven Mesh Tokenization" |

- Introduction 以 "mesh tokenization 的根本问题" 开篇（不依赖生成）
- 生成作为 natural downstream application 引入
- 两条路径均可用同一 introduction 框架

### 5.3 竞品风险

| 潜在竞品方向 | 风险 | 理由 |
|-------------|------|------|
| Graph BPE → Mesh | 低 | Graph Tokenization 2026.03 刚发表，连续特征是公认障碍 |
| MeshLLM → BPE 分区 | 低 | MeshLLM 用 KNN，路线完全不同 |
| FACE/BPT → patch-level | 中 | 基础是坐标序列化，非图结构 |
| 分子 BPE → 3D | 低 | 分子图与 mesh 对偶图结构差异大 |

**缓解**: 加速执行，Phase 0 为零成本验证，快速获取 Go/No-Go 信号。

---

## 6. 与 MeshLex v1 的关系

MeshLex v1 Section 5.3 列出的 5 个 limitations：

| Limitation | MeshLex-HMT 的回应 |
|-----------|-------------------|
| 1. Reconstruction only | Phase 4 新增生成 pipeline |
| 2. Boundary stitching | 未直接解决（orthogonal 问题） |
| **3. Fixed patch size** | **BPE 自动发现自适应粒度** — 平面区域大 patch、曲面细节小 patch |
| 4. Codebook size | 双 codebook (struct + geo) 提供更大表达空间 |
| 5. Single decoder | 未直接解决（保留 cross-attention decoder） |

---

## 7. 支撑文献

| 论文 | 关键贡献 | 与本工作的关系 |
|------|---------|--------------|
| Graph Tokenization (Guo et al., 2026) arXiv:2603.11099 | BPE on labeled graphs → 14 benchmarks SOTA | 核心灵感来源；我们扩展到连续特征 mesh |
| CamS (Jiang et al., 2026) arXiv:2601.02530 | Multi-scale BPE motifs on molecular graphs | Multi-scale 思路参考 |
| RGVQ (Zhai et al., 2025) arXiv:2508.06588 | Graph-specific VQ collapse fix | SimVQ 与 RGVQ 的互补可能性 |
| HQA-GAE (Zeng et al., 2025) arXiv:2504.12715 | Hierarchical two-layer codebook | 层次化 codebook 思路 |
| MeshLLM (Fang et al., 2025) arXiv:2508.01242 | Primitive-Mesh decomposition | 分区基线对比 |
| DiffPool (Ying et al., 2018) | Learned graph pooling | MinCutPool 基线来源 |

---

## 8. 审稿迭代记录

本提案经历 2 轮 NeurIPS/ICML 级别模拟审稿（详见 context/AUTO_REVIEW.md）：

### Round 1: 5/10 → No

**主要问题** (10 weaknesses):
1. **(Critical)** 无生成 pipeline — 标题含 "generation" 但只做 reconstruction
2. **(High)** 离散化信息损失未量化
3. **(High)** "Topology" 命名错误 — 实为离散化几何特征，非拓扑不变量
4. **(Medium-High)** METIS 是弱基线
5. **(Medium-High)** 只有 CD 指标
6. **(Medium)** Fusion 机制未 ablate
7. **(Medium)** BPE patch 大小未分析
8. **(Low-Medium)** Go/No-Go 标准任意
9. **(Low-Medium)** 缺 MeshLLM 对比
10. **(Low)** 竞品风险乐观

**修复**: 全部 10 项已修复 — 新增 Phase 4 生成、修正命名、增强基线/指标/融合、操作化 Go/No-Go。

### Round 2: 7/10 → Almost

**残余问题** (5 remaining weaknesses):
1. **(Medium-High)** 缺已发表生成基线数值 → 已添加 PolyGen/MeshGPT/FACE
2. **(Medium)** 缺 fallback 论文结构 → 已添加双轨策略
3. **(Medium)** Graph BPE 算法未精确规范 → 已添加完整算法段落
4. **(Medium)** Ablation 矩阵过大 → 已区分 primary (11 runs) vs secondary (5 runs)
5. **(Low-Medium)** H1 析取条件过松 → 已拆分为 H1a (必要) + H1b (加分)

**评审结论**: "Fix RW3 (algorithm spec), RW1 (generation baselines), RW4 (ablation priority), and this is ready to execute."

---

## 9. 下一步行动

### Phase 0: BPE 可行性验证 (~2h CPU)

1. 从 Objaverse 预处理数据提取面邻接对偶图
2. 测试 3 种离散化粒度 (coarse / medium / fine)
3. 计算离散化互信息 MI
4. 实现 Graph BPE (Section 3.1 算法)
5. 统计 BPE 词汇表分布 + patch 大小分布
6. 可视化 top-K tokens 对应的 mesh 子结构
7. 操作性验证: token 内法向方差 vs METIS patch 中位数
8. **产出**: 可行性报告 + 可视化 + MI 表 → H1a/H5 Go/No-Go 决策

### 后续阶段

- **Phase 1** (8h): 多基线分区对比 → H2 Go/No-Go
- **Phase 2-3** (50h GPU): Dual-token 训练 + ablation → H3/H4 Go/No-Go
- **Phase 4** (40h GPU): AR 生成 → 论文路径决策 (主路径 vs 备用)
- **Buffer** (20h): 重跑 + 补充

**推进顺序**: Phase 0 → 1 → 2 → 3 → 4，每阶段末尾 Go/No-Go 决策。
