# Gap Analysis: Graph Tokenization × MeshLex

> **Date**: 2026-03-17
> **Scope**: 评估将 Graph BPE Tokenization 引入 MeshLex 的可行性，识别研究空白，提炼具体 idea
> **核心文献**: 6 篇高相关论文 + MeshLex v1 验证报告

---

## 1. 知识基础 — 核心文献摘要

### 1.1 Graph Tokenization (Guo et al., 2026) — arXiv:2603.11099

**核心机制**: Labeled Graph → Frequency-Guided Euler Circuit (可逆序列化) → BPE (学习子结构词汇) → Token Sequence → BERT/GTE

**关键设计决策**:
- 统计 edge pattern `(l_u, l_e, l_v)` 全局频率 `F(p)`，引导 Euler 电路遍历顺序
- BPE 迭代合并最高频相邻符号对 → 发现层次化子结构词汇
- **可逆性** + **确定性**：序列可完美还原原图
- BPE 压缩率 ~10×，同时提升下游性能

**结果**: 14 个图基准 SOTA，标准 BERT 超越 GNN 和 Graph Transformer

**致命局限**: **仅支持离散标签图**。论文 Appendix A.1 明确承认连续特征是 key limitation，建议用 VQ 桥接。

### 1.2 CamS — Multi-scale Graph AR Modeling (Jiang et al., 2026) — arXiv:2601.02530

**核心机制**: BPE-style Motif Mining on molecular graphs → Multi-scale serialization (fine→coarse) → LLaMA NTP

**关键创新**:
- **Motif Scale `s`**: 通过控制 BPE merge 次数实现粒度可调
- **Cross-Scale Concatenation**: fine scale → coarse scale 拼接，让 coarse 以 fine 为上下文
- **Single-Atom Vocabulary Closure (SAVC)**: 保证每个原子都有覆盖

**启示**: Multi-scale 思想对 MeshLex 有直接参考价值

### 1.3 RGVQ (Zhai et al., 2025) — arXiv:2508.06588

**核心发现**: 图数据的 VQ codebook collapse 比图像/语言严重得多

- **原因**: GNN message-passing 使相邻节点 embedding 高度相似 (feature redundancy) + 密集连接图有重叠计算树 (structural redundancy)
- **解决方案**:
  1. Gumbel-Softmax 替代 hard argmax → 所有 codeword 都能接收梯度
  2. Structure-aware contrastive regularization → 相似节点共享分配，不相似节点分离
- **结果**: Vanilla Graph VQ perplexity 低至 1.00 (512 码本只用 1 个)；RGVQ 提升至 211-446

**与 MeshLex 的关联**: 我们的 SimVQ 已解决 collapse (util 95%+)，但 RGVQ 的 graph-specific contrastive loss 可能进一步提升 codebook 质量。

### 1.4 HQA-GAE (Zeng et al., 2025) — arXiv:2504.12715

**核心创新**: Hierarchical two-layer codebook + Annealing-based encoding

- **Layer 1**: 标准 VQ codebook (size M)
- **Layer 2**: 对 Layer 1 codes 做 k-means 聚类 (size C < M)，强制相似 codes 靠近
- **Annealing**: 温度 T 指数衰减，早期均匀探索、后期聚焦最优 codes
- **结果**: 7-8 个图基准上 link prediction + node classification SOTA

**与 MeshLex 的关联**: 层次化 codebook 可以天然对应 mesh 的多尺度结构 (small patch → large region)。

### 1.5 MeshLLM (Fang et al., 2025) — arXiv:2508.01242

**核心创新**: Primitive-Mesh decomposition 将 mesh 切分为有意义的子单元

- **KNN-based**: FPS 选种子点 → KNN 聚类周围面 → 产生局部子网格 (2-10 个 cluster)
- **Semantic-based**: 3DSAMPart 语义分割 → 语义级别子网格
- **序列化**: 继承 LLaMA-Mesh 的 text serialization (坐标量化为 0-64 整数)
- **关键训练任务**: Vertex-Face Prediction — 从顶点预测面连接关系

**与 MeshLex 的差异**: MeshLLM 的 Primitive-Mesh 是几何近邻 (KNN) 或语义驱动，不是拓扑驱动；且不学习离散 codebook。

### 1.6 Graph Pooling 领域 (DiffPool, MinCutPool, MaxCutPool 等)

**核心范式**: GNN 学习 soft assignment matrix S → 将 N 节点 coarsen 为 K 超节点

- **DiffPool** (Ying et al., 2018): S = softmax(GNN(X, A))，O(N²K) 复杂度
- **MinCutPool** (Bianchi et al., 2020): 最小化 spectral cut loss
- **MaxCutPool** (Abate et al., 2024): 基于 max-cut 的特征感知 pooling
- **LGRPool** (2025): 局部-全局正则化，缓解 DiffPool 复杂度

**与 MeshLex 的关联**: Graph pooling = learned graph coarsening，可以替代 METIS 作为 end-to-end 的 mesh 分区方案。但 O(N²) 复杂度对 ~1000 面的 mesh 可能可接受。

---

## 2. Gap Analysis — 5 个研究空白

### Gap 1: Graph BPE Tokenization 从未应用于 3D Mesh 对偶图

| 维度 | 现状 |
|------|------|
| Graph Tokenization | 仅在分子图 / 社交网络 / 学术网络上验证 (离散标签) |
| CamS | 仅在分子图上验证 (原子/键类型 = 离散) |
| Mesh 领域 | 全部使用启发式分区 (METIS, KNN, 坐标排序) |

**空白**: 没有人在 mesh 面邻接 (dual) 图上做过 BPE-style 的数据驱动子结构发现。

**障碍**: Mesh 面有连续特征 (法向量, 面积, 曲率)，Graph Tokenization 需要离散标签。

**可行性评估**: **中-高**。面特征可通过以下方式离散化：
- 法向量 → 球面均匀分桶 (e.g., 64 bins via icosphere)
- 面积 → 对数分桶 (e.g., 8 bins)
- 二面角 → 角度分桶 (e.g., 16 bins for 0°-180°)
- 组合后 alphabet size = 64 × 8 × 16 = 8,192（可通过降低分辨率控制）

### Gap 2: Mesh 分区永远是启发式的，从未数据驱动

| 方法 | 分区策略 | 驱动方式 |
|------|---------|---------|
| MeshLex v1 | METIS k-way | 图论启发式 (面法向余弦权重) |
| MeshLLM | KNN / FPS | 几何近邻 |
| BPT/DeepMesh | Block-patch | 坐标排序 |
| FACE | N/A (per-face) | 无分区 |
| **缺失** | BPE-learned | **数据驱动频率统计** |

**空白**: 没有人从数据中学习 mesh 应该怎么切——切在哪里、切多大。

**意义**: 如果 mesh 的局部拓扑确实存在 universal vocabulary (MeshLex 的核心假设)，那么 BPE 应该能自动发现这些 recurring patterns，产生的分区可能比 METIS 更有语义意义。

### Gap 3: 两步 tokenization (分区 + 编码) 无人统一

**当前范式**:
```
Step 1: Mesh → Patches (METIS / KNN / block) [固定，不可微]
Step 2: Patches → Tokens (GCN / VQ / coordinate quantization) [可学习]
```

**问题**: Step 1 的质量直接影响 Step 2，但 Step 1 无法从 Step 2 的 loss 中获得梯度反馈。如果 METIS 切坏了，后面的 VQ 再好也救不回来。

**空白**: 没有 end-to-end 的机制让分区和编码联合优化。

**潜在方案**:
- a) Learned graph pooling (DiffPool-style) 替代 METIS → 产生可微分区
- b) BPE + VQ 联合训练：BPE 发现拓扑 pattern，VQ 编码几何细节
- c) 层次化 contraction：在对偶图上迭代合并节点 (类似 BPE merge)，每步保留 geometric embedding

### Gap 4: Graph-specific VQ 技术未进入 Mesh 生成领域

| VQ 技术 | 应用领域 | 核心机制 |
|---------|---------|---------|
| SimVQ (Li et al., 2025) | 通用 VQ | 冻结 C + 学习 W，防止 collapse |
| RGVQ (Zhai et al., 2025) | Graph 节点 VQ | Gumbel-Softmax + 对比正则化 |
| HQA-GAE (Zeng et al., 2025) | Graph 节点 VQ | 层次化双层 codebook + 退火 |
| VQGAN-LC (Zhu et al., 2024) | 图像 VQ | 预训练初始化 |
| **MeshLex v1** | Mesh Patch VQ | SimVQ + dead-code revival |

**空白**: RGVQ 和 HQA-GAE 的 graph-specific 技术从未在 mesh generation 场景测试。

**机会**:
- RGVQ 的 contrastive regularization 利用图结构信息 → mesh 的面邻接关系正好提供这种结构
- HQA-GAE 的 hierarchical codebook → 天然适配 mesh 的多尺度性质

### Gap 5: 多尺度 Mesh Tokenization 不存在

| 方法 | 尺度策略 |
|------|---------|
| CamS (分子) | Multi-scale: fine motifs → coarse scaffolds, 拼接后 NTP |
| MeshLex v1 | 单尺度: ~35 faces/patch (固定) |
| 所有 Mesh 方法 | 单一粒度 tokenization |

**空白**: mesh 有天然的多尺度结构 (细节 → 局部 → 全局)，但没有人利用过。

**机会**: CamS 的 multi-scale concatenation 策略可以迁移到 mesh:
- Scale 1: 单面级别 (atomic)
- Scale 2: ~10 faces (small patch)
- Scale 3: ~35 faces (current MeshLex patch)
- Scale 4: ~100+ faces (large region)

---

## 3. 方法比较矩阵

| 维度 | MeshLex v1 | Graph Tokenization | CamS | MeshLLM | RGVQ |
|------|-----------|-------------------|------|---------|------|
| **数据类型** | 3D mesh (连续) | 标签图 (离散) | 分子图 (离散) | 3D mesh (连续) | 通用图 (连续) |
| **分区方式** | METIS (固定) | N/A | BPE merge (learned) | KNN/Semantic | N/A |
| **编码方式** | GCN → SimVQ | BPE → Transformer | BPE → LLaMA | 坐标量化 → LLM | GNN → VQ |
| **Token 粒度** | ~35 faces/token | 可变子图/token | 可变 motif/token | 1 face/多token | 1 node/token |
| **多尺度** | 否 | 否 | 是 (fine→coarse) | 否 | 否 |
| **E2E 可训练** | 部分 (编码器E2E，分区固定) | 否 (BPE 离线) | 否 (BPE 离线) | 是 (LLM E2E) | 是 |
| **解决 collapse** | SimVQ + dead-code | N/A | N/A | N/A | Gumbel + contrastive |
| **跨类别泛化** | 验证: ratio 1.019× | 跨数据集测试 | 跨分子测试 | 跨类别生成 | 跨图测试 |

---

## 4. Idea Refinement — 3 个候选方案

### Idea A: MeshDualBPE — 在 Mesh 对偶图上的 BPE Tokenization

**核心思想**: 将 Graph Tokenization 的 BPE pipeline 直接移植到 mesh 面邻接对偶图上，用 BPE 自动发现 mesh 的 recurring topological patterns，替代 METIS 分区。

**Pipeline**:
```
1. 面特征离散化: 每面 → (normal_bin, area_bin) → label ∈ Σ
   每共边 → (dihedral_bin) → edge label
2. 构建 Labeled Dual Graph: 面=节点, 共边=边, 离散标签
3. 全数据集统计 edge pattern 频率 F(p)
4. Frequency-Guided Euler Circuit 序列化
5. BPE 学习合并词汇 (vocabulary size K = 1000~4000)
6. 每个 BPE token = 一组面 = 一个 "data-driven patch"
7. 对比 METIS patch 的质量 (patch 内连通性、大小均匀性、语义一致性)
```

**评分**:
| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | 9/10 | Graph BPE → Mesh 是首次，零直接竞品 |
| 可行性 | 6/10 | 离散化信息损失不确定；BPE patch 大小不可控 |
| 影响力 | 8/10 | 如果成功，证明 mesh 拓扑有类 NLP 的子结构规律 |
| 清晰度 | 8/10 | 步骤明确，可直接实现 |
| 证据支撑 | 7/10 | Graph Tokenization 14 基准 SOTA；MeshLex v1 已验证 universal vocabulary |
| **总分** | **38/50** | |

**风险**:
- 离散化粒度：太粗 → 丢失几何信息 → BPE 学到的 pattern 没有意义；太细 → alphabet 爆炸 → BPE 效率低
- Patch 大小不可控：BPE 产生的 token 可能非常不均匀 (1 面 ~ 100+ 面)

### Idea B: PoolLex — Learned Graph Pooling 替代 METIS 实现 E2E

**核心思想**: 用可微分的图 pooling (MinCutPool-style) 替代 METIS，使分区和 VQ codebook 联合端到端训练。

**Pipeline**:
```
1. 输入: 面邻接对偶图 (面特征: 法向+面积+位置, 连续)
2. GNN encoder → 面级别 embedding
3. MinCutPool layer → soft assignment matrix S ∈ R^{N×K}
   → 将 N 个面分配到 K 个 cluster (=patches)
   → 自动学习分区
4. Coarsened graph: K 个超节点 (=patches), 每个的 embedding = 聚合后的面 embedding
5. 超节点 embedding → SimVQ codebook → discrete token
6. Decoder 从 token 重建 patch 几何
7. 全流程 E2E: recon loss 同时优化分区 (S) 和编码 (VQ)
```

**评分**:
| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | 8/10 | Learned pooling for mesh tokenization 未见过 |
| 可行性 | 5/10 | MinCutPool O(N²K) 对 1000 面 mesh 可能吃力；分区+VQ 联合优化可能不稳定 |
| 影响力 | 9/10 | E2E mesh tokenization 是范式级创新 |
| 清晰度 | 6/10 | 多组件联合优化，调参复杂 |
| 证据支撑 | 6/10 | Graph pooling 在图分类上成熟；但未用于 mesh VQ-VAE |
| **总分** | **34/50** | |

**风险**:
- 计算复杂度：1000 面 → N=1000, K~30 → assignment matrix 30K 参数/mesh，可能勉强可行
- 训练稳定性：pooling + VQ 双重离散化，梯度路径复杂

### Idea C (推荐): MeshLex-HMT — Hierarchical Mesh Tokenization with Structure-Geometry Decoupling

> **术语说明**: 此处 "structure" 指面邻接对偶图上的连接模式 (connectivity patterns)，由离散化的局部微分几何特征（法向、面积、二面角）标注。严格来说这是 "局部几何结构" 而非拓扑不变量（如亏格、边界环）。我们用 "structural channel" 而非 "topology channel" 以避免歧义。BPE 在对偶图上发现的是 **recurring connectivity-geometry motifs** — 即哪些面以什么样的几何关系反复相连。

**核心思想**: 将 mesh tokenization 分为结构通道 (structural channel) 和几何通道 (geometry channel)，在结构通道上应用 BPE 发现面连接模式 (connectivity-geometry motifs)，在几何通道上用 VQ 编码精确几何，再通过多种融合策略联合表示。

**Graph BPE 算法规范** (核心贡献):
```
输入: 训练集中所有 mesh 的 labeled dual graphs {G_i = (V_i, E_i, l_V, l_E)}
      其中 l_V: 面 → 离散标签, l_E: 共边 → 离散标签 (二面角 bin)

定义:
- "Bigram" = 对偶图中相邻的一对节点 (u, v)，表示为 triple (l_V(u), l_E(u,v), l_V(v))
- "Merge" = 将匹配某 bigram pattern 的所有 (u, v) 对合并为单节点 w:
  - w 继承 u, v 的所有外部边 (不含 u-v 边本身)
  - w 获得新标签 l_V(w) = "merged_{l_V(u)}_{l_E}_{l_V(v)}"
  - 若合并后 w 与某邻居有多重边，保留所有边 (multigraph)
- "频率统计" = 遍历所有训练图，统计每种 bigram pattern 的全局出现次数

算法 (标准 BPE 扩展到图):
1. 初始词汇表 V_struct = 所有离散面标签 (base alphabet)
2. 统计所有训练图中的 bigram 频率
3. 选择最高频 bigram (l_a, l_e, l_b)
4. 在所有训练图中执行 merge: 将所有匹配的 (u,v) 对合并
   - 注意: 同一节点可能参与多个匹配 → 采用贪心左优先: 按节点 ID 排序，先匹配先合并，已合并节点不再参与
5. 将新符号加入 V_struct
6. 重复 2-5 直到 |V_struct| 达到目标大小 (e.g., 2000)

已知 complication:
- 图中 merge 顺序不唯一 (vs 线性序列 BPE 的左→右确定性) → 贪心 ID 排序提供确定性
- 合并可能产生 multigraph → 后续 bigram 统计需考虑多重边
- 图同构问题: 不同 mesh 中结构相同但 ID 不同的子图应视为同一 pattern → 通过 label-based pattern matching (而非 subgraph isomorphism) 规避，复杂度 O(|E|)
```

**Pipeline**:
```
=== 离线阶段 (一次性) ===
1. 面特征离散化 → labeled dual graph
   - 离散化方案需 ablation: {32, 64, 128} normal bins × {4, 8, 16} area bins × {8, 16, 32} dihedral bins
   - 报告: 离散化互信息 MI(discrete_label, continuous_features) 量化信息损失
2. Graph BPE on dual graph (上述算法) → 学习 structural vocabulary V_struct (size ~2000)
3. 每个 BPE token 对应一组面 = "struct-patch"
4. 统计 BPE patch 大小分布 (mean, std, min, max, percentiles)
   → 评估压缩率变化 vs METIS 固定 ~35 faces/patch

=== 训练阶段 ===
5. 对每个 mesh:
   a) BPE encode → struct-patch 划分 + struct-token IDs (离散，来自 V_struct)
   b) 对每个 struct-patch: GCN encode patch geometry → z_geo ∈ R^d (连续)
   c) z_geo → SimVQ → geo-token IDs (离散，来自 geo codebook)

6. Dual-Token Representation:
   - Struct-token: 捕捉面的连接模式 (哪些面以什么几何关系相连)
   - Geo-token: 捕捉精确几何 (顶点坐标在局部坐标系下的形状)

7. Fusion Strategies (ablation):
   a) Additive: h = struct_embed(struct_id) + geo_embed(geo_id)
   b) Concatenation: h = [struct_embed; geo_embed], proj to d
   c) Gated: g = σ(W_g[struct_embed; geo_embed]), h = g⊙struct_embed + (1-g)⊙geo_embed
   d) Cross-attention: struct tokens attend to geo tokens, then fuse

8. Sequence Model (ablation):
   a) Transformer self-attention across patches
   b) GCN on patch-level coarsened graph (MeshLex v1 baseline)
   → 判断是否需要 Transformer，还是 GCN 已足够

9. Decoder: Cross-attention from patch embeddings → reconstruct vertices

=== 生成阶段 (Phase 4) ===
10. 将 dual-token 序列 (struct_id, geo_id) 排列为 patch 序列
11. 训练轻量 AR 模型 (GPT-2 scale, ~50M params) 进行 next-token prediction
12. 从 AR 模型采样 → decode → 生成新 mesh
13. 报告: FID, COV, MMD on ShapeNet Chair/Table
    - 生成基线 (使用已发表数值): PolyGen, MeshGPT, FACE
    - 若指标协议不完全一致，注明差异 (e.g., 采样点数, normalization)

=== 评估 ===

--- 主实验 (Main Paper Tables) ---
14. 重建指标: CD, Normal Consistency (NC), F-Score@{0.01, 0.02, 0.05}
15. Mesh 质量: 非流形边比例, 自交面比例
16. 生成指标 (Phase 4): FID, COV, MMD on ShapeNet Chair/Table (vs PolyGen/MeshGPT/FACE)

--- Primary Ablations (Main Paper) ---
17a. Decoupling ablation (fixing best fusion + BPE partition):
     Struct-only vs Geo-only vs Struct+Geo → 3 runs
17b. Fusion ablation (fixing BPE partition + Struct+Geo):
     add vs concat vs gated vs cross-attn → 4 runs
17c. Partition comparison (separate table, not crossed):
     METIS vs spectral vs MinCutPool vs BPE → 4 configs on same reconstruction task

--- Secondary Ablations (Appendix) ---
17d. Discretization granularity: coarse / medium / fine → 3 runs
17e. Sequence model: Transformer vs GCN → 2 runs

--- 分析 ---
18. 与 MeshLex v1 (GCN-only) + MeshLLM primitive decomposition 对比
19. BPE vocabulary 分析:
    - 可视化 top-K tokens 对应的 mesh 子结构
    - 对比 PartNet 语义标注 → 计算 semantic purity (同一 token 内面属于同一语义部件的比例)
    - 操作性标准: token 内法向方差 < τ_n (τ_n = METIS patch 法向方差中位数)

--- 总实验量估算 ---
Primary: 3 + 4 + 4 = 11 runs
Secondary: 3 + 2 = 5 runs
Generation: 1 run (AR model)
Total: ~17 runs (远少于 full factorial 216 cells, 可在 ~100h 内完成)
```

**评分** (Round 1 review 后修订):
| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | 9/10 | Structure-Geometry decoupling + BPE 首次出现在 mesh |
| 可行性 | 6/10 | BPE 离线低风险；但 Phase 4 生成需额外 ~2 周开发 |
| 影响力 | 9/10 | 如含生成结果，可直接与 MeshGPT/FACE 竞争 |
| 清晰度 | 8/10 | 步骤清晰，fusion/discretization ablation 明确 |
| 证据支撑 | 8/10 | Graph Tokenization SOTA + MeshLex v1 验证 + CamS 验证 multi-scale |
| **总分** | **40/50** | (可行性因生成需求微降，但影响力因生成大幅提升) |

**核心优势**:
- 结构和几何解耦 → 自然解决连续特征问题 (离散化只影响结构通道，几何通道保持连续)
- BPE 是离线的 → 训练时不增加计算开销
- 可增量构建在 MeshLex v1 之上 → 最小化实现风险
- 丰富的 ablation 空间 → 出论文友好
- 包含生成 pipeline → 可直接与 SOTA mesh generation 方法对比

**已知风险与缓解**:
- **离散化信息损失**: 通过 3-level granularity ablation 量化 → 如果所有粒度都差，说明 BPE 不适合 mesh
- **BPE patch 大小不均**: 报告分布统计 → 如果极端不均，可加 BPE vocab size 约束或后处理分裂/合并
- **"结构" vs "拓扑" 语义**: 已修正命名为 structural channel，明确 BPE 发现的是 connectivity-geometry motifs
- **生成质量**: Phase 4 使用简单 AR 模型，目标是证明 dual-token 可生成，不追求 SOTA 生成质量
- **竞品时间窗口**: Graph Tokenization (2026.03) 刚发布，3-6 个月内可能有竞品 → 加速执行，优先完成核心贡献

**论文结构双轨策略** (Fallback):
- **主路径** (Phase 4 成功): 完整 paper = BPE tokenization + structure-geometry decoupling + reconstruction + generation
  - 定位: Mesh Generation (与 MeshGPT/FACE 竞争)
  - 标题侧重: "... for Compositional Mesh Generation"
- **备用路径** (Phase 4 生成质量不足): Paper = BPE tokenization + structure-geometry decoupling + reconstruction + BPE vocabulary analysis
  - 定位: Mesh Representation/Tokenization (侧重 analysis 贡献)
  - 标题侧重: "... a Study on Data-Driven Mesh Tokenization"
  - 生成结果作为 "preliminary" 放入 appendix
- **Introduction 写法**: 以 "mesh tokenization 的根本问题" 开篇 (不依赖生成)，生成作为 natural downstream application 引入 → 两种路径均可用

**可分阶段实验**:
- Phase 0: BPE on dual graph 可行性验证 (纯离线分析，无训练) — **~2h CPU**
- Phase 1: BPE patches vs METIS vs spectral vs MinCutPool 分区质量对比 — **~8h**
- Phase 2: Dual-token 训练 + fusion/discretization ablation — **~20h GPU**
- Phase 3: 完整 MeshLex-HMT vs MeshLex v1 对比 (reconstruction) — **~30h GPU**
- Phase 4: AR 生成模型训练 + 标准生成指标评估 — **~40h GPU**

---

## 5. 推荐方案: Idea C (MeshLex-HMT)

### 5.1 为什么选 C

1. **最高总分 (40/50)** 且各维度均衡
2. **解决了核心矛盾**: Graph Tokenization 需要离散标签，但 mesh 有连续特征 → Structure-Geometry decoupling 天然化解
3. **可增量验证**: Phase 0 几乎零成本 (纯数据分析)，Phase 1 低成本 (只需 BPE 离线训练)
4. **故事完整**: "NLP 有 BPE for text → Graph Tokenization 证明 BPE for graphs → 我们证明 BPE for meshes → 并用它生成新 mesh"
5. **包含生成 pipeline**: Phase 4 AR 生成让本文不只是 tokenizer 论文，可与 MeshGPT/FACE 直接竞争

### 5.2 关键假设 & Go/No-Go 标准

| 假设 | 验证方法 | Go 标准 | No-Go 标准 |
|------|---------|---------|-----------|
| H1a: BPE tokens 几何一致性 | token 内法向方差 < τ_n (τ_n = METIS patch 法向方差中位数) 的 token 比例 | ≥60% tokens 满足 | <30% tokens 满足 |
| H1b: BPE tokens 语义一致性 (若有 PartNet 标注) | PartNet semantic purity (同一 token 内面属于同一部件的比例) | ≥50% semantic purity | <25% semantic purity |
| **H1 综合判定** | H1a 为必要条件; H1b 为加分项 (无 PartNet 标注时仅看 H1a) | H1a Go 即可继续; H1a+H1b 均 Go = strong signal | H1a No-Go → 停止 |
| H2: BPE patches 在多基线对比中不劣 | Patch 内连通性、大小方差、CD; 对比 METIS + spectral clustering + MinCutPool | BPE CD ≤ best baseline × 1.2 | BPE CD > best baseline × 2 |
| H3: Struct+Geo dual token 优于 Geo-only | 配对 Wilcoxon signed-rank test across shapes | p < 0.05 且 median CD 改善 ≥ 3% | p > 0.1 或 median CD 无改善 |
| H4: 最佳 fusion strategy 优于 simple addition | Ablation: add vs concat vs gated vs cross-attn | 至少一种 fusion > add 且 p < 0.05 | 所有 fusion 策略无显著差异 |
| H5: 离散化粒度可控 | 3-level granularity ablation (coarse/medium/fine) | 存在 sweet spot 且 MI > 0.5 | 所有粒度 MI < 0.3 或 CD 单调恶化 |

### 5.3 资源估算

| 阶段 | 内容 | 计算量 | 时间估算 (RTX 4090) |
|------|------|--------|-------------------|
| Phase 0 | BPE 可行性 + patch 统计 + 离散化 MI | CPU-only | ~2h |
| Phase 1 | BPE vs METIS vs spectral vs MinCutPool | CPU preprocess + GPU 轻量训练 | ~8h |
| Phase 2 | Dual-token 训练 + fusion/discretization ablation | GPU 训练 200 epochs × 多配置 | ~20h |
| Phase 3 | 完整 reconstruction 对比 (MeshLex v1 / MeshLLM baseline) | GPU 训练 × 3 配置 | ~30h |
| Phase 4 | AR 生成模型 + FID/COV/MMD 评估 | GPU 训练 ~50M param AR model | ~40h |
| Buffer | 重跑失败实验 / 额外 ablation | | ~20h |
| **Total** | | | **~120h GPU** |

> **注**: Phase 4 是 Round 1 review 后新增的关键阶段。论文采用双轨策略：主路径含生成 (mesh generation venue)，备用路径仅含 tokenization + analysis (representation venue)。120h 含 20h buffer 应对实验失败。

---

## 6. 与论文定位的关系

### 6.1 对 MeshLex 论文的增强

MeshLex 当前论文 (context/22_final_report.md) 的 **Section 5.3 Limitations** 明确列出:

> "1. Reconstruction only... 2. Boundary stitching... 3. **Fixed patch size**... 4. Codebook size... 5. Single decoder architecture"

MeshLex-HMT 直接回应 Limitation 3 ("Fixed patch size"):
- METIS 使用固定 ~35 faces/patch → BPE 自动发现最优粒度
- 可能发现平面区域需要大 patch (多面)、曲面细节需要小 patch (少面)

### 6.2 新增贡献点

在原有 5 个贡献之上，可新增:

6. **首次在 3D mesh 对偶图上应用 BPE tokenization**，发现 mesh 面的连接模式存在 NLP-like 的频率分布规律，并量化了离散化信息损失
7. **Structure-Geometry decoupled tokenization**: 结构通道 BPE + 几何通道 VQ 的双通道表示，含 4 种融合策略对比
8. **Data-driven vs heuristic partition**: BPE patches 与 METIS / spectral / learned pooling 多基线对比
9. **Dual-token mesh generation**: 基于 struct+geo dual tokens 的 AR 生成模型，报告标准生成指标 (FID, COV, MMD)
10. **Comprehensive evaluation**: CD + NC + F-Score + 流形质量 + 语义纯度的多维评估体系

### 6.3 竞品风险分析

| 潜在竞品方向 | 风险评估 |
|-------------|---------|
| 有人也想到 Graph BPE → Mesh? | 低 — Graph Tokenization 刚发表 (2026.03)，且连续特征是公认障碍 |
| MeshLLM 扩展到 BPE 分区? | 低 — MeshLLM 用 KNN 分区，路线完全不同 |
| FACE/BPT 升级到 patch-level? | 中 — 但他们的基础是坐标序列化，不是图结构 |
| 分子领域 BPE 迁移到 3D? | 低 — 分子图和 mesh 对偶图结构差异很大 |

---

## 7. 下一步行动

**Phase 0 (BPE 可行性 + 离散化分析, ~2h CPU)**:

1. 从现有 Objaverse 预处理数据中，提取面邻接对偶图
2. 设计面特征离散化方案，**测试 3 种粒度** (coarse/medium/fine)
3. 计算离散化互信息 MI(discrete_label, continuous_features)
4. 实现 Frequency-Guided Euler Circuit on dual graph
5. 运行 BPE，统计词汇表分布
6. **统计 BPE patch 大小分布** (mean, std, min, max, p5/p25/p50/p75/p95)
7. 可视化: BPE 合并过程 + top-K tokens 对应的 mesh 子结构
8. 操作性验证: token 内法向方差 vs METIS patch 法向方差
9. 产出: 可行性报告 + 可视化 + 离散化 MI 表 → Go/No-Go 决策

**Phase 1 (多基线分区对比, ~8h)**:

1. 实现分区基线: METIS, spectral clustering on mesh Laplacian, MinCutPool
2. 统一评估: CD, patch 连通性, 大小方差, 语义纯度 (若有 PartNet 标注)
3. 如有 MeshLLM primitive decomposition 代码，加入对比

**Phase 2-3 (Dual-token 训练 + 完整对比, ~50h GPU)**:

1. Fusion ablation: add / concat / gated / cross-attn
2. Discretization ablation: 3 粒度
3. Sequence model ablation: Transformer vs GCN
4. 多维指标: CD + NC + F-Score + 流形质量

**Phase 4 (生成, ~40h GPU)**:

1. 训练 AR 模型 on dual-token sequences
2. 标准生成指标: FID, COV, MMD (ShapeNet Chair/Table)
3. 与 MeshGPT, FACE 对比

**推进顺序**: Phase 0 → 1 → 2 → 3 → 4，每阶段末尾 Go/No-Go 决策。

**需要用户确认**: 是否按此顺序推进？Phase 4 (生成) 是否为本次投稿的必要条件？
