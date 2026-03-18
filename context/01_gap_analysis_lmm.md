<!-- markdownlint-disable -->
# Gap Analysis: Large Mesh Model (LMM)

- [Gap Analysis: Large Mesh Model (LMM)](#gap-analysis-large-mesh-model-lmm)
  - [执行摘要](#执行摘要)
  - [Method Comparison Matrix](#method-comparison-matrix)
    - [A. 自回归 (AR) 网格生成方法](#a-自回归-ar-网格生成方法)
    - [B. 扩散/流匹配 (Diffusion/Flow) 3D生成方法](#b-扩散流匹配-diffusionflow-3d生成方法)
    - [C. 大型重建模型 (LRM) 家族](#c-大型重建模型-lrm-家族)
    - [D. LLM/多模态统一方法](#d-llm多模态统一方法)
    - [E. SSM/Mamba 方法](#e-ssmmamba-方法)
  - [Contradictions Found](#contradictions-found)
    - [Contradiction 1: 离散 vs 连续 tokens — 哪个更好？](#contradiction-1-离散-vs-连续-tokens--哪个更好)
    - [Contradiction 2: 数据scaling vs 架构scaling — 哪个更重要？](#contradiction-2-数据scaling-vs-架构scaling--哪个更重要)
    - [Contradiction 3: Face-by-face 生成 vs Coarse-to-fine 生成 — 哪种序列化更优？](#contradiction-3-face-by-face-生成-vs-coarse-to-fine-生成--哪种序列化更优)
    - [Contradiction 4: Tokenization压缩 vs 信息保真](#contradiction-4-tokenization压缩-vs-信息保真)
  - [Blank Identification (空白识别)](#blank-identification-空白识别)
    - [Blank 1: 重建与生成从未在单一显式mesh模型中统一](#blank-1-重建与生成从未在单一显式mesh模型中统一)
    - [Blank 2: 显式mesh生成中从未验证 Scaling Laws](#blank-2-显式mesh生成中从未验证-scaling-laws)
    - [Blank 3: AR+Diffusion 混合架构从未应用于显式mesh生成](#blank-3-ardiffusion-混合架构从未应用于显式mesh生成)
    - [Blank 4: Mamba/SSM 从未应用于通用mesh生成（仅限固定拓扑）](#blank-4-mambassm-从未应用于通用mesh生成仅限固定拓扑)
    - [Blank 5: 多模态条件生成显式mesh的统一框架不存在](#blank-5-多模态条件生成显式mesh的统一框架不存在)
    - [Blank 6: 纹理感知的显式mesh生成从未实现](#blank-6-纹理感知的显式mesh生成从未实现)
    - [Blank 7: Coarse-to-fine 显式mesh生成缺少conditional generation验证](#blank-7-coarse-to-fine-显式mesh生成缺少conditional-generation验证)
  - [Field Trends (趋势分析)](#field-trends-趋势分析)
    - [升温趋势 (2024-2025)](#升温趋势-2024-2025)
    - [降温趋势](#降温趋势)
    - [新兴问题](#新兴问题)
  - [Research Gaps (按分数排名)](#research-gaps-按分数排名)
    - [Gap 1: 统一多模态条件的大规模显式mesh生成模型 (Unified Large Mesh Model)](#gap-1-统一多模态条件的大规模显式mesh生成模型-unified-large-mesh-model)
    - [Gap 2: 显式mesh生成的 Scaling Laws 验证](#gap-2-显式mesh生成的-scaling-laws-验证)
    - [Gap 3: Coarse-to-fine 混合架构（AR结构规划 + Flow细节生成）用于显式mesh](#gap-3-coarse-to-fine-混合架构ar结构规划--flow细节生成用于显式mesh)
    - [Gap 4: Mamba/SSM 用于通用显式mesh生成](#gap-4-mambassm-用于通用显式mesh生成)
    - [Gap 5: 纹理感知的显式mesh端到端生成](#gap-5-纹理感知的显式mesh端到端生成)
    - [Gap 6: 连续 vs 离散 mesh tokens 的公平对比](#gap-6-连续-vs-离散-mesh-tokens-的公平对比)
    - [Gap 7: LOD Progressive Mesh Generation 的 Conditional 验证](#gap-7-lod-progressive-mesh-generation-的-conditional-验证)
  - [Gap Ranking Summary](#gap-ranking-summary)
  - [研究日志](#研究日志)
  - [未验证问题](#未验证问题)
  - [结论](#结论)

## 执行摘要

- **研究主题**: 统一3D重建与生成的大规模显式网格模型 (Large Mesh Model)
- **总迭代轮数**: 3
- **总阅读论文数**: 75+ (含 Stage 1)
- **初始问题数**: 3
- **识别的研究空白数**: 7
- **停止原因**: 已识别足够数量的研究空白且达到论文目标
- **新增关键论文 (Stage 2)**: EdgeRunner, ARMesh, BPT/DeepMesh, FastMesh, ShapeLLM-Omni, Unifi3D, CraftsMan3D, TriMM, AssetFormer, MeshLLM

---

## Method Comparison Matrix

### A. 自回归 (AR) 网格生成方法

| Paper | Year | Tokenization | Tokens/Face | Max Faces | Quant Res | Conditioning | Backbone | Output |
|-------|------|-------------|-------------|-----------|-----------|-------------|----------|--------|
| MeshGPT | 2023 | VQ-VAE (GraphConv) | 9 | ~800 | 128 | Category | GPT-2 | Tri mesh |
| MeshXL | 2024 | Direct coord | 9 | ~800 | 128 | Category | GPT-2 (~350M) | Tri mesh |
| MeshAnything V2 | 2024 | AMT (adjacent) | ~3 | 1,600 | 128 | Point cloud | Decoder-only TF | Tri mesh |
| EdgeRunner | 2024 | EdgeBreaker-based | ~4-5 | 4,000 | 512 | Point cloud / Image (ArAE+LDM) | OPT + DiT | Tri mesh |
| BPT (DeepMesh) | 2024 | Block+Patch | ~2.25 (75% compress) | 8,000+ | 128 | Point cloud / Image | Decoder-only TF | Tri mesh |
| Meshtron | 2024 | Direct coord 1024-bin | 9 | 64,000 | 1024 | Point cloud (Perceiver) | Hourglass TF (1.1B) | Tri/Quad mesh |
| ARMesh | 2025 | PSC (LOD vertex-split) | Variable | Flexible (LOD) | Continuous | Unconditional | TF (AR) | Simplicial complex |
| FastMesh | 2025 | Vertex-only AR + parallel face | ~23% of BPT | 4,000+ | 128→continuous | Point cloud | AR TF + BiTF | Tri mesh |
| TreeMeshGPT | 2025 | Tree sequencing | - | - | - | - | AR TF | Tri mesh |
| LLaMA-Mesh | 2024 | Plain text OBJ | ~27/face | ~500 | 64 | Text / Image | LLaMA-3.1 (8B) | Tri mesh |
| MeshLLM | 2025 | Primitive-Mesh decomp | Variable | 1,500+ faces | - | Text / Image | LLM | Tri mesh |

### B. 扩散/流匹配 (Diffusion/Flow) 3D生成方法

| Paper | Year | Representation | Latent Type | Model Scale | Conditioning | Mesh Extraction | Texture |
|-------|------|---------------|-------------|-------------|-------------|----------------|---------|
| Direct3D | 2024 | Triplane | KL-continuous (32×32×16) | ~675M (DiT-XL) | Image (DINO+CLIP) | Occupancy → MC | No |
| TRELLIS | 2024 | Sparse voxel SLat | Sparse VAE (8ch) | 342M/1.1B/2B | Image/Text (CLIP+DINOv2) | FlexiCubes 256³ | Yes (Gaussian) |
| TripoSG | 2025 | SDF latent | Hybrid hash-grid | 1.5B→4B MoE | Image | SDF → MC | No |
| Step1X-3D | 2025 | SDF latent | FLUX-style MMDiT | ~1B | Image | TSDF → MC | Albedo |
| Hunyuan3D 2.0 | 2025 | SDF + texture | DiT latent | ~1B each | Image/Text | SDF → MC | Albedo/PBR-like |
| MAR-3D | 2025 | Point cloud occupancy | Continuous (Pyramid VAE) | ~260K data | Image | Occupancy → MC | No |
| MeshCraft | 2025 | Face-level continuous | KL-8dim per face | ~MeshGPT scale | Image + face count | Direct mesh decode | No |
| CraftsMan3D | 2024 | SDF latent | VAE | - | Image (multi-view guided) | SDF → MC + Normal refiner | No |
| OctGPT | 2025 | Octree binary | BSQ (codebook-free) | Consumer 4090 | Text/Image/Sketch | SDF → MC | No |

### C. 大型重建模型 (LRM) 家族

| Paper | Year | Representation | Model Scale | Speed | Mesh Extraction | Texture |
|-------|------|---------------|-------------|-------|----------------|---------|
| LRM | 2023 | Triplane-NeRF | ~500M | <5s | MC from NeRF | Yes |
| GRM | 2024 | Pixel-aligned Gaussians | ~500M | 0.1s | TSDF fusion | Yes |
| Long-LRM | 2024 | Mamba2+TF hybrid | ~500M | - | MC | Yes |
| TripoSR | 2024 | Triplane-NeRF (camera-free) | ~500M | <1s | MC | Yes |
| CRM | 2024 | Triplane (Conv U-Net) | - | 10s | FlexiCubes | Yes |
| SF3D | 2024 | UV-unwrapped mesh | - | 0.5s | Direct mesh opt | Yes (UV+delighting) |

### D. LLM/多模态统一方法

| Paper | Year | 3D Repr | Backbone | Input | Output | Gen+Understand |
|-------|------|---------|----------|-------|--------|---------------|
| ShapeGPT | 2023 | SDF 64³ VQ (512 tok) | T5-base (220M) | Text/Image/Shape | Shape/Text | Yes (limited) |
| LLaMA-Mesh | 2024 | Text OBJ | LLaMA-3.1 (8B) | Text/Image | Mesh/Text | Yes |
| MeshLLM | 2025 | Primitive-Mesh | LLM | Text/Image | Mesh/Text | Yes |
| SAR3D | 2024 | Multi-scale triplane VQ | GPT-style + Vicuna-7B | Image/Text | Triplane→mesh + Caption | Yes (separate models) |
| ShapeLLM-Omni | 2025 | Voxel 64³ VQ (1024 tok) | Qwen-2.5-VL (7B) | Text/Image/3D | 3D/Text | Yes (unified, first ChatGPT-4o-style) |

### E. SSM/Mamba 方法

| Paper | Year | Representation | Topology | Max Vertices | Speed vs TF |
|-------|------|---------------|----------|-------------|-------------|
| MeshMamba | 2024 | Raw vertex coords | Fixed (template) | 10,475 (SMPL-X) | 6-9x faster |

---

## Contradictions Found

### Contradiction 1: 离散 vs 连续 tokens — 哪个更好？

- **MeshCraft** 明确实验证明连续 KL-8dim tokens 大幅优于离散 RVQ tokens（Objaverse: 99.66% vs 65.12% triangle accuracy），主张连续表示是mesh生成的正确方向。
- **G3PT** 使用离散 LFQ tokens（codebook 8192），实现了97-99%码本利用率，在3D生成中首次展示 scaling law 行为，IoU达87.6。
- **SAR3D** 使用离散 VQ 16384，实现 0.82s 极速生成。
- **MAR-3D** 显式避免 VQ，使用连续 Pyramid VAE，论证 VQ 带来不可接受的压缩损失。
- **EdgeRunner** 使用连续 latent（L2正则化，非 VQ），通过 ArAE 将变长mesh压缩为定长latent code。
- **矛盾本质**: 在**mesh重建质量**上连续明显胜出；但在**生成可控性和scaling**上离散尚未被证伪。没有工作在同一框架内公平对比两者用于**显式mesh**生成的差异。

### Contradiction 2: 数据scaling vs 架构scaling — 哪个更重要？

- **TripoSG** 明确实验结论：数据scaling（180K→2M）的影响是架构scaling的**3倍**。主张数据质量和数量是3D生成的第一瓶颈。
- **G3PT** 是唯一展示3D生成 scaling law 的工作（0.1B→0.5B→1.5B），证明模型参数增加带来可预测的质量提升，主张架构scaling是核心驱动力。
- **矛盾本质**: 两者实验设置不同（TripoSG用SDF隐式表示，G3PT用离散occupancy tokens），结论可能都正确但各自适用范围未明确。**显式mesh领域**同时验证数据+架构scaling的工作不存在。

### Contradiction 3: Face-by-face 生成 vs Coarse-to-fine 生成 — 哪种序列化更优？

- **主流方法**（MeshGPT, BPT, EdgeRunner, FastMesh, Meshtron）全部采用 face-by-face 词法排序，沿 z-y-x 坐标轴排列面，逐面生成。
- **ARMesh** 首次提出 next-LOD prediction，通过反转mesh simplification过程，从单点逐步增加几何细节，主张这种coarse-to-fine方式"更符合人类对3D的感知"。
- **G3PT/SAR3D** 采用 next-scale prediction（但在隐式token空间，非显式mesh上）。
- **矛盾本质**: ARMesh在unconditional generation上展示了可行性，但**尚无conditional generation**（image/text→mesh）的验证。Face-by-face方法在conditional生成上成熟但固有序列长度瓶颈。两种范式在同一条件下未被公平对比。

### Contradiction 4: Tokenization压缩 vs 信息保真

- **BPT** 实现75%压缩率，8K面网格可处理，主张"压缩是scaling的关键"。
- **Meshtron** 刻意不压缩（9 tokens/face），主张"架构效率（Hourglass TF + truncated training）是比表示压缩更好的路径"，64K面。
- **FastMesh** 提出vertex-face解耦，vertex AR + face并行生成，tokens仅为BPT的23%，4K+面，8x加速。
- **矛盾本质**: 三种完全不同的scaling策略（压缩token / 高效架构 / 解耦生成）都声称有效，但从未在相同规模和数据集上对比。

---

## Blank Identification (空白识别)

### Blank 1: 重建与生成从未在单一显式mesh模型中统一

**类型**: Method combination blank

**描述**: 所有现有方法严格分为两类：(a) 重建模型（LRM/GRM/TripoSR/CRM）— 确定性feed-forward，从观测图像回归3D，但输出隐式表示+MC后处理；(b) 生成模型（MeshGPT/Meshtron/BPT/TRELLIS）— 概率性采样，从条件信号生成新3D。没有任何模型能同时：
- 给定清晰多视角图像 → 高保真重建显式mesh（确定性模式）
- 给定文本/草图 → 创造性生成显式mesh（生成模式）

**证据**:
- SAR3D 尝试统一生成+理解，但生成和理解用**不同模型**（generation TF vs SAR3D-LLM），输出triplane而非mesh
- ShapeLLM-Omni 最接近统一，但输出64³ voxel（1024 tokens），几何分辨率极低，且依赖外部TRELLIS将voxel重建为mesh
- LLaMA-Mesh 同时支持text→mesh和mesh→text，但mesh质量极低（max 500 faces, 64-level quantization）
- MeshLLM 扩展到1.5M样本，但仍受LLM token length限制
- Unifi3D 仅是**评测框架**，未提出统一模型

**空白确认**: 经acd_search验证，无论文标题或摘要同时包含 "unified reconstruction generation explicit mesh" 的关键组合。

### Blank 2: 显式mesh生成中从未验证 Scaling Laws

**类型**: Scale gap

**描述**: G3PT 首次在3D生成中展示power-law scaling（0.1B→1.5B），但其tokenizer输出占据field tokens，最终mesh通过MC提取。Meshtron scaling到1.1B参数和64K面，但**未报告scaling law**（未测试多个模型规模的loss曲线）。所有显式mesh AR方法（MeshGPT/BPT/EdgeRunner/FastMesh）均在<1B参数、<100K数据规模运作，无scaling行为研究。

**证据**:
- G3PT: 0.1B/0.5B/1.5B三点验证power-law，但表示是occupancy field tokens
- Meshtron: 仅0.5B和1.1B两个规模，未绘制scaling curve
- BPT: 验证了**数据scaling**（更多面数→更好质量），但未验证模型参数scaling
- MeshCraft: 单一规模（~MeshGPT size），未讨论scaling
- 显式mesh领域的 scaling law 完全空白

### Blank 3: AR+Diffusion 混合架构从未应用于显式mesh生成

**类型**: Method combination blank

**描述**: 在隐式3D生成领域，MAR-3D 成功展示了 masked AR + per-token diffusion 的混合范式；EdgeRunner 将 AR auto-encoder + latent diffusion 结合。但在**显式mesh**生成中：
- 纯AR方法（MeshGPT/Meshtron/BPT）：逐面生成，序列长、推理慢
- 纯Diffusion/Flow方法（MeshCraft）：并行生成快，但面数受限、拓扑质量差
- 从未有方法尝试：**AR做粗粒度结构规划 + Diffusion做细粒度几何细化**

**证据**:
- EdgeRunner 最接近：ArAE (AR encoder) 压缩为固定latent → DiT (diffusion) 做条件生成。但其AR部分是**编码器**而非生成器，diffusion在latent空间而非mesh空间操作
- CraftsMan3D 用 DiT 生成粗SDF + Normal refiner 细化，但操作在SDF空间，非mesh
- ARMesh 的 LOD 渐进生成是coarse-to-fine，但纯AR无diffusion
- FastMesh 的 vertex AR + face parallel 是解耦但非 AR+Diffusion 混合
- 真正的 "AR coarse mesh structure + Diffusion fine geometry refinement" 在显式mesh上从未尝试

### Blank 4: Mamba/SSM 从未应用于通用mesh生成（仅限固定拓扑）

**类型**: Method transfer blank

**描述**: MeshMamba 展示了 Mamba-SSM 替代 Transformer 处理10K+顶点mesh的能力（6-9x加速），但**严格限于固定拓扑**（SMPL/SMPL-X模板mesh，所有训练样本共享相同连接关系）。将 Mamba 的近线性复杂度优势应用于**通用mesh生成**（可变拓扑、可变面数）从未被探索。

**证据**:
- MeshMamba 论文 Limitation 明确指出: "Limited to fixed-topology meshes with tight clothing"
- Long-LRM 将 Mamba2 + Transformer hybrid 用于250K token的3D重建，但输出隐式表示
- Point Cloud Mamba, 3DMambaComplete 将 Mamba 用于点云，但非mesh
- 在显式mesh序列（face token序列）上，Mamba 的选择性状态空间机制是否能有效替代 causal attention：**完全未知**

### Blank 5: 多模态条件生成显式mesh的统一框架不存在

**类型**: Conditioning gap

**描述**: 现有显式mesh生成方法的conditioning极为受限：
- Meshtron: 仅 point cloud（无text/image直接条件）
- MeshGPT/MeshXL: 仅 category label
- BPT: point cloud + image
- EdgeRunner: point cloud (ArAE) + image (LDM)
- MeshCraft: image + face count
- LLaMA-Mesh: text + image，但mesh质量极低
- **无任何方法**能在高质量显式mesh生成（>4K面）中同时支持 text/image/point cloud/multi-view 多种输入

**证据**:
- OctGPT 支持 text/image/sketch/category，但输出SDF→MC，非显式mesh拓扑
- TRELLIS 支持 text/image，输出多格式（含mesh），但mesh via FlexiCubes后处理
- Hunyuan3D 2.0 支持 text/image，但SDF→MC
- 高面数显式mesh + 多模态条件 = 空白

### Blank 6: 纹理感知的显式mesh生成从未实现

**类型**: Feature gap

**描述**: 所有显式mesh AR方法**仅生成几何**（顶点坐标+面连接），不生成材质/纹理。纹理需要单独的pipeline（如SyncMVD, Paint3D）。而隐式方法（TRELLIS, Hunyuan3D, SF3D）能生成带纹理的mesh。

**证据**:
- Meshtron, BPT, EdgeRunner, MeshGPT, FastMesh: 全部 geometry-only
- G3PT, Direct3D: "No texture generation, requires separate external method"
- MeshCraft: geometry-only
- 唯一例外是 SF3D（直接mesh优化+UV unwrapping），但它是重建模型而非生成模型
- 将vertex color或UV coordinates纳入AR mesh token序列从未被尝试

### Blank 7: Coarse-to-fine 显式mesh生成缺少conditional generation验证

**类型**: Validation gap

**描述**: ARMesh 提出了优雅的 LOD-based progressive mesh generation（从单点→渐进细化），但：
- 仅验证了 unconditional generation 和 mesh encoding
- 未验证 text-conditioned 或 image-conditioned 生成
- 未在 Objaverse-scale 数据上训练
- 未与 BPT/Meshtron 等SOTA方法在conditional任务上对比

**证据**:
- ARMesh 论文 Section 5.3 仅展示 unconditional generation（ShapeNet categories）
- Section 6 "Limitation, Future Work" 未明确讨论conditional generation
- G3PT/SAR3D 的 next-scale prediction 在隐式空间有conditional验证，但ARMesh的mesh-native LOD方法没有

---

## Field Trends (趋势分析)

### 升温趋势 (2024-2025)

1. **Tokenization压缩竞赛**: MeshGPT(9tok/face) → AMT(3tok) → EdgeRunner(4-5tok) → BPT(2.25tok) → FastMesh(vertex-only, 23% of BPT)。每隔几个月就有新压缩方案，竞赛白热化。

2. **Coarse-to-fine / Multi-scale 范式兴起**: G3PT(next-scale), SAR3D(next-scale), ARMesh(next-LOD), MAR-3D(cascaded LR→HR)。从一维词法排序转向层次化多尺度生成是明确趋势。

3. **LLM+3D 统一框架**: LLaMA-Mesh → MeshLLM → ShapeLLM-Omni，将3D mesh作为LLM的"外语"，统一理解+生成。ShapeLLM-Omni标志着 ChatGPT-4o-style 3D模型的起步。

4. **数据scaling > 架构创新**: TripoSG(2M curated)、BPT(scaling面数)、MeshLLM(1.5M样本) 均强调数据质量和规模。高质量3D数据成为核心竞争力。

5. **Vertex-Face解耦**: FastMesh 和 SpaceMesh 开创vertex AR + face并行的新范式，突破传统"全序列化"思路。

6. **Flow matching / Rectified flow 取代传统 DDPM**: TRELLIS, TripoSG, MeshCraft 全部使用 rectified flow，在3D生成中已成为默认扩散范式。

### 降温趋势

1. **SDS优化方法**: DreamFusion风格的逐物体SDS优化已被大规模feed-forward模型取代。
2. **纯VQ-VAE tokenization**: MeshGPT式的graph-conv VQ-VAE被证明是瓶颈（lossy, 限制面数），连续latent和高效序列化正在取代它。
3. **简单category-conditional生成**: 早期方法仅支持类别条件（ShapeNet 5-13类），已无竞争力。
4. **小规模ShapeNet-only评测**: 越来越多方法在Objaverse全集上训练和评测，ShapeNet-only结果不再令人信服。

### 新兴问题

1. **显式mesh生成的scaling law**: G3PT在隐式空间破冰，显式mesh领域紧随其后。
2. **Mesh + Texture端到端生成**: 目前是最大的practical gap — 隐式方法能做，显式方法不能。
3. **交互式mesh编辑**: ShapeLLM-Omni引入了3D编辑数据集（62K pairs），但基于voxel而非显式mesh。
4. **非三角形mesh**: 除Meshtron支持quad-dominant外，几乎所有方法限于triangular mesh。

---

## Research Gaps (按分数排名)

**排名公式**: score = feasibility × 0.4 + impact × 0.4 + novelty × 0.2

### Gap 1: 统一多模态条件的大规模显式mesh生成模型 (Unified Large Mesh Model)
- **类型**: Method combination blank (Blank 1 + Blank 5)
- **描述**: 构建一个单一模型，能在不同条件（text/image/point cloud/multi-view）下生成高质量显式mesh（>4K faces），同时支持重建模式（确定性）和生成模式（创造性）。核心思路是：用一个强大的mesh tokenizer将显式mesh压缩为compact latent tokens，在此latent空间上训练conditional generation model，支持多种输入模态。
- **证据**: Blank 1 + Blank 5 综合。SAR3D/ShapeLLM-Omni展示了统一方向但质量不足；EdgeRunner展示了ArAE+LDM的可行性但条件单一；BPT展示了高面数mesh的可处理性。
- **可行性**: **High** — EdgeRunner已验证ArAE→定长latent→LDM的pipeline；BPT已验证8K面mesh的可压缩性；DINOv2/CLIP条件注入机制成熟。核心技术组件均已存在，需要的是正确组合和规模化。
- **潜在影响**: **High** — 这是3D生成领域的"GPT-4 moment"——统一多任务多模态于单一mesh模型。直接对标CCF-A。
- **新颖性**: **High** — 无先例工作同时满足：显式mesh + 多模态条件 + 大规模 + 重建/生成统一。
- **置信度**: High
- **Score**: High(3)×0.4 + High(3)×0.4 + High(3)×0.2 = **3.0**

### Gap 2: 显式mesh生成的 Scaling Laws 验证
- **类型**: Scale gap (Blank 2)
- **描述**: 系统性验证显式mesh AR模型在参数规模（100M→500M→1B+）和数据规模（10K→100K→1M）上的scaling行为，绘制power-law曲线，确定最优的compute-data allocation。
- **证据**: G3PT首次在隐式3D tokens上展示scaling law；Meshtron到1.1B但未测scaling；BPT验证面数scaling但非参数scaling。
- **可行性**: **Medium** — 需要大量计算资源（G3PT用136 H20 GPU训练2周）。但可以从小规模开始验证趋势。
- **潜在影响**: **High** — 如果显式mesh也遵循scaling law，将为整个领域提供资源分配指导，意义堪比Chinchilla/Kaplan对LLM的贡献。
- **新颖性**: **High** — 显式mesh领域零先例。
- **置信度**: High
- **Score**: Medium(2)×0.4 + High(3)×0.4 + High(3)×0.2 = **2.6**

### Gap 3: Coarse-to-fine 混合架构（AR结构规划 + Flow细节生成）用于显式mesh
- **类型**: Method combination blank (Blank 3 + Blank 7)
- **描述**: 用AR模型生成粗粒度mesh结构（LOD-0，few hundred faces），然后用Flow/Diffusion模型对mesh进行渐进细化（LOD-0→LOD-1→...→LOD-N），每级添加几何细节。结合ARMesh的LOD思想 + MeshCraft的Flow-based mesh refinement + G3PT的multi-scale token设计。
- **证据**: ARMesh验证了LOD progressive生成可行（unconditional）；MAR-3D验证了cascaded LR→HR在隐式3D中有效；CraftsMan3D验证了DiT粗生成+法线细化的pipeline。但三者从未在显式mesh上组合。
- **可行性**: **Medium** — ARMesh的PSC tokenization比较复杂，实现难度中等。LOD生成需要mesh simplification作为数据预处理。
- **潜在影响**: **High** — 解决AR mesh生成的两大核心问题：序列长度（粗级别短序列）和推理速度（细化可并行）。
- **新颖性**: **High** — 显式mesh上的 AR+Flow coarse-to-fine 从未被尝试。
- **置信度**: Medium
- **Score**: Medium(2)×0.4 + High(3)×0.4 + High(3)×0.2 = **2.6**

### Gap 4: Mamba/SSM 用于通用显式mesh生成
- **类型**: Method transfer blank (Blank 4)
- **描述**: 将Mamba-SSM的近线性复杂度优势应用于通用（可变拓扑）mesh token序列生成，替代或辅助Transformer，突破长序列瓶颈。
- **证据**: MeshMamba验证了Mamba处理10K+顶点mesh的能力（6-9x加速），但限于固定拓扑；Long-LRM验证了Mamba2+TF hybrid处理250K tokens的可行性。
- **可行性**: **Medium** — Mamba的选择性扫描机制对有序序列（如mesh tokens按z-y-x排序）的适用性需要验证。vertex serialization策略（MeshMamba的body-part UV sorting）可能不直接适用于通用mesh。
- **潜在影响**: **Medium** — 加速效果显著，但可能不改变生成质量，主要是efficiency improvement。
- **新颖性**: **High** — 通用mesh生成中零先例。
- **置信度**: Medium
- **Score**: Medium(2)×0.4 + Medium(2)×0.4 + High(3)×0.2 = **2.2**

### Gap 5: 纹理感知的显式mesh端到端生成
- **类型**: Feature gap (Blank 6)
- **描述**: 在AR mesh生成序列中同时预测顶点坐标和vertex color/UV coordinates，实现geometry+texture一体化生成。
- **证据**: 所有AR mesh方法均geometry-only；隐式方法（TRELLIS, Hunyuan3D）通过渲染loss或单独texture model生成纹理。
- **可行性**: **Low-Medium** — vertex color增加3个channel/vertex（RGB），序列长度增加~33%。UV coordinates更复杂，需要UV unwrapping作为数据预处理。但已有SF3D展示了可微分UV unwrapping。
- **潜在影响**: **High** — 巨大的practical impact，解决mesh生成的"最后一公里"问题。
- **新颖性**: **Medium** — 概念上直观（只是增加每个token的预测维度），技术挑战在于数据和训练。
- **置信度**: Medium
- **Score**: Low-Medium(1.5)×0.4 + High(3)×0.4 + Medium(2)×0.2 = **2.2**

### Gap 6: 连续 vs 离散 mesh tokens 的公平对比
- **类型**: Contradiction resolution (Contradiction 1)
- **描述**: 在统一框架（相同数据、相同backbone、相同评测）中系统对比连续KL-latent和离散VQ tokens用于显式mesh生成的质量、速度、可控性差异。
- **证据**: MeshCraft在面级别验证了连续>>离散（99.66% vs 65.12%），但其他维度（generation diversity, controllability, scaling）未比较。
- **可行性**: **High** — 实验设计清晰，核心是消融研究。
- **潜在影响**: **Medium** — 学术价值高但可能不构成独立顶会论文，更适合作为大系统的ablation。
- **新颖性**: **Medium** — 问题已知，只是缺少系统性对比。
- **置信度**: High
- **Score**: High(3)×0.4 + Medium(2)×0.4 + Medium(2)×0.2 = **2.4**

### Gap 7: LOD Progressive Mesh Generation 的 Conditional 验证
- **类型**: Validation gap (Blank 7)
- **描述**: 将ARMesh的LOD渐进生成范式扩展到conditional generation（image/text→mesh），在Objaverse-scale数据上验证，并与BPT/Meshtron等SOTA对比。
- **证据**: ARMesh仅unconditional; G3PT/SAR3D在隐式空间有conditional验证。
- **可行性**: **High** — ARMesh codebase可扩展，只需加入cross-attention条件机制。
- **潜在影响**: **Medium** — 如果成功，为coarse-to-fine mesh generation提供关键验证。
- **新颖性**: **Medium** — 方法已知（ARMesh），缺的是条件化验证。
- **置信度**: High
- **Score**: High(3)×0.4 + Medium(2)×0.4 + Medium(2)×0.2 = **2.4**

---

## Gap Ranking Summary

| Rank | Gap | Score | Type |
|------|-----|-------|------|
| 1 | Unified Large Mesh Model (多模态+重建/生成统一) | **3.0** | Combination |
| 2 | 显式mesh Scaling Laws | **2.6** | Scale |
| 3 | AR+Flow Coarse-to-fine Mesh Generation | **2.6** | Combination |
| 4 | 连续 vs 离散 mesh tokens 公平对比 | **2.4** | Contradiction |
| 5 | LOD Progressive Conditional Mesh Generation | **2.4** | Validation |
| 6 | Mamba/SSM 通用mesh生成 | **2.2** | Transfer |
| 7 | 纹理感知端到端mesh生成 | **2.2** | Feature |

**Top 3 Gaps 进入 Idea Generation**:
1. **Unified Large Mesh Model** — 最高可行性×影响×新颖性组合
2. **显式mesh Scaling Laws** — 学术价值最高，可与Gap 1合并验证
3. **AR+Flow Coarse-to-fine** — 技术创新度最高，解决核心效率瓶颈

---

## 研究日志

**第1轮 SEARCH**: 针对"重建与生成的统一模型"，执行6次搜索（3×acd_search + 3×web_search），找到15+篇新论文。关键发现：Unifi3D（统一评测框架，非统一模型）、EdgeRunner（ArAE+LDM hybrid）、ShapeLLM-Omni（首个ChatGPT-4o-style 3D LLM）。

**第1轮 READ**: 深入阅读EdgeRunner（compact tokenization + ArAE + latent diffusion pipeline）、ShapeLLM-Omni（Qwen-2.5-VL + 3D VQVAE, 1024 tokens, 7B params）、Unifi3D（5种表示的公平对比）。确认：统一重建+生成在显式mesh上仍是空白。

**第1轮 REFLECT**: 发现EdgeRunner的ArAE架构是统一模型的关键技术组件——将变长mesh压缩为定长latent，使diffusion可介入。发现ShapeLLM-Omni虽然统一但几何质量受限（64³ voxel）。新gap: tokenization压缩策略的系统对比。

**第2轮 SEARCH**: 针对"mesh生成scaling和高效架构"，执行6次搜索，找到BPT/DeepMesh（75%压缩, 8K面）、ARMesh（next-LOD prediction）、FastMesh（vertex-face解耦, 8x加速）、AssetFormer（模块化3D生成）、TreeMeshGPT。

**第2轮 READ**: 深入阅读BPT（block-wise indexing + patch aggregation, 75% compression milestone）、ARMesh（GSlim→PSC→AR learning, coarse-to-fine从单点到完整mesh）、FastMesh（vertex AR + bidirectional TF face prediction, 23% tokens of BPT）。确认：三种完全不同的scaling策略（压缩/架构/解耦）都有效但未对比。

**第2轮 REFLECT**: 识别Contradiction 4（压缩 vs 架构 vs 解耦）。发现ARMesh的LOD范式是突破性的但缺conditional验证。识别Blank 3（AR+Diffusion混合）和Blank 7（LOD conditional验证）。

**第3轮 SEARCH**: 针对"多模态条件3D mesh生成和统一框架"，执行6次搜索，找到TriMM（collaborative multi-modal coding）、CraftsMan3D（DiT粗生成+法线细化）、CAD-MLLM。

**第3轮 READ**: 阅读TriMM（RGB+RGBD+PC多模态编码→triplane diffusion）、CraftsMan3D（3D native DiT + normal-based geometry refiner, coarse-to-fine pipeline）。确认：多模态条件在隐式3D有进展，但显式mesh上仍空白。

**第3轮 EVALUATE**: 7个Gaps全部确认为真实研究空白，每个至少有3+篇论文的交叉证据支撑。完成排名。

---

## 未验证问题

1. **Quad mesh / 混合多边形mesh的AR生成**: Meshtron支持quad-dominant但未有后续工作深入，市场需求（游戏/影视行业偏好quad mesh）是否足以支撑独立研究方向？
2. **非流形(non-manifold)和开放surface mesh的生成**: 几乎所有方法假设watertight manifold mesh，但实际应用中大量mesh是non-watertight的（如衣服、头发）。ARMesh的simplicial complex框架理论上可处理，但未验证。
3. **Scene-level mesh generation**: OctGPT展示了初步的scene-level能力（Synthetic Rooms, 5K scenes），但场景级显式mesh生成是一个更大的未探索领域。

---

## 结论

本Gap Analysis基于75+篇论文的系统性分析，识别了7个经过验证的研究空白。**核心发现**：

1. 显式mesh生成领域正处于从"小规模category-conditional"向"大规模multi-modal conditional"转型的关键节点
2. Tokenization压缩、架构效率、生成范式（face-by-face vs coarse-to-fine）三大技术路线正在激烈竞争，尚无统一胜者
3. 隐式3D生成（TRELLIS/TripoSG/G3PT）在规模化和多模态上已大幅领先，但显式mesh的拓扑质量优势（artist-like tessellation）使其不可替代
4. **最大的机会**在于将隐式3D生成的成功经验（scaling, multi-modal conditioning, coarse-to-fine）迁移到显式mesh领域，同时保持mesh的拓扑优势

**Top 3 Gaps ready for Idea Generation**: Unified Large Mesh Model (3.0), Scaling Laws for Explicit Mesh (2.6), AR+Flow Coarse-to-fine (2.6)
