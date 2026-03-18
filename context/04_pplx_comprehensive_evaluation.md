<!-- markdownlint-disable -->
```bash
pplx: 88/78/82
gpt:  95/63/97
kimi: 89/88/87
综合: 3h -- 31.81￥ -- 90.6/75.0/88.8
```

- [AI Scientist Gap Analysis 验证报告：Large Mesh Model (LMM) 研究空白核实](#ai-scientist-gap-analysis-验证报告large-mesh-model-lmm-研究空白核实)
  - [执行摘要](#执行摘要)
  - [Gap 1：统一多模态条件的大规模显式 mesh 生成模型](#gap-1统一多模态条件的大规模显式-mesh-生成模型)
    - [原文断言](#原文断言)
    - [核实结果：**基本属实，但需补充最新进展**](#核实结果基本属实但需补充最新进展)
  - [Gap 2：显式 mesh 生成中从未验证 Scaling Laws](#gap-2显式-mesh-生成中从未验证-scaling-laws)
    - [原文断言](#原文断言-1)
    - [核实结果：**属实**](#核实结果属实)
  - [Gap 3：AR + Diffusion 混合架构从未应用于显式 mesh 生成](#gap-3ar--diffusion-混合架构从未应用于显式-mesh-生成)
    - [原文断言](#原文断言-2)
    - [核实结果：**基本属实，但需关注 TSSR 和 MARS**](#核实结果基本属实但需关注-tssr-和-mars)
  - [Gap 4：Mamba/SSM 从未应用于通用 mesh 生成（仅限固定拓扑）](#gap-4mambassm-从未应用于通用-mesh-生成仅限固定拓扑)
    - [原文断言](#原文断言-3)
    - [核实结果：**属实，但需区分两个 "MeshMamba"**](#核实结果属实但需区分两个-meshmamba)
  - [Gap 5：多模态条件生成显式 mesh 的统一框架不存在](#gap-5多模态条件生成显式-mesh-的统一框架不存在)
    - [原文断言](#原文断言-4)
    - [核实结果：**基本属实，但需更新 Nautilus 的进展**](#核实结果基本属实但需更新-nautilus-的进展)
  - [Gap 6：纹理感知的显式 mesh 端到端生成从未实现](#gap-6纹理感知的显式-mesh-端到端生成从未实现)
    - [原文断言](#原文断言-5)
    - [核实结果：**部分被挑战，需修订**](#核实结果部分被挑战需修订)
  - [Gap 7：Coarse-to-fine 显式 mesh 生成缺少 conditional generation 验证](#gap-7coarse-to-fine-显式-mesh-生成缺少-conditional-generation-验证)
    - [原文断言](#原文断言-6)
    - [核实结果：**属实，但需补充 MARS 的进展**](#核实结果属实但需补充-mars-的进展)
  - [矛盾识别的核实](#矛盾识别的核实)
    - [Contradiction 1：离散 vs 连续 tokens](#contradiction-1离散-vs-连续-tokens)
    - [Contradiction 2：数据 scaling vs 架构 scaling](#contradiction-2数据-scaling-vs-架构-scaling)
    - [Contradiction 3：Face-by-face vs Coarse-to-fine](#contradiction-3face-by-face-vs-coarse-to-fine)
    - [Contradiction 4：Tokenization 压缩 vs 信息保真](#contradiction-4tokenization-压缩-vs-信息保真)
  - [需要更新的遗漏论文](#需要更新的遗漏论文)
  - [综合评估](#综合评估)
- [Idea 文档质量评估](#idea-文档质量评估)
  - [综合质量得分：**78/100**](#综合质量得分78100)
    - [分项打分](#分项打分)
    - [关键问题](#关键问题)
    - [总结](#总结)
- [Experiment Design 文档评估](#experiment-design-文档评估)
  - [实验设计质量得分：**82/100**](#实验设计质量得分82100)
    - [分项打分](#分项打分-1)
    - [关键优点](#关键优点)
    - [关键问题](#关键问题-1)
    - [总评](#总评)

# AI Scientist Gap Analysis 验证报告：Large Mesh Model (LMM) 研究空白核实

## 执行摘要

本报告对 AI Scientist 调研产出的 Large Mesh Model (LMM) Gap Analysis 文档中识别的 **7 个研究空白（Gaps）** 进行了系统性核实。通过检索 2024–2026 年的最新论文和技术进展，逐项验证每个 Gap 是否仍然成立，以及是否有新工作已经填补或部分填补了这些空白。

**总体结论：7 个 Gap 中，5 个基本属实、1 个部分被挑战（需修订）、1 个已有显著进展但仍未完全解决。** 该 Gap Analysis 整体质量较高，对文献的覆盖面广泛，核心判断大方向正确，但个别细节需要根据最新进展进行更新。

***

## Gap 1：统一多模态条件的大规模显式 mesh 生成模型

### 原文断言

没有任何模型能同时实现：(a) 给定清晰多视角图像→高保真重建显式 mesh；(b) 给定文本/草图→创造性生成显式 mesh。SAR3D 用不同模型、ShapeLLM-Omni 输出 64³ voxel 质量极低、LLaMA-Mesh 最多 500 面。

### 核实结果：**基本属实，但需补充最新进展**

ShapeLLM-Omni 确实是基于 Qwen-2.5-VL-7B 架构、使用 8192 entry 的 3D VQVAE 将 64³ voxel 压缩为 1024 tokens 进行统一生成，其几何分辨率确实受限。LLaMA-Mesh 以纯文本 OBJ 格式表示 mesh，最多约 500 面，64 级量化。[1][2][3]

但需要补充几项新进展：
- **FACE (2026.03)** 提出 "one-face-one-token" 策略，压缩比达 0.11（比以往最优翻倍），并通过 VecSet encoder + Autoregressive Autoencoder 实现了高保真 mesh 重建 + 下游 image-to-mesh 生成。不过 FACE 仍是单模态（image→mesh），并非多模态统一模型。[4][5]
- **MeshWeaver (2025.09)** 利用稀疏体素引导实现顶点级 tokenization，支持最多 16K 面的 mesh 生成，压缩率 18%。但该工作也未实现多模态条件统一。[6]
- **UniLat3D (2025)** 提出将几何与外观编码在统一 latent 空间中的单阶段 3D 生成框架，但输出通过 3D Gaussians/meshes 解码，仍属于隐式中间表示路线。[7]

**结论**：Gap 1 **仍然成立**。目前没有单一模型能够在显式 mesh 表示上同时支持多模态条件（text/image/point cloud）的高质量（>4K 面）重建与生成。FACE 和 MeshWeaver 推进了显式 mesh 的可处理规模，但都是单条件模态。

***

## Gap 2：显式 mesh 生成中从未验证 Scaling Laws

### 原文断言

G3PT 首次在 3D 生成中展示 power-law scaling（0.1B→1.5B），但其 tokenizer 输出 occupancy field tokens，最终 mesh 通过 Marching Cubes 提取。所有显式 mesh AR 方法（MeshGPT/BPT/EdgeRunner/FastMesh）均在 <1B 参数、<100K 数据规模运作，无 scaling 行为研究。

### 核实结果：**属实**

G3PT 确实是第一个在 3D 生成中展示 scaling law 的工作，使用 cross-scale vector quantization 和 cross-scale autoregressive modeling，在 0.1B/0.5B/1.5B 参数规模上展示了 power-law 行为。但其表示是 occupancy field tokens，非显式 mesh 拓扑。[8][9]

ScaMo (CVPR 2025) 在 motion generation 领域验证了 scaling law，但与 mesh 生成无关。TripoSG (2025) 将模型从 1.5B 扩展并使用 2M 高质量 3D 数据训练，证明了数据 scaling 的重要性，但表示为 SDF latent，非显式 mesh，且未绘制 scaling curve。[10][11][12]

在显式 mesh 的 AR 方法中：Meshtron (1.1B 参数) 是最大规模的显式 mesh 模型，但仅有两个规模点（0.5B 和 1.1B），未做系统性 scaling 分析。BPT/DeepMesh 验证了面数（即数据复杂度）的 scaling 效果，但非模型参数 scaling。

**结论**：Gap 2 **完全属实**。显式 mesh 生成领域至今没有 scaling law 的系统性验证。

***

## Gap 3：AR + Diffusion 混合架构从未应用于显式 mesh 生成

### 原文断言

在显式 mesh 生成中，纯 AR（MeshGPT/Meshtron/BPT）与纯 Diffusion/Flow（MeshCraft）各有优劣，但从未有方法尝试 "AR 做粗粒度结构规划 + Diffusion 做细粒度几何细化"。

### 核实结果：**基本属实，但需关注 TSSR 和 MARS**

新发现的相关工作：
- **TSSR (Topology Sculptor, Shape Refiner, 2025.10)** 使用 Discrete Diffusion Model 实现了 mesh 生成的 "拓扑雕刻 + 形状精炼" 两阶段解耦。它将 DDM 分为 topology sculpting 和 shape refinement 两个阶段，在某种意义上实现了 coarse-to-fine，且支持并行生成达 10,000 面。但 TSSR 全程使用 DDM（非 AR+Diffusion 混合），两个阶段都基于离散扩散。[13][14]
- **MARS (2025.02)** 提出 multi-LOD mesh autoregressive model，通过 next-LOD token prediction 实现粗到细的 mesh detailization。但 MARS 全程使用 AR（非 AR+Diffusion 混合），且侧重于 detailization 而非完整生成。[15][16]
- **FastMesh (2025.08)** 的 vertex AR + bidirectional transformer face generation 是一种解耦但非 AR+Diffusion 混合。[17][18]

**结论**：Gap 3 **基本属实**。TSSR 虽然实现了两阶段解耦（粗拓扑→细几何），但两阶段都是 DDM，不是 AR+Diffusion 混合。MARS 的 multi-LOD AR 也不是 AR+Diffusion 混合。真正的 "AR 生成粗结构 mesh + Flow/Diffusion 细化几何" 在显式 mesh 上确实从未被尝试。

***

## Gap 4：Mamba/SSM 从未应用于通用 mesh 生成（仅限固定拓扑）

### 原文断言

MeshMamba 展示了 Mamba-SSM 替代 Transformer 处理 10K+ 顶点 mesh 的能力（6-9x 加速），但严格限于固定拓扑（SMPL/SMPL-X 模板）。将 Mamba 用于通用（可变拓扑）mesh 生成从未被探索。

### 核实结果：**属实，但需区分两个 "MeshMamba"**

搜索发现存在两篇不同的 "MeshMamba" 论文：
1. **MeshMamba (ICCV 2025, Yoshiyasu et al.)** — 用于 articulated 3D mesh generation and reconstruction（人体），基于 Mamba SSM，处理 10K+ 顶点，但使用 SMPL/SMPL-X **固定拓扑模板**。其论文明确限制为 fixed-topology meshes with tight clothing。[19][20]
2. **Mesh Mamba (2024, 显著性预测)** — 一个用于 mesh 显著性预测的 SSM 模型，而非生成模型，与 mesh 生成无关。[21]

在通用（可变拓扑、可变面数）mesh 生成领域，确实没有任何方法使用 Mamba/SSM 架构。所有 SOTA 方法（BPT、FastMesh、EdgeRunner、Nautilus 等）均使用 Transformer。

**结论**：Gap 4 **完全属实**。Mamba/SSM 在通用 mesh 生成领域是零先例。

***

## Gap 5：多模态条件生成显式 mesh 的统一框架不存在

### 原文断言

无任何方法能在高质量显式 mesh 生成（>4K 面）中同时支持 text/image/point cloud/multi-view 多种输入。

### 核实结果：**基本属实，但需更新 Nautilus 的进展**

**Nautilus (ICCV 2025)** 实现了 point cloud conditioned + image conditioned 的显式 mesh 生成，支持高达 5,000 面，使用 locality-aware tokenization 将压缩比提升至 0.275。这是显式 mesh 领域在条件多样性上的重要进步，但仍不支持 text conditioning。[22][23]

**FACE (2026.03)** 展示了 image-to-mesh 的生成能力，但同样不支持 text 或 point cloud。[24]

**MeshWeaver (2025.09)** 使用稀疏体素引导，支持 16K 面，但条件也限于 point cloud/shape。[6]

**Mesh Silksong (2025.07)** 实现了约 22% 的压缩率和 manifold topology 保证，支持 point cloud 条件，但无 text 支持。[25][26]

| 方法 | 最大面数 | Text | Image | Point Cloud | Multi-view |
|------|---------|------|-------|-------------|------------|
| BPT/DeepMesh | 8,000+ | ❌ | ✅ | ✅ | ❌ |
| Nautilus | 5,000 | ❌ | ✅ | ✅ | ❌ |
| MeshWeaver | 16,000 | ❌ | ❌ | ✅ | ❌ |
| FastMesh | 4,000+ | ❌ | ❌ | ✅ | ❌ |
| FACE | — | ❌ | ✅ | ❌ | ❌ |
| LLaMA-Mesh | ~500 | ✅ | ✅ | ❌ | ❌ |
| MeshLLM | 1,500+ | ✅ | ✅ | ❌ | ❌ |

**结论**：Gap 5 **属实**。高面数（>4K）显式 mesh 生成 + 同时支持 text/image/point cloud 多模态条件的统一框架确实不存在。支持多模态的方法（LLaMA-Mesh、MeshLLM）面数极低；面数高的方法（BPT、Nautilus、MeshWeaver）条件单一。

***

## Gap 6：纹理感知的显式 mesh 端到端生成从未实现

### 原文断言

所有显式 mesh AR 方法仅生成几何（顶点坐标+面连接），不生成材质/纹理。将 vertex color 或 UV coordinates 纳入 AR mesh token 序列从未被尝试。

### 核实结果：**部分被挑战，需修订**

原文断言 "从未被尝试" 需要修正：
- **InstantTexture** 等工具已展示将 vertex-colored meshes 转换为 UV-mapped textured meshes 的后处理方案，但这是后处理而非端到端生成。[27][28]
- **ARM (Appearance Reconstruction Model, 2024.11)** 展示了通过 UV texture space 的 back-projection 实现高质量纹理重建，但属于重建而非生成。[29]
- **DreamMesh** 使用 coarse-to-fine 方案在 triangle mesh 上联合操控 geometry 和 texture，但通过 SDS 优化而非 AR 生成。[30]
- **Point-UV Diffusion** 使用 coarse-to-fine pipeline 在 3D mesh 上生成纹理，但纹理生成与几何生成是分离的两个步骤。[31][32]

核实确认：在 **AR mesh 生成序列中同时预测顶点坐标和 vertex color/UV** 的端到端方法确实不存在。所有 2024–2026 年的显式 mesh AR 方法（BPT、FastMesh、EdgeRunner、Nautilus、Mesh Silksong、FACE、TreeMeshGPT）仍然是 geometry-only。

但 Gap Analysis 原文中 "将 vertex color 或 UV coordinates 纳入 AR mesh token 序列从未被尝试" 这一说法的措辞略需软化，因为 **SF3D** 已在重建模型中实现了可微分 UV unwrapping + texture generation，以及 **SeamCrafter (2025.09)** 已开始探索 artist-style UV unwrapping 的 AR 生成。[33][34]

**结论**：Gap 6 **大方向属实，但需修订措辞**。在 AR mesh 生成中同时预测 geometry + texture 确实是空白，但已有零星相关探索（SeamCrafter 做 UV 生成、SF3D 做可微分 UV）开始接近这个方向。原文 "从未被尝试" 应改为 "在 AR mesh 几何+纹理联合生成中从未实现端到端训练"。

***

## Gap 7：Coarse-to-fine 显式 mesh 生成缺少 conditional generation 验证

### 原文断言

ARMesh 提出了 LOD-based progressive mesh generation，但仅验证了 unconditional generation，未验证 text/image conditioned 生成，未在 Objaverse-scale 数据上训练。

### 核实结果：**属实，但需补充 MARS 的进展**

ARMesh (2025.09) 确实仅展示了 unconditional generation（ShapeNet categories），其论文的 Section 5.3 和 Section 6 均未涉及 conditional generation。[35][36]

MARS (2025.02) 的 multi-LOD autoregressive model 在 3D Shape Detailization benchmark 上取得了 SOTA，但其 "conditional" 是以粗 mesh 为输入的 detailization，而非 image/text→mesh 的多模态条件生成。[15]

TSSR (2025.10) 虽然实现了 coarse-to-fine（topology sculpting → shape refinement），但主要是 point cloud conditioned，尚未展示 text/image conditioning。[14]

TreeMeshGPT (CVPR 2025) 使用 autoregressive tree sequencing 支持 point cloud conditioning 和 5,500 面，但不是 LOD progressive 范式。[37]

**结论**：Gap 7 **属实**。LOD 渐进式显式 mesh 生成在 conditional generation（尤其是 image/text 条件）上仍未被验证。MARS 和 TSSR 各有部分进展但都不满足完整条件。

***

## 矛盾识别的核实

### Contradiction 1：离散 vs 连续 tokens

原文判断 "没有工作在同一框架内公平对比两者用于显式 mesh 生成" — **属实**。MeshCraft 的面级对比（99.66% vs 65.12%）仍是唯一的直接证据。FACE (2026.03) 使用连续 latent 表示取得 SOTA 重建质量，但未与离散 token 方案在生成任务上对比。[4]

### Contradiction 2：数据 scaling vs 架构 scaling

原文判断 "显式 mesh 领域同时验证数据+架构 scaling 的工作不存在" — **属实**。TripoSG 强调数据 scaling 3 倍于架构 scaling 的影响，G3PT 展示模型参数 scaling law，但两者都不是显式 mesh 表示。[10][8]

### Contradiction 3：Face-by-face vs Coarse-to-fine

原文判断 "两种范式在同一条件下未被公平对比" — **属实**。ARMesh 的 LOD 方法仅有 unconditional 验证。[36]

### Contradiction 4：Tokenization 压缩 vs 信息保真

原文列举的三种策略（BPT 压缩 token / Meshtron 高效架构 / FastMesh 解耦生成）需补充 2025 年的新进展。Nautilus（压缩比 0.275, 5K 面）、Mesh Silksong（压缩比 ~22%, manifold 保证）、FACE（压缩比 0.11, SOTA 重建） 推动了 tokenization 竞赛持续白热化。但 **四种策略仍未在相同规模和数据集上对比** 这一核心矛盾仍然成立。[24][22][25]

***

## 需要更新的遗漏论文

Gap Analysis 中未覆盖的重要新工作：

| 论文 | 时间 | 核心贡献 | 与哪个 Gap 相关 |
|------|------|---------|---------------|
| **FACE** | 2026.03 | one-face-one-token, 压缩比 0.11, ARAE 框架 | Gap 1/6 |
| **MeshWeaver** | 2025.09 | 稀疏体素引导, 16K 面, 压缩率 18% | Gap 1/5 |
| **TSSR** | 2025.10 | DDM-based topology sculpting + shape refinement, 10K 面 | Gap 3 |
| **Mesh Silksong** | 2025.07 | 每顶点仅访问一次, 压缩率 22%, manifold 保证 | Contradiction 4 |
| **UniLat3D** | 2025 | 几何+外观统一 latent, 单阶段 3D 生成 | Gap 1/6 |
| **SeamCrafter** | 2025.09 | Artist-style UV unwrapping 的 AR 生成 | Gap 6 |

***

## 综合评估

| Gap | 原文评分 | 验证结论 | 修正建议 |
|-----|---------|---------|---------|
| Gap 1: 统一大规模显式 mesh 模型 | 3.0 | ✅ 属实 | 补充 FACE、MeshWeaver 的进展 |
| Gap 2: 显式 mesh Scaling Laws | 2.6 | ✅ 完全属实 | 无需修正 |
| Gap 3: AR+Flow Coarse-to-fine | 2.6 | ✅ 基本属实 | 补充 TSSR（DDM 两阶段但非 AR+Diffusion 混合） |
| Gap 4: Mamba/SSM 通用 mesh 生成 | 2.2 | ✅ 完全属实 | 区分两篇 MeshMamba |
| Gap 5: 多模态条件统一框架 | — | ✅ 属实 | 更新 Nautilus（支持 PC+Image, 5K 面） |
| Gap 6: 纹理感知端到端生成 | 2.2 | ⚠️ 需修订 | 软化 "从未尝试" 措辞，补充 SeamCrafter |
| Gap 7: LOD Conditional 验证 | 2.4 | ✅ 属实 | 补充 MARS 的 detailization 进展 |

**评分合理性**：Gap 排名的相对顺序合理。Gap 1（统一模型，3.0 分）确实是最具影响力和可行性的方向；Gap 2 和 Gap 3 并列第二也符合实际——scaling law 的学术价值极高，AR+Flow 混合的技术创新度最高。

**整体评价**：该 Gap Analysis 的核心结论 **可信度高**，7 个 Gap 中 5 个完全属实、1 个基本属实需轻微修订、1 个需补充最新进展。对 75+ 篇论文的覆盖面在当时（2025 年中期）是全面的，但 2025 下半年至 2026 初的新工作（TSSR、FACE、MeshWeaver、Mesh Silksong）需要纳入更新版本。

**整体属实度**：88%.

# Idea 文档质量评估

Now I have sufficient information. Let me provide the analysis directly as requested (a simple scoring question).

基于我之前的 Gap Analysis 验证以及对最新论文的进一步核查，以下是对该 AI Scientist 生成的 Idea 文档的质量评估：

## 综合质量得分：**78/100**

### 分项打分

| 维度 | 得分 | 说明 |
|------|------|------|
| **技术洞察力** | 85/100 | 核心洞察极为精准——将 EdgeRunner 的 ArAE 识别为"桥梁架构"，连接显式 mesh 与隐式 latent 世界，这是整个 idea 中最有价值的判断 [openreview](https://openreview.net/forum?id=81cta3WQVI) |
| **文献基础** | 80/100 | 80+ 篇论文覆盖面好，对 EdgeRunner、BPT、TRELLIS 等关键论文的技术理解准确。但遗漏了 **FACE (2026.03)** [arxiv](https://arxiv.org/abs/2603.01515)、**VertexRegen (ICCV 2025)** [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VertexRegen_Mesh_Generation_with_Continuous_Level_of_Detail_ICCV_2025_paper.pdf) 和 **Nautilus (ICCV 2025)** [arxiv](https://arxiv.org/abs/2501.14317) 三篇重要竞争对手 |
| **新颖性** | 70/100 | Idea 1 (MeshFoundation) 本质是**组合创新**——将 ArAE + BPT + Rectified Flow + 多模态条件拼装在一起。各组件均已有验证 [iclr](https://iclr.cc/virtual/2025/poster/30789)，创新点在组合而非基础方法突破。自评 7/10 novelty 基本合理 |
| **可行性判断** | 82/100 | 对计算资源需求（~64 A100）、技术风险（主要在集成和 scaling 而非单点假设）的评估务实。Idea 2 的 patch boundary 风险识别准确 |
| **竞争态势判断** | 65/100 | **最大扣分项**。FACE (2026.03) 已实现压缩比 0.11（比 BPT 的 0.25 优 2 倍）+ ARAE 框架 + image-to-mesh latent diffusion [arxiv](https://arxiv.org/abs/2603.01515)，这与 Idea 1 的 MAE+LDM pipeline **高度重叠**。VertexRegen (ICCV 2025) 也实现了 coarse-to-fine progressive generation [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VertexRegen_Mesh_Generation_with_Continuous_Level_of_Detail_ICCV_2025_paper.pdf)，与 Idea 2 的 LOD 思路部分重叠。如果今天开始做，FACE 已经抢先发表了核心 pipeline |
| **实验设计** | 80/100 | 4 个模型规模 × 3 个数据规模 = 12 个实验的 scaling law 设计合理。评估指标（Chamfer Distance + FID + 用户研究 + scaling curve）覆盖全面 |
| **想法层次感** | 82/100 | 3 个保留 + 2 个丢弃的筛选过程透明；MeshFoundation v2（合并 Idea 1+2 的 coarse-to-fine latent）展示了深入思考；丢弃 LLM-native 和 GNN face 的理由充分 |
| **落地风险评估** | 75/100 | 对 Idea 1 过于乐观（自评 Confidence: High），没有充分考虑：(a) ArAE decoder 在高面数(8K+)上的重建质量衰减；(b) 多模态 CFG 在 mesh latent 空间中的调参难度；(c) BPT tokenization 的 manifold 假设限制 |

### 关键问题

**1. Idea 1 (MeshFoundation) 的差异化不足** — FACE (2026.03) 已发表了 ARAE + latent diffusion 的核心 pipeline，且压缩比（0.11）远优于 BPT（0.25）。如果要推进 Idea 1，必须在**多模态条件统一**和 **scaling law** 两个方向上做出显著超越 FACE 的差异化贡献。 [arxiv](https://arxiv.org/html/2603.01515v1)

**2. Idea 2 (MeshCascade) 的竞品** — VertexRegen (ICCV 2025) 和 ARMesh (NeurIPS 2025) 都已实现了 coarse-to-fine progressive mesh generation。Idea 2 的核心差异——patch-wise flow refinement——未被验证，且 patch boundary 一致性是真实工程难题。 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VertexRegen_Mesh_Generation_with_Continuous_Level_of_Detail_ICCV_2025_paper.pdf)

**3. Idea 3 (MeshSSM) 评估准确** — 自评 31/50 并建议不独立投稿而作为 backbone ablation，这个判断很合理。

### 总结

> **78分**——属于**良好但非优秀**的 idea generation。技术洞察力强，组件选择合理，但**对竞争态势的感知有滞后**（未覆盖 FACE、VertexRegen 等 2025 下半年至 2026 年初的关键竞品）。如果基于此 idea 立即开始研究，需要首先重新定位 vs FACE 的差异化，否则 novelty 会大幅下降。建议将 **多模态条件统一** + **scaling law 验证** 作为核心卖点，tokenizer 选型从 BPT 更新为 FACE 级别的压缩方案。

# Experiment Design 文档评估

Now I have enough information to evaluate the experiment design comprehensively.

基于对实验设计文档的详细阅读和相关论文的核实，以下是我的评估：

## 实验设计质量得分：**82/100**

### 分项打分

| 维度 | 得分 | 说明 |
|------|------|------|
| **假设清晰度** | 85/100 | 4 个假设明确可证伪（统一重建/生成、多模态条件、Scaling Law、SOTA 质量），结构严谨 |
| **架构设计合理性** | 80/100 | 4 组件分解清晰（BPT tokenizer → MAE → Cascaded LFT → AR decoder），模块化设计便于调试。z_coarse/z_fine 的多尺度 latent 设计有理论依据 |
| **Baseline 覆盖度** | 90/100 | 11 个 baseline 分三层（primary/secondary/tertiary），覆盖了 DeepMesh (ICCV 2025) [arxiv](https://arxiv.org/html/2503.15265)、EdgeRunner (ICLR 2025) [openreview](https://openreview.net/forum?id=81cta3WQVI)、Meshtron、TRELLIS、TripoSG 等核心竞品。分层设计合理 |
| **评估指标体系** | 88/100 | CD/F-Score/NC/ECD + FID/CLIP + IoU + Triangle Quality + User Study，覆盖几何精度、生成质量、mesh 拓扑质量和人类感知。评估协议对齐 DeepMesh 标准 [youtube](https://www.youtube.com/watch?v=OpobsCvXt-E) |
| **Ablation 设计** | 92/100 | **最强亮点**。6 组 ablation 直接对应核心贡献和已识别矛盾（多尺度 latent、L2 vs KL vs VQ、多模态、Flow vs DDPM、tokenizer 对比、Scaling Law），设计精准 |
| **数据集选择** | 85/100 | Objaverse++ (500K curated) [arxiv](https://arxiv.org/html/2504.07334v1) 作为训练集合理，GSO/ABO 作为 OOD 测试集体现泛化验证意识。但 Objaverse++ 实际是 CVPR 2025 Workshop 论文 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025W/CDEL/html/Lin_Objaverse_Curated_3D_Object_Dataset_with_Quality_Annotations_ICCVW_2025_paper.html)，并非 ICCV 2025 正文 |
| **资源估算** | 78/100 | 计算量估算基本合理（MVP: 8×A100×1周, 完整: 64×A100×5周），但偏乐观——MAE 从 5000 tokens 压缩到 256 tokens 的重建质量需大量调参，3-5 天可能不够 |
| **时间线规划** | 70/100 | 18 周完成从数据预处理到论文提交**过于激进**。Phase 3 (Scaling Study) 12 个实验在 5 周内完成需要极高的并行度和零失败率，实际可能需要 8-10 周 |
| **风险管理** | 85/100 | 6 个风险识别准确，尤其 Risk 1 (MAE 重建瓶颈) 和 Risk 6 (竞品抢先) 是真实核心风险。Mitigation 方案务实（如 MAE 维度调整、先 arXiv 占位） |
| **竞品感知更新** | 65/100 | **主要扣分项**。未纳入 FACE (2026.03) [arxiv](https://arxiv.org/abs/2603.01515) 作为 baseline——FACE 的 ARAE + latent diffusion pipeline 与 MeshFoundation 架构高度相似，压缩比 0.11 远优于 BPT 的 0.25。也未纳入 VertexRegen (ICCV 2025) [vertexregen.github](https://vertexregen.github.io) 和 Nautilus (ICCV 2025) [arxiv](https://arxiv.org/abs/2501.14317) |

### 关键优点

1. **Ablation 设计是整个文档最强部分** — Ablation 2 (L2/KL/VQ 对比) 直接解决 Gap Analysis 中的 Contradiction 1；Ablation 6 (Scaling Law) 直接贡献 Gap 2。这种 "ablation = contribution" 的设计思路非常成熟
2. **MVP 策略明智** — 先在 ShapeNet 上 8×A100×1 周验证 pipeline，再扩展到 Objaverse 的分阶段策略降低了资金风险
3. **已有代码/框架表格** 详列了每个组件的开源来源和 License，工程准备充分 [github](https://github.com/zhaorw02/DeepMesh)
4. **FlashMesh speculative decoding 和 DeepMesh DPO** 被正确标记为 "未解决问题/future work"，没有贪多嚼不烂 [arxiv](https://arxiv.org/html/2503.15265)

### 关键问题

1. **FACE 缺失是最大隐患** — FACE (2026.03) 实现了 one-face-one-token + ARAE + latent diffusion，压缩比 0.11 大幅优于 BPT (0.25)。如果 reviewer 问 "为什么不用 FACE tokenizer 替代 BPT"，实验设计中没有答案。**建议将 FACE tokenizer 加入 Ablation 5** [arxiv](https://arxiv.org/html/2603.01515v1)
2. **18 周时间线不现实** — Phase 1 的 MAE 训练（从 5000 tokens 压缩到 256）涉及大量超参搜索（M=128/256/512, D=32/64/128），3 天不够。合理估计应为 **24-28 周**
3. **Cascaded Flow 的 Stage A→B 错误累积** — 文档承认了 Risk 2 但 mitigation 过于简略。TRELLIS 的成功是单阶段 flow，DALL-E 3 的 cascaded 方案基于像素空间有丰富先验，而 mesh latent 空间的 cascaded flow **没有任何先例验证**，风险应标为 High 而非 Medium
4. **Objaverse++ 实际是 Workshop paper** — 文档将其标注为 ICCV 2025，实际是 ICCV 2025 **Workshop** 论文，约 500K（后期标注扩展到 790K）模型。经过 quality filtering 后可用于训练的高质量 mesh 数量可能显著少于预期 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025W/CDEL/html/Lin_Objaverse_Curated_3D_Object_Dataset_with_Quality_Annotations_ICCVW_2025_paper.html)

### 总评

> **82分** — 属于**良好的实验设计**，模块化架构分解清晰、ablation 对齐 contribution、风险管理意识到位。主要短板是**竞品更新滞后**（缺 FACE/VertexRegen/Nautilus）和**时间线过于乐观**。如果补充 FACE 作为 tokenizer baseline、将时间线调整为 24-28 周，可升至 88 分左右。
