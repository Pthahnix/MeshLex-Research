<!-- markdownlint-disable -->
# Experiment Plan: MeshFoundation v2

- [Experiment Plan: MeshFoundation v2](#experiment-plan-meshfoundation-v2)
  - [执行摘要](#执行摘要)
  - [Hypothesis](#hypothesis)
  - [Method](#method)
    - [核心架构: MeshFoundation v2](#核心架构-meshfoundation-v2)
      - [Component 1: BPT Mesh Tokenizer (existing, frozen)](#component-1-bpt-mesh-tokenizer-existing-frozen)
      - [Component 2: Mesh Auto-Encoder (MAE) — 核心创新](#component-2-mesh-auto-encoder-mae--核心创新)
      - [Component 3: Cascaded Latent Flow Transformer (LFT) — 核心生成模型](#component-3-cascaded-latent-flow-transformer-lft--核心生成模型)
      - [Component 4: AR Mesh Decoder (from MAE, frozen at generation time)](#component-4-ar-mesh-decoder-from-mae-frozen-at-generation-time)
    - [已有代码/框架](#已有代码框架)
  - [Evaluation](#evaluation)
    - [Datasets](#datasets)
      - [Training Data](#training-data)
      - [Evaluation Data](#evaluation-data)
    - [Baselines](#baselines)
      - [Primary Baselines (显式 mesh 方法, 直接竞争)](#primary-baselines-显式-mesh-方法-直接竞争)
      - [Secondary Baselines (隐式方法, 跨范式对比)](#secondary-baselines-隐式方法-跨范式对比)
      - [Tertiary Baselines (LLM-based, 参考对比)](#tertiary-baselines-llm-based-参考对比)
    - [Metrics](#metrics)
      - [Primary Metrics (核心评估)](#primary-metrics-核心评估)
      - [Secondary Metrics (生成质量)](#secondary-metrics-生成质量)
      - [Mesh-Specific Metrics](#mesh-specific-metrics)
      - [User Study](#user-study)
    - [Ablations](#ablations)
      - [Ablation 1: Multi-scale Latent Design](#ablation-1-multi-scale-latent-design)
      - [Ablation 2: Latent Regularization](#ablation-2-latent-regularization)
      - [Ablation 3: Conditioning Modality](#ablation-3-conditioning-modality)
      - [Ablation 4: Flow vs Diffusion](#ablation-4-flow-vs-diffusion)
      - [Ablation 5: Tokenizer Comparison](#ablation-5-tokenizer-comparison)
      - [Ablation 6: Scaling Law Study (核心贡献)](#ablation-6-scaling-law-study-核心贡献)
  - [Resources](#resources)
    - [Compute](#compute)
      - [Phase 1: MAE Training](#phase-1-mae-training)
      - [Phase 2: LFT Training (Base Model, 400M)](#phase-2-lft-training-base-model-400m)
      - [Phase 3: Scaling Law Study (12 experiments)](#phase-3-scaling-law-study-12-experiments)
      - [Phase 4: Evaluation \& Ablations](#phase-4-evaluation--ablations)
      - [Total Compute Estimate](#total-compute-estimate)
    - [Storage](#storage)
    - [Frameworks](#frameworks)
    - [Timeline](#timeline)
      - [Phase 1: Data Preparation \& MAE (Weeks 1-3)](#phase-1-data-preparation--mae-weeks-1-3)
      - [Phase 2: Latent Flow Transformer (Weeks 4-7)](#phase-2-latent-flow-transformer-weeks-4-7)
      - [Phase 3: Scaling Law Study (Weeks 8-12)](#phase-3-scaling-law-study-weeks-8-12)
      - [Phase 4: Evaluation \& Ablations (Weeks 13-16)](#phase-4-evaluation--ablations-weeks-13-16)
      - [Phase 5: Paper Submission (Weeks 17-18)](#phase-5-paper-submission-weeks-17-18)
  - [Risks](#risks)
    - [Risk 1: MAE Reconstruction Quality Bottleneck](#risk-1-mae-reconstruction-quality-bottleneck)
    - [Risk 2: Cascaded Flow Training Instability](#risk-2-cascaded-flow-training-instability)
    - [Risk 3: Compute Resource Insufficiency for Scaling Study](#risk-3-compute-resource-insufficiency-for-scaling-study)
    - [Risk 4: BPT Tokenizer Limitation](#risk-4-bpt-tokenizer-limitation)
    - [Risk 5: Multi-modal Conditioning Conflicts](#risk-5-multi-modal-conditioning-conflicts)
    - [Risk 6: Novel Competitors During Research Period](#risk-6-novel-competitors-during-research-period)
  - [研究日志](#研究日志)
  - [未解决问题](#未解决问题)
  - [传递给 Experiment Execution 的关键信息](#传递给-experiment-execution-的关键信息)

## 执行摘要

- **总迭代轮数**: 2 (Experiment Design stage)
- **总阅读论文数**: 95+ (含 Stages 1-3)
- **初始问题数**: 4
- **已完成组件**: Hypothesis, Method, Datasets, Baselines, Metrics, Ablations, Resources, Risks
- **停止原因**: 实验设计已完整
- **新增关键论文 (Stage 4)**: DeepMesh (ICCV 2025, RL for mesh), FlashMesh (speculative decoding), Objaverse++ (curated dataset), Scaling Rectified Flow Transformers (SD3), TreeMeshGPT, iFlame, Mesh RAG

---

## Hypothesis

**核心假设**: 通过将 BPT 压缩的显式 mesh token 序列编码为固定长度的多尺度 latent code（z_coarse + z_fine），并在该 latent 空间上训练 Cascaded Rectified Flow Transformer，可以实现：
1. **统一重建与生成**: 单一模型同时支持确定性重建（encoder→decoder）和创造性生成（flow sampling→decoder）
2. **多模态条件**: 在 latent flow 空间通过 cross-attention 统一支持 image/text/point cloud 条件
3. **Scaling Law**: 显式 mesh 生成首次展示 power-law scaling 行为（参数规模 150M→1.5B）
4. **SOTA 质量**: 在 4K+ 面显式 mesh 生成上超越现有方法（BPT/DeepMesh、EdgeRunner、Meshtron）

**预期结果**: MeshFoundation v2 在 Chamfer Distance、F-Score、FID 等指标上达到或超越 SOTA，同时首次在显式 mesh 领域验证 scaling law。

---

## Method

### 核心架构: MeshFoundation v2

#### Component 1: BPT Mesh Tokenizer (existing, frozen)

- **来源**: BPT/DeepMesh (Weng et al., 2024)
- **功能**: 将三角mesh序列化为1D token序列，75%压缩率
- **支持面数**: 8K+ faces
- **量化分辨率**: 128 bins（坐标量化）
- **代码**: 使用 BPT 开源实现（GitHub: whaohan/bpt）
- **修改**: 无需修改，直接作为预处理pipeline

#### Component 2: Mesh Auto-Encoder (MAE) — 核心创新

- **Encoder**: Transformer encoder with learnable query tokens
  - 输入: BPT mesh token embeddings (variable length, up to ~5000 tokens for 8K faces)
  - Query tokens: M_coarse=64 + M_fine=256 个可学习query token
  - Cross-attention: query tokens attend to mesh token embeddings
  - 输出: z_coarse ∈ R^{64×D}, z_fine ∈ R^{256×D}, D=64
  - 正则化: L2 regularization on z (following EdgeRunner, not VQ)
  - Encoder layers: 12 transformer blocks

- **Decoder**: Autoregressive transformer
  - 输入: z_fine as prefix (256 tokens) → generate BPT mesh tokens
  - Architecture: 24 transformer blocks, hidden dim 1024, 16 attention heads
  - 训练: Cross-entropy loss on next-token prediction + L2 loss on latent
  - Causal mask: z_fine tokens can attend to each other (bidirectional), mesh tokens are causal

- **Multi-scale design**: z_coarse captures global structure (overall shape, proportions); z_fine captures local details (surface curvature, edge sharpness). z_coarse is derived from z_fine via a learned linear projection (z_coarse = proj(mean_pool(z_fine, groups=4))).

- **训练策略**:
  - Stage 1: Train MAE end-to-end on mesh reconstruction
  - Loss: L_recon = CE(predicted BPT tokens, ground truth) + λ_L2 * ||z||²
  - λ_L2 = 1e-4 (following EdgeRunner)
  - 数据: All training meshes, face count 500-8000

#### Component 3: Cascaded Latent Flow Transformer (LFT) — 核心生成模型

- **Stage A — Coarse Flow (z_coarse generation)**:
  - Model: Rectified Flow Transformer (DiT-style)
  - 输入: Noise ε ~ N(0,I) ∈ R^{64×D}
  - 输出: z_coarse ∈ R^{64×D}
  - Conditioning: Multi-modal cross-attention (image/text/point cloud)
  - ODE sampling: 50 steps (Euler method)
  - 参数规模: ~200M (base), scales to 400M/800M/1.5B

- **Stage B — Fine Flow (z_fine generation, conditioned on z_coarse)**:
  - Model: Rectified Flow Transformer
  - 输入: Noise ε ~ N(0,I) ∈ R^{256×D}, concatenated with z_coarse (expanded to 256 via repeat+project)
  - 输出: z_fine ∈ R^{256×D}
  - Conditioning: z_coarse (self-attention) + multi-modal cross-attention
  - ODE sampling: 50 steps
  - 参数规模: ~200M (base), scales with Stage A

- **Multi-modal conditioning**:
  - Image: DINOv2 ViT-L/14 patch tokens (pixel-level, 257 tokens) + CLIP ViT-L/14 class token (semantic, 1 token)
  - Text: CLIP text encoder (77 tokens max)
  - Point cloud: PointNet++ encoder → 256 tokens (following Michelangelo/CLAY design)
  - Injection: Cross-attention in every other transformer block
  - CFG: 10% random drop of each condition independently during training; guidance scale w=3.0-7.5 at inference

- **Rectified Flow formulation** (following TRELLIS/TripoSG/SD3):
  - Forward: x_t = (1-t) * x_0 + t * ε, t ∈ [0,1]
  - Velocity prediction: v_θ(x_t, t, c) predicts ε - x_0
  - Loss: L_flow = E[||v_θ(x_t, t, c) - (ε - x_0)||²]
  - Logit-normal time sampling (following SD3): bias towards t=0.5 for perceptually relevant scales

#### Component 4: AR Mesh Decoder (from MAE, frozen at generation time)

- 与 MAE decoder 相同，在生成时接收 LFT 生成的 z_fine，自回归解码为 BPT tokens → 显式 mesh
- Top-p sampling (p=0.9) + temperature (τ=0.8) for generation
- Nucleus sampling ensures diversity while maintaining quality

### 已有代码/框架

| Component | Source | License |
|-----------|--------|---------|
| BPT tokenizer | github.com/whaohan/bpt | Apache 2.0 |
| EdgeRunner ArAE | github.com/NVlabs/EdgeRunner | CC-BY-NC-SA 4.0 |
| Rectified Flow | torchdiffeq / custom implementation | MIT |
| DINOv2 | github.com/facebookresearch/dinov2 | Apache 2.0 |
| CLIP | github.com/openai/CLIP | MIT |
| PointNet++ | github.com/erikwijmans/Pointnet2_PyTorch | MIT |
| DiT architecture | github.com/facebookresearch/DiT | CC-BY-NC 4.0 |

**注意**: EdgeRunner (CC-BY-NC-SA 4.0) 仅用于架构设计参考，不直接使用其代码。我们的 MAE 将从头实现。

---

## Evaluation

### Datasets

#### Training Data

1. **Objaverse++ High-Quality Subset** (~200K meshes after filtering)
   - Source: Objaverse++ (Lin et al., ICCV 2025), 500K curated from Objaverse 1.0
   - 过滤标准: aesthetic quality >= "high", non-monochromatic, non-transparent, single object, manifold mesh
   - 面数范围: 500-8000 faces (BPT 支持范围)
   - 预处理: BPT tokenization, normalize to unit cube, watertight check
   - **Why**: Objaverse++ 已验证 quality > quantity (用户研究中 50K 高质量 > 100K 随机)

2. **Objaverse Full Filtered** (~500K meshes, for scaling study)
   - Objaverse++ 全部 500K curated models
   - 用于 scaling law 实验的大数据集

3. **ShapeNet Core** (51K, 13 categories)
   - 用于与早期方法 (MeshGPT, MeshXL) 的公平对比
   - 仅用于 category-conditional 实验

#### Evaluation Data

4. **GSO (Google Scanned Objects)** (~1K high-quality scans)
   - OOD (out-of-distribution) 测试集
   - 高质量真实扫描，验证泛化能力
   - Source: Google Research

5. **Objaverse Test Split** (~20K, held out from training)
   - In-distribution 测试集
   - 随机 10% hold-out

6. **ABO (Amazon Berkeley Objects)** (~8K)
   - 电商场景的工业级 3D 模型
   - 用于 domain transfer 评测

### Baselines

#### Primary Baselines (显式 mesh 方法, 直接竞争)

1. **BPT/DeepMesh** (Weng et al., 2024 + Zhao et al., ICCV 2025)
   - SOTA AR mesh generation, BPT tokenization, 8K+ faces
   - DeepMesh adds RL (DPO) for quality alignment
   - Conditions: point cloud, image
   - 对比原因: 使用相同 tokenizer (BPT)，最公平的 apple-to-apple 对比

2. **EdgeRunner** (Tang et al., ICLR 2025)
   - ArAE + latent diffusion, 4K faces, 512³ resolution
   - Conditions: point cloud, image (via LDM)
   - 对比原因: 最接近我们架构的已有方法（ArAE 思想的原始来源）

3. **Meshtron** (Hao et al., 2024)
   - Hourglass Transformer, 1.1B params, 64K faces
   - Conditions: point cloud
   - 对比原因: 当前最大规模 AR mesh 模型

4. **MeshGPT** (Siddiqui et al., 2023)
   - VQ-VAE + GPT-2, ~800 faces
   - Conditions: category
   - 对比原因: 经典 baseline

5. **FastMesh** (2025)
   - Vertex-face decoupled generation, 8x speedup
   - Conditions: point cloud
   - 对比原因: 代表不同 scaling 策略（解耦 vs 压缩）

6. **TreeMeshGPT** (Lionar et al., 2025)
   - Tree-structured sequencing, ~22% compression
   - Conditions: point cloud
   - 对比原因: 最新的 tokenization 竞品

#### Secondary Baselines (隐式方法, 跨范式对比)

7. **TRELLIS** (Xiang et al., 2024)
   - Sparse voxel SLat + Rectified Flow, up to 2B params
   - Conditions: image, text
   - 对比原因: 隐式方法 SOTA，验证显式 mesh 是否能匹配隐式质量

8. **TripoSG** (2025)
   - SDF + hybrid hash-grid, 1.5B-4B (MoE)
   - Conditions: image
   - 对比原因: 数据 scaling 代表

9. **Direct3D** (Wu et al., 2024)
   - Triplane LDM, ~675M DiT-XL
   - Conditions: image (DINO+CLIP dual conditioning)
   - 对比原因: 我们的 multi-modal conditioning 设计参考

#### Tertiary Baselines (LLM-based, 参考对比)

10. **LLaMA-Mesh** (Wang et al., 2024) — text/image → mesh, 质量较低，参考
11. **MeshLLM** (Fang et al., 2025) — primitive-mesh decomposition, LLM-based

### Metrics

#### Primary Metrics (核心评估)

1. **Chamfer Distance (CD)** — 几何精度
   - 在 mesh 表面均匀采样 10K 点，计算 L2 CD
   - Lower is better
   - 标准: ×10³ 报告

2. **F-Score (F1@τ)** — 阈值精度
   - τ = 0.01 (unit cube normalized)
   - Precision + Recall 的调和平均
   - Higher is better

3. **Normal Consistency (NC)** — 表面质量
   - 采样点法向量的余弦相似度
   - Higher is better

4. **Edge Chamfer Distance (ECD)** — mesh 边缘质量
   - 针对 mesh 边（非面）的 Chamfer Distance
   - 评估 tessellation 质量和边缘锐度
   - Lower is better

#### Secondary Metrics (生成质量)

5. **FID (Fréchet Inception Distance)** — 生成分布质量
   - 在 24 个视角渲染 512×512 图像
   - 使用 InceptionV3 特征
   - Lower is better

6. **CLIP Score** — 条件一致性
   - Image condition: CLIP similarity between input image and rendered views
   - Text condition: CLIP similarity between text prompt and rendered views
   - Higher is better

7. **IoU (Intersection over Union)** — 体积精度
   - 256³ voxelization, 计算交集/并集
   - 与隐式方法对比时使用
   - Higher is better

#### Mesh-Specific Metrics

8. **Triangle Quality Score** — mesh 拓扑质量
   - Average triangle aspect ratio (closer to 1.0 = equilateral = better)
   - Minimum angle distribution
   - 评估生成的 mesh 是否具有 artist-like 拓扑

9. **Mesh Completeness Rate** — 生成完整度
   - 非截断 mesh 的比例（面数达到目标的 90%+）
   - 百分比
   - Higher is better

#### User Study

10. **Pairwise Preference** — 人类感知
    - 20 名评审者对比 MeshFoundation vs top-3 baselines
    - 每对 50 个样本（文本提示 + 随机选择的两个结果）
    - 评审维度: 整体质量、几何精度、拓扑质量
    - 报告 win rate %

### Ablations

#### Ablation 1: Multi-scale Latent Design
- **Variant A (Full)**: z_coarse (64×D) + z_fine (256×D), cascaded flow
- **Variant B (Single-scale)**: z_fine only (256×D), single-stage flow
- **Variant C (Small latent)**: z only (64×D), single-stage flow
- **Expected insight**: Multi-scale cascaded 设计是否比 single-scale 更好？多大的 latent 是最优平衡点？

#### Ablation 2: Latent Regularization
- **Variant A (L2)**: L2 regularization on latent (EdgeRunner style)
- **Variant B (KL)**: KL divergence regularization (VAE style)
- **Variant C (VQ)**: Vector quantization (G3PT style)
- **Expected insight**: 验证 Contradiction 1（连续 vs 离散），在统一框架中公平对比

#### Ablation 3: Conditioning Modality
- **Image only** (DINOv2 + CLIP)
- **Text only** (CLIP text)
- **Point cloud only** (PointNet++)
- **All modalities**
- **Expected insight**: 各条件模态的贡献，多模态是否互补

#### Ablation 4: Flow vs Diffusion
- **Variant A (Rectified Flow)**: 默认设计
- **Variant B (DDPM)**: 标准去噪扩散，1000 steps
- **Variant C (EDM)**: Karras et al. 2022 formulation
- **Expected insight**: 验证 Rectified Flow 在 mesh latent 空间是否同样优于 DDPM（如同在隐式 3D 中的表现）

#### Ablation 5: Tokenizer Comparison
- **BPT** (默认, 75% compression)
- **EdgeRunner tokenizer** (EdgeBreaker-based, 50% compression)
- **DeepMesh tokenizer** (improved BPT variant)
- **Expected insight**: tokenizer 对最终生成质量的影响

#### Ablation 6: Scaling Law Study (核心贡献)
- **MAE fixed scale** (冻结 MAE, 仅 scale LFT)
- **LFT model sizes**: 150M, 400M, 800M, 1.5B parameters
- **Data sizes**: 80K, 200K, 500K meshes (from Objaverse++)
- **Plot**: Loss vs compute (FLOPs) curves for all 12 combinations (4 sizes × 3 data scales)
- **Verify**: Power-law behavior L(C) = αC^{-β}
- **Compare**: With G3PT's implicit token scaling curve (if data available)
- **Expected insight**: 首次验证显式 mesh 生成是否遵循 scaling law，确定最优 compute-data allocation

---

## Resources

### Compute

#### Phase 1: MAE Training
- **GPU**: 8× A100 80GB (or 8× H100 80GB)
- **Training time**: ~3-5 days
- **Justification**: EdgeRunner trains ArAE on 64 A100s for 1 week; our MAE is simpler (no EdgeBreaker tokenization), and we train on smaller data initially. 8 GPUs × 5 days is conservative.
- **Batch size**: 4 per GPU × 8 GPUs = 32 effective
- **Mixed precision**: bf16

#### Phase 2: LFT Training (Base Model, 400M)
- **GPU**: 16× A100 80GB
- **Training time**: ~5-7 days
- **Justification**: TRELLIS trains 342M model on 8 A100s; Direct3D trains 675M DiT on 32 A100s. Our 400M model on 16 GPUs is proportional.
- **Batch size**: 8 per GPU × 16 GPUs = 128 effective
- **Mixed precision**: bf16
- **Optimizer**: AdamW, lr=1e-4, cosine decay, 10K warmup steps

#### Phase 3: Scaling Law Study (12 experiments)
- **GPU**: 32-64× A100 80GB (shared across experiments, sequential)
- **Training time**: ~3-4 weeks total
  - 150M models (×3 data scales): 1-2 days each = 6 days
  - 400M models (×3 data scales): 3-5 days each = 15 days
  - 800M models (×3 data scales): 5-7 days each = 21 days
  - 1.5B models (×3 data scales): 7-10 days each = 30 days
  - **With parallelism** (2-3 experiments concurrent on 64 GPUs): ~3-4 weeks
- **Justification**: G3PT used 136 H20 GPUs for 2 weeks for 3 model sizes. Our 12 experiments on 64 A100s for 3-4 weeks is comparable.

#### Phase 4: Evaluation & Ablations
- **GPU**: 8× A100 80GB
- **Time**: ~1 week (inference + rendering + metric computation)

#### Total Compute Estimate
- **Minimum viable** (base model only): 16× A100 × 2 weeks ≈ 5,376 GPU-hours
- **Full paper** (with scaling study): 64× A100 × 5 weeks ≈ 53,760 GPU-hours
- **Cost estimate** (RunPod A100 80GB @ $1.64/hr):
  - Minimum: ~$8,800
  - Full: ~$88,000
- **Alternative**: Apply for academic compute grants (Google TPU Research Cloud, NVIDIA Academic Program)

### Storage
- **Dataset**: Objaverse++ filtered meshes (~200K × ~1MB avg) ≈ 200 GB
- **BPT tokenized data**: ~50 GB (compressed token sequences)
- **Rendered images** (for FID): 200K × 24 views × 512² ≈ 2 TB
- **Model checkpoints**: 12 experiments × 5 checkpoints × ~6GB ≈ 360 GB
- **Total**: ~3 TB

### Frameworks
- **PyTorch** 2.x (with torch.compile for training speedup)
- **Key libraries**:
  - `transformers` (HuggingFace, for DiT architecture reference)
  - `torchdiffeq` (ODE solvers for rectified flow)
  - `trimesh` / `pymeshlab` (mesh processing)
  - `pytorch3d` (differentiable rendering for evaluation)
  - `open_clip` (CLIP/DINOv2 feature extraction)
  - `kaolin` (NVIDIA 3D deep learning library)
  - `deepspeed` / `FSDP` (distributed training)

### Timeline

#### Phase 1: Data Preparation & MAE (Weeks 1-3)
- Week 1: Download Objaverse++, quality filtering, BPT preprocessing pipeline
- Week 2: Implement MAE (encoder + decoder), unit tests
- Week 3: Train MAE, validate reconstruction quality

#### Phase 2: Latent Flow Transformer (Weeks 4-7)
- Week 4: Implement Cascaded LFT (Stage A coarse + Stage B fine)
- Week 5: Implement multi-modal conditioning (DINOv2/CLIP/PointNet++)
- Week 6-7: Train base model (400M), iterate on hyperparameters

#### Phase 3: Scaling Law Study (Weeks 8-12)
- Week 8-9: Train 150M and 400M models on 3 data scales (6 experiments)
- Week 10-11: Train 800M models on 3 data scales (3 experiments)
- Week 12: Train 1.5B models on 3 data scales (3 experiments, may extend)

#### Phase 4: Evaluation & Ablations (Weeks 13-16)
- Week 13: Full evaluation on all test sets, baseline comparisons
- Week 14: Ablation studies (6 ablation groups)
- Week 15: User study, scaling curve analysis, visualization
- Week 16: Paper writing, figure generation

#### Phase 5: Paper Submission (Weeks 17-18)
- Draft + revision cycle
- Target: CVPR 2027 (deadline ~Nov 2026) or NeurIPS 2026 (deadline ~May 2026)

---

## Risks

### Risk 1: MAE Reconstruction Quality Bottleneck
- **描述**: MAE 将 variable-length mesh (5000+ tokens) 压缩为固定 256 tokens 可能导致信息损失，重建质量不足
- **Likelihood**: Medium
- **Impact**: High (整个 pipeline 依赖 MAE 质量)
- **Mitigation**:
  1. 先在小数据集 (ShapeNet) 上验证 MAE 重建质量
  2. 调整 latent 维度 (M=128/256/512, D=32/64/128) 寻找最优
  3. 如果 256 tokens 不够，增加到 512 (仍比原始 5000+ tokens 短 10x)
  4. 添加 reconstruction loss warm-up: 先高 L2 权重，逐渐降低
  5. Fallback: 使用 EdgeRunner 的 ArAE 架构（已验证 4K faces 重建质量）

### Risk 2: Cascaded Flow Training Instability
- **描述**: 两阶段 flow (coarse→fine) 的级联训练可能不稳定，Stage B 依赖 Stage A 的质量
- **Likelihood**: Medium
- **Impact**: Medium
- **Mitigation**:
  1. 先独立训练 Stage A 和 Stage B (用 ground truth z_coarse 训练 Stage B)
  2. 然后 joint fine-tuning with small learning rate
  3. Fallback: 放弃 cascaded design，使用 single-stage flow on z_fine only (Ablation Variant B)

### Risk 3: Compute Resource Insufficiency for Scaling Study
- **描述**: 完整 scaling law study (12 experiments) 需要 ~54K GPU-hours，可能超出预算
- **Likelihood**: High
- **Impact**: Medium (scaling study 是重要贡献但非必需)
- **Mitigation**:
  1. 优先训练 3 个规模 (150M/400M/800M) 在 2 个数据规模上 = 6 experiments
  2. 1.5B 模型仅训练最大数据规模 = 1 experiment
  3. 总共 7 experiments，减少 42% 计算量
  4. 使用 chinchilla-optimal compute allocation 减少浪费训练
  5. 申请学术计算资源 (NVIDIA Academic, Google TRC)

### Risk 4: BPT Tokenizer Limitation
- **描述**: BPT 的 128-bin 量化可能限制几何精度上限
- **Likelihood**: Low-Medium
- **Impact**: Medium
- **Mitigation**:
  1. 尝试提高量化分辨率到 256 或 512 bins (如 EdgeRunner 使用 512)
  2. 使用 DeepMesh 的改进 tokenizer (2025, higher efficiency)
  3. 在 ablation 中对比不同 tokenizer

### Risk 5: Multi-modal Conditioning Conflicts
- **描述**: 同时注入 image/text/point cloud 条件可能导致条件冲突或模式坍塌
- **Likelihood**: Low
- **Impact**: Medium
- **Mitigation**:
  1. 独立 CFG dropout (10% per modality) 确保模型对每种条件都有鲁棒性
  2. 先单模态训练，再多模态 fine-tuning
  3. 使用 separate cross-attention layers per modality (不共享参数)

### Risk 6: Novel Competitors During Research Period
- **描述**: 2025-2026 mesh generation 领域发展极快（DeepMesh/FlashMesh/TreeMeshGPT 均为 2025 新作），可能在我们完成前出现类似工作
- **Likelihood**: Medium-High
- **Impact**: High (novelty 被抢占)
- **Mitigation**:
  1. **速度优先**: 先发布 base model 结果 (不等 scaling study)，在 arXiv 占位
  2. **差异化**: scaling law study 是独特贡献，即使架构被类似工作覆盖，scaling 分析仍有价值
  3. 持续关注 arXiv 新发表，及时调整 contribution 定位

---

## 研究日志

**第1轮 SEARCH**: 针对4个初始问题（评估方法、baseline、数据集、实验参数），执行6次并行搜索（3×acd_search + 3×web_search）。找到15+篇新论文。关键发现：DeepMesh (ICCV 2025, RL+DPO for mesh quality alignment, Hourglass TF)、FlashMesh (speculative decoding, 2x speedup)、Objaverse++ (500K curated, quality > quantity 实验验证)、TreeMeshGPT (tree sequencing, 22% compression)、iFlame (linear attention for mesh)、Mesh RAG (retrieval augmentation for mesh AR)。

**第1轮 READ**: 通过 paper_content 和 web_content 获取 DeepMesh、FlashMesh、Objaverse++ 的完整实验设置。DeepMesh 使用 Hourglass TF + BPT tokenization + DPO，训练设置与 Meshtron 类似。FlashMesh 实现 2x 推理加速。Objaverse++ 提供 500K curated 3D models，用户研究证实 quality > quantity。

**第1轮 REFLECT**: 确认实验设计的核心组件：
- 数据集: Objaverse++ (500K curated) 作为主要训练集，解决了数据质量问题
- Baselines: BPT/DeepMesh (ICCV 2025 SOTA) 成为最重要的直接竞品
- 评估: DeepMesh 使用 CD/F-Score/NC/ECD + 用户研究，成为标准评估协议
- 参数: EdgeRunner 在 64 A100 上训练约 1 周，提供了计算量基准

**第2轮 SEARCH**: 搜索 Rectified Flow 训练超参数和 EdgeRunner 训练细节。找到 Scaling Rectified Flow Transformers (SD3 paper)，确认 logit-normal time sampling 和 scaling trends。EdgeRunner GitHub README 确认: 64 A100 GPUs, batch size 4, ~1 week training。

**第2轮 EVALUATE**: 4个初始问题全部回答完成：
1. 评估方法: CD/F-Score/NC/ECD + FID/CLIP + IoU + User Study (基于 DeepMesh/BPT 标准)
2. Baselines: 11 个方法，分3层 (primary/secondary/tertiary)
3. 数据集: Objaverse++ (train) + GSO/ABO (OOD test) + ShapeNet (legacy comparison)
4. 实验参数: MAE 8×A100×5d, LFT 16×A100×7d, Scaling 64×A100×4w

---

## 未解决问题

1. **DeepMesh DPO 是否可以集成**: DeepMesh 引入 RL (DPO) 提升 mesh 质量，是否可在 MeshFoundation 的 AR decoder 中使用？需要进一步评估 DPO 对 latent flow pipeline 的适用性。

2. **FlashMesh speculative decoding 集成**: AR decoder 部分是否可以使用 FlashMesh 的 structured speculation 加速推理？这是 orthogonal optimization，可作为 future work。

3. **Texture generation 扩展**: 当前设计仅生成几何。是否可在 z_fine latent 中编码 vertex color 信息？留作 MeshFoundation v3 的扩展方向。

4. **Non-manifold mesh 处理**: BPT 假设 watertight manifold mesh。实际 Objaverse 数据中非流形 mesh 占比不明，需要在数据预处理阶段统计和处理。

---

## 传递给 Experiment Execution 的关键信息

**最小可行实验 (MVP)**:
1. 在 ShapeNet (51K, 5 categories) 上训练 MAE + 单阶段 LFT (400M) → 验证 pipeline 可行性
2. 预计: 8× A100 × 1 week, ~$2,000

**完整实验**:
1. Objaverse++ 数据预处理 + BPT tokenization
2. MAE training (8× A100 × 5 days)
3. LFT training - base 400M (16× A100 × 7 days)
4. Scaling study - 7-12 experiments (64× A100 × 3-4 weeks)
5. Evaluation + ablations + user study (8× A100 × 1 week)

**目标 venue**: CVPR 2027 / NeurIPS 2026 / ICCV 2027 (CCF-A)
