# MeshLex Theory-Driven Design Spec

**Date**: 2026-03-20
**Target**: NeurIPS 2027
**Type**: Theory-Heavy (Type R)

---

## 1. Executive Summary

MeshLex v1 发现了一个关键现象：用 ~500 个 patch token 可以跨 1156 个类别泛化，CD ratio 仅 1.019×。这暗示 mesh patch 空间存在 universal structure。

本 spec 提出一个**理论驱动的系统重设计**方案：

1. **实验层**：系统性表征 mesh patch 空间的等价结构（相变曲线 + 幂律分布 + 跨数据集 universality）
2. **理论层**：从 Gauss-Bonnet 定理推导高曲率 patch 数量上界，用 Lean4 形式化证明
3. **系统层**：基于理论设计曲率感知的非均匀 codebook

---

## 2. Problem Statement

### 2.1 核心问题

> 为什么 mesh 的拓扑结构会存在 universal vocabulary？

已知：
- MeshLex v1 用 512 个 token 实现 CD ratio 1.019× 的跨类别泛化
- 如果 patch 类型均匀分布，512 个 token 无法支撑这个泛化能力

推断：
- Patch 类型频率一定遵循重尾分布（幂律）
- 少数几种 patch 类型极高频，大量类型极稀少

### 2.2 核心假说

**Hypothesis**: Mesh patch 频率遵循幂律分布 f(r) ∝ r^{-α}，其来源是物理世界 3D 表面的微分几何结构。

从微分几何推导：

光滑 2-流形的每一点可用 Gaussian 曲率 K 和 Mean 曲率 H 描述，分成 5 种本质类型：

| 类型 | K 和 H | 物理直觉 | 预期频率 |
|------|--------|----------|----------|
| Flat | K≈0, H≈0 | 平面、盒子的面 | 极高频 |
| Elliptic | K>0 | 球面、凸起 | 高频 |
| Cylindrical | K≈0, H≠0 | 圆柱侧面 | 中频 |
| Hyperbolic | K<0 | 凹陷、马鞍形 | 低频 |
| Corner | K>>0 | 立方体角 | 稀少 |

物理世界的 3D 物体绝大部分表面是平的或微凸的，尖锐特征是少数。离散化后自然导致 flat/elliptic patch 极高频。

### 2.3 Gauss-Bonnet 约束

Gauss-Bonnet 定理：任意闭合亏格-0 曲面的总 Gaussian 曲率 = 4π（固定值）。

这意味着：
- Elliptic patch (K>0) 和 hyperbolic patch (K<0) 的数量受约束
- Flat patch (K≈0) 可以任意多 → 高频
- 高曲率 patch 的总量被锁住 → 低频

---

## 3. Core Contributions

### 3.1 Five Contributions

| # | 贡献 | 类型 |
|---|------|------|
| C1 | 首个 mesh patch 空间等价结构的系统性实验研究：相变 + 幂律 + universality | 实验理论 |
| C2 | 连接离散微分几何：Gaussian 曲率类型 ↔ patch 频率分布 | 理论 |
| C3 | Lean4 形式化证明：Gauss-Bonnet → 高曲率 patch 上界 O(χ/κ) | 形式化 |
| C4 | 曲率感知非均匀 codebook：理论直接驱动系统设计 | 方法 |
| C5 | 全量数据（97K meshes）上的完整生成+重建实验，逼近 SOTA | 应用 |

### 3.2 Contribution Chain

```
[理论 C1,C2,C3]
    实验测量相变曲线 + 幂律分布
    Lean4 证明高曲率 patch 上界
         ↓ 解释了为什么 MeshLex v1 能泛化
[方法 C4]
    曲率感知的非均匀 codebook
         ↓ 用更好的 codebook
[应用 C5]
    全量数据重训，逼近 SOTA
```

---

## 4. Theory Layer Design

### 4.1 Experiment: Rate-Distortion Curve + Phase Transitions

**核心想法**：VQ-VAE 本身就是 ε-等价聚类的近似。训练 K 个 codeword 的 VQ，就是在找 K 个等价类。

**实验设计**：

```
For K ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}:
    1. 训练 VQ-VAE (fixed architecture, only K varies)
    2. 记录 mean reconstruction CD = distortion D(K)
    3. 画出 D(K) vs K 曲线
```

**期望信号**：
- D(K) 下降不均匀，在某些 K* 处急剧下降
- 这些 K* 是"新增 token 刚好能区分一种新曲率类型"的阈值

**可视化**：

```
D(K)
high |  \
     |   \
     |    *---- 相变：突然快速下降
     |         \
     |          *---- 另一个相变
     |               \___________
low  +---32--128--512--2048---> K
```

### 4.2 Experiment: Power Law Distribution + Universality

**操作**：用训练好的 K=1024 VQ-VAE 编码所有 patch，统计 token 频率。

```python
# 对全量 dataset 的所有 patches
token_freq = Counter()
for patch in all_patches:
    token_id = vqvae.encode(patch)
    token_freq[token_id] += 1

# 画 Zipf 图 (log-log)
sorted_freqs = sorted(token_freq.values(), reverse=True)
# 期望: log(freq) ∝ -α × log(rank)
```

**跨数据集检验（Universality 关键实验）**：
1. 在 Objaverse only 上训练 → 得到 codebook
2. 用这个 codebook 编码 ShapeNet 的 patches（零 fine-tuning）
3. 如果 ShapeNet 的频率分布与 Objaverse 幂律一致（同一个 α），是 universal 的有力证据

### 4.3 Experiment: Curvature Correlation

**目标**：把 token 频率和 patch 的离散 Gaussian 曲率关联。

**离散曲率计算**（angle defect）：

$$K_v = 2\pi - \sum_{f \ni v} \theta_{vf}$$

每个 patch 的曲率 $\bar{K}_P = \text{mean}_{v \in P} |K_v|$。

**期望结果**：

| Token rank（按频率） | 平均 $\bar{K}_P$ |
|----------------------|------------------|
| Top 10%（最高频） | ≈ 0（平面状） |
| Middle 40% | 小正值（微曲） |
| Bottom 50%（最低频） | 大正值（角点/边缘） |

### 4.4 Lean4 Formalization Design

**目标定理**：

> 对于一个封闭三角网格 M，满足 angle defect 超过 κ 的顶点数，上界是 $2\pi|\chi(M)| / \kappa$。与网格的面数无关，只取决于拓扑类型（χ）。

**证明链**：

**Step 1**: [公理，引用 Descartes/Gauss-Bonnet 经典结论]

$$\sum_{v} K_v = 2\pi \cdot \chi(M)$$

**Step 2**: [Lean4 证明 — 有限求和的 Markov 不等式]

设 $S = \sum_v K_v$，若 $S \geq 0$，则：

$$|\{v : K_v > \kappa\}| \cdot \kappa < \sum_{v: K_v > \kappa} K_v \leq S$$

$$\Rightarrow |\{v : K_v > \kappa\}| < S / \kappa$$

**Step 3**: [Lean4 证明 — 组合两步]

$$|\{v : K_v > \kappa\}| \leq 2\pi|\chi(M)| / \kappa \quad \square$$

**Lean4 实现骨架**：

```lean
theorem finite_sum_markov {α : Type*} (s : Finset α)
    (f : α → ℝ) (hf : ∀ x ∈ s, f x ≥ 0) (κ : ℝ) (hκ : κ > 0) :
    (s.filter (fun x => f x > κ)).card ≤
    (s.sum f) / κ := by
  -- 核心步骤：filter 集合的 sum 被整体 sum 控制
  -- 然后除以 κ
  ...
```

**估计工作量**：50-100 行 Lean4 代码，2-3 周实现。

**离散 Gauss-Bonnet 处理**：作为 axiom 标注引用经典文献（Descartes 1637 + Polyhedral Gauss-Bonnet 现代证明）。

---

## 5. System Layer Design

### 5.1 Curvature-Aware Codebook（核心改动）

**现状**：MeshLex 用均匀 VQ，512 个 codeword 平等对待每种 patch。

**改进**：按曲率分配 codeword，高频的平面 patch 分得多，低频的角点 patch 分得少。

**具体方案**：

**Step 1**: 预计算所有 patch 的曲率 $|\bar{K}_P|$

**Step 2**: 按曲率大小分成 B=5 个 bin

| Bin | 曲率范围 | 历史频率 | Codeword 分配 |
|-----|----------|----------|---------------|
| 1 (flat) | $\|K\| < 0.1$ | ~40% | 200 |
| 2 (mild) | $\|K\| < 0.3$ | ~25% | 130 |
| 3 (medium) | $\|K\| < 0.6$ | ~20% | 100 |
| 4 (sharp) | $\|K\| < 1.0$ | ~10% | 52 |
| 5 (extreme) | $\|K\| \geq 1.0$ | ~5% | 30 |
| **Total** | | 100% | **512** |

**Step 3**: 每个 bin 单独训练一个小 SimVQ sub-codebook

**Step 4**: 推理时先按 $|\bar{K}_P|$ 分配到 bin，再在 bin 内查询最近 codeword

### 5.2 Key Ablation Table

| 方案 | Codebook 设计 | 理论依据 |
|------|---------------|----------|
| Baseline | 均匀 512 tokens | 无 |
| **Ours** | 曲率感知非均匀 512 tokens | Lean4 定理 |
| Upper bound | 均匀 1024 tokens | 无（两倍大小） |

**成功标准**："Ours" 用同等参数量（512）超过均匀 512，接近均匀 1024。

### 5.3 Complete Pipeline

曲率感知 codebook 只改动 M2（Tokenizer），其余保持不变：

| 模块 | 内容 | 变化 |
|------|------|------|
| M1 | Patch Partitioning (METIS) | 不变 |
| M2 | Patch Tokenizer | **曲率感知 SimVQ（新）** |
| M3 | AR Generation (GPT-2, 20.4M params) | 不变 |
| M4 | Assembly (StitchingMLP) | 不变 |

### 5.4 Training Schedule

| 阶段 | 内容 | 数据 | 时间估算 |
|------|------|------|----------|
| Pre-compute | 计算所有 patch 的 $\bar{K}$ | 全量 | ~2h |
| VQ-VAE | 训练曲率感知 codebook | 全量 | ~15h GPU |
| Baseline | 训练均匀 VQ baseline（对比用） | 全量 | ~15h GPU |
| AR model | 训练 AR 生成模型 | 全量 token sequences | ~30h GPU |
| Eval | 重建 + 生成质量评估 | ShapeNet Chair/Table | ~5h |

---

## 6. Evaluation Plan

### 6.1 Reconstruction Evaluation

**Metrics**: CD, Normal Consistency, F-Score@{0.01, 0.02, 0.05}

**Baseline**: MeshLex v1（5% 数据训练）

**预期**: 全量数据 + 曲率感知 codebook 大幅超越旧版

### 6.2 Generation Evaluation

**Metrics**: FID, COV, MMD on ShapeNet Chair + Table

**对比**: PolyGen, MeshGPT, FACE（用发表数值）

**目标**: 不一定超越 SOTA，但差距合理（在同量级内），足以反驳"理论没用"

### 6.3 Ablation Experiments

| 实验 | 目的 |
|------|------|
| 均匀 512 vs 曲率感知 512 | 证明非均匀分配有效 |
| 各 bin codeword 数量 ablation | 找最优分配比例 |
| 有无理论先验 vs 纯 data-driven 分配 | 验证理论的指导价值 |

---

## 7. Paper Structure

```
§1 Introduction
    - Problem: 为什么 mesh 生成这么难？
    - Observation (MeshLex v1): 512 token 跨 1156 类别泛化
    - Question: 为什么？这个结构是什么？
    - This paper: 3 层贡献（实验 + 理论 + 系统）

§2 Related Work
    - Mesh generation
    - Graph tokenization
    - Formal methods in ML

§3 Theory
    §3.1 实验：相变曲线 + 幂律测量
    §3.2 连接曲率：从 Gauss-Bonnet 到频率分布
    §3.3 Lean4 形式化：高曲率 patch 上界证明

§4 Method
    §4.1 曲率感知 codebook 设计
    §4.2 完整 MeshLex pipeline

§5 Experiments
    §5.1 理论验证（相变 + 幂律）
    §5.2 Codebook ablation（曲率感知 vs 均匀）
    §5.3 重建质量（CD / F-Score / NC）
    §5.4 生成质量（FID / COV / MMD）

§6 Conclusion
```

---

## 8. Timeline

**首选**: NeurIPS 2027（deadline 约 2027 年 5 月）→ 约 13 个月

**备选**: ICLR 2027（deadline 约 2026 年 9 月）→ 约 6 个月

### 里程碑

| 阶段 | 内容 | 时间 |
|------|------|------|
| Phase T1 | 全量数据集准备完成 | 当前进行中 |
| Phase T2 | 理论实验（相变 + 幂律） | ~2 周 |
| Phase T3 | Lean4 形式化证明 | ~3 周 |
| Phase T4 | 曲率感知 codebook 实现 + 训练 | ~2 周 |
| Phase T5 | AR 模型全量重训 | ~1 周 |
| Phase T6 | 完整评估 + 论文撰写 | ~4 周 |

---

## 9. Risk Mitigation

### 9.1 理论风险

**风险**: 相变曲线不明显，幂律拟合差

**缓解**:
- 尝试不同的 K 值范围
- 检查是否是数据处理问题
- 如果确实没有相变，修改叙事为"均匀 patch 空间的发现"

### 9.2 系统风险

**风险**: 曲率感知 codebook 效果不如均匀 baseline

**缓解**:
- 调整 bin 数量和分配比例
- 尝试更细粒度的曲率划分
- 作为 negative result 报告，仍有理论价值

### 9.3 Lean4 风险

**风险**: Mathlib 基础设施不足，证明受阻

**缓解**:
- 使用 axiom 标注经典结论，不从头证明
- 降低形式化目标，只证明 Markov 不等式应用部分

---

## 10. Success Criteria

| 层次 | 标准 | 判定 |
|------|------|------|
| 理论 C1 | 相变曲线有 ≥2 个明显相变点 | GO |
| 理论 C2 | 幂律拟合 R² > 0.9 | GO |
| 理论 C3 | Lean4 proof 编译通过 | GO |
| 方法 C4 | 曲率感知 > 均匀 baseline（同等参数） | GO |
| 应用 C5 | 生成 FID 与 SOTA 同量级（<2× 差距） | GO |

**整体判定**: ≥3 GO + 无 FAIL → 论文可投

---

## Appendix A: Key References

1. **Gauss-Bonnet**: Descartes (1637), Polyhedral Gauss-Bonnet (Banchoff, 1970)
2. **Discrete Curvature**: Meyer et al., "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds" (2002)
3. **Rate-Distortion**: Cover & Thomas, "Elements of Information Theory"
4. **Lean4 Mathlib**: https://github.com/leanprover-community/mathlib4
5. **SimVQ**: Li et al., 2025

---

## Appendix B: Lean4 Proof Sketch (Full)

```lean
-- 高曲率顶点上界定理
-- 对于封闭三角网格 M，满足 angle defect > κ 的顶点数上界

import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.Order.Ring.Lemmas

namespace MeshLex

-- 离散 Gauss-Bonnet 作为公理
axiom discreteGaussBonnet {M : Type*} [Fintype M]
  (K : M → ℝ) (χ : ℤ) :
  (∑ v : M, K v) = 2 * Real.pi * χ

-- 有限集合的 Markov 不等式
theorem finite_markov {α : Type*} (s : Finset α)
    (f : α → ℝ) (hf : ∀ x ∈ s, f x ≥ 0) (κ : ℝ) (hκ : κ > 0) :
    (s.filter (fun x => f x > κ)).card ≤ ((s.sum f) / κ).toNat := by
  -- 证明略，利用 filter 集合的 sum 性质
  sorry

-- 主定理：高曲率顶点上界
theorem high_curvature_bound {M : Type*} [Fintype M]
    (K : M → ℝ) (χ : ℤ) (hχ : χ > 0) (κ : ℝ) (hκ : κ > 0) :
    (Finset.univ.filter (fun v : M => K v > κ)).card ≤
    Int.natAbs ((2 * Real.pi * χ) / κ) := by
  have h_sum : ∑ v : M, K v = 2 * Real.pi * χ := discreteGaussBonnet K χ
  have h_pos : 0 ≤ ∑ v : M, K v := by
    rw [h_sum]
    positivity
  have h_markov := finite_markov Finset.univ K (by intro x _; positivity) κ hκ
  -- 完成证明
  sorry

end MeshLex
```

---

## Appendix C: Curvature Computation

```python
import numpy as np
import trimesh

def compute_discrete_gaussian_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    计算每个顶点的离散 Gaussian 曲率 (angle defect)

    K_v = 2π - Σ θ_vf (所有包含 v 的面的内角)
    """
    K = np.zeros(len(mesh.vertices))

    for face in mesh.faces:
        v0, v1, v2 = face
        # 获取三条边向量
        e01 = mesh.vertices[v1] - mesh.vertices[v0]
        e02 = mesh.vertices[v2] - mesh.vertices[v0]
        e12 = mesh.vertices[v2] - mesh.vertices[v1]

        # 计算三个内角
        theta0 = np.arccos(np.clip(np.dot(e01, e02) / (np.linalg.norm(e01) * np.linalg.norm(e02)), -1, 1))
        theta1 = np.arccos(np.clip(np.dot(-e01, e12) / (np.linalg.norm(e01) * np.linalg.norm(e12)), -1, 1))
        theta2 = np.pi - theta0 - theta1

        # 累加 angle defect
        K[v0] -= theta0
        K[v1] -= theta1
        K[v2] -= theta2

    K += 2 * np.pi  # 最终 angle defect
    return K

def compute_patch_curvature(mesh: trimesh.Trimesh, patch_faces: np.ndarray) -> float:
    """
    计算 patch 的平均曲率
    """
    # 获取 patch 内的所有顶点
    patch_vertices = set()
    for face_idx in patch_faces:
        patch_vertices.update(mesh.faces[face_idx])

    K = compute_discrete_gaussian_curvature(mesh)
    patch_K = np.mean([np.abs(K[v]) for v in patch_vertices])
    return patch_K
```
