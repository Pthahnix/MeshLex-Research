# Literature Survey: MeshLex Theory-Driven Design

**Date**: 2026-03-21
**Scope**: Novelty/feasibility assessment for MeshLex's 5 contributions (C1-C5)
**Papers Surveyed**: 14 (12 deeply read, 2 noted)
**Search Angles**: (1) VQ token frequency distributions, (2) Discrete differential geometry + mesh DL, (3) Non-uniform codebook allocation, (4) Lean4 formal verification in ML, (5) Mesh generation with patch/token approaches

---

## Paper Reading Notes

### HIGH Relevance (direct impact on novelty claims)

#### P1: Analyzing the Language of Visual Tokens
- **Authors**: (2024), image domain
- **Key Finding**: Image VQ-VAE tokens do NOT follow Zipf's law. Instead follow **lognormal** distribution. Measured alpha=4.37 (vs natural language alpha~1.71).
- **No theoretical derivation** of why the distribution takes this form.
- **No geometric/curvature connection**.
- **Impact on MeshLex C1**: MeshLex cannot claim "first to study Zipf in VQ tokens" globally. BUT: (a) this is images only, (b) the NEGATIVE result (not Zipf) actually strengthens MeshLex if mesh tokens DO follow Zipf — it shows domain matters. (c) MeshLex provides a geometric explanation, which this paper completely lacks.
- **Action**: Must cite. Reframe C1 as "first in 3D mesh domain + first geometric explanation."
- **Risk**: If mesh tokens also turn out lognormal, the Zipf narrative needs pivoting.

#### P2: The Language of Time
- **Authors**: Xie et al. (2025), time-series domain
- **Key Finding**: Time-series VQ tokens DO follow Zipf's law (alpha~1.025). Provides theoretical explanation via GEM/Pitman-Yor (rich-get-richer) process.
- **No geometric interpretation**, no curvature, no mesh data.
- **Impact on MeshLex C2**: A competing theory exists for WHY VQ tokens follow power law. GEM/Pitman-Yor is a generic statistical mechanism that could apply to any domain.
- **Action**: Must cite. Must run competing theories experiment (GEM vs geometric model). If curvature model fits better → geometric explanation wins.
- **Risk**: If GEM fits mesh data equally well, the curvature explanation loses unique value.

#### P3: FreeMesh — Boosting Mesh Generation with Coordinates Merging
- **Authors**: Liu et al. (2025, ICML)
- **Key Finding**: Introduces Per-Token-Mesh-Entropy (PTME) as training-free theoretical metric for mesh tokenizers. Uses BPE for coordinate merging. Strong correlation (r=0.965) between PTME and Chamfer Distance.
- **Theoretical framework**: Shannon entropy, PMI-based merging conditions.
- **No curvature analysis**, no frequency distribution analysis, no power law.
- **Per-coordinate** approach (sub-word level), not per-patch.
- **Impact on MeshLex**: FreeMesh occupies the "theoretical mesh tokenization" space but from information-theoretic angle. MeshLex's geometric angle is complementary, not competing. However, reviewers may ask "why not just use PTME to evaluate your codebook?"
- **Action**: Cite as complementary theoretical work. Consider computing PTME for MeshLex codebooks as additional validation metric.

#### P4: GPSToken — Gaussian Parameterized Spatially-adaptive Tokenization
- **Authors**: Zhang et al. (2025), image domain
- **Key Finding**: Non-uniform spatial tokenization for images using 2D Gaussians. Complex regions get more tokens.
- **Mechanism**: Entropy-driven partitioning (Sobel edges), then transformer refinement. **Continuous** features, no discrete codebook.
- **No curvature**, no mesh, no codebook allocation.
- **Impact on MeshLex C4**: Validates the general principle that non-uniform tokenization outperforms uniform. But fundamentally different mechanism (spatial token density vs codebook capacity allocation). Not a competitor.
- **Action**: Cite as cross-domain evidence for non-uniform tokenization principle.

#### P5: TorchLean — Formalizing Neural Networks in Lean
- **Authors**: George et al. (2026)
- **Key Finding**: First framework to formalize end-to-end ML pipeline in Lean4 (execution, AD, verification). High mathematical sophistication.
- **Impact on MeshLex C3**: Sets precedent that Lean4 proofs are appearing in ML research. But TorchLean formalizes numerical semantics (floating-point correctness), while MeshLex formalizes mathematical properties (Gauss-Bonnet → bounds). Different philosophy.
- **Action**: Cite as evidence that formal verification is gaining traction in ML. Position MeshLex's Lean4 proof as "domain-theoretic formalization" vs TorchLean's "numerical formalization."

### MEDIUM Relevance (contextual)

#### P6: Riemannian Graph Tokenizer (RGT)
- Curvature-aware tokenization for **abstract graphs** (not meshes). Uses embedding space curvature (constant κ ∈ {-1, 0, +1}), not surface Gaussian curvature. Zero threat.

#### P7: Differentiable Topology from Curvatures
- Uses Gauss-Bonnet to estimate Euler characteristic from point clouds. Key insight: they use curvature to compute a **single global scalar** (χ), while MeshLex uses curvature to **preserve local variation**. Orthogonal use of same theorem.
- **Action**: Cite for discrete curvature computation methodology (Weingarten map approach could be superior to angle defect for robustness).

#### P8: CurvaNet
- Uses directional curvature for 3D shape **classification** (KDD 2020, 60 citations). No generation, no codebook, no tokenization. Context only.

#### P9: MeshCraft
- Continuous VAE for mesh generation (flow-based DiT). No codebook, no frequency analysis. Represents "codebook-free" alternative approach.

#### P10: Rate-Adaptive Quantization
- Multi-rate VQ with Seq2Seq codebook generation network. Different mechanism from curvature-aware allocation.

#### P11: Representation Collapsing in VQ
- Taxonomy of VQ failure modes (token collapse, embedding collapse). Useful for MeshLex: adopt perplexity diagnostics, consider two-stage pre-training.

### LOW Relevance

#### P12: VP-VAE
- Perturbation-based VQ training. No overlap.

#### P13: Compression Tells Intelligence (survey)
- No R-D curves with varying K, no phase transitions. Only high-level information bottleneck framing.

#### P14: MeshGPT
- Baseline competitor for mesh generation. Per-face VQ approach with uniform codebook.

---

## Domain Landscape

### Major Threads

1. **Token frequency analysis in VQ systems**: Emerging field. Image tokens → lognormal (P1). Time-series → Zipf (P2). **3D mesh tokens → UNEXPLORED.** This is MeshLex's primary opening.

2. **Non-uniform tokenization**: Growing trend across modalities. GPSToken (images), FreeMesh (mesh coordinates). But all are **data-driven** (entropy, frequency). **Theory-driven** non-uniform allocation (from differential geometry) is unexplored.

3. **Formal verification in ML**: Rapidly expanding. TorchLean (2026) shows Lean4 is ready for complex ML formalization. But all existing work formalizes **numerical properties** (AD correctness, bound propagation). Formalizing **domain-theoretic properties** (geometric constraints → system design) is new.

4. **Mesh generation approaches**: Split between (a) per-face coordinate tokenization (MeshGPT, FreeMesh, Edgerunner), (b) continuous latent (MeshCraft, Fluid), and (c) per-patch vocabulary (MeshLex). Per-patch is least crowded.

### Key Debates

- **Discrete vs Continuous tokens** for mesh generation (MeshGPT vs MeshCraft)
- **Per-face vs Per-patch** granularity (most competitors vs MeshLex)
- **Training-free vs learned** tokenizer evaluation (FreeMesh's PTME vs learned VQ-VAE)
- **Statistical vs geometric** explanation of token distributions (GEM/Pitman-Yor vs curvature)

### Development Trajectory

- 2024: MeshGPT establishes per-face VQ + AR as dominant paradigm
- 2025: FreeMesh, MeshCraft explore alternatives (BPE merging, continuous latent)
- 2025-2026: Theoretical analysis emerging (PTME, Zipf analysis in other domains)
- **Gap**: No one connects mesh geometry (curvature) to tokenizer design theory
