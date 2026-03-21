# Gap Analysis: MeshLex Theory-Driven Design

**Date**: 2026-03-21
**Input**: 14-paper literature survey
**Method**: Comparison matrix + contradiction detection + blank identification + trend analysis

---

## 1. Method Comparison Matrix

| Paper | Domain | Token Type | Codebook | Frequency Analysis | Curvature | Theory | Non-uniform |
|-------|--------|-----------|----------|-------------------|-----------|--------|-------------|
| MeshLex (ours) | 3D Mesh | Discrete (VQ) | SimVQ/RVQ | Power law (proposed) | Gaussian curvature | Gauss-Bonnet + Lean4 | Curvature-aware bins |
| Analyzing Visual Tokens | 2D Image | Discrete (VQ) | VQGAN | Lognormal (measured) | None | None | None |
| Language of Time | Time-series | Discrete (VQ) | VQ-VAE | Zipf (measured) | None | GEM/Pitman-Yor | None |
| FreeMesh | 3D Mesh | Merged coords | BPE vocab | None | None | Shannon entropy (PTME) | BPE merging (implicit) |
| GPSToken | 2D Image | Continuous | None | None | None | Heuristic (Sobel entropy) | Spatial density |
| MeshGPT | 3D Mesh | Discrete (VQ) | ResVQ | None | None | None | None (uniform) |
| MeshCraft | 3D Mesh | Continuous | None (KL-VAE) | None | None | None | None |
| TorchLean | N/A (ML theory) | N/A | N/A | N/A | N/A | Lean4 (numerical) | N/A |
| DiffTopo | 3D Shapes | N/A | N/A | N/A | Gaussian (Weingarten) | Gauss-Bonnet | N/A |
| CurvaNet | 3D Mesh | N/A | N/A | N/A | Directional | None | N/A |
| RGT | Graphs | Discrete | Learned | None | Embedding space κ | Riemannian geometry | Manifold selection |
| Rate-Adaptive VQ | 2D Image | Discrete (VQ) | Multi-rate | None | None | None | Rate-adaptive |
| VQ Collapsing | General VQ | Discrete | Various | None | None | Taxonomy | None |

### Key Observations from Matrix

1. **No paper combines curvature + frequency analysis + codebook design.** MeshLex is the only one attempting this triple connection.
2. **Frequency analysis exists in images and time-series, NOT in 3D meshes.** Clear domain gap.
3. **Non-uniform codebook allocation** is addressed by Rate-Adaptive VQ (image, data-driven) and GPSToken (image, heuristic). None uses geometry to drive allocation.
4. **Lean4 formalization** in ML exists (TorchLean) but for numerical properties, not domain-theoretic.
5. **FreeMesh** is the closest competitor in "theoretical mesh tokenization" but uses information theory (entropy), not geometry.

---

## 2. Contradiction Detection

### Contradiction 1: Zipf vs Lognormal in VQ Tokens
- **Language of Time**: VQ tokens follow Zipf's law (α≈1.025)
- **Analyzing Visual Tokens**: VQ tokens follow lognormal (NOT Zipf, α=4.37)
- **Resolution**: Domain-dependent. Time-series structure (temporal patterns) → Zipf. Image structure (spatial redundancy) → lognormal.
- **Implication for MeshLex**: Mesh tokens could go either way. Must test BOTH fits. If Zipf → strengthens theory. If lognormal → need different theoretical frame.

### Contradiction 2: Discrete vs Continuous Tokens for Mesh Generation
- **MeshGPT, FreeMesh**: Discrete tokens are the way forward
- **MeshCraft**: Continuous tokens with flow-based generation are superior (claims better quality, faster)
- **Resolution**: Both viable; discrete enables AR generation, continuous enables flow/diffusion
- **Implication for MeshLex**: MeshLex uses discrete VQ. If continuous approaches dominate by 2027, the codebook theory becomes less relevant. **Risk: medium.**

### Contradiction 3: Theory-Free vs Theory-Driven Tokenizer Design
- **Most mesh generation papers**: Purely empirical, no theoretical justification for design choices
- **FreeMesh**: Provides training-free theoretical metric (PTME) but purely information-theoretic
- **MeshLex**: Proposes geometric theory (curvature → frequency → codebook)
- **Resolution**: Not really a contradiction — it's a gap. The field has not yet seen theory-driven mesh tokenizer design.

---

## 3. Blank Identification

### Gap G1: Token Frequency Distribution in 3D Mesh VQ (HIGH priority)
- **Type**: Dataset/scenario gap
- **Description**: Zipf/power-law analysis has been done for image tokens (P1) and time-series tokens (P2), but NEVER for 3D mesh patch tokens.
- **Evidence**: P1 and P2 explicitly study this; no mesh paper does.
- **Feasibility**: HIGH — MeshLex already has trained VQ-VAEs and can immediately compute token frequencies.
- **Validation**: Searched "mesh token frequency distribution power law" — zero results.

### Gap G2: Geometric Explanation of VQ Token Statistics (HIGH priority)
- **Type**: Method combination blank (geometry + VQ statistics)
- **Description**: No paper connects differential geometry (curvature) to VQ token frequency distributions. P2 uses GEM/Pitman-Yor (statistical), P1 provides no explanation at all.
- **Evidence**: All 14 papers — none attempts this connection.
- **Feasibility**: HIGH — discrete Gaussian curvature is well-understood (angle defect); correlation with token frequency is a straightforward experiment.
- **Validation**: No paper found connecting curvature to codebook design.

### Gap G3: Theory-Driven Non-Uniform Codebook for 3D (HIGH priority)
- **Type**: Method combination blank (geometric theory + codebook allocation)
- **Description**: Non-uniform codebook allocation exists in images (Rate-Adaptive VQ, GPSToken) but is always data-driven. No one uses geometric theory to drive allocation.
- **Evidence**: GPSToken uses Sobel entropy (heuristic), Rate-Adaptive uses learned Seq2Seq. Neither is theory-driven.
- **Feasibility**: HIGH — curvature bins are easy to compute; allocation is a hyperparameter choice.

### Gap G4: Formal Verification of Domain Theory in ML (MEDIUM priority)
- **Type**: Scale gap (Lean4 used for numerical ML properties, never for domain-specific geometric constraints)
- **Description**: TorchLean formalizes numerical semantics. No paper formalizes domain-specific mathematical constraints (like Gauss-Bonnet) and uses them to justify ML system design.
- **Evidence**: TorchLean (P5), Lean4 survey — all focus on proof verification, code verification, or numerical correctness.
- **Feasibility**: MEDIUM — Lean4 proof of Markov inequality is simple (~50 lines), but Mathlib4 infrastructure for discrete geometry may be limited.

### Gap G5: Bound-to-Distribution Theoretical Bridge (HIGH priority, INTERNAL)
- **Type**: MeshLex-internal theoretical gap
- **Description**: Gauss-Bonnet + Markov inequality gives an UPPER BOUND on high-curvature vertices. But MeshLex claims a POWER LAW distribution. An upper bound ≠ a distribution. Missing: maximum entropy argument.
- **Evidence**: The spec's derivation chain (§4.4) proves |{v: K_v > κ}| ≤ 2π|χ|/κ but never derives f(r) ∝ r^{-α}.
- **Feasibility**: HIGH — MaxEnt under fixed total curvature is a standard statistical mechanics argument (Jaynes 1957).
- **Action**: MUST fix before submission.

### Gap G6: Cross-Domain Universality of VQ Token Distributions (MEDIUM priority)
- **Type**: Direction mentioned in "future work" but not pursued
- **Description**: P1 and P2 study individual domains. No one systematically compares VQ token distributions ACROSS domains (image, time-series, mesh, audio, etc.) to identify universal vs domain-specific patterns.
- **Feasibility**: MEDIUM — requires trained VQ-VAEs across multiple domains.

### Gap G7: FreeMesh PTME as Complementary Validation (LOW priority)
- **Type**: Potential enhancement
- **Description**: FreeMesh's PTME metric could validate MeshLex codebooks from an information-theoretic angle, providing dual theoretical validation (geometric + information-theoretic).
- **Feasibility**: HIGH — PTME is training-free, can compute immediately.

---

## 4. Trend Analysis

### Heating Up (2024-2026)
- Theoretical analysis of VQ tokenizers (PTME, Zipf analysis, language analogies)
- Mesh generation moving beyond per-vertex coordinate tokenization
- Lean4 formal verification in ML research
- Non-uniform/adaptive tokenization across modalities

### Cooling Down
- Pure per-face VQ approaches (MeshGPT-style) — being challenged by BPE merging (FreeMesh) and continuous methods (MeshCraft)
- Uniform codebook design without theoretical justification

### Emerging Problems
- "Why does VQ work?" — fundamental question being asked across domains
- Bridging geometry and learning — curvature-aware methods gaining traction in classification (CurvaNet), topology (DiffTopo), but NOT yet in generation

---

## 5. Ranked Gap List

| Rank | Gap | Type | Impact | Feasibility | Novelty | Score |
|------|-----|------|--------|-------------|---------|-------|
| 1 | G2: Geometric explanation of VQ token stats | Method combination | 10 | 9 | 10 | 29 |
| 2 | G5: Bound-to-distribution bridge (MaxEnt) | Internal theory gap | 9 | 9 | 8 | 26 |
| 3 | G1: Token frequency in 3D mesh VQ | Dataset gap | 9 | 10 | 7 | 26 |
| 4 | G3: Theory-driven non-uniform codebook | Method combination | 8 | 9 | 9 | 26 |
| 5 | G4: Lean4 domain theory in ML | Scale gap | 7 | 6 | 9 | 22 |
| 6 | G6: Cross-domain VQ distribution comparison | Future work | 8 | 5 | 7 | 20 |
| 7 | G7: PTME complementary validation | Enhancement | 4 | 9 | 3 | 16 |
