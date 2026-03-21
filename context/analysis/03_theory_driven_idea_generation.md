# Idea Generation: MeshLex Theory-Driven Design

**Date**: 2026-03-21
**Input**: Ranked gap list from Gap Analysis
**Focus**: Top 5 gaps (G1, G2, G3, G4, G5)

---

## Ideas from Gap G2: Geometric Explanation of VQ Token Statistics

### Idea I1: MaxEnt Curvature Distribution Theory
- **Strategy**: Combination (Gauss-Bonnet + Maximum Entropy principle)
- **Description**: Derive that under the Gauss-Bonnet constraint (ΣK_v = 2πχ), the maximum entropy distribution of curvature magnitudes is exponential. When discretized into VQ bins with logarithmic thresholds, this produces a power law in rank-frequency space. This closes Gap G5 simultaneously.
- **Scores**: Novelty: 9, Feasibility: 8, Impact: 9, Clarity: 9, Evidence: 8 | **Total: 43/50**
- **Key Risk**: MaxEnt argument requires assuming independent curvature values, which is only approximately true for mesh patches.
- **Mitigation**: Frame as "approximate model" validated empirically.

### Idea I2: Competing Theories Experiment (Geometric vs Statistical)
- **Strategy**: Inversion (challenge the assumption that Zipf = rich-get-richer)
- **Description**: Fit BOTH the curvature-derived model AND the GEM/Pitman-Yor model (from "Language of Time") to mesh token frequencies. Compare AIC/BIC. If geometric model wins, it's a strong result against the generic statistical explanation.
- **Scores**: Novelty: 8, Feasibility: 9, Impact: 8, Clarity: 9, Evidence: 7 | **Total: 41/50**
- **Key Risk**: GEM might fit equally well.
- **Mitigation**: Even if GEM fits, the curvature model provides INTERPRETABILITY (which tokens are flat vs sharp), which GEM cannot.

### Idea I3: Cross-Modality Distribution Comparison
- **Strategy**: Scale (apply frequency analysis methodology across domains)
- **Description**: Systematically compare VQ token distributions across 3 modalities: images (expect lognormal per P1), time-series (expect Zipf per P2), and meshes (measure). Show that mesh tokens have a unique distributional signature explainable by geometry.
- **Scores**: Novelty: 7, Feasibility: 6, Impact: 8, Clarity: 7, Evidence: 8 | **Total: 36/50**
- **Key Risk**: Requires access to trained VQ-VAEs for image and time-series.
- **Mitigation**: Use published numbers from P1 and P2; only train mesh VQ.

---

## Ideas from Gap G1: Token Frequency in 3D Mesh VQ

### Idea I4: Dual Distribution Test (Power Law vs Lognormal)
- **Strategy**: Transfer (apply P1's methodology to mesh domain)
- **Description**: For the K=1024 VQ-VAE, compute token frequencies and fit both power law and lognormal. Use Kolmogorov-Smirnov test + Vuong's closeness test (as P1 used) to determine the correct distributional family. This is a direct empirical contribution.
- **Scores**: Novelty: 7, Feasibility: 10, Impact: 7, Clarity: 10, Evidence: 9 | **Total: 43/50**
- **Key Risk**: If lognormal wins, must pivot the narrative.
- **Mitigation**: Lognormal with curvature explanation is still novel (no one has done it).

### Idea I5: Rate-Distortion Phase Transition with Curvature Annotation
- **Strategy**: Combination (rate-distortion analysis + curvature labeling)
- **Description**: Sweep K ∈ {32..4096}, measure D(K). At each K, label each codeword by its average patch curvature. Track which curvature types "unlock" at each phase transition. This provides mechanistic evidence that phase transitions correspond to new curvature types being distinguishable.
- **Scores**: Novelty: 9, Feasibility: 8, Impact: 9, Clarity: 8, Evidence: 7 | **Total: 41/50**
- **Key Risk**: Phase transitions may not be sharp.
- **Mitigation**: Even gradual transitions with curvature annotations are informative.

---

## Ideas from Gap G3: Theory-Driven Non-Uniform Codebook

### Idea I6: Curvature-Proportional Codebook Allocation
- **Strategy**: Direct application of theory
- **Description**: As specified in the design doc: 5 curvature bins with allocation proportional to empirical frequency. Compare against uniform baseline and "oracle" (2x uniform).
- **Scores**: Novelty: 8, Feasibility: 9, Impact: 8, Clarity: 10, Evidence: 7 | **Total: 42/50**
- **Already in spec.** Main concern: may not outperform data-driven allocation.

### Idea I7: Information-Geometric Dual Validation
- **Strategy**: Combination (MeshLex curvature theory + FreeMesh PTME)
- **Description**: Validate curvature-aware codebook from BOTH geometric (curvature correlation) AND information-theoretic (PTME) perspectives. Show that curvature-aware allocation achieves lower PTME than uniform allocation. This dual validation is novel.
- **Scores**: Novelty: 7, Feasibility: 9, Impact: 7, Clarity: 8, Evidence: 8 | **Total: 39/50**

---

## Ideas from Gap G4: Lean4 Domain Theory in ML

### Idea I8: Lean4 Proof as "Verified Design Rationale"
- **Strategy**: Transfer (software engineering concept of "design rationale" to ML)
- **Description**: Frame the Lean4 proof not as a standalone contribution but as a "verified design rationale" — the first instance where an ML system design choice (codebook allocation) is formally justified by a machine-checked mathematical proof. This differentiates from TorchLean (which verifies numerical properties).
- **Scores**: Novelty: 8, Feasibility: 7, Impact: 7, Clarity: 8, Evidence: 6 | **Total: 36/50**
- **Key Risk**: Reviewers may dismiss as trivial math.
- **Mitigation**: Emphasize the methodology (theory → formal proof → system design) not the theorem difficulty.

---

## Ideas from Gap G5: Bound-to-Distribution Bridge

### Idea I9: Constrained MaxEnt Derivation + Empirical Validation
- **Strategy**: Combination (same as I1, but more specific)
- **Description**:
  1. State Gauss-Bonnet constraint: Σ K_v = 2πχ(M)
  2. Apply MaxEnt: the distribution maximizing entropy subject to fixed total curvature is exponential: p(K) ∝ exp(-λK)
  3. Show that exponential curvature → power law in token rank-frequency (via discretization + Pareto approximation)
  4. Validate empirically by measuring λ and comparing predicted vs actual frequency curves
- **Scores**: Novelty: 9, Feasibility: 8, Impact: 9, Clarity: 8, Evidence: 7 | **Total: 41/50**
- **Note**: This is I1 with more detail. They should be merged.

---

## Ranking and Top 3 Recommendations

| Rank | Idea | Score | Gap |
|------|------|-------|-----|
| 1 | **I1/I9: MaxEnt Curvature Theory** | 43 | G2+G5 |
| 2 | **I4: Dual Distribution Test** | 43 | G1 |
| 3 | **I6: Curvature-Proportional Codebook** | 42 | G3 |
| 4 | I2: Competing Theories Experiment | 41 | G2 |
| 5 | I5: R-D Phase Transition + Curvature Annotation | 41 | G1 |
| 6 | I7: Dual Validation (Curvature + PTME) | 39 | G3 |
| 7 | I3: Cross-Modality Comparison | 36 | G2 |
| 8 | I8: Lean4 Verified Design Rationale | 36 | G4 |

---

## Top 3 Recommendations with Rationale

### Recommendation 1: MaxEnt Curvature Distribution Theory (I1/I9)
**Why**: Closes the most critical theoretical gap (bound ≠ distribution). Without this, the theory chain has a logical hole that sharp reviewers WILL exploit. The MaxEnt argument is well-established in physics (Jaynes 1957) and can be adapted in ~1 page. This is the single highest-impact fix to the spec.

### Recommendation 2: Dual Distribution Test (I4)
**Why**: This is the first experiment to run and is make-or-break. If mesh tokens are lognormal (like images), the entire power-law narrative needs pivoting BEFORE investing in theory. Extremely fast to execute (just analyze existing VQ-VAE outputs). Must include Vuong's test for model comparison.

### Recommendation 3: Curvature-Proportional Codebook (I6)
**Why**: Already in the spec. The key enhancement: add PTME validation (I7) and competing theories comparison (I2) as ablation experiments to strengthen the empirical case.

### Recommended Execution Order
```
I4 (Dual Distribution Test)     ← FIRST: go/no-go gate
    ↓ if power law confirmed
I1/I9 (MaxEnt Theory)            ← fill theoretical gap
    ↓
I5 (R-D + Curvature Annotation)  ← mechanistic evidence
I2 (Competing Theories)           ← defend against GEM
I6 (Curvature Codebook)           ← the actual system
I7 (Dual Validation)              ← extra evidence
I8 (Lean4 Proof)                  ← parallel track
```
