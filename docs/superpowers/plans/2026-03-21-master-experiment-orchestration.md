# Master Experiment Orchestration Plan

> **For agentic workers:** This is an **entry-point document**. It does NOT contain implementation steps directly. Instead, it tells you which branch to check out and which plan file to follow for each phase. Use superpowers:executing-plans or superpowers:subagent-driven-development on the referenced plan files.

**Goal:** Orchestrate the execution of three research plans on the full 72K mesh / 10.8M patch dataset (`Pthahnix/MeshLex-Patches`), producing: (1) full-scale baseline models + evaluation, (2) theory-driven analysis + curvature-aware codebook, (3) PatchDiffusion masked discrete diffusion variants.

**Execution Mode:** Serial (one plan at a time). Hardware: 3× RTX 5090.

**Dataset:** HuggingFace `Pthahnix/MeshLex-Patches` — 72,555 meshes, 10.8M patches, splits: seen_train=53,492 / seen_test=13,372 / unseen=5,541.

---

## Execution Order

```
Phase A: fullscale-experiment     ── foundation (VQ-VAE ×4, AR ×2, analysis, evaluation)
    │
    │  outputs: checkpoints, encoded sequences, preliminary results, dashboard
    ▼
Phase B: theory-driven-design     ── theory analysis + curvature-aware codebook + Lean4
    │                                  (skip tasks already done in Phase A)
    │
    │  outputs: Vuong's test, R-D curves, curvature codebook, Lean4 proofs
    ▼
Phase C: innovation-brainstorm    ── PatchDiffusion variants (Block, Hierarchical RVQ)
                                      (skip basic MDLM, already done in Phase A Task 12+14)

    outputs: 3 diffusion variants trained + evaluated
```

---

## Phase A: Full-Scale Experiment (Foundation)

| Item | Value |
|------|-------|
| **Branch** | `fullscale-experiment` |
| **Plan file** | `docs/superpowers/plans/2026-03-21-fullscale-experiment.md` |
| **Spec** | `docs/superpowers/specs/2026-03-21-fullscale-experiment-design.md` |
| **Tasks** | 16 tasks (Task 1–16) |
| **Estimated time** | 2–3 days |

### What this covers

- Phase 0: Hardware audit + dependency install
- Phase 1: VQ-VAE foundation training ×4 (PCA K=1024, noPCA K=1024, K=512, K=2048)
- Phase 2: Token encoding + AR training ×2 (PCA 11-token, noPCA 7-token)
- Phase 3a: Preliminary experiments rerun (Exp 1–5 on full data)
- Phase 3b: Theory-driven analysis (K ablation, VQ method comparison, curvature-frequency correlation)
- Phase 3c: MDLM full-scale training (Pure MDLM, ~20-40M params)
- Phase 4: Evaluation + ablation (reconstruction CD, generation quality, PCA vs noPCA dashboard)
- Phase 5: Paper-ready figures

### Execution instructions

```bash
git checkout fullscale-experiment
# Follow docs/superpowers/plans/2026-03-21-fullscale-experiment.md
# Execute all 16 tasks sequentially (Task 1 → Task 16)
# All commits stay on this branch
```

### Key outputs needed by later phases

| Output | Path | Used by |
|--------|------|---------|
| PCA VQ-VAE checkpoint | `data/checkpoints/rvq_full_pca/` | Phase B (curvature analysis), Phase C (MDLM data) |
| noPCA VQ-VAE checkpoint | `data/checkpoints/rvq_full_nopca/` | Phase B (baseline comparison) |
| K=512/K=2048 VQ-VAE checkpoints | `data/checkpoints/rvq_full_pca_k*` | Phase B (K ablation deep-dive) |
| Encoded PCA sequences | `data/sequences/rvq_full_pca/` | Phase B, Phase C |
| Encoded noPCA sequences | `data/sequences/rvq_full_nopca/` | Phase C |
| Vanilla/EMA VQ checkpoints | `data/checkpoints/rvq_full_pca_vanilla/`, `_ema/` | Phase B (FM1 test) |
| Preliminary analysis results | `results/fullscale_preliminary/` | Phase B (comparison baseline) |
| MDLM checkpoint (Pure MDLM) | `data/checkpoints/mdlm_full/` | Phase C (baseline for variants) |
| Full evaluation dashboard | `results/fullscale_eval/DASHBOARD.md` | Phase B, C (append to) |

### Completion gate

Before proceeding to Phase B, verify:

```bash
# All 4 VQ-VAE checkpoints exist and are uploaded to HF
ls data/checkpoints/rvq_full_*/checkpoint_final.pt

# Both AR models trained
ls data/checkpoints/ar_full_*/checkpoint_final.pt

# Evaluation dashboard exists
cat results/fullscale_eval/DASHBOARD.md

# MDLM checkpoint exists
ls data/checkpoints/mdlm_full/checkpoint_final.pt
```

---

## Phase B: Theory-Driven Analysis + Curvature-Aware Codebook

| Item | Value |
|------|-------|
| **Branch** | `theory-driven-design` |
| **Plan file** | `docs/superpowers/plans/2026-03-20-meshlex-theory-driven-implementation.md` |
| **Spec** | `docs/superpowers/specs/2026-03-20-meshlex-theory-driven-design.md` |
| **Tasks** | 15 tasks (Task 0.1–12, including 3.5 and 3.6), **skip duplicates** (see below) |
| **Estimated time** | 1–2 days |

### Switch branch and sync data

```bash
git checkout theory-driven-design

# IMPORTANT: Phase A outputs live on fullscale-experiment branch's working directory.
# If the pod was reset or data was cleared between phases, re-download from HF:
python -c "
from huggingface_hub import hf_hub_download
import os
for name in ['rvq_full_pca', 'rvq_full_nopca', 'rvq_full_pca_k512', 'rvq_full_pca_k2048',
             'rvq_full_pca_vanilla', 'rvq_full_pca_ema', 'ar_full_pca', 'ar_full_nopca', 'mdlm_full']:
    try:
        path = hf_hub_download('Pthahnix/MeshLex-Research', f'checkpoints/{name}/checkpoint_final.pt',
                               local_dir='data')
        print(f'✅ {name}: {path}')
    except Exception as e:
        print(f'⚠️ {name}: {e}')
"

# Re-download patch data if needed (same as Phase A Task 7 Step 1)
# Re-encode sequences if needed (same as Phase A Task 8)
```

### Skip map — tasks already covered by Phase A

| Theory-Driven Task | Overlaps with Phase A | Action |
|--------------------|-----------------------|--------|
| Task 0.1: Dual Distribution Fitting | Phase A Task 11 (Exp 1 rerun, lognormal fitting) | **SKIP** — use `results/fullscale_preliminary/exp1_per_category/` |
| Task 1-2: Curvature computation + binning | Phase A Task 13 Step 7-8 (curvature analysis) | **SKIP if** `results/fullscale_theory/curvature/` exists. **RUN if** Phase A curvature module is insufficient (check if `discrete_gaussian_curvature()` function exists in codebase) |
| Task 3: Dual Distribution + Vuong's test | Phase A Task 11 + Task 13 Step 1 (K ablation includes fitting) | **PARTIAL SKIP** — Phase A does lognormal fitting but may not have Vuong's test. Run only the Vuong's test portion |
| Task 5: Curvature Correlation | Phase A Task 13 Step 7-8 | **SKIP** — use `results/fullscale_theory/curvature/curvature_results.json` |

### Tasks to execute (unique to theory-driven)

| Task | Content | Why unique |
|------|---------|------------|
| Task 3.5 | MaxEnt curvature distribution derivation | Pure theory — not in fullscale |
| Task 3.6 | Competing theories (GEM vs Geometric) | Hypothesis testing — not in fullscale |
| Task 4 | Rate-Distortion experiment | R-D curves — not in fullscale |
| Task 6 | Cross-dataset universality | Multi-dataset comparison — not in fullscale |
| Task 7 | Curvature-Aware VQ-VAE model | New model architecture — not in fullscale |
| Task 8 | Lean4 formalization | Formal proofs — not in fullscale |
| Task 9 | Curvature-aware training | Training new model — not in fullscale |
| Task 10 | Uniform baseline training | Comparison baseline — not in fullscale |
| Task 11 | PTME dual validation | Novel validation method — not in fullscale |
| Task 12 | Enhanced reconstruction evaluation | PTME-enhanced metrics — not in fullscale |

### Execution instructions

```bash
# On branch: theory-driven-design
# Follow docs/superpowers/plans/2026-03-20-meshlex-theory-driven-implementation.md
# Skip tasks marked SKIP above
# For PARTIAL SKIP tasks, read the task and only run the un-covered portions
# All commits stay on this branch
```

### Completion gate

```bash
# Curvature-aware VQ-VAE trained
ls data/checkpoints/rvq_curvature_aware/checkpoint_final.pt

# Lean4 proofs compile
cd lean4/ && lake build

# PTME validation results exist
cat results/theory_driven/ptme_validation.json
```

---

## Phase C: PatchDiffusion Variants

| Item | Value |
|------|-------|
| **Branch** | `innovation-brainstorm` |
| **Plan file** | `docs/superpowers/plans/2026-03-21-patchdiffusion-implementation.md` |
| **Spec** | `docs/superpowers/specs/2026-03-21-patchdiffusion-design.md` |
| **Tasks** | 12 tasks (Task 1–12), **skip duplicates** (see below) |
| **Estimated time** | 1–2 days |

### Switch branch and sync data

```bash
git checkout innovation-brainstorm

# Same checkpoint/data sync as Phase B — download from HF if needed
# Additionally, the Pure MDLM checkpoint from Phase A is the starting point:
# data/checkpoints/mdlm_full/checkpoint_final.pt
```

### Skip map — tasks already covered by Phase A

> **IMPORTANT**: Phase A creates `src/mdlm_model.py` (class `FullMDLM`, function `apply_masking()`).
> PatchDiffusion Tasks 7–12 import from `src/mdlm.py` (class `MaskedDiffusionTransformer`).
> These are **different files with different APIs**. Tasks 1–6 cannot be simply skipped.
> Instead, they should be **adapted** to wrap/extend Phase A's `FullMDLM` while creating
> the `src/mdlm.py` module that Tasks 7–12 expect.

| PatchDiffusion Task | Overlaps with Phase A | Action |
|--------------------|-----------------------|--------|
| Task 1: Masking schedule utilities | Phase A Task 12 has `apply_masking()` | **ADAPT** — reuse Phase A's masking logic, but create `src/mdlm.py` with PatchDiffusion's expected API (`cosine_schedule()`, `forward_mask()`) wrapping Phase A's `src/mdlm_model.py` |
| Task 2: Forward process (masking) | Phase A Task 12 | **ADAPT** — thin wrapper in `src/mdlm.py` |
| Task 3: MaskedDiffusionTransformer | Phase A Task 12 (`FullMDLM` class) | **ADAPT** — create `MaskedDiffusionTransformer` in `src/mdlm.py` that extends or wraps `FullMDLM` |
| Task 4: Iterative unmasking (sampling) | Phase A Task 12 (`FullMDLM.generate()`) | **ADAPT** — expose as PatchDiffusion API |
| Task 5: Token sequence dataset | Phase A Task 12 (`MDLMTokenDataset`) | **ADAPT** — re-export or extend |
| Task 6: Training script (Pure MDLM) | Phase A Task 14 (`scripts/train_mdlm.py`) | **SKIP** — already trained, checkpoint exists |

### Tasks to execute (unique to PatchDiffusion)

| Task | Content | Why unique |
|------|---------|------------|
| Task 7 | Block Diffusion variant | AR over blocks + MDLM within blocks — new architecture |
| Task 8 | Hierarchical RVQ Diffusion variant | Coarse-to-fine across RVQ levels — new architecture |
| Task 9 | Generation script (all 3 variants) | Unified generation — needs all variants |
| Task 10 | Extended training script (all variants) | Train Block + Hierarchical variants |
| Task 11 | Diffusion-specific evaluation metrics | FDD, diversity, quality metrics unique to diffusion |
| Task 12 | Integration test — full pipeline | End-to-end validation |

### Execution instructions

```bash
# On branch: innovation-brainstorm
# Follow docs/superpowers/plans/2026-03-21-patchdiffusion-implementation.md
# Tasks 1-5: ADAPT — don't build from scratch, wrap Phase A's src/mdlm_model.py
# Task 6: SKIP (Pure MDLM already trained in Phase A)
# Tasks 7-12: Run fully
# All commits stay on this branch
```

### Completion gate

```bash
# All 3 variant checkpoints exist
ls data/checkpoints/mdlm_full/checkpoint_final.pt          # Pure (from Phase A)
ls data/checkpoints/block_diffusion/checkpoint_final.pt     # Block
ls data/checkpoints/hierarchical_rvq/checkpoint_final.pt    # Hierarchical

# Evaluation results
cat results/patchdiffusion/evaluation_results.json
```

---

## Post-Execution: Merge Strategy

After all three phases complete, merge results back to `main`:

```bash
git checkout main

# Merge fullscale results first (foundation)
git merge fullscale-experiment --no-ff -m "merge: full-scale experiment results"

# Merge theory-driven (may have conflicts in shared files — resolve favoring theory-driven)
git merge theory-driven-design --no-ff -m "merge: theory-driven analysis + curvature codebook"

# Merge patchdiffusion (may have conflicts in MDLM files — resolve favoring patchdiffusion)
git merge innovation-brainstorm --no-ff -m "merge: PatchDiffusion variants"

git push
```

**Conflict resolution priority**: If files conflict across branches, prefer the branch that was worked on later (it has the most up-to-date version). Specifically: `src/mdlm_model.py` may conflict between fullscale and patchdiffusion — prefer patchdiffusion's version (it extends the model).

**Post-merge cleanup**: After merging, the codebase will have both `src/mdlm_model.py` (Phase A) and `src/mdlm.py` (Phase C). Consolidate into a single module if needed, or keep both if `src/mdlm.py` is a thin wrapper.

---

## Summary

| Phase | Branch | Plan File | Tasks (total) | Tasks (to run) | Est. Time |
|-------|--------|-----------|---------------|-----------------|-----------|
| A | `fullscale-experiment` | `2026-03-21-fullscale-experiment.md` | 16 | **16** (all) | 2–3 days |
| B | `theory-driven-design` | `2026-03-20-meshlex-theory-driven-implementation.md` | 15 | **~10** (skip 0.1, 1-2, 5; partial 3) | 1–2 days |
| C | `innovation-brainstorm` | `2026-03-21-patchdiffusion-implementation.md` | 12 | **~11** (adapt 1-5, skip 6) | 1–2 days |
| **Total** | | | **43** | **~37** | **4–7 days** |
