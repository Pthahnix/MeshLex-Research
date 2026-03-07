# MeshLex Research

**MeshLex: Learning a Topology-aware Patch Vocabulary for Compositional Mesh Generation**

A research project exploring whether 3D triangle meshes possess a finite, reusable "vocabulary" of local topological patterns — analogous to how BPE tokens form a vocabulary for natural language.

## Motivation

All current mesh generation methods serialize meshes into 1D token sequences and feed them to transformers. They differ only in *how* they serialize (BPT, EdgeBreaker, FACE, etc.) and *what* backbone they use (GPT, DiT, Mamba, etc.). But mesh is fundamentally a graph — forcing it into a sequence is like cutting a map into strips and asking a model to reassemble it.

MeshLex takes a different approach: instead of generating meshes face-by-face, we learn a **codebook of ~4096 topology-aware patches** (each covering 20-50 faces) and generate meshes by selecting, deforming, and assembling patches from this codebook. A 4000-face mesh becomes ~130 tokens — an order of magnitude more compact than the state-of-the-art (FACE, ICML 2026: ~400 tokens).

## Core Hypothesis

> Mesh local topology is low-entropy and universal across object categories. A finite codebook of ~4096 topology prototypes, combined with continuous deformation parameters, can reconstruct arbitrary meshes with high fidelity.

## Research Evolution

This project evolved through multiple stages of literature analysis, idea generation, and competitive evaluation:

| Stage | Document | Summary |
|-------|----------|---------|
| 0 | `00_original_prompt.md` | Initial vision: Large Mesh Model (LMM) for unified reconstruction + generation |
| 1 | `01_gap_analysis_lmm.md` | 75+ paper survey, 7 research gaps identified |
| 2 | `02_idea_generation_lmm.md` | 5 candidate ideas → MeshFoundation v2 selected |
| 3 | `03_experiment_design_lmm.md` | Full experiment design for MeshFoundation v2 |
| 4 | `04_pplx_comprehensive_evaluation.md` | Independent review (Gap 88% accuracy, Idea 78/100, Exp 82/100) |
| 5 | `05_cc_pplx_debate.md` | Paradigm shift: from "better serialization" to "should we serialize at all?" → MeshLex |
| 6 | `06_plan_meshlex_validation.md` | Validation experiment plan for MeshLex feasibility |

## Current Status

**Phase: Validation experiment design complete, pending execution.**

The next step is a 2-3 day feasibility experiment on ShapeNet to verify the core hypothesis before committing to full-scale development. See `06_plan_meshlex_validation.md` for details.

## Key Differentiators

| | MeshMosaic | FreeMesh | FACE | **MeshLex** |
|---|---|---|---|---|
| Approach | Divide-and-conquer | BPE on coordinates | One-face-one-token | **Topology patch codebook** |
| Still per-face generation? | Yes (within each patch) | Yes (merged coordinates) | Yes | **No** |
| Has codebook? | No | Yes (coordinate-level) | No | **Yes (topology-level)** |
| Compression (4K faces) | N/A | ~300 tokens | ~400 tokens | **~130 tokens** |

## Repository Structure

```
.context/                          # Research documents (chronological)
├── 00-06_*.md                     # Research evolution documents
├── material/                      # Analysis summaries of key papers
└── paper/                         # [gitignored] 300+ paper markdown files
CLAUDE.md                          # Project conventions for Claude Code
README.md                          # This file
```

## Target Venue

CCF-A conferences: CVPR / NeurIPS / ICCV

## License

Research use only. Not yet published.
