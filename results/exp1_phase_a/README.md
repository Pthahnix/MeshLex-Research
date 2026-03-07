# Experiment 1 Phase A: 5-Category Data Preparation Report

## Summary

| Category | Meshes | Patches | Median P/M | Median Faces |
|----------|--------|---------|------------|-------------|
| airplane | 29 | 2,088 | 62 | 2,180 |
| car | 30 | 1,850 | 53 | 1,853 |
| chair | 400 | 12,475 | 29 | 1,000 |
| lamp | 51 | 2,215 | 31 | 1,082 |
| table | 53 | 1,645 | 27 | 932 |
| **TOTAL** | **563** | **20,273** | - | - |

## Train/Test Split

**Train categories** (80/20 mesh-level split):
- airplane: 24 train (1,769 patches) / 5 test (319 patches)
- chair: 320 train (9,956 patches) / 80 test (2,519 patches)
- table: 43 train (1,370 patches) / 10 test (275 patches)

**Cross-category test** (unseen categories):
- car: 30 meshes (1,850 patches)
- lamp: 51 meshes (2,215 patches)

**Total training patches: 13,095** | **Same-cat test: 3,113** | **Cross-cat test: 4,065**

## Notes

- Decimation filter: `target_faces * 5` (5000 faces max) — allows higher-poly meshes that pyfqmr can't fully reduce
- Airplane/car meshes tend to have higher poly counts from Objaverse (median ~2000 faces vs target 1000)
- Chair dominates training data (76%), but airplane/table provide meaningful diversity
- Data is sufficient for validation experiment (Go/No-Go threshold, not paper-quality)

## Files

- `patch_distribution.png` — patches/mesh histogram per category
- `face_distribution.png` — face count histogram per category
- `preview_{category}_{0,1}.png` — mesh preview renders (4 views each)
