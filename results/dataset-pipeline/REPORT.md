# Dataset Pipeline Progress Report — ShapeNet Phase

**Generated:** 2026-03-21 06:00 UTC

## Overall Status
- **Phase**: ShapeNet streaming (Phase D-2)
- **Pipeline PID**: 223161 (running)
- **Categories completed**: 13 / 55 (24%) — 11 OK, 2 errors
- **Currently processing**: bus (02924116) — 939 models

## Stats
- Meshes OK: 6,079
- Meshes fail: 2,616
- Success rate: 69.9%
- Total patches: 1,089,655
- Avg patches/mesh: 179.2

## Per-Category Breakdown
| Category | Synset | OK | Fail | Patches |
|----------|--------|----|------|---------|
| airplane | 02691156 | 1,957 | 2,088 | 549,219 |
| trash_bin | 02747177 | 302 | 41 | 60,980 |
| bag | 02773838 | 79 | 4 | 14,038 |
| basket | 02801938 | 88 | 25 | 10,941 |
| bathtub | 02808440 | 803 | 53 | 87,883 |
| bed | 02818832 | 213 | 20 | 33,632 |
| bench | 02828884 | 1,503 | 310 | 251,431 |
| bicycle | 02834778 | — | — | ERROR: 404 (zip not on HF) |
| birdhouse | 02843684 | 72 | 1 | 3,724 |
| boat | 02858304 | — | — | ERROR: 404 (zip not on HF) |
| bookshelf | 02871439 | 389 | 63 | 29,181 |
| bottle | 02876657 | 491 | 7 | 36,917 |
| bowl | 02880940 | 182 | 4 | 11,709 |

## Disk Usage
- Used: 17GB / 80GB (21%)
- Free: 64GB

## Latest Log
```
[bowl] Done: 182 ok, 4 fail, 11709 patches
[bus] Downloading category 02924116.zip...
[bus] Downloaded + extracted in 35s
[bus] Found 939 models
```

## Notes
- 2 categories (bicycle, boat) returned 404 — their zips are not available on HF ShapeNet repo.
- airplane had high fail rate (51.6%) due to complex/non-manifold meshes; other categories ~90%+ success.
- Pipeline started at 04:35 UTC (~85 min elapsed).
- Objaverse phase (D-1) completed earlier: 93 batches, 32,136 OK, 4,619,061 patches.
