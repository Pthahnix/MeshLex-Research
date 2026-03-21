# Dataset Pipeline Progress Report — ShapeNet Phase

**Generated:** 2026-03-21 06:33 UTC

## Overall Status
- **Phase**: ShapeNet streaming (Phase D-2)
- **Pipeline PID**: 223161 (running)
- **Categories completed**: 18 / 55 (33%) — 16 OK, 2 errors
- **Currently processing**: car (02958343) — 3514 models

## Stats
- Meshes OK: 8,208
- Meshes fail: 3,274
- Success rate: 71.5%
- Total patches: 1,408,059
- Avg patches/mesh: 171.5

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
| bicycle | 02834778 | — | — | ERROR: 404 |
| birdhouse | 02843684 | 72 | 1 | 3,724 |
| boat | 02858304 | — | — | ERROR: 404 |
| bookshelf | 02871439 | 389 | 63 | 29,181 |
| bottle | 02876657 | 491 | 7 | 36,917 |
| bowl | 02880940 | 182 | 4 | 11,709 |
| bus | 02924116 | 404 | 535 | 115,134 |
| cabinet | 02933112 | 1,467 | 104 | 173,482 |
| camera | 02942699 | 99 | 14 | 15,777 |
| can | 02946921 | 104 | 4 | 8,203 |
| cap | 02954340 | 55 | 1 | 5,808 |

## Disk Usage
- Used: 41GB / 80GB (51%)
- Free: 40GB
- Note: car zip extraction is large (~27GB in /tmp/meshlex_shapenet/)

## Latest Log
```
[cap] Done: 55 ok, 1 fail, 5808 patches
[car] Downloaded + extracted in 158s
[car] Found 3514 models
```

## Notes
- 2 categories (bicycle, boat) returned 404 — zips not on HF ShapeNet repo.
- Pipeline started at 04:35 UTC (~118 min elapsed).
- car is a large category (3514 models), expect ~30 min to process.
- Objaverse phase (D-1) completed earlier: 93 batches, 32,136 OK, 4,619,061 patches.
