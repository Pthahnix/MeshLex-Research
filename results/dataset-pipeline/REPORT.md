# Dataset Pipeline Progress Report — ShapeNet Phase

**Generated:** 2026-03-21 05:48 UTC

## Overall Status
- **Phase**: ShapeNet streaming (Phase D-2)
- **Pipeline PID**: 223161 (running)
- **Categories completed**: 6 / 55 (11%)
- **Currently processing**: bench (02828884) — 1813 models

## Stats
- Meshes OK: 3,442
- Meshes fail: 2,231
- Success rate: 60.7%
- Total patches: 756,693
- Avg patches/mesh: 219.8

## Per-Category Breakdown
| Category | Synset | OK | Fail | Patches |
|----------|--------|----|------|---------|
| airplane | 02691156 | 1,957 | 2,088 | 549,219 |
| trash_bin | 02747177 | 302 | 41 | 60,980 |
| bag | 02773838 | 79 | 4 | 14,038 |
| basket | 02801938 | 88 | 25 | 10,941 |
| bathtub | 02808440 | 803 | 53 | 87,883 |
| bed | 02818832 | 213 | 20 | 33,632 |

## Disk Usage
- Used: 16GB / 80GB (19%)
- Free: 65GB

## Latest Log
```
[bed] Done: 213 ok, 20 fail, 33632 patches
[bench] Downloading category 02828884.zip...
[bench] Downloaded + extracted in 18s
[bench] Found 1813 models
```

## Notes
- Pipeline started at 04:35 UTC. Sub-batches of 500 meshes written to HF per category.
- airplane had high fail rate (51.6%) likely due to complex/non-manifold meshes.
- Smaller categories (bag, basket) processed in under a minute each.
- Objaverse phase (D-1) completed earlier: 93 batches, 32,136 OK, 4,619,061 patches.
