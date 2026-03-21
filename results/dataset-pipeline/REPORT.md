# Dataset Pipeline Progress Report — ShapeNet Phase

**Generated:** 2026-03-21 08:48 UTC

## Overall Status
- **Phase**: ShapeNet streaming (Phase D-2)
- **Pipeline PID**: 223161 (running)
- **Categories completed**: 28 / 55 (51%) — 26 OK, 2 errors
- **Currently processing**: guitar (03467517)

## Stats
- Meshes OK: 18,310
- Meshes fail: 7,312
- Success rate: 71.5%
- Total patches: 2,908,970
- Avg patches/mesh: 158.9

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
| car | 02958343 | 492 | 3,022 | 254,232 |
| cellphone | 02992529 | 786 | 45 | 121,597 |
| chair | 03001627 | 6,132 | 646 | 802,911 |
| clock | 03046257 | 581 | 70 | 82,860 |
| keyboard | 03085013 | 47 | 18 | 11,394 |
| dishwasher | 03207941 | 80 | 13 | 8,245 |
| display | 03211117 | 983 | 110 | 92,235 |
| earphone | 03261776 | 61 | 12 | 11,107 |
| faucet | 03325088 | 709 | 35 | 100,964 |
| file_cabinet | 03337140 | 231 | 67 | 15,366 |

## Disk Usage
- Used: 14GB / 80GB (17%)
- Free: 67GB

## Latest Log
```
[file_cabinet] Done: 231 ok, 67 fail, 15366 patches
[guitar] Downloading category 03467517.zip...
```

## Timing
- Pipeline started: 04:35 UTC (~4.2h elapsed, 238 min CPU)
- Processing rate: ~107 meshes/min (speeding up with smaller categories)
- Remaining: ~27 categories, biggest are table(8509), sofa(3173), rifle(2373), loudspeaker(2382), lamp(2318)
- 2 categories (bicycle, boat) 404 errors — zips not on HF
- Objaverse phase (D-1) completed: 32,136 OK, 4,619,061 patches
