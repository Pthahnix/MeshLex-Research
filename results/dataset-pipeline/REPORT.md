# Dataset Pipeline — COMPLETE

**Generated:** 2026-03-21 12:31 UTC

## Combined Statistics (Objaverse + ShapeNet)

|                  | Objaverse    | ShapeNet     | Combined      |
|------------------|-------------|-------------|---------------|
| Meshes OK        | 32,136      | 40,419      | **72,555**    |
| Meshes Fail      | 14,364      | 12,053      | 26,417        |
| Success Rate     | 69.1%       | 77.0%       | **73.3%**     |
| Total Patches    | 4,619,061   | 6,186,952   | **10,806,013**|
| Avg Patches/Mesh | 143.7       | 153.1       | 148.9         |
| Categories       | ~1000+      | 55 (53 ok)  | —             |

## ShapeNet Details
- **Categories**: 57 attempted → 55 successful, 2 errors (bicycle/boat: 404 on HF)
- **Largest categories**: table (7,301 ok), chair (6,132 ok), sofa (2,716 ok)
- **Lowest success rates**: car (14.0%), motorbike (22.8%), train (33.7%) — complex vehicle meshes

## Timeline
- Pipeline started: 04:35 UTC
- HF rate limit pause: 11:15–11:20 UTC (128 commits/hour, fixed with sub_batch_size=2000)
- Pipeline completed: 12:31 UTC
- **Total wall time: ~8 hours**

## HF Dataset
- Repo: `Pthahnix/MeshLex-Patches`
- Format: Parquet (via Daft)
- Dual normalization: PCA + noPCA stored per patch
