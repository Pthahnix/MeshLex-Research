# MeshLex v2 Generation Evaluation Report

## Overview
- Generated meshes evaluated across 4 temperature settings
- Each setting: 10 meshes, 130 patches/mesh, 3900 points/mesh

## Token Distribution

| Temperature | Patches | Unique Tokens | CB L1 Util | CB L2 Util | CB L3 Util |
|-------------|---------|---------------|------------|------------|------------|
| T=0.7 | 1300 | 1403 | 32.8% | 41.8% | 47.8% |
| T=0.8 | 1300 | 1376 | 31.0% | 35.3% | 42.0% |
| T=0.9 | 1300 | 1286 | 29.0% | 45.2% | 49.8% |
| T=1.0 | 1300 | 1469 | 40.5% | 43.5% | 50.2% |

## Point Cloud Quality

| Temperature | Mean Pairwise CD | Spatial Spread | Point Density |
|-------------|-----------------|----------------|---------------|
| T=0.7 | 0.4489 | 11.420 ± 7.967 | 302.2 |
| T=0.8 | 1.0786 | 11.377 ± 8.987 | 847349.7 |
| T=0.9 | 0.2475 | 11.290 ± 8.344 | 206.6 |
| T=1.0 | 0.2626 | 7.175 ± 6.290 | 677.5 |

## Key Observations

- All meshes generate exactly 130 patches (max_len=910, 7 tokens/patch)
- Codebook utilization indicates how many unique codes the AR model uses
- Higher pairwise CD = more diverse generations
- Spatial spread measures the bounding box diagonal of each generated mesh

## Files
- `evaluation_results.json` — full numeric results
- `evaluation_dashboard.png` — visual dashboard