# Dry-Run Dataset Pipeline Visualization

**Date:** 2026-03-20

## Dataset Summary

- Total meshes: 53
- Total patches: 2997
- Sources: {'shapenet': np.int64(2964), 'objaverse': np.int64(33)}
- Categories: ['Band_Aid', 'rocket']

## Visualizations

### Statistics
![Stats](stats_summary.png)

### Objaverse Samples

#### d4c9180a46cf401f...
![Overview](objaverse/d4c9180a46cf401f_overview.png)
![Normalization](objaverse/d4c9180a46cf401f_norm_compare.png)

#### dc19a68329ab435f...
![Overview](objaverse/dc19a68329ab435f_overview.png)
![Normalization](objaverse/dc19a68329ab435f_norm_compare.png)

### ShapeNet Samples

#### 04099429_15474cf9caa...
![Overview](shapenet/04099429_15474cf9caa_overview.png)
![Normalization](shapenet/04099429_15474cf9caa_norm_compare.png)

#### 04099429_1a3ef9b0c9c...
![Overview](shapenet/04099429_1a3ef9b0c9c_overview.png)
![Normalization](shapenet/04099429_1a3ef9b0c9c_norm_compare.png)

#### 04099429_1ab4a282a80...
![Overview](shapenet/04099429_1ab4a282a80_overview.png)
![Normalization](shapenet/04099429_1ab4a282a80_norm_compare.png)

#### 04099429_2407c2684ee...
![Overview](shapenet/04099429_2407c2684ee_overview.png)
![Normalization](shapenet/04099429_2407c2684ee_norm_compare.png)

#### 04099429_24d392e5178...
![Overview](shapenet/04099429_24d392e5178_overview.png)
![Normalization](shapenet/04099429_24d392e5178_norm_compare.png)

#### 04099429_3c43ddee5e1...
![Overview](shapenet/04099429_3c43ddee5e1_overview.png)
![Normalization](shapenet/04099429_3c43ddee5e1_norm_compare.png)

#### 04099429_3e75a7a2f8f...
![Overview](shapenet/04099429_3e75a7a2f8f_overview.png)
![Normalization](shapenet/04099429_3e75a7a2f8f_norm_compare.png)

#### 04099429_3f3232433c2...
![Overview](shapenet/04099429_3f3232433c2_overview.png)
![Normalization](shapenet/04099429_3f3232433c2_norm_compare.png)

#### 04099429_4c553d60964...
![Overview](shapenet/04099429_4c553d60964_overview.png)
![Normalization](shapenet/04099429_4c553d60964_norm_compare.png)

#### 04099429_53009165a8a...
![Overview](shapenet/04099429_53009165a8a_overview.png)
![Normalization](shapenet/04099429_53009165a8a_norm_compare.png)

### Processing Pipeline
Shows: World Space → Centered → PCA Rotated → PCA+Scale → No-PCA+Scale

![Pipeline](pipeline/objaverse_d4c9180a46cf401f_pipeline.png)
![Pipeline](pipeline/objaverse_dc19a68329ab435f_pipeline.png)
![Pipeline](pipeline/shapenet_04099429_15474cf9caa_pipeline.png)
![Pipeline](pipeline/shapenet_04099429_1a3ef9b0c9c_pipeline.png)
![Pipeline](pipeline/shapenet_04099429_1ab4a282a80_pipeline.png)
