# 3D Pose Refinement

## Overview
This project contains two implementations of 3D pose refinement using identical optimization algorithms:

1. **utils/fit_pose3d.py** - Clean PyTorch implementation 
2. **external/egohumans/lib/models/fit_pose3d.py** - Original mmcv-dependent implementation

## Implementation Comparison

### Key Similarities:
- **Same optimization objective**: Both use identical loss functions (limb length, symmetry, init pose, temporal)
- **Same skeleton structure**: Both use COCO 17-keypoint format
- **Same optimizer**: Both use LBFGS with closure function
- **Same early stopping**: Both use `_compute_relative_change` with ftol threshold
- **Same multi-epoch structure**: Both have global iterations and epochs

### Key Differences:

#### Framework:
- **utils/**: Pure PyTorch implementation with custom `build_optimizer`
- **external/**: Uses mmcv's `build_optimizer` function

#### Minor Implementation Details:
- **Temporal loss**: external/ missing weight multiplication (line 164 vs utils/ line 146)
- **Symmetry loss**: external/ uses `.sum()` (line 155) vs utils/ uses `.mean()` (line 134)  
- **Verbose output**: Different formatting styles and content
- **Constructor**: external/ takes `human_name, global_iter, total_global_iters` vs utils/ takes just `cfg`
- **Input handling**: utils/ has numpy-to-tensor conversion and device handling (lines 78-82)
- **Assertions**: utils/ has additional shape validation (lines 139-140, 168)

## Conclusion
Both implementations use the **same pose refinement algorithm** - the utils/ version is a cleaner, standalone PyTorch implementation while external/ is the original mmcv-dependent version. The core optimization logic is identical.