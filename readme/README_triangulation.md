# Triangulation Methods for Human Pose Estimation

This document compares two triangulation approaches found in the codebase and provides recommendations for human pose estimation.

## Comparison of Existing Methods

### aniposelib/cameras.py - `triangulate_ransac`

**Location**: Method of `CameraGroup` class (line 632)

**Call Flow**: `triangulate_ransac` → `triangulate_possible` → `triangulate` (with `fast=False`)

**Key Features**:
- **Input**: CxNx2 array (cameras x points x 2D coordinates)
- **RANSAC Strategy**: Uses `triangulate_possible` with exhaustive combination search
- **Core Algorithm**: 
  - Reshapes points to CxNx1x2 format
  - Uses `itertools.product` to try all camera combinations
  - For each combination, calls `triangulate` with `fast=False` (default)
  - Picks best solution based on reprojection error threshold (0.5 pixels)
- **Triangulation Method**: **SVD-based Direct Linear Transformation (DLT)** via `triangulate_simple`
- **Output**: Returns 3D points, picked point indices, 2D points used, and errors

**Advantages**:
- **_Uses same SVD-based DLT as egohumans (not OpenCV pairwise)_**
- Robust through exhaustive combination search
- Well-tested for calibration scenarios
- Handles multiple point options per camera
- Early stopping when reprojection error < 0.5 pixels

**Limitations**:
- **_Actually uses multi-view DLT, not pairwise triangulation_**
- No confidence weighting in the DLT construction
- Computationally expensive due to exhaustive combination testing
- Always uses slower `fast=False` mode (more accurate but slower)

### egohumans/triangulation.py - `triangulate_ransac`

**Location**: Method of `Triangulator` class (line 332)

**Key Features**:
- **Input**: Projection matrices + 2D points with confidence scores
- **RANSAC Strategy**: Systematic iteration through all view pairs
- **Core Algorithm**:
  - Creates all possible view pairs
  - Triangulates using Direct Linear Transformation (DLT)
  - Finds inliers based on reprojection error threshold
  - Optional direct optimization using `least_squares`
- **Triangulation Method**: Direct Linear Transformation (DLT) with SVD
- **Output**: Returns 3D point, inlier view indices, and reprojection error vector

**Camera Selection Strategy**:
The algorithm implements a hierarchical approach with primary and secondary cameras:
1. **Primary cameras** are tried first (lines 115-138)
2. **Secondary cameras** are used as fallback if:
   - Primary cameras don't have enough inliers after triangulation (`< min_views`)
   - OR insufficient primary cameras initially but enough combined (`>= secondary_min_views`)
3. **Strategy hierarchy**:
   - **Best case**: Use only primary cameras (most trusted)
   - **Fallback 1**: Use primary + secondary if primary alone fails
   - **Fallback 2**: Use primary + secondary if insufficient primary cameras

**Advantages**:
- Native multi-view support (N ≥ 2 cameras)
- Incorporates confidence scores naturally
- Optional non-linear refinement
- More principled for multi-view scenarios


**Limitations**:
- DLT minimizes algebraic error (less accurate than geometric error)
- More sensitive to noise than optimal methods


## Technical Differences: Combination Strategy (Both Use DLT)

**Important Note**: Both methods actually use the same underlying **SVD-based Direct Linear Transformation (DLT)** for triangulation. The main difference is in their **combination strategy** for handling multiple cameras.

### aniposelib's `triangulate_ransac` Strategy
- **Method**: Exhaustive combination search
- **Algorithm**: 
  - Tests all possible camera combinations using `itertools.product`
  - For each combination, uses SVD-based DLT via `triangulate_simple`
  - Selects combination with lowest reprojection error
- **Advantages**: Guaranteed to find optimal camera combination, early stopping at 0.5px threshold
- **Limitations**: Computationally expensive (O(2^n) combinations), no confidence weighting

### egohumans' `triangulate_ransac` Strategy  
- **Method**: Systematic view pair iteration with inlier consensus
- **Algorithm**:
  - Iterates through all view pairs systematically
  - Uses SVD-based DLT for triangulation
  - Builds consensus based on reprojection error threshold
  - Optional non-linear refinement with `least_squares`
- **Advantages**: Confidence weighting, hierarchical camera selection, non-linear refinement
- **Limitations**: May not find globally optimal combination, more complex parameter tuning

### Underlying Triangulation Method (Same for Both)
**Direct Linear Transformation (DLT) with SVD**:
- **Method**: Solves triangulation as linear system using SVD
- **Algorithm**: Constructs constraint matrix A, solves A*X=0 using SVD
- **Advantages**: Multi-view support, easy confidence weighting, closed-form solution
- **Limitations**: Minimizes algebraic error, less robust to noise than geometric methods
