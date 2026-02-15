# Point Cloud Compression - Superquadric Fitting

This project implements a superquadric-based point cloud compression pipeline for rock surface mapping.

## Project Structure

```
src/
├── superquadrics/          # Main superquadric fitting module
│   ├── main_sq.py          # Entry point (SuperquadricProcessor class)
│   ├── output/             # Generated results (output.txt)
│   └── src/
│       ├── ems_core.py                      # EMS algorithm implementation
│       ├── super_quadrics_reduction.py      # SuperquadricFitter class (single + iterative)
│       └── iterative_sq_reduction.py        # Legacy iterative fitting (kept for reference)
├── utils/                  # Utility functions
│   └── utils.py            # Point cloud processing helpers
└── initial_code/           # Legacy code
    └── visualize_pcd.py    # Basic point cloud visualization
```

## What We Accomplished

### 1. **Removed Visualization Dependencies**
- Stripped all Open3D GUI and rendering code from processing modules
- Made the pipeline headless (no windows, no multiprocessing visualization)
- Processing modules are now pure computation

### 2. **Unified Output**
- Both single and iterative modes output to the same file: `output/output.txt`
- Different headers distinguish the mode:
  - Single: `=== Superquadric Fitting Results ===`
  - Iterative: `=== Iterative Superquadric Fitting Results ===`

### 3. **Class-Based Architecture**
- **`SuperquadricProcessor`** (main_sq.py): Entry point class, handles configuration
- **`SuperquadricFitter`** (super_quadrics_reduction.py): Core fitting logic
  - `fit()`: Dispatcher method
  - `single_superquadric()`: Fits one superquadric
  - `iterative_sq()`: Hierarchical fitting (up to N levels)

### 4. **Flexible Configuration**
- Global constants at top of `main_sq.py`:
  ```python
  MODE = 'single'           # 'single' or 'iterative'
  PCD_FILENAME = 'rock1.ply'
  MAX_ITERATIONS = 5        # For iterative mode
  MIN_POINTS = 500
  THRESHOLD = 0.10          # 10cm inlier threshold
  ```

### 5. **Clean Import Structure**
- Fixed sys.path manipulation to set paths before importing local modules
- Ensured `utils` and `ems_core` are importable from nested directory structure

## Usage

### Standalone Execution
```bash
cd /home/kanav/workspaces/point_cloud_compression
python3 src/superquadrics/main_sq.py
```

### Programmatic Usage
```python
from src.superquadrics.main_sq import SuperquadricProcessor

# Use default config
processor = SuperquadricProcessor()
processor.run()

# Override config
processor = SuperquadricProcessor(
    pcd_filename='rock2.ply',
    mode='iterative',
    max_iterations=10,
    threshold=0.05  # 5cm threshold
)
processor.run()
```

## Algorithms

### Single Mode
1. Load point cloud
2. Fit one superquadric using EMS (Expectation-Maximization-Switching)
3. Classify points as inliers (<10cm) or outliers (>10cm)
4. Output SQ parameters, statistics, PCA analysis

### Iterative Mode
1. Fit superquadric to entire point cloud (Level 1)
2. Remove inliers, keep outliers (deviations)
3. Fit another superquadric to outliers (Level 2)
4. Repeat until `MAX_ITERATIONS` or `num_outliers < MIN_POINTS`
5. Output hierarchy of superquadrics capturing progressive detail

## Output Format

**Single Mode:**
```
=== Superquadric Fitting Results ===

1. Superquadric Parameters:
   ax, ay, az, e1, e2

2. Point Cloud Statistics:
   Total Points, Inliers, Outliers

3. Oriented Bounding Box:
   Center, Extent

4. PCA Analysis:
   Eigenvalues, Eigenvectors
```

**Iterative Mode:**
```
=== Iterative Superquadric Fitting Results ===

Total Levels Fitted: N
Original Points: M

--- Level 1 ---
   Parameters, Center, Coverage, Inliers, Outliers
   
--- Level 2 ---
   ...
```

## Future Work
- B-spline implementation for deviation encoding (currently deferred)
- Integration of superquadrics + B-splines for full compression pipeline
- Change detection using compressed representations
