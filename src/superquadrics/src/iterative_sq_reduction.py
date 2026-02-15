
import numpy as np
import open3d as o3d
import os
import sys
import copy
from tqdm import tqdm

# Add parent dir to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(os.path.join(grandparent_dir, 'utils'))

from ems_core import Superquadric, EMSFitter
import utils

# CONFIG
MAX_ITERATIONS = 5
MIN_POINTS = 500
DEVIATION_THRESHOLD = 0.10  # 10 cm

def fit_iterative_superquadrics(pcd_path):
    """
    Fits multiple superquadrics iteratively to capture hierarchical detail.
    
    Args:
        pcd_path: Absolute path to the point cloud file (.ply)
    """
    print(f"Loading {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    if pcd.is_empty():
        print("Error: Point cloud is empty.")
        return

    current_pcd = copy.deepcopy(pcd)
    final_sq_list = []
    
    print(f"\n=== Starting Iterative Superquadric Reduction (Max {MAX_ITERATIONS} Levels) ===")
    
    for i in range(MAX_ITERATIONS):
        n_points = len(current_pcd.points)
        print(f"\n--- Iteration {i+1}: Fitting on {n_points} points ---")
        
        if n_points < MIN_POINTS:
            print("Stopping: Too few points remaining.")
            break
            
        # 1. Fit SQ
        fitter = EMSFitter(current_pcd, init_type='BBOX')
        fitter.w_o_prior = 0.6  # Higher outlier probability for recursive steps
        
        # Fit
        pbar = tqdm(total=50, desc=f"Fitting Level {i+1}")
        sq_model = fitter.fit(max_iters=50, external_pbar=pbar)
        pbar.close()
        
        # 2. Identify Inliers / Outliers (Deviations)
        pts_canonical = fitter.points
        mu = sq_model.radial_distance_approximation(pts_canonical)
        dists = np.linalg.norm(pts_canonical - mu, axis=1)
        
        inlier_mask = dists <= DEVIATION_THRESHOLD
        outlier_mask = ~inlier_mask
        
        num_inliers = np.sum(inlier_mask)
        num_outliers = np.sum(outlier_mask)
        
        print(f"Result: {num_inliers} Inliers (Surface), {num_outliers} Outliers (Deviations)")
        
        # 3. Store SQ parameters
        final_sq_list.append({
            'level': i + 1,
            'params': [sq_model.ax, sq_model.ay, sq_model.az, sq_model.e1, sq_model.e2],
            'center': fitter.center,
            'rotation': fitter.R_init,
            'num_inliers': num_inliers,
            'num_outliers': num_outliers,
            'coverage_points': n_points
        })
        
        # 4. Update Point Cloud for Next Iteration (use outliers)
        if num_outliers < MIN_POINTS:
            print("Stopping: Deviation set too small.")
            break
            
        next_pcd = current_pcd.select_by_index(np.where(outlier_mask)[0])
        current_pcd = next_pcd
    
    # Save results to output file
    output_dir = os.path.join(parent_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'iterative_output.txt')
    
    with open(output_file, 'w') as f:
        f.write("=== Iterative Superquadric Fitting Results ===\n\n")
        f.write(f"Total Levels Fitted: {len(final_sq_list)}\n")
        f.write(f"Original Points: {len(pcd.points)}\n\n")
        
        for sq_data in final_sq_list:
            f.write(f"--- Level {sq_data['level']} ---\n")
            f.write(f"   Parameters (ax, ay, az, e1, e2): {sq_data['params']}\n")
            f.write(f"   Center: {sq_data['center']}\n")
            f.write(f"   Coverage: {sq_data['coverage_points']} points\n")
            f.write(f"   Inliers: {sq_data['num_inliers']}\n")
            f.write(f"   Outliers: {sq_data['num_outliers']}\n\n")
    
    print(f"\n✓ Results saved to: {output_file}")

