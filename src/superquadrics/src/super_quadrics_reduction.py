import numpy as np
import open3d as o3d
import os
import sys
import copy
from tqdm import tqdm

# Path setup BEFORE importing local modules
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(src_dir, 'utils'))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # For ems_core

import utils
from ems_core import Superquadric, EMSFitter


class SuperquadricFitter:
    def __init__(self, pcd_path, output_dir):

        self.pcd_path = pcd_path
        self.output_dir = output_dir
        self.pcd = None
    
    def _load_point_cloud(self):
        """Load the point cloud from the specified path."""
        print(f"Loading {self.pcd_path}...")
        try:
            self.pcd = o3d.io.read_point_cloud(self.pcd_path)
        except:
            print(f"Error: Could not load file {self.pcd_path}")
            return False
        
        if self.pcd.is_empty():
            print("Error: Point cloud is empty.")
            return False
        
        return True
    
    def fit(self, mode='single', **kwargs):
        if not self._load_point_cloud():
            return
        
        if mode == 'single':
            self.single_superquadric()
        elif mode == 'iterative':
            self.iterative_sq(**kwargs)
        else:
            print(f"Error: Unknown mode '{mode}'. Use 'single' or 'iterative'.")
    
    def single_superquadric(self):
        """Fits a single superquadric to the point cloud."""
        print("\n=== Starting Superquadric Fitting ===")
        fitter = EMSFitter(self.pcd, init_type='BBOX')
        fitter.w_o_prior = 0.5
        
        pbar = tqdm(total=100, desc="EMS Fitting")
        sq_model = fitter.fit(max_iters=100, external_pbar=pbar)
        pbar.close()
        
        # Analyze results
        num_inliers, num_outliers, _ = utils.analyze_inliers(
            sq_model, fitter.points, distance_threshold=0.1
        )
        
        # Get OBB and PCA info
        obb = self.pcd.get_oriented_bounding_box()
        mean, cov = self.pcd.compute_mean_and_covariance()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Save output
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, 'output.txt')
        
        with open(output_file, 'w') as f:
            f.write("=== Superquadric Fitting Results ===\n\n")
            
            f.write("1. Superquadric Parameters:\n")
            f.write(f"   ax: {sq_model.ax:.6f}\n")
            f.write(f"   ay: {sq_model.ay:.6f}\n")
            f.write(f"   az: {sq_model.az:.6f}\n")
            f.write(f"   e1: {sq_model.e1:.6f}\n")
            f.write(f"   e2: {sq_model.e2:.6f}\n\n")
            
            f.write("2. Point Cloud Statistics:\n")
            f.write(f"   Total Points: {len(fitter.points)}\n")
            f.write(f"   Surface Inliers (<10cm): {num_inliers}\n")
            f.write(f"   Deviation Outliers: {num_outliers}\n\n")
            
            f.write("3. Oriented Bounding Box:\n")
            f.write(f"   Center: [{obb.center[0]:.6f}, {obb.center[1]:.6f}, {obb.center[2]:.6f}]\n")
            f.write(f"   Extent: [{obb.extent[0]:.6f}, {obb.extent[1]:.6f}, {obb.extent[2]:.6f}]\n\n")
            
            f.write("4. PCA Analysis:\n")
            f.write(f"   Eigenvalues: {eigenvalues}\n")
            f.write("   Eigenvectors (Columns):\n")
            f.write(f"{eigenvectors}\n")
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def iterative_sq(self, max_iterations=5, min_points=500, threshold=0.10):
        """
        Fits multiple superquadrics iteratively to capture hierarchical detail.
        
        Args:
            max_iterations: Maximum number of SQ levels to fit
            min_points: Minimum points required to continue
            threshold: Distance threshold in meters for inlier classification
        """
        current_pcd = copy.deepcopy(self.pcd)
        final_sq_list = []
        
        print(f"\n=== Starting Iterative Superquadric Reduction (Max {max_iterations} Levels) ===")
        
        for i in range(max_iterations):
            n_points = len(current_pcd.points)
            print(f"\n--- Iteration {i+1}: Fitting on {n_points} points ---")
            
            if n_points < min_points:
                print("Stopping: Too few points remaining.")
                break
                
            # Fit SQ
            fitter = EMSFitter(current_pcd, init_type='BBOX')
            fitter.w_o_prior = 0.6
            
            pbar = tqdm(total=50, desc=f"Fitting Level {i+1}")
            sq_model = fitter.fit(max_iters=50, external_pbar=pbar)
            pbar.close()
            
            # Identify Inliers / Outliers
            pts_canonical = fitter.points
            mu = sq_model.radial_distance_approximation(pts_canonical)
            dists = np.linalg.norm(pts_canonical - mu, axis=1)
            
            inlier_mask = dists <= threshold
            outlier_mask = ~inlier_mask
            
            num_inliers = np.sum(inlier_mask)
            num_outliers = np.sum(outlier_mask)
            
            print(f"Result: {num_inliers} Inliers (Surface), {num_outliers} Outliers (Deviations)")
            
            # Store SQ parameters
            final_sq_list.append({
                'level': i + 1,
                'params': [sq_model.ax, sq_model.ay, sq_model.az, sq_model.e1, sq_model.e2],
                'center': fitter.center,
                'rotation': fitter.R_init,
                'num_inliers': num_inliers,
                'num_outliers': num_outliers,
                'coverage_points': n_points
            })
            
            # Update for next iteration
            if num_outliers < min_points:
                print("Stopping: Deviation set too small.")
                break
                
            next_pcd = current_pcd.select_by_index(np.where(outlier_mask)[0])
            current_pcd = next_pcd
        
        # Save results
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, 'output.txt')
        
        with open(output_file, 'w') as f:
            f.write("=== Iterative Superquadric Fitting Results ===\n\n")
            f.write(f"Total Levels Fitted: {len(final_sq_list)}\n")
            f.write(f"Original Points: {len(self.pcd.points)}\n\n")
            
            for sq_data in final_sq_list:
                f.write(f"--- Level {sq_data['level']} ---\n")
                f.write(f"   Parameters (ax, ay, az, e1, e2): {sq_data['params']}\n")
                f.write(f"   Center: {sq_data['center']}\n")
                f.write(f"   Coverage: {sq_data['coverage_points']} points\n")
                f.write(f"   Inliers: {sq_data['num_inliers']}\n")
                f.write(f"   Outliers: {sq_data['num_outliers']}\n\n")
        
        print(f"\n✓ Results saved to: {output_file}")
