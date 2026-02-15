#!/usr/bin/env python3
"""
Main entry point for Superquadric-based point cloud compression.
Can be used as a standalone script or imported as a class.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from src.super_quadrics_reduction import SuperquadricFitter


# ============ CONFIGURATION ============
MODE = 'single'  # 'single' or 'iterative'
PCD_FILENAME = 'rock1.ply'

# Iterative-specific parameters (only used if MODE == 'iterative')
MAX_ITERATIONS = 5
MIN_POINTS = 500
THRESHOLD = 0.10  # 10cm


class SuperquadricProcessor:

    
    def __init__(self, pcd_filename=None, mode=None, output_dir=None,
                 max_iterations=None, min_points=None, threshold=None):

        # Use global config as defaults
        self.pcd_filename = pcd_filename or PCD_FILENAME
        self.mode = mode or MODE
        self.max_iterations = max_iterations or MAX_ITERATIONS
        self.min_points = min_points or MIN_POINTS
        self.threshold = threshold or THRESHOLD
        
        # Set paths
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.pcd_dir = os.path.join(project_root, 'point_clouds')
        self.output_dir = output_dir or os.path.join(current_dir, 'output')
        
        self.pcd_path = os.path.join(self.pcd_dir, self.pcd_filename)
    
    def run(self):
        """Execute the superquadric fitting process."""
        # Verify file exists
        if not os.path.exists(self.pcd_path):
            print(f"Error: Point cloud not found at {self.pcd_path}")
            return
        
        print(f"Processing: {self.pcd_path}")
        print(f"Mode: {self.mode}")
        
        # Create fitter and run
        fitter = SuperquadricFitter(self.pcd_path, self.output_dir)
        fitter.fit(mode=self.mode, max_iterations=self.max_iterations, 
                   min_points=self.min_points, threshold=self.threshold)


def main():
    """Standalone execution entry point."""
    processor = SuperquadricProcessor()
    processor.run()


if __name__ == "__main__":
    main()
