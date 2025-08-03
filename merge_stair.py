#!/usr/bin/env python3
"""
merge_stairs.py

Aligns and merges multiple CSV‑exported stair scans into one dense PLY.

Dependencies:
    pip install open3d
"""

import glob
import numpy as np
import open3d as o3d
import os

def load_csv_cloud(path):
    # Robust CSV loader: find header, skip metadata, handle hex values
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Find header row with X,Y,Z columns
    header_row = 0
    header = None
    for i, line in enumerate(lines):
        cols = line.strip().split(',')
        if any(c.upper() in ['X', 'Y', 'Z'] for c in cols):
            header = cols
            header_row = i
            break
    
    if header is None:
        raise ValueError(f"Could not find X,Y,Z columns in {path}")
    
    # Map column names to indices
    col_map = {name.strip().upper(): idx for idx, name in enumerate(header)}
    try:
        x_col = col_map['X']
        y_col = col_map['Y'] 
        z_col = col_map['Z']
    except KeyError:
        raise ValueError(f"Missing X,Y,Z columns in {path}. Available: {header}")
    
    # Load data, handle hex values and conversion errors
    all_data = []
    for line in lines[header_row+1:]:
        cols = line.strip().split(',')
        if len(cols) >= max(x_col, y_col, z_col) + 1:
            try:
                row_data = []
                for i, col in enumerate(cols):
                    if col.startswith('0x'):  # hex values
                        row_data.append(0.0)
                    else:
                        row_data.append(float(col))
                all_data.append(row_data)
            except ValueError:
                continue  # skip rows with conversion errors
    
    if not all_data:
        raise ValueError(f"No valid numeric data found in {path}")
    
    data = np.array(all_data, dtype=np.float32)
    pts = data[:, [x_col, y_col, z_col]]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return pcd

def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    return pcd_down

def pairwise_register(source_down, target_down, voxel_size):
    fpfh_src = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    fpfh_tgt = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    ransac_res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        fpfh_src, fpfh_tgt,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)
    )

    icp_res = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        max_correspondence_distance=voxel_size * 0.5,
        init=ransac_res.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return icp_res.transformation

def merge_views(csv_pattern, voxel_size=0.05):
    files = sorted(glob.glob(csv_pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {csv_pattern}")

    print(f"Reference: {files[0]}")
    model = load_csv_cloud(files[0])
    model_down = preprocess(model, voxel_size)

    for path in files[1:]:
        print(f"Merging: {os.path.basename(path)}")
        view = load_csv_cloud(path)
        view_down = preprocess(view, voxel_size)

        tf = pairwise_register(view_down, model_down, voxel_size)
        view.transform(tf)
        model += view
        model_down = preprocess(model, voxel_size)

    merged = model.voxel_down_sample(voxel_size)
    return merged

if __name__ == "__main__":
    # Match files stair_9.1.csv … stair_9.9.csv
    input_pattern = r"/home/vincent/ros2_ws/Stairs_Dataset/stair_9/stair_9.[1-9].csv"
    output_ply    = r"/home/vincent/ros2_ws/Stairs_Dataset/stair_9/stair9_merged.ply"
    voxel         = 0.05  # meters

    print("Starting merge of stair_3 scans…")
    merged_cloud = merge_views(input_pattern, voxel_size=voxel)
    print(f"Saving merged cloud ({len(merged_cloud.points)} points) to:\n  {output_ply}")
    o3d.io.write_point_cloud(output_ply, merged_cloud)
    print("Done.")
