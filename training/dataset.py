#!/usr/bin/env python3
"""
Stairs Dataset Loader for RandLA-Net Training
==============================================

This module handles loading and preprocessing of the ETH Zurich stairs dataset
for training the RandLA-Net stair detection model.

Features:
- Load PLY files with point coordinates and labels
- Multi-scale point cloud preprocessing
- Data augmentation (rotation, jittering, scaling)
- Balanced sampling for class imbalance
- Batch generation with proper padding

Dataset Structure Expected:
    data_path/
    ├── train/
    │   ├── stair_1.ply
    │   ├── stair_2.ply
    │   └── ...
    ├── val/
    │   ├── stair_test_1.ply
    │   └── ...
    └── labels/
        ├── stair_1_labels.txt
        ├── stair_2_labels.txt
        └── ...

Author: Vincent Yeung
"""

# ─── System Imports ──────────────────────────────────────────────────────────────
import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional

# ─── Deep Learning Framework ─────────────────────────────────────────────────────
import torch
from torch.utils.data import Dataset

# ─── Scientific Computing ───────────────────────────────────────────────────────
import numpy as np
from scipy.spatial import cKDTree

# ─── 3D Processing ──────────────────────────────────────────────────────────────
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not available. Using fallback PLY reader.")

# ─── Local Utilities ─────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent / "src" / "my_terrain_seg" / "src"))
from my_terrain_seg.my_model_project.utils import compute_multiscale_indices
from my_terrain_seg.my_model_project.hyperparameters import K_N, SUB_SAMPLING_RATIO


# ─── Dataset Class ───────────────────────────────────────────────────────────────

class StairsDataset(Dataset):
    """
    PyTorch Dataset for ETH Zurich stairs point clouds
    
    This dataset loads PLY files and their corresponding labels for training
    the RandLA-Net stair segmentation model.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        num_points: int = 16384,
        patches_per_scan: int = 50,
        augment: bool = True,
        class_weights: Optional[List[float]] = None
    ):
        """
        Initialize stairs dataset
        
        Args:
            data_path: Path to dataset root directory
            split: 'train' or 'val'
            num_points: Number of points to sample per patch
            patches_per_scan: Number of patches to extract per scan
            augment: Whether to apply data augmentation
            class_weights: Weights for balanced sampling [background, riser, tread]
        """
        self.data_path = Path(data_path)
        self.split = split
        self.num_points = num_points
        self.patches_per_scan = patches_per_scan
        self.augment = augment and (split == 'train')
        self.class_weights = class_weights or [1.0, 1.0, 1.0]
        
        # Load file lists
        self.scan_files = self._load_file_list()
        
        # Calculate total number of patches
        self.total_patches = len(self.scan_files) * self.patches_per_scan
        
        print(f"Loaded {len(self.scan_files)} {split} scans")
        print(f"Total patches: {self.total_patches}")
    
    def _load_file_list(self) -> List[Dict[str, Path]]:
        """Load list of PLY files and their corresponding labels"""
        split_dir = self.data_path / self.split
        label_dir = self.data_path / 'labels'
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        files = []
        for ply_file in split_dir.glob('*.ply'):
            label_file = label_dir / f"{ply_file.stem}_labels.txt"
            
            if label_file.exists():
                files.append({
                    'ply': ply_file,
                    'labels': label_file
                })
            else:
                print(f"Warning: Label file not found for {ply_file.name}")
        
        if len(files) == 0:
            raise RuntimeError(f"No valid PLY/label pairs found in {split_dir}")
        
        return files
    
    def _load_ply(self, ply_path: Path) -> np.ndarray:
        """Load point cloud from PLY file"""
        if HAS_OPEN3D:
            # Use Open3D for robust PLY loading
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = np.asarray(pcd.points, dtype=np.float32)
        else:
            # Fallback: simple PLY reader
            points = self._read_ply_fallback(ply_path)
        
        return points
    
    def _read_ply_fallback(self, ply_path: Path) -> np.ndarray:
        """Fallback PLY reader without Open3D"""
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        
        # Find vertex count
        vertex_count = 0
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.strip() == 'end_header':
                header_end = i + 1
                break
        
        # Read vertex data
        points = []
        for i in range(header_end, header_end + vertex_count):
            coords = lines[i].strip().split()[:3]  # x, y, z
            points.append([float(c) for c in coords])
        
        return np.array(points, dtype=np.float32)
    
    def _load_labels(self, label_path: Path) -> np.ndarray:
        """Load point labels from text file"""
        return np.loadtxt(label_path, dtype=np.int32)
    
    def _augment_points(self, points: np.ndarray) -> np.ndarray:
        """Apply data augmentation to point cloud"""
        if not self.augment:
            return points
        
        # Random rotation around Z-axis (stairs are typically horizontal)
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=np.float32)
        
        points = points @ rotation_matrix.T
        
        # Random jittering
        jitter = np.random.normal(0, 0.01, points.shape).astype(np.float32)
        points += jitter
        
        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        points *= scale
        
        return points
    
    def _sample_patch(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a patch of points with balanced class distribution"""
        if len(points) <= self.num_points:
            # If we have fewer points than needed, pad with repetition
            indices = np.arange(len(points))
            while len(indices) < self.num_points:
                indices = np.concatenate([indices, indices[:self.num_points - len(indices)]])
        else:
            # Weighted sampling based on class distribution
            class_counts = np.bincount(labels, minlength=3)
            sample_weights = np.zeros(len(labels))
            
            for class_id in range(3):
                mask = labels == class_id
                if class_counts[class_id] > 0:
                    sample_weights[mask] = self.class_weights[class_id] / class_counts[class_id]
            
            # Normalize weights
            sample_weights /= sample_weights.sum()
            
            # Sample points
            indices = np.random.choice(
                len(points),
                size=self.num_points,
                replace=False,
                p=sample_weights
            )
        
        return points[indices], labels[indices]
    
    def _compute_features(self, points: np.ndarray) -> np.ndarray:
        """Compute input features for each point"""
        # Basic features: x, y, z coordinates
        features = points.copy()
        
        # Add height feature (z-coordinate relative to minimum)
        z_min = points[:, 2].min()
        height_feature = (points[:, 2] - z_min).reshape(-1, 1)
        features = np.concatenate([features, height_feature], axis=1)
        
        # Add distance to center
        center = points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1).reshape(-1, 1)
        features = np.concatenate([features, distances], axis=1)
        
        return features.astype(np.float32)
    
    def __len__(self) -> int:
        """Return total number of patches in dataset"""
        return self.total_patches
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single training sample
        
        Returns:
            points: Dictionary containing multiscale point cloud data
            labels: Point-wise class labels
        """
        # Determine which scan and patch within that scan
        scan_idx = idx // self.patches_per_scan
        patch_idx = idx % self.patches_per_scan
        
        # Load scan data
        scan_info = self.scan_files[scan_idx]
        points = self._load_ply(scan_info['ply'])
        labels = self._load_labels(scan_info['labels'])
        
        # Ensure labels match points
        if len(points) != len(labels):
            min_len = min(len(points), len(labels))
            points = points[:min_len]
            labels = labels[:min_len]
        
        # Apply augmentation
        points = self._augment_points(points)
        
        # Sample patch
        points, labels = self._sample_patch(points, labels)
        
        # Compute features
        features = self._compute_features(points)
        
        # Compute multiscale indices for RandLA-Net
        multiscale_data = compute_multiscale_indices(
            features,
            k_n=K_N,
            sub_sampling_ratio=SUB_SAMPLING_RATIO
        )
        
        # Convert to tensors
        point_data = {
            'features': torch.from_numpy(features),
            'xyz': [torch.from_numpy(xyz.astype(np.float32)) for xyz in multiscale_data['xyz']],
            'neigh_idx': [torch.from_numpy(idx.astype(np.int64)) for idx in multiscale_data['neigh_idx']],
            'sub_idx': [torch.from_numpy(idx.astype(np.int64)) for idx in multiscale_data['sub_idx']],
            'interp_idx': [torch.from_numpy(idx.astype(np.int64)) for idx in multiscale_data['interp_idx']]
        }
        
        labels_tensor = torch.from_numpy(labels.astype(np.int64))
        
        return point_data, labels_tensor


# ─── Utility Functions ───────────────────────────────────────────────────────────

def create_data_splits(data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.2):
    """Create train/val splits from a directory of PLY files"""
    data_path = Path(data_path)
    ply_files = list(data_path.glob('*.ply'))
    
    if len(ply_files) == 0:
        raise RuntimeError(f"No PLY files found in {data_path}")
    
    # Shuffle files
    np.random.shuffle(ply_files)
    
    # Split indices
    n_files = len(ply_files)
    n_train = int(n_files * train_ratio)
    
    train_files = ply_files[:n_train]
    val_files = ply_files[n_train:]
    
    # Create directories
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Move files
    for f in train_files:
        (train_dir / f.name).write_bytes(f.read_bytes())
    
    for f in val_files:
        (val_dir / f.name).write_bytes(f.read_bytes())
    
    print(f"Created {len(train_files)} train files and {len(val_files)} val files")


def analyze_dataset_statistics(dataset: StairsDataset) -> Dict[str, float]:
    """Analyze dataset class distribution and statistics"""
    all_labels = []
    all_points = []
    
    print("Analyzing dataset statistics...")
    for i in range(min(100, len(dataset))):  # Sample first 100 patches
        _, labels = dataset[i]
        all_labels.extend(labels.numpy())
        all_points.append(len(labels))
    
    all_labels = np.array(all_labels)
    
    # Class distribution
    class_counts = np.bincount(all_labels, minlength=3)
    total_points = len(all_labels)
    
    stats = {
        'total_points_sampled': total_points,
        'avg_points_per_patch': np.mean(all_points),
        'background_ratio': class_counts[0] / total_points,
        'riser_ratio': class_counts[1] / total_points,  
        'tread_ratio': class_counts[2] / total_points,
        'class_counts': class_counts.tolist()
    }
    
    print("\nDataset Statistics:")
    print(f"Total points sampled: {stats['total_points_sampled']:,}")
    print(f"Average points per patch: {stats['avg_points_per_patch']:.1f}")
    print(f"Class distribution:")
    print(f"  Background (0): {stats['background_ratio']:.3f} ({class_counts[0]:,} points)")
    print(f"  Riser (1): {stats['riser_ratio']:.3f} ({class_counts[1]:,} points)")
    print(f"  Tread (2): {stats['tread_ratio']:.3f} ({class_counts[2]:,} points)")
    
    return stats


# ─── Testing ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test dataset loading
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to stairs dataset')
    parser.add_argument('--create_splits', action='store_true', help='Create train/val splits')
    args = parser.parse_args()
    
    if args.create_splits:
        create_data_splits(args.data_path)
    
    # Test dataset
    try:
        dataset = StairsDataset(args.data_path, split='train')
        print(f"Dataset loaded successfully: {len(dataset)} patches")
        
        # Test sample
        points, labels = dataset[0]
        print(f"Sample shape - Features: {points['features'].shape}, Labels: {labels.shape}")
        print(f"Multiscale levels: {len(points['xyz'])}")
        
        # Analyze statistics
        stats = analyze_dataset_statistics(dataset)
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
