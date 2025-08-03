#!/usr/bin/env python3
"""
Utility Functions for Point Cloud Processing

This file contains helper functions used throughout the stair detection pipeline.
These utilities handle common operations like data movement, PLY file export,
and geometric computations needed for the RandLA-Net model.

Key Functions:
- Device management: Moving data between CPU/GPU
- Point cloud export: Saving results as colored PLY files  
- Geometric operations: KNN search, neighbor finding, sampling
- Model utilities: Index handling, interpolation

All functions are optimized for real-time performance in robotic applications.
"""

# ─── Standard Library ────────────────────────────────────────────────────────────
import os

# ─── Third-Party Libraries ───────────────────────────────────────────────────────
import torch
import numpy as np
from scipy.spatial import cKDTree

# ─── Device Management Functions ─────────────────────────────────────────────────

def move_to_device(batch, device):
    """
    Move data structures containing PyTorch tensors to specified device (CPU/GPU).
    
    Recursively processes dictionaries, lists, and tuples to move all tensor
    data to the target device. Non-tensor data is left unchanged.
    Essential for GPU acceleration in deep learning pipelines.
    
    Args:
        batch: Data structure (dict, list, tuple, or tensor) to move
        device: Target device ('cpu', 'cuda', etc.)
        
    Returns:
        Data structure with all tensors moved to target device
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k,v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)
    else:
        return batch

# ─── Point Cloud Export Functions ────────────────────────────────────────────────
def save_colored_point_cloud(xyz: np.ndarray,
                             labels: np.ndarray,
                             filename: str,
                             color_map: dict=None):
    default_map = {
        0: (200,200,200),
        1: (255,50,50),
        2: (50,255,50),
    }
    cmap = color_map or default_map
    N = xyz.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    lines = [" ".join(header)]
    for (x,y,z), lab in zip(xyz, labels):
        r,g,b = cmap.get(int(lab),(0,0,0))
        lines.append(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename,"w") as f:
        f.write("\n".join(lines))

# ─── Neighbour Helpers ─────────────────────────────────────────────────────────
def ensure_long(idx: torch.Tensor, ref: torch.Tensor) -> torch.LongTensor:
    return idx.to(device=ref.device, dtype=torch.long)

def relative_pos_encoding(xyz: torch.Tensor, neigh_idx: torch.LongTensor):
    B,N,K = neigh_idx.shape
    batch_idx = torch.arange(B,device=xyz.device).view(B,1,1).expand(B,N,K)
    neigh_pts = xyz[batch_idx, neigh_idx]    # (B,N,K,3)
    return neigh_pts - xyz.unsqueeze(2)      # (B,N,K,3)

def gather_neighbour(feat: torch.Tensor, neigh_idx: torch.LongTensor):
    B,N,C = feat.shape
    K = neigh_idx.size(-1)
    batch_idx = torch.arange(B,device=feat.device).view(B,1,1).expand(B,N,K)
    return feat[batch_idx, neigh_idx]        # (B,N,K,C)

def random_sample(x: torch.Tensor, sub_idx: torch.LongTensor):
    B,_,C = x.shape
    batch_idx = torch.arange(B,device=x.device).view(B,1).expand(B,sub_idx.size(1))
    return x[batch_idx, sub_idx]             # (B,M,C)

def nearest_interpolation(x: torch.Tensor, interp_idx: torch.LongTensor):
    """
    x: (B,M,C) or (M,C)
    interp_idx: (B,N,K) or (N,K)
    returns (B,N,K,C) or (N,K,C)
    """
    unbatched = False
    if interp_idx.dim()==2:
        interp_idx = interp_idx.unsqueeze(0)  # → (1,N,K)
        if x.dim()==2:
            x = x.unsqueeze(0)                # → (1,M,C)
        unbatched = True

    B,M,C = x.shape
    _,N,K = interp_idx.shape

    # clamp any out‑of‑bounds index into [0, M-1]
    interp_idx = interp_idx.clamp(min=0, max=M-1)

    batch_idx = torch.arange(B,device=x.device).view(B,1,1).expand(B,N,K)
    out = x[batch_idx, interp_idx]          # → (B,N,K,C)

    if unbatched:
        out = out.squeeze(0)                 # → (N,K,C)
    return out

# ─── Multiscale Indexing ──────────────────────────────────────────────────────────
def compute_multiscale_indices(xyz: np.ndarray, k_n: int, sub_sampling_ratio: list):
    """
    Returns dict of lists of numpy arrays:
      'xyz'       : [ (N0,3), (N1,3), … ]
      'neigh_idx' : [ (N0,K0), (N1,K1), … ] with Ki=min(k_n,Ni)
      'sub_idx'   : [ (N1,), (N2,), … ]
      'interp_idx': [ (N0,1), (N1,1), … ]  — maps *original* cloud → nearest in subsampled
    """
    pts = xyz.copy()
    out = {'xyz':[], 'neigh_idx':[], 'sub_idx':[], 'interp_idx':[]}

    for ratio in sub_sampling_ratio:
        Ni = pts.shape[0]
        Ki = min(k_n, Ni)

        # 1) k-NN graph at this scale
        tree = cKDTree(pts)
        _, neigh = tree.query(pts, k=Ki)
        out['xyz'].append(pts)
        out['neigh_idx'].append(neigh.astype(np.int64))

        # 2) pick M = Ni // ratio samples
        M = max(1, Ni // ratio)
        choice = np.random.choice(Ni, M, replace=False)
        out['sub_idx'].append(choice.astype(np.int64))

        # 3) for **each original point** find its nearest in the NEW subsampled cloud
        sub_pts = pts[choice]
        tree2   = cKDTree(sub_pts)
        _, interp_full = tree2.query(pts, k=1)     # shape (Ni,)
        # reshape to (Ni,1) so nearest_interpolation can handle it
        out['interp_idx'].append(interp_full.astype(np.int64).reshape(-1,1))

        # move to next (subsampled) scale
        pts = sub_pts

    return out

# ─── Misc ─────────────────────────────────────────────────────────────────────────
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
