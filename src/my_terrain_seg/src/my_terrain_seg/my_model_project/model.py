#!/usr/bin/env python3
"""
RandLA-Net Model Architecture

This file defines the RandLA-Net neural network for 3D point cloud segmentation.
RandLA-Net is designed for efficient processing of large point clouds using 
random sampling and local feature aggregation.

Architecture Overview:
1. Encoder: Progressive downsampling with dilated residual blocks
2. Decoder: Feature propagation with skip connections  
3. Output: 3-class segmentation (Background, Riser, Tread)

Key Components:
- Dilated Residual Blocks: Capture multi-scale local features
- Random Sampling: Efficient point reduction for real-time processing
- Feature Propagation: Recover spatial resolution in decoder

Reference: 
Hu, Q., Yang, B., Xie, L., Rosa, S., Guo, Y., Wang, Z., ... & Markham, A. (2020). 
RandLA-Net: Efficient semantic segmentation of large-scale point clouds. 
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 11108-11117.

Implementation Notes:
- Adapted for stair detection (3 classes instead of original multi-class)
- Optimized for real-time robotic applications
- Integrated with ROS2 for seamless robot navigation
"""

# ─── Deep Learning Framework ─────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ─── Local Utilities ─────────────────────────────────────────────────────────────
from my_terrain_seg.my_model_project.utils import (
    ensure_long,
    relative_pos_encoding,
    gather_neighbour,
    random_sample,
    nearest_interpolation
)

# ─── Model Configuration ─────────────────────────────────────────────────────────
from my_terrain_seg.my_model_project.hyperparameters import D_OUT, NUM_LAYERS, NUM_CLASSES


# ─── RandLA-Net Main Architecture ────────────────────────────────────────────────

class RandlaNet(nn.Module):
    """
    RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
    
    A lightweight neural network designed for real-time point cloud processing.
    Uses random sampling and local feature aggregation to handle large scenes
    while maintaining spatial context through skip connections.
    
    Architecture Flow:
    Input (XYZ) → Encoder (downsample + extract features) → Decoder (upsample + classify)
    """
    
    def __init__(self):
        super().__init__()
        
        # ─── Model Configuration ─────────────────────────────────────────────────
        feature_dims = D_OUT      # [8, 64, 128, 256, 512, 512] - feature dimensions per layer
        num_layers = NUM_LAYERS   # 6 layers total
        num_classes = NUM_CLASSES # 3 classes: Background, Riser, Tread
        
        assert len(feature_dims) == num_layers, "Feature dimensions must match number of layers"

        # ─── Initial Feature Extraction ──────────────────────────────────────────
        # Convert 3D coordinates (X,Y,Z) to initial feature representation  
        initial_features = feature_dims[0] // 2  # 4 features from 8
        self.coordinate_to_features = nn.Linear(3, initial_features)

        # ─── Encoder: Progressive Downsampling ───────────────────────────────────
        # Each encoder layer reduces point count while increasing feature richness
        self.encoder_layers = nn.ModuleList()
        input_channels = initial_features
        
        for output_features in feature_dims:
            self.encoder_layers.append(DilatedResidualBlock(input_channels, output_features))
            input_channels = output_features * 2  # Dilated blocks double the channels

        # ─── Feature Transition ──────────────────────────────────────────────────
        # Bridge between encoder and decoder with same dimensionality
        self.feature_bridge = nn.Linear(input_channels, input_channels)

        # ─── Decoder: Progressive Upsampling ─────────────────────────────────────
        # Each decoder layer increases spatial resolution while refining features
        self.decoder_layers = nn.ModuleList()
        
        for output_features in reversed(feature_dims):
            # Combine current features with skip connection from encoder
            combined_channels = input_channels + (output_features * 2)
            self.decoder_layers.append(FeatureDecoder(combined_channels, output_features * 2))
            input_channels = output_features * 2

        # ─── Final Classification Head ────────────────────────────────────────────
        # Multi-layer perceptron to produce per-point class predictions
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dims[0] * 2, 64),  # First dense layer
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, 32),                   # Second dense layer  
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)           # Output layer: 3 classes
        )

    def forward(self, inputs: dict):
        """
        Forward pass through the RandLA-Net architecture.
        
        Args:
            inputs: Dictionary containing:
                - 'features': Input point coordinates (B, N, 3)
                - 'xyz': Point coordinates at each level  
                - 'neigh_idx': Neighbor indices for local aggregation
                - 'sub_idx': Subsampling indices for downsampling
                - 'interp_idx': Interpolation indices for upsampling
                
        Returns:
            Class predictions for each input point (B, N, num_classes)
        """
        
        # ─── Initial Feature Extraction ──────────────────────────────────────────
        # Convert 3D coordinates to initial feature representation
        x = F.leaky_relu(self.coordinate_to_features(inputs['features']), 0.2)  # (B, N₀, initial_features)
        encoder_features = []  # Store features from each encoder layer for skip connections

        # ─── Encoder: Hierarchical Feature Extraction ────────────────────────────
        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            xyz = inputs['xyz'][layer_idx]                                         # Current layer coordinates
            neighbors = ensure_long(inputs['neigh_idx'][layer_idx], xyz)           # Neighbor indices
            
            # Extract features using dilated residual block
            layer_features = checkpoint(encoder_layer, x, xyz, neighbors)          # (B, Nᵢ, features)
            encoder_features.append(layer_features)

            # Downsample points for next layer
            subsample_indices = ensure_long(inputs['sub_idx'][layer_idx], layer_features)  # (B, Mᵢ)
            x = random_sample(layer_features, subsample_indices)                   # (B, Mᵢ, features)

        # ─── Feature Bridge ──────────────────────────────────────────────────────
        # Transition between encoder and decoder
        x = F.leaky_relu(self.feature_bridge(x), 0.2)                            # (B, M_final, features)

        # ─── Decoder: Progressive Upsampling with Skip Connections ───────────────
        num_decoder_layers = len(self.decoder_layers)
        for layer_idx, decoder_layer in enumerate(self.decoder_layers):
            # Get corresponding encoder features (reverse order)
            skip_features = encoder_features[-(layer_idx+1)]                       # (B, Nⱼ, skip_features)
            
            # Get interpolation indices for upsampling
            interpolation_indices = ensure_long(inputs['interp_idx'][-(layer_idx+1)], x)  # (B, Nⱼ, K)
            
            # Upsample and combine with skip connection
            x = decoder_layer(x, skip_features, interpolation_indices)            # (B, Nⱼ, combined_features)

        # ─── Final Classification ────────────────────────────────────────────────
        # Generate per-point class predictions
        batch_size, num_points, _ = inputs['features'].shape
        class_predictions = self.classification_head(x)                          # (B, N₀, num_classes)
        
        return class_predictions.view(batch_size, num_points, NUM_CLASSES)


# ─── Encoder Block ───────────────────────────────────────────────────────────────
class DilatedResidualBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.fc1      = nn.Linear(fin, fout)
        self.bn1      = nn.BatchNorm1d(fout)
        self.block    = BuildingBlock(fout)
        self.fc2      = nn.Linear(fout, fout * 2)
        self.bn2      = nn.BatchNorm1d(fout * 2)
        # Changed to Sequential with BatchNorm to match your model
        self.shortcut = nn.Sequential(
            nn.Linear(fin, fout * 2),
            nn.BatchNorm1d(fout * 2)
        )

    def forward(self, x, xyz, neigh_idx):
        B, N, _ = x.shape
        # Reshape for BatchNorm1d: (B, N, C) -> (B, C, N)
        f  = self.fc1(x)
        f  = self.bn1(f.transpose(1, 2)).transpose(1, 2)
        f  = F.leaky_relu(f, 0.2)
        f  = self.block(xyz, f, neigh_idx)
        f  = self.fc2(f)
        f  = self.bn2(f.transpose(1, 2)).transpose(1, 2)
        
        sc = self.shortcut[0](x)  # Linear layer
        sc = self.shortcut[1](sc.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        return F.leaky_relu(f + sc, 0.2)


class BuildingBlock(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        # relative_pos_encoding now produces (B,N,K,3)
        self.fc1  = nn.Linear(3, d_out)
        self.att1 = AttentivePooling(d_out * 2, d_out)
        self.fc2  = nn.Linear(d_out, d_out)
        self.att2 = AttentivePooling(d_out * 2, d_out)

    def forward(self, xyz, feat, neigh_idx):
        pe     = relative_pos_encoding(xyz, neigh_idx)  # (B,N,K,3)
        f_xyz  = F.leaky_relu(self.fc1(pe), 0.2)        # (B,N,K,d_out)
        f_nei  = gather_neighbour(feat, neigh_idx)      # (B,N,K,d_out)
        f_cat  = torch.cat([f_nei, f_xyz], dim=-1)      # (B,N,K,2*d_out)
        agg1   = self.att1(f_cat)                       # (B,N,d_out)

        f_xyz2 = F.leaky_relu(self.fc2(f_xyz), 0.2)        # (B,N,K,d_out)
        f_nei2 = gather_neighbour(agg1, neigh_idx)      # (B,N,K,d_out)
        f_cat2 = torch.cat([f_nei2, f_xyz2], dim=-1)    # (B,N,K,2*d_out)
        return self.att2(f_cat2)                        # (B,N,d_out)


class AttentivePooling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.attn_fc = nn.Linear(in_dim, in_dim)
        self.out_fc  = nn.Linear(in_dim, out_dim)
        # Add BatchNorm to match your model structure
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        B, N, K, C = x.shape
        attn  = F.softmax(self.attn_fc(x), dim=2)  # over K
        x_sum = (x * attn).sum(dim=2)              # (B,N,in_dim)
        out = self.out_fc(x_sum)                   # (B,N,out_dim)
        # Apply BatchNorm: (B,N,C) -> (B,C,N) -> (B,N,C)
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        return F.leaky_relu(out, 0.2)


# ─── Decoder Block ───────────────────────────────────────────────────────────────
class FeatureDecoder(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Changed to match your model structure: fc1 + bn1 + skip_proj
        self.fc1 = nn.Linear(fin, fout)
        self.bn1 = nn.BatchNorm1d(fout)
        # Skip connection projection
        self.skip_proj = nn.Sequential(
            nn.Linear(fin, fout),
            nn.BatchNorm1d(fout)
        )

    def forward(self, x, skip, interp_idx):
        # x: (B,M,Cₘ), skip: (B,N,Cₛ), interp_idx: (B,N,K)
        x_up  = nearest_interpolation(x, interp_idx).mean(dim=2)  # (B,N,Cₘ)
        x_cat = torch.cat([x_up, skip], dim=-1)                   # (B,N,Cₘ+Cₛ)
        
        # Main path
        out = self.fc1(x_cat)  # (B,N,fout)
        out = self.bn1(out.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        
        # Skip connection
        skip_out = self.skip_proj[0](x_cat)  # Linear
        skip_out = self.skip_proj[1](skip_out.transpose(1, 2)).transpose(1, 2)  # BatchNorm
        
        return F.leaky_relu(out + skip_out, 0.2)


# ─── Model Factory ───────────────────────────────────────────────────────────────
def build_model():
    return RandlaNet()
