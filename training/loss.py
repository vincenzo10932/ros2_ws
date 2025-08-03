#!/usr/bin/env python3
"""
Loss Functions for RandLA-Net Stair Detection Training
======================================================

This module implements various loss functions optimized for point cloud
segmentation with class imbalance, specifically for stair detection.

Available Loss Functions:
- WeightedCrossEntropyLoss: Standard weighted cross-entropy
- FocalLoss: Focal loss for handling class imbalance
- DiceLoss: Dice loss for segmentation
- CombinedLoss: Combination of multiple loss functions

Author: Vincent Yeung
"""

# ─── Deep Learning Framework ─────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Scientific Computing ───────────────────────────────────────────────────────
import numpy as np
from typing import List, Optional


# ─── Weighted Cross-Entropy Loss ─────────────────────────────────────────────────

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for handling class imbalance
    
    This loss function applies different weights to different classes
    to compensate for class imbalance in the stair dataset.
    """
    
    def __init__(self, class_weights: List[float], ignore_index: int = -1):
        """
        Initialize weighted cross-entropy loss
        
        Args:
            class_weights: List of weights for each class [background, riser, tread]
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss
        
        Args:
            predictions: Model predictions (N, num_classes)
            targets: Ground truth labels (N,)
            
        Returns:
            Loss value
        """
        # Ensure weights are on the same device
        if self.class_weights.device != predictions.device:
            self.class_weights = self.class_weights.to(predictions.device)
            self.criterion.weight = self.class_weights
        
        return self.criterion(predictions, targets)


# ─── Focal Loss ──────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard negative mining
    
    Reference: Lin, T. Y., et al. "Focal loss for dense object detection." ICCV 2017.
    """
    
    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        ignore_index: int = -1
    ):
        """
        Initialize focal loss
        
        Args:
            alpha: Class balancing weights [background, riser, tread]
            gamma: Focusing parameter (higher = more focus on hard examples)
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            predictions: Model predictions (N, num_classes)
            targets: Ground truth labels (N,)
            
        Returns:
            Loss value
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            predictions,
            targets,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # Compute probabilities and focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != predictions.device:
                self.alpha = self.alpha.to(predictions.device)
            
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


# ─── Dice Loss ───────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    
    Dice loss is particularly effective for segmentation with class imbalance
    as it focuses on the overlap between prediction and ground truth.
    """
    
    def __init__(self, smooth: float = 1e-6, ignore_background: bool = False):
        """
        Initialize Dice loss
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_background: Whether to ignore background class in loss
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss
        
        Args:
            predictions: Model predictions (N, num_classes)
            targets: Ground truth labels (N,)
            
        Returns:
            Loss value
        """
        num_classes = predictions.shape[1]
        
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Flatten for easier computation
        predictions_flat = predictions.view(-1, num_classes)
        targets_flat = targets_one_hot.view(-1, num_classes)
        
        # Compute Dice coefficient for each class
        dice_scores = []
        start_class = 1 if self.ignore_background else 0
        
        for class_idx in range(start_class, num_classes):
            pred_class = predictions_flat[:, class_idx]
            target_class = targets_flat[:, class_idx]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)
        
        # Return average Dice loss
        avg_dice = torch.stack(dice_scores).mean()
        return 1.0 - avg_dice


# ─── Combined Loss ───────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Combined loss function using multiple loss components
    
    This combines cross-entropy, focal, and dice losses for robust training.
    """
    
    def __init__(
        self,
        class_weights: List[float],
        ce_weight: float = 1.0,
        focal_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_gamma: float = 2.0
    ):
        """
        Initialize combined loss
        
        Args:
            class_weights: Weights for each class
            ce_weight: Weight for cross-entropy component
            focal_weight: Weight for focal loss component
            dice_weight: Weight for dice loss component
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.ce_loss = WeightedCrossEntropyLoss(class_weights)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.dice_loss = DiceLoss(ignore_background=True)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            predictions: Model predictions (N, num_classes)
            targets: Ground truth labels (N,)
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        loss_components = {}
        
        if self.ce_weight > 0:
            ce_loss = self.ce_loss(predictions, targets)
            total_loss += self.ce_weight * ce_loss
            loss_components['ce'] = ce_loss.item()
        
        if self.focal_weight > 0:
            focal_loss = self.focal_loss(predictions, targets)
            total_loss += self.focal_weight * focal_loss
            loss_components['focal'] = focal_loss.item()
        
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(predictions, targets)
            total_loss += self.dice_weight * dice_loss
            loss_components['dice'] = dice_loss.item()
        
        return total_loss


# ─── IoU Loss ────────────────────────────────────────────────────────────────────

class IoULoss(nn.Module):
    """
    IoU (Intersection over Union) Loss for segmentation
    
    Directly optimizes the IoU metric which is commonly used for evaluation.
    """
    
    def __init__(self, smooth: float = 1e-6, ignore_background: bool = False):
        """
        Initialize IoU loss
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            ignore_background: Whether to ignore background class
        """
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss
        
        Args:
            predictions: Model predictions (N, num_classes)
            targets: Ground truth labels (N,)
            
        Returns:
            IoU loss value
        """
        num_classes = predictions.shape[1]
        
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Flatten for easier computation
        predictions_flat = predictions.view(-1, num_classes)
        targets_flat = targets_one_hot.view(-1, num_classes)
        
        # Compute IoU for each class
        iou_scores = []
        start_class = 1 if self.ignore_background else 0
        
        for class_idx in range(start_class, num_classes):
            pred_class = predictions_flat[:, class_idx]
            target_class = targets_flat[:, class_idx]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum() - intersection
            
            iou_score = (intersection + self.smooth) / (union + self.smooth)
            iou_scores.append(iou_score)
        
        # Return negative mean IoU as loss
        mean_iou = torch.stack(iou_scores).mean()
        return 1.0 - mean_iou


# ─── Utility Functions ───────────────────────────────────────────────────────────

def calculate_class_weights(class_counts: List[int], method: str = 'inverse') -> List[float]:
    """
    Calculate class weights for handling imbalanced datasets
    
    Args:
        class_counts: Number of samples for each class
        method: Weighting method ('inverse', 'sqrt_inverse', 'log_inverse')
        
    Returns:
        List of class weights
    """
    class_counts = np.array(class_counts, dtype=float)
    total_samples = class_counts.sum()
    
    if method == 'inverse':
        # Inverse frequency weighting
        weights = total_samples / (len(class_counts) * class_counts)
    elif method == 'sqrt_inverse':
        # Square root of inverse frequency
        weights = np.sqrt(total_samples / (len(class_counts) * class_counts))
    elif method == 'log_inverse':
        # Logarithmic inverse weighting
        weights = np.log(total_samples / class_counts)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights
    weights = weights / weights.min()
    
    return weights.tolist()


# ─── Testing ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test loss functions
    torch.manual_seed(42)
    
    # Create dummy data
    batch_size = 1000
    num_classes = 3
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test class weights calculation
    class_counts = [5000, 800, 1200]  # Typical stair dataset distribution
    weights = calculate_class_weights(class_counts, method='inverse')
    print(f"Class counts: {class_counts}")
    print(f"Calculated weights: {[f'{w:.3f}' for w in weights]}")
    
    # Test loss functions
    print("\nTesting loss functions:")
    
    # Weighted Cross-Entropy
    ce_loss = WeightedCrossEntropyLoss(weights)
    ce_value = ce_loss(predictions, targets)
    print(f"Weighted CE Loss: {ce_value:.4f}")
    
    # Focal Loss
    focal_loss = FocalLoss(alpha=weights, gamma=2.0)
    focal_value = focal_loss(predictions, targets)
    print(f"Focal Loss: {focal_value:.4f}")
    
    # Dice Loss
    dice_loss = DiceLoss(ignore_background=True)
    dice_value = dice_loss(predictions, targets)
    print(f"Dice Loss: {dice_value:.4f}")
    
    # IoU Loss
    iou_loss = IoULoss(ignore_background=True)
    iou_value = iou_loss(predictions, targets)
    print(f"IoU Loss: {iou_value:.4f}")
    
    # Combined Loss
    combined_loss = CombinedLoss(weights, ce_weight=1.0, focal_weight=0.5, dice_weight=0.5)
    combined_value = combined_loss(predictions, targets)
    print(f"Combined Loss: {combined_value:.4f}")
    
    print("\nAll loss functions working correctly!")
