#!/usr/bin/env python3
"""
Evaluation Metrics for RandLA-Net Stair Detection
=================================================

This module implements evaluation metrics for point cloud segmentation,
specifically designed for stair detection with 3 classes:
- Class 0: Background
- Class 1: Riser
- Class 2: Tread

Available Metrics:
- IoU (Intersection over Union) per class and mean
- Accuracy (overall and per class)
- Precision and Recall per class
- F1 Score per class
- Confusion Matrix

Author: Vincent Yeung
"""

# ─── Scientific Computing ───────────────────────────────────────────────────────
import numpy as np
from typing import List, Dict, Tuple, Optional

# ─── Deep Learning Framework ─────────────────────────────────────────────────────
import torch


# ─── IoU Calculator ──────────────────────────────────────────────────────────────

class IoUCalculator:
    """
    Calculate Intersection over Union (IoU) for point cloud segmentation
    
    IoU is the primary metric for evaluating segmentation quality,
    measuring the overlap between predicted and ground truth regions.
    """
    
    def __init__(self, num_classes: int = 3, ignore_index: int = -1):
        """
        Initialize IoU calculator
        
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Class index to ignore in calculations
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_points = 0
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update confusion matrix with batch predictions
        
        Args:
            predictions: Predicted class labels (N,)
            targets: Ground truth class labels (N,)
        """
        # Flatten arrays if needed
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Filter out ignored indices
        if self.ignore_index >= 0:
            valid_mask = targets != self.ignore_index
            predictions = predictions[valid_mask]
            targets = targets[valid_mask]
        
        # Ensure predictions and targets are in valid range
        valid_mask = (predictions >= 0) & (predictions < self.num_classes) & \
                    (targets >= 0) & (targets < self.num_classes)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        for pred, target in zip(predictions, targets):
            self.confusion_matrix[target, pred] += 1
        
        self.total_points += len(predictions)
    
    def compute_class_ious(self) -> np.ndarray:
        """
        Compute IoU for each class
        
        Returns:
            Array of IoU values for each class
        """
        ious = np.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            # True positives: diagonal element
            tp = self.confusion_matrix[class_idx, class_idx]
            
            # False positives: sum of column minus true positives
            fp = self.confusion_matrix[:, class_idx].sum() - tp
            
            # False negatives: sum of row minus true positives
            fn = self.confusion_matrix[class_idx, :].sum() - tp
            
            # IoU = TP / (TP + FP + FN)
            if tp + fp + fn > 0:
                ious[class_idx] = tp / (tp + fp + fn)
            else:
                ious[class_idx] = 0.0
        
        return ious
    
    def compute_mean_iou(self) -> float:
        """
        Compute mean IoU across all classes
        
        Returns:
            Mean IoU value
        """
        class_ious = self.compute_class_ious()
        return class_ious.mean()
    
    def compute_weighted_iou(self, class_weights: Optional[List[float]] = None) -> float:
        """
        Compute weighted IoU
        
        Args:
            class_weights: Weights for each class (default: uniform)
            
        Returns:
            Weighted IoU value
        """
        class_ious = self.compute_class_ious()
        
        if class_weights is None:
            class_weights = [1.0] * self.num_classes
        
        weights = np.array(class_weights)
        weights = weights / weights.sum()  # Normalize
        
        return (class_ious * weights).sum()


# ─── Accuracy Calculator ─────────────────────────────────────────────────────────

class AccuracyCalculator:
    """
    Calculate accuracy metrics for point cloud segmentation
    """
    
    def __init__(self, num_classes: int = 3, ignore_index: int = -1):
        """
        Initialize accuracy calculator
        
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Class index to ignore in calculations
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_points = 0
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update confusion matrix with batch predictions
        
        Args:
            predictions: Predicted class labels (N,)
            targets: Ground truth class labels (N,)
        """
        # Flatten arrays if needed
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Filter out ignored indices
        if self.ignore_index >= 0:
            valid_mask = targets != self.ignore_index
            predictions = predictions[valid_mask]
            targets = targets[valid_mask]
        
        # Ensure predictions and targets are in valid range
        valid_mask = (predictions >= 0) & (predictions < self.num_classes) & \
                    (targets >= 0) & (targets < self.num_classes)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        for pred, target in zip(predictions, targets):
            self.confusion_matrix[target, pred] += 1
        
        self.total_points += len(predictions)
    
    def compute_overall_accuracy(self) -> float:
        """
        Compute overall accuracy (correct predictions / total predictions)
        
        Returns:
            Overall accuracy value
        """
        if self.total_points == 0:
            return 0.0
        
        correct_predictions = np.diag(self.confusion_matrix).sum()
        return correct_predictions / self.total_points
    
    def compute_class_accuracies(self) -> np.ndarray:
        """
        Compute per-class accuracy (recall)
        
        Returns:
            Array of accuracy values for each class
        """
        accuracies = np.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            # True positives
            tp = self.confusion_matrix[class_idx, class_idx]
            
            # Total ground truth for this class
            total_gt = self.confusion_matrix[class_idx, :].sum()
            
            if total_gt > 0:
                accuracies[class_idx] = tp / total_gt
            else:
                accuracies[class_idx] = 0.0
        
        return accuracies
    
    def compute_precision(self) -> np.ndarray:
        """
        Compute precision for each class
        
        Returns:
            Array of precision values for each class
        """
        precisions = np.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            # True positives
            tp = self.confusion_matrix[class_idx, class_idx]
            
            # Total predictions for this class
            total_pred = self.confusion_matrix[:, class_idx].sum()
            
            if total_pred > 0:
                precisions[class_idx] = tp / total_pred
            else:
                precisions[class_idx] = 0.0
        
        return precisions
    
    def compute_f1_scores(self) -> np.ndarray:
        """
        Compute F1 score for each class
        
        Returns:
            Array of F1 scores for each class
        """
        recall = self.compute_class_accuracies()
        precision = self.compute_precision()
        
        f1_scores = np.zeros(self.num_classes)
        
        for class_idx in range(self.num_classes):
            if recall[class_idx] + precision[class_idx] > 0:
                f1_scores[class_idx] = 2 * (recall[class_idx] * precision[class_idx]) / \
                                      (recall[class_idx] + precision[class_idx])
            else:
                f1_scores[class_idx] = 0.0
        
        return f1_scores


# ─── Comprehensive Evaluator ─────────────────────────────────────────────────────

class SegmentationEvaluator:
    """
    Comprehensive evaluator for point cloud segmentation
    
    Combines IoU and accuracy calculations with detailed reporting.
    """
    
    def __init__(self, num_classes: int = 3, class_names: Optional[List[str]] = None):
        """
        Initialize evaluator
        
        Args:
            num_classes: Number of segmentation classes
            class_names: Names for each class (for reporting)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
        self.iou_calc = IoUCalculator(num_classes)
        self.acc_calc = AccuracyCalculator(num_classes)
    
    def reset(self):
        """Reset all metrics"""
        self.iou_calc.reset()
        self.acc_calc.reset()
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update metrics with batch predictions
        
        Args:
            predictions: Predicted class labels
            targets: Ground truth class labels
        """
        self.iou_calc.update(predictions, targets)
        self.acc_calc.update(predictions, targets)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all available metrics
        
        Returns:
            Dictionary containing all computed metrics
        """
        # IoU metrics
        class_ious = self.iou_calc.compute_class_ious()
        mean_iou = self.iou_calc.compute_mean_iou()
        
        # Accuracy metrics
        overall_acc = self.acc_calc.compute_overall_accuracy()
        class_accs = self.acc_calc.compute_class_accuracies()
        class_precisions = self.acc_calc.compute_precision()
        class_f1s = self.acc_calc.compute_f1_scores()
        
        # Organize results
        metrics = {
            'mean_iou': mean_iou,
            'overall_accuracy': overall_acc,
            'mean_precision': class_precisions.mean(),
            'mean_recall': class_accs.mean(),
            'mean_f1': class_f1s.mean()
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name.lower()}_iou'] = class_ious[i]
            metrics[f'{class_name.lower()}_accuracy'] = class_accs[i]
            metrics[f'{class_name.lower()}_precision'] = class_precisions[i]
            metrics[f'{class_name.lower()}_f1'] = class_f1s[i]
        
        return metrics
    
    def print_detailed_report(self):
        """Print detailed evaluation report"""
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*60)
        print("SEGMENTATION EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Mean IoU:        {metrics['mean_iou']:.4f}")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Mean Precision:   {metrics['mean_precision']:.4f}")
        print(f"  Mean Recall:      {metrics['mean_recall']:.4f}")
        print(f"  Mean F1:          {metrics['mean_f1']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<12} {'IoU':<8} {'Acc':<8} {'Prec':<8} {'F1':<8}")
        print("-" * 45)
        
        for i, class_name in enumerate(self.class_names):
            iou = metrics[f'{class_name.lower()}_iou']
            acc = metrics[f'{class_name.lower()}_accuracy']
            prec = metrics[f'{class_name.lower()}_precision']
            f1 = metrics[f'{class_name.lower()}_f1']
            
            print(f"{class_name:<12} {iou:<8.4f} {acc:<8.4f} {prec:<8.4f} {f1:<8.4f}")
        
        print("\nConfusion Matrix:")
        self._print_confusion_matrix()
    
    def _print_confusion_matrix(self):
        """Print confusion matrix"""
        cm = self.acc_calc.confusion_matrix
        
        # Header
        print(f"{'True\\Pred':<12}", end="")
        for class_name in self.class_names:
            print(f"{class_name:<12}", end="")
        print()
        
        # Matrix rows
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<12}", end="")
            for j in range(self.num_classes):
                print(f"{cm[i,j]:<12}", end="")
            print()


# ─── Testing ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test metrics with dummy data
    np.random.seed(42)
    
    # Create dummy predictions and targets
    n_points = 10000
    num_classes = 3
    
    # Simulate realistic stair detection scenario
    targets = np.random.choice(num_classes, size=n_points, p=[0.7, 0.15, 0.15])  # More background
    
    # Simulate model predictions (with some noise)
    predictions = targets.copy()
    # Add some classification errors
    error_mask = np.random.random(n_points) < 0.2  # 20% error rate
    predictions[error_mask] = np.random.choice(num_classes, size=error_mask.sum())
    
    print("Testing segmentation metrics...")
    
    # Test individual calculators
    iou_calc = IoUCalculator(num_classes)
    iou_calc.update(predictions, targets)
    
    class_ious = iou_calc.compute_class_ious()
    mean_iou = iou_calc.compute_mean_iou()
    
    print(f"Class IoUs: {[f'{iou:.4f}' for iou in class_ious]}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # Test comprehensive evaluator
    class_names = ['Background', 'Riser', 'Tread']
    evaluator = SegmentationEvaluator(num_classes, class_names)
    evaluator.update(predictions, targets)
    evaluator.print_detailed_report()
    
    print("\nAll metrics working correctly!")
