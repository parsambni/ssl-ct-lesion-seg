"""
Metrics: Dice, Jaccard, Sensitivity, Specificity, etc.
"""
import numpy as np
import torch
from typing import Dict


class SegmentationMetrics:
    """Compute segmentation metrics."""
    
    @staticmethod
    def dice_score(pred: np.ndarray, target: np.ndarray, class_id: int = 1) -> float:
        """
        Compute Dice score for a specific class.
        
        Args:
            pred: Predicted labels (H, W) or (B, H, W)
            target: Target labels (H, W) or (B, H, W)
            class_id: Class to compute Dice for (default: 1 = tumor/vessel)
            
        Returns:
            Dice score in [0, 1]
        """
        pred_binary = (pred == class_id).astype(np.float32)
        target_binary = (target == class_id).astype(np.float32)
        
        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary) + np.sum(target_binary)
        
        if union == 0:
            return 1.0 if np.array_equal(pred_binary, target_binary) else 0.0
        
        dice = 2.0 * intersection / union
        return float(dice)
    
    @staticmethod
    def jaccard_score(pred: np.ndarray, target: np.ndarray, class_id: int = 1) -> float:
        """
        Compute Jaccard (IoU) score for a specific class.
        
        Args:
            pred: Predicted labels
            target: Target labels
            class_id: Class ID
            
        Returns:
            Jaccard score in [0, 1]
        """
        pred_binary = (pred == class_id).astype(np.float32)
        target_binary = (target == class_id).astype(np.float32)
        
        intersection = np.sum(pred_binary * target_binary)
        union = np.sum(pred_binary + target_binary) - intersection
        
        if union == 0:
            return 1.0 if np.array_equal(pred_binary, target_binary) else 0.0
        
        jaccard = intersection / union
        return float(jaccard)
    
    @staticmethod
    def sensitivity(pred: np.ndarray, target: np.ndarray, class_id: int = 1) -> float:
        """
        Compute sensitivity (recall) for a specific class.
        
        Args:
            pred: Predicted labels
            target: Target labels
            class_id: Class ID
            
        Returns:
            Sensitivity in [0, 1]
        """
        pred_binary = (pred == class_id).astype(np.float32)
        target_binary = (target == class_id).astype(np.float32)
        
        tp = np.sum(pred_binary * target_binary)
        fn = np.sum((1 - pred_binary) * target_binary)
        
        if tp + fn == 0:
            return 0.0
        
        return float(tp / (tp + fn))
    
    @staticmethod
    def specificity(pred: np.ndarray, target: np.ndarray, class_id: int = 1) -> float:
        """
        Compute specificity for a specific class.
        
        Args:
            pred: Predicted labels
            target: Target labels
            class_id: Class ID
            
        Returns:
            Specificity in [0, 1]
        """
        pred_binary = (pred == class_id).astype(np.float32)
        target_binary = (target == class_id).astype(np.float32)
        
        tn = np.sum((1 - pred_binary) * (1 - target_binary))
        fp = np.sum(pred_binary * (1 - target_binary))
        
        if tn + fp == 0:
            return 0.0
        
        return float(tn / (tn + fp))
    
    @staticmethod
    def hausdorff_distance(pred: np.ndarray, target: np.ndarray, class_id: int = 1) -> float:
        """
        Compute Hausdorff distance for a specific class.
        
        Args:
            pred: Predicted labels
            target: Target labels
            class_id: Class ID
            
        Returns:
            Hausdorff distance
        """
        from scipy.ndimage import distance_transform_edt
        
        pred_binary = (pred == class_id).astype(np.uint8)
        target_binary = (target == class_id).astype(np.uint8)
        
        if not (pred_binary.max() > 0 and target_binary.max() > 0):
            # One or both are empty
            if np.array_equal(pred_binary, target_binary):
                return 0.0
            else:
                return np.inf
        
        # Distance from pred to target
        dist_pred = distance_transform_edt(1 - pred_binary)
        dist_target = distance_transform_edt(1 - target_binary)
        
        hausdorff_dist = max(dist_pred[target_binary > 0].max(),
                             dist_target[pred_binary > 0].max())
        
        return float(hausdorff_dist)
    
    @staticmethod
    def compute_all_metrics(pred: np.ndarray, target: np.ndarray, class_id: int = 1) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            pred: Predicted labels
            target: Target labels
            class_id: Class ID
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "dice": SegmentationMetrics.dice_score(pred, target, class_id),
            "jaccard": SegmentationMetrics.jaccard_score(pred, target, class_id),
            "sensitivity": SegmentationMetrics.sensitivity(pred, target, class_id),
            "specificity": SegmentationMetrics.specificity(pred, target, class_id),
        }
        
        # Hausdorff distance is slow, skip by default
        # metrics["hausdorff"] = SegmentationMetrics.hausdorff_distance(pred, target, class_id)
        
        return metrics


class TeacherConfidence:
    """Compute teacher model confidence for reliability estimation."""
    
    @staticmethod
    def entropy_confidence(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence as 1 - normalized entropy.
        
        Args:
            logits: Model logits (B, C, H, W)
            
        Returns:
            Confidence map (B, H, W) in [0, 1]
        """
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # (B, H, W)
        max_entropy = np.log(logits.shape[1])
        
        # Normalize and invert
        normalized_entropy = entropy / (max_entropy + 1e-8)
        confidence = 1.0 - normalized_entropy
        
        return confidence
    
    @staticmethod
    def max_prob_confidence(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence as max probability.
        
        Args:
            logits: Model logits (B, C, H, W)
            
        Returns:
            Confidence map (B, H, W) in [0, 1]
        """
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
        confidence = probs.max(dim=1)[0]  # (B, H, W)
        return confidence
    
    @staticmethod
    def combined_confidence(
        logits: torch.Tensor,
        entropy_weight: float = 0.5,
        prob_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Combine entropy and max-prob confidence.
        
        Args:
            logits: Model logits
            entropy_weight: Weight for entropy confidence
            prob_weight: Weight for max-prob confidence
            
        Returns:
            Combined confidence map (B, H, W)
        """
        entropy_conf = TeacherConfidence.entropy_confidence(logits)
        prob_conf = TeacherConfidence.max_prob_confidence(logits)
        
        combined = entropy_weight * entropy_conf + prob_weight * prob_conf
        return combined / (entropy_weight + prob_weight)
