"""
Losses: Dice loss, cross-entropy, and weighted combinations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation (also known as F1 loss)."""
    
    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
            reduction: "mean" or "sum"
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Target labels of shape (B, H, W), values in [0, C-1]
            
        Returns:
            Scalar loss
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets.long(), num_classes=logits.shape[1])  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Compute Dice per class, ignoring background
        dice_losses = []
        for c in range(logits.shape[1]):
            if c == 0:  # Skip background
                continue
            
            pred_c = probs[:, c, :, :]
            targ_c = targets_one_hot[:, c, :, :]
            
            intersection = (pred_c * targ_c).sum()
            union = (pred_c + targ_c).sum()
            
            if union == 0:
                dice = torch.tensor(1.0, device=logits.device)
            else:
                dice = 1.0 - (2 * intersection + self.smooth) / (union + self.smooth)
            
            dice_losses.append(dice)
        
        if len(dice_losses) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        dice_loss = torch.stack(dice_losses).mean()
        
        return dice_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy loss."""
    
    def __init__(self, class_weights: torch.Tensor = None, reduction: str = "mean"):
        """
        Args:
            class_weights: Tensor of shape (C,) with weights per class
            reduction: "mean" or "none"
        """
        super().__init__()
        self.class_weights = class_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy.
        
        Args:
            logits: Predicted logits (B, C, H, W)
            targets: Target labels (B, H, W)
            
        Returns:
            Loss
        """
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        
        return self.ce_loss(logits_flat, targets_flat)


class ConfidenceGatedPseudoLoss(nn.Module):
    """
    Confidence-gated pseudo-label loss for semi-supervised learning.
    
    Only computes loss where teacher confidence > threshold.
    """
    
    def __init__(self, confidence_threshold: float = 0.9, base_loss: str = "dice"):
        """
        Args:
            confidence_threshold: Minimum confidence to use pseudo-label
            base_loss: "dice" or "ce"
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.base_loss_name = base_loss
        
        if base_loss == "dice":
            self.base_loss = DiceLoss(smooth=1.0)
        elif base_loss == "ce":
            self.base_loss = nn.CrossEntropyLoss(reduction="none")
        else:
            raise ValueError(f"Unknown base_loss: {base_loss}")
    
    def forward(
        self,
        logits: torch.Tensor,
        pseudo_labels: torch.Tensor,
        teacher_confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute confidence-gated pseudo-label loss.
        
        Args:
            logits: Student logits (B, C, H, W)
            pseudo_labels: Teacher pseudo-labels (B, H, W)
            teacher_confidence: Teacher confidence scores (B, H, W) in [0, 1]
            
        Returns:
            Scalar loss
        """
        # Create mask where confidence > threshold
        confidence_mask = teacher_confidence > self.confidence_threshold  # (B, H, W)
        
        if not confidence_mask.any():
            # No high-confidence predictions, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        if self.base_loss_name == "dice":
            # For Dice loss, we need to manually apply the mask
            pseudo_labels_masked = pseudo_labels.clone()
            pseudo_labels_masked[~confidence_mask] = 0  # Ignore low-confidence regions
            
            loss = self.base_loss(logits, pseudo_labels_masked)
            
            # Weight by confidence mask
            return loss * confidence_mask.float().mean()
        
        else:  # "ce"
            B, C, H, W = logits.shape
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
            pseudo_labels_flat = pseudo_labels.reshape(-1)
            mask_flat = confidence_mask.reshape(-1)
            
            loss = self.base_loss(logits_flat, pseudo_labels_flat)
            loss_masked = loss[mask_flat]
            
            if len(loss_masked) == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            return loss_masked.mean()


class ConsistencyLoss(nn.Module):
    """
    Consistency loss between student and teacher predictions.
    Uses KL divergence or MSE on probability maps.
    """
    
    def __init__(self, method: str = "mse", temperature: float = 1.0):
        """
        Args:
            method: "mse" or "kl"
            temperature: Temperature for softmax
        """
        super().__init__()
        self.method = method
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss.
        
        Args:
            student_logits: Student predictions (B, C, H, W)
            teacher_logits: Teacher predictions (B, C, H, W)
            
        Returns:
            Scalar loss
        """
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        if self.method == "mse":
            loss = F.mse_loss(student_probs, teacher_probs)
        elif self.method == "kl":
            loss = self.kl_loss(torch.log(student_probs + 1e-8), teacher_probs)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return loss
