"""
SSL: Semi-supervised learning with Mean Teacher and reliability-aware augmentation.
"""
import torch
import torch.nn as nn
import copy
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MeanTeacher(nn.Module):
    """
    Mean Teacher for semi-supervised learning.
    
    Maintains a teacher network as an exponential moving average (EMA) of the student.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        ema_decay: float = 0.99
    ):
        """
        Args:
            student_model: Student network
            ema_decay: EMA decay rate (typically 0.99 or 0.999)
        """
        super().__init__()
        self.student = student_model
        self.teacher = self._create_teacher(student_model)
        self.ema_decay = ema_decay
        self.update_counter = 0
    
    def _create_teacher(self, student_model: nn.Module) -> nn.Module:
        """Create teacher as a copy of student."""
        teacher = copy.deepcopy(student_model)
        # Disable gradients for teacher
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher
    
    def update_teacher(self):
        """Update teacher weights using EMA."""
        alpha = self.ema_decay
        
        for student_param, teacher_param in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data
        
        self.update_counter += 1
    
    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student."""
        return self.student(x)
    
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher (no grad)."""
        with torch.no_grad():
            return self.teacher(x)
    
    def get_teacher_confidence(
        self,
        x: torch.Tensor,
        method: str = "entropy"
    ) -> torch.Tensor:
        """
        Compute teacher confidence for reliability gating.
        
        Args:
            x: Input tensor
            method: "entropy" or "maxprob"
            
        Returns:
            Confidence map (B, H, W) in [0, 1]
        """
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        probs = torch.softmax(teacher_logits, dim=1)
        
        if method == "entropy":
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            max_entropy = torch.tensor(probs.shape[1]).float().log()
            confidence = 1.0 - (entropy / max_entropy.to(entropy.device))
        elif method == "maxprob":
            confidence = probs.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown confidence method: {method}")
        
        return confidence


class ReliabilityGating:
    """
    Reliability-aware gating for pseudo-labels and augmentation.
    
    Uses teacher confidence to:
    1. Gate pseudo-label usage (confidence > threshold)
    2. Modulate augmentation strength (lower confidence -> stronger augmentation)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        min_augmentation_strength: float = 0.3,
        max_augmentation_strength: float = 1.0
    ):
        """
        Args:
            confidence_threshold: Min confidence to use pseudo-label
            min_augmentation_strength: Min augmentation strength (0 = no aug)
            max_augmentation_strength: Max augmentation strength
        """
        self.confidence_threshold = confidence_threshold
        self.min_strength = min_augmentation_strength
        self.max_strength = max_augmentation_strength
    
    def gate_pseudo_label(
        self,
        confidence_map: torch.Tensor,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Create binary gating mask for pseudo-labels.
        
        Args:
            confidence_map: Teacher confidence (B, H, W)
            threshold: Confidence threshold (use self.confidence_threshold if None)
            
        Returns:
            Gate mask (B, H, W) of 0/1
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        gate = (confidence_map > threshold).float()
        return gate
    
    def compute_augmentation_strength(
        self,
        confidence_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-pixel augmentation strength based on confidence.
        
        Lower confidence -> stronger augmentation
        
        Args:
            confidence_map: Teacher confidence (B, H, W)
            
        Returns:
            Augmentation strength (B, H, W) in [min_strength, max_strength]
        """
        # Invert confidence: low confidence -> high strength
        inverse_confidence = 1.0 - confidence_map
        
        # Scale to [min_strength, max_strength]
        strength = (
            self.min_strength +
            inverse_confidence * (self.max_strength - self.min_strength)
        )
        
        return strength


class PseudoLabelGenerator:
    """Generate pseudo-labels from teacher predictions with optional filtering."""
    
    @staticmethod
    def generate_pseudo_labels(
        teacher_logits: torch.Tensor,
        confidence_threshold: float = 0.0,
        return_confidence: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate pseudo-labels from teacher logits.
        
        Args:
            teacher_logits: Teacher logits (B, C, H, W)
            confidence_threshold: Min confidence (0 = use all)
            return_confidence: If True, also return confidence map
            
        Returns:
            (pseudo_labels, confidence_map) or just pseudo_labels
            pseudo_labels: (B, H, W) class indices
            confidence_map: (B, H, W) in [0, 1]
        """
        probs = torch.softmax(teacher_logits, dim=1)  # (B, C, H, W)
        
        # Pseudo-labels: argmax
        pseudo_labels = probs.argmax(dim=1)  # (B, H, W)
        
        # Confidence: max probability
        confidence_map = probs.max(dim=1)[0]  # (B, H, W)
        
        if return_confidence:
            return pseudo_labels, confidence_map
        else:
            return pseudo_labels


class MeanTeacherLoss(nn.Module):
    """
    Combine supervised loss and semi-supervised consistency loss.
    
    Total loss = supervised_loss + lambda_consistency * consistency_loss
    """
    
    def __init__(
        self,
        supervised_loss: nn.Module,
        consistency_loss: nn.Module,
        lambda_consistency: float = 0.1,
        rampup_epochs: int = 0
    ):
        """
        Args:
            supervised_loss: Loss for labeled data
            consistency_loss: Loss for unlabeled data
            lambda_consistency: Weight for consistency loss
            rampup_epochs: Ramp up consistency gradually from 0
        """
        super().__init__()
        self.supervised_loss = supervised_loss
        self.consistency_loss = consistency_loss
        self.lambda_consistency = lambda_consistency
        self.rampup_epochs = rampup_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for rampup scheduling."""
        self.current_epoch = epoch
    
    def get_consistency_weight(self) -> float:
        """Get current consistency loss weight with rampup."""
        if self.rampup_epochs == 0:
            return self.lambda_consistency
        
        # Linear rampup: 0 to lambda_consistency over rampup_epochs
        rampup_factor = min(1.0, self.current_epoch / max(1, self.rampup_epochs))
        return self.lambda_consistency * rampup_factor
    
    def forward(
        self,
        student_logits: torch.Tensor,
        target_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        labeled_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            student_logits: Student predictions (B, C, H, W)
            target_labels: Target labels (B, H, W)
            teacher_logits: Teacher predictions (B, C, H, W)
            labeled_mask: Binary mask (B, H, W) indicating labeled pixels
            
        Returns:
            (total_loss, sup_loss, cons_loss)
        """
        # Supervised loss on labeled data
        supervised_loss = self.supervised_loss(student_logits, target_labels)
        
        # Consistency loss (all data)
        consistency_loss = self.consistency_loss(student_logits, teacher_logits)
        
        # Weight consistency loss
        consistency_weight = self.get_consistency_weight()
        
        total_loss = supervised_loss + consistency_weight * consistency_loss
        
        return total_loss, supervised_loss, consistency_loss
