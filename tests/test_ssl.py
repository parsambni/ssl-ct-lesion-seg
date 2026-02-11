"""
test_ssl.py: Tests for semi-supervised learning modules.
"""
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ssl import MeanTeacher, ReliabilityGating, PseudoLabelGenerator, MeanTeacherLoss
from src.models import UNet2D
from src.losses import DiceLoss, ConsistencyLoss


def test_mean_teacher_initialization():
    """Test Mean Teacher initialization."""
    student = UNet2D(in_channels=1, out_channels=2)
    mt = MeanTeacher(student, ema_decay=0.99)
    
    assert mt.student is not None
    assert mt.teacher is not None
    assert mt.ema_decay == 0.99


def test_teacher_ema_update():
    """Test teacher EMA update."""
    student = UNet2D(in_channels=1, out_channels=2)
    mt = MeanTeacher(student, ema_decay=0.99)
    
    # Get initial teacher param value
    teacher_param_before = mt.teacher.enc1.conv1.weight.clone()
    
    # Modify student
    with torch.no_grad():
        for param in student.parameters():
            param.data += 1.0
    
    # Update teacher
    mt.update_teacher()
    
    # Teacher should have moved slightly
    teacher_param_after = mt.teacher.enc1.conv1.weight
    
    diff = (teacher_param_after - teacher_param_before).abs().max()
    assert diff > 0, "Teacher should have been updated"


def test_reliability_gating():
    """Test reliability gating."""
    rg = ReliabilityGating(confidence_threshold=0.9)
    
    confidence = torch.rand(2, 64, 64)  # (B, H, W)
    
    gate = rg.gate_pseudo_label(confidence)
    assert gate.shape == confidence.shape
    assert gate.min() == 0.0 and gate.max() == 1.0


def test_augmentation_strength():
    """Test augmentation strength computation."""
    rg = ReliabilityGating(
        min_augmentation_strength=0.3,
        max_augmentation_strength=1.0
    )
    
    confidence = torch.linspace(0, 1, 64 * 64).reshape(1, 64, 64)
    
    strength = rg.compute_augmentation_strength(confidence)
    
    assert strength.shape == confidence.shape
    assert strength.min() >= 0.3, f"Min strength should be >= 0.3, got {strength.min()}"
    assert strength.max() <= 1.0, f"Max strength should be <= 1.0, got {strength.max()}"


def test_pseudo_label_generation():
    """Test pseudo-label generation."""
    logits = torch.randn(2, 2, 64, 64)
    
    pseudo_labels, confidence = PseudoLabelGenerator.generate_pseudo_labels(
        logits, return_confidence=True
    )
    
    assert pseudo_labels.shape == (2, 64, 64)
    assert confidence.shape == (2, 64, 64)
    assert pseudo_labels.dtype == torch.long
    assert confidence.min() >= 0.0 and confidence.max() <= 1.0


def test_mean_teacher_loss():
    """Test Mean Teacher combined loss."""
    sup_loss = DiceLoss()
    cons_loss = ConsistencyLoss(method="mse")
    
    mt_loss = MeanTeacherLoss(sup_loss, cons_loss, lambda_consistency=0.1)
    
    student_logits = torch.randn(2, 2, 64, 64)
    teacher_logits = torch.randn(2, 2, 64, 64)
    target_labels = torch.randint(0, 2, (2, 64, 64))
    labeled_mask = torch.randint(0, 2, (2, 64, 64)).float()
    
    total_loss, sup_loss_val, cons_loss_val = mt_loss(
        student_logits, target_labels, teacher_logits, labeled_mask
    )
    
    assert total_loss > 0
    assert sup_loss_val >= 0
    assert cons_loss_val >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
