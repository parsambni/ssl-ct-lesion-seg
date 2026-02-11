"""
test_transforms.py: Tests for augmentation transforms.
"""
import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.transforms import WeakAugmentation, StrongAugmentation, TargetedBoundaryAugmentation


def test_weak_augmentation():
    """Test weak augmentation."""
    aug = WeakAugmentation(patch_size=(256, 256))
    
    img = np.random.randn(1, 64, 64).astype(np.float32)
    aug_img = aug(img)
    
    assert aug_img.shape == img.shape
    assert aug_img.dtype == np.float32


def test_strong_augmentation():
    """Test strong augmentation."""
    aug = StrongAugmentation(patch_size=(256, 256))
    
    img = np.random.randn(1, 64, 64).astype(np.float32)
    aug_img = aug(img)
    
    assert aug_img.shape == img.shape
    assert aug_img.dtype == np.float32


def test_targeted_boundary_augmentation():
    """Test targeted boundary augmentation."""
    aug = TargetedBoundaryAugmentation(patch_size=(256, 256))
    
    img = np.random.randn(1, 64, 64).astype(np.float32)
    label = np.zeros((64, 64), dtype=np.int32)
    label[20:40, 20:40] = 1  # Small square
    
    reliability = np.random.rand(64, 64).astype(np.float32)
    
    aug_img, aug_label = aug(img, label, reliability, augmentation_strength=0.8)
    
    assert aug_img.shape == img.shape
    assert aug_label.shape == label.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
