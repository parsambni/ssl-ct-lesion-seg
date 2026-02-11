"""
Transforms: weak and strong augmentations for semi-supervised learning.
"""
import os
# Try to use headless backend for OpenCV if needed
try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from typing import Tuple
from scipy import ndimage
from PIL import Image


class WeakAugmentation:
    """Weak augmentations: light transforms suitable for consistency loss."""
    
    def __init__(self, patch_size: Tuple[int, int] = (256, 256)):
        self.patch_size = patch_size
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply weak augmentation.
        
        Args:
            img: Image of shape (1, H, W)
            
        Returns:
            Augmented image of same shape
        """
        # Light horizontal flip
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=2)  # Flip along width
        
        # Light vertical flip
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=1)  # Flip along height
        
        # Small rotation (±15 degrees) using scipy
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)
            img[0] = ndimage.rotate(img[0], angle, order=1, reshape=False)
        
        return img.copy()


class StrongAugmentation:
    """Strong augmentations: aggressive transforms for pseudo-label training."""
    
    def __init__(self, patch_size: Tuple[int, int] = (256, 256)):
        self.patch_size = patch_size
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply strong augmentation.
        
        Args:
            img: Image of shape (1, H, W)
            
        Returns:
            Augmented image of same shape
        """
        img = img.copy()
        H, W = img.shape[1], img.shape[2]
        
        # Strong rotation (±30 degrees)
        if np.random.rand() < 0.8:
            angle = np.random.uniform(-30, 30)
            img[0] = ndimage.rotate(img[0], angle, order=1, reshape=False)
        
        # Elastic deformation
        if np.random.rand() < 0.5:
            img = self._elastic_deform(img, alpha=30, sigma=5)
        
        # Gaussian blur using scipy
        if np.random.rand() < 0.5:
            sigma = np.random.choice([0.5, 1.0])
            img[0] = ndimage.gaussian_filter(img[0], sigma=sigma)
        
        # Intensity variation
        if np.random.rand() < 0.5:
            intensity_factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img * intensity_factor, 0, 1)
        
        # Horizontal and vertical flips
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=2)
        if np.random.rand() < 0.5:
            img = np.flip(img, axis=1)
        
        return img
    
    def _elastic_deform(self, img: np.ndarray, alpha: float = 30, sigma: float = 5) -> np.ndarray:
        """Apply elastic deformation using scipy."""
        H, W = img.shape[1], img.shape[2]
        
        # Generate random displacement fields
        dx = np.random.randn(H, W) * sigma
        dy = np.random.randn(H, W) * sigma
        
        # Smooth displacement using Gaussian filter
        dx = ndimage.gaussian_filter(dx.astype(np.float32), sigma=sigma)
        dy = ndimage.gaussian_filter(dy.astype(np.float32), sigma=sigma)
        
        # Scale by alpha
        dx = dx * alpha / (sigma ** 2 + 1e-8)
        dy = dy * alpha / (sigma ** 2 + 1e-8)
        
        # Create meshgrid and apply displacement
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x_new = np.clip((x + dx).astype(np.float32), 0, W - 1)
        y_new = np.clip((y + dy).astype(np.float32), 0, H - 1)
        
        # Remap using scipy map_coordinates
        coords = np.array([y_new, x_new])
        img[0] = ndimage.map_coordinates(img[0], coords, order=1, cval=0.0)
        
        return img


class TargetedBoundaryAugmentation:
    """
    Reliability-aware targeted augmentation focused on tumor boundaries.
    
    Uses reliability scores (from teacher entropy/confidence) to:
    1. Gate augmentation strength at boundary regions
    2. Apply more aggressive augmentation in uncertain boundary regions
    """
    
    def __init__(self, patch_size: Tuple[int, int] = (256, 256)):
        self.patch_size = patch_size
    
    def __call__(
        self,
        img: np.ndarray,
        label: np.ndarray,
        reliability_map: np.ndarray,
        augmentation_strength: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply targeted boundary augmentation.
        
        Args:
            img: Image of shape (1, H, W)
            label: Label of shape (H, W)
            reliability_map: Reliability scores in [0, 1] of shape (H, W)
            augmentation_strength: Scale factor for augmentation strength [0, 1]
            
        Returns:
            (aug_img, aug_label) tuple
        """
        img = img.copy()
        label = label.copy()
        
        # Find boundary regions (where label changes)
        boundary_map = self._compute_boundary(label)
        
        # Combine boundary and reliability: augment uncertain boundary regions more
        augmentation_mask = boundary_map * (1.0 - reliability_map)
        augmentation_mask = augmentation_mask / (augmentation_mask.max() + 1e-8)
        
        # Apply spatially-varying augmentation
        aug_img, aug_label = self._apply_spatial_augmentation(
            img, label, augmentation_mask, augmentation_strength
        )
        
        return aug_img, aug_label
    
    def _compute_boundary(self, label: np.ndarray) -> np.ndarray:
        """
        Compute boundary map (distance from label boundary).
        
        Args:
            label: Label mask (H, W)
            
        Returns:
            Boundary map (H, W) where 1 = boundary, 0 = interior
        """
        # Use binary dilation and erosion to find boundary
        from scipy.ndimage import binary_dilation, binary_erosion
        
        label_binary = label > 0
        
        dilated = binary_dilation(label_binary, iterations=2)
        eroded = binary_erosion(label_binary, iterations=2)
        
        boundary = (dilated ^ eroded).astype(np.float32)  # XOR for boundary
        return np.clip(boundary, 0, 1)
    
    def _apply_spatial_augmentation(
        self,
        img: np.ndarray,
        label: np.ndarray,
        strength_map: np.ndarray,
        max_strength: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation with spatially-varying strength.
        
        Args:
            img: Image (1, H, W)
            label: Label (H, W)
            strength_map: Local augmentation strength (H, W)
            max_strength: Maximum strength factor
            
        Returns:
            (aug_img, aug_label) tuple
        """
        H, W = img.shape[1], img.shape[2]
        
        # Rotation (only in high-strength regions) using scipy
        if np.random.rand() < (0.5 * max_strength):
            angle = np.random.uniform(-15, 15) * max_strength
            img[0] = ndimage.rotate(img[0], angle, order=1, reshape=False)
            label = ndimage.rotate(label.astype(np.float32), angle, order=1, reshape=False)
        
        # Intensity: spatial variation based on strength_map
        if np.random.rand() < (0.5 * max_strength):
            intensity_factor = 1.0 + np.random.uniform(-0.2, 0.2) * max_strength
            # Apply scaling weighted by strength_map
            local_scale = 1.0 + strength_map * (intensity_factor - 1.0)
            img[0] = img[0] * local_scale
            img[0] = np.clip(img[0], 0, 1)
        
        return img, label


class RandomCrop:
    """Random crop to patch size."""
    
    def __init__(self, patch_size: Tuple[int, int] = (256, 256)):
        self.patch_size = patch_size
    
    def __call__(self, img: np.ndarray, label: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop both image and label."""
        # Make sure patch is not larger than image
        H, W = img.shape[1], img.shape[2]
        pH, pW = self.patch_size
        
        if H <= pH and W <= pW:
            return img, label
        
        # Random offsets
        h_offset = np.random.randint(0, max(1, H - pH + 1))
        w_offset = np.random.randint(0, max(1, W - pW + 1))
        
        img_crop = img[:, h_offset:h_offset + pH, w_offset:w_offset + pW]
        
        if label is not None:
            label_crop = label[h_offset:h_offset + pH, w_offset:w_offset + pW]
            return img_crop, label_crop
        
        return img_crop, None
