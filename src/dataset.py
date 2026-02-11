"""
Dataset module: PyTorch dataset for loading 2D slices from 3D volumes.

Supports:
- Flexible image normalization (CT windowing, min-max, z-score)
- Augmentation at load time
- Labeled/unlabeled metadata tracking for SSL
- Robust error handling with warnings
"""
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable, Dict
import json
import numpy as np
import logging

from src.data import DatasetDiscovery, SliceExtractor, CTPreprocessor


logger = logging.getLogger(__name__)


class SliceDataset(Dataset):
    """
    PyTorch dataset for 2D axial slices.
    
    Loads 3D medical volumes, extracts 2D axial slices, and caches them.
    Supports augmentation and labeled/unlabeled metadata.
    """
    
    def __init__(
        self,
        patient_ids: List[str],
        discovery: DatasetDiscovery,
        mode: str = "ct",
        slice_thickness: int = 1,
        transform: Optional[Callable] = None,
        min_slice_coverage: float = 0.0,
        track_label_status: bool = True
    ):
        """
        Initialize SliceDataset.
        
        Args:
            patient_ids: List of patient IDs to load
            discovery: DatasetDiscovery instance with metadata
            mode: Image normalization mode:
                - "ct": Hounsfield windowing (default for hepatic vessel)
                - "minmax": Min-max normalization to [0, 1]
                - "zscore": Z-score normalization
            slice_thickness: Load every Nth slice (1=all, 2=every other, etc.)
            transform: Optional augmentation transform (function or tuple of functions)
            min_slice_coverage: Min fraction of non-zero labels to keep slice [0.0, 1.0]
            track_label_status: Track which patients are labeled vs unlabeled
        """
        self.patient_ids = patient_ids
        self.discovery = discovery
        self.mode = mode
        self.slice_thickness = slice_thickness
        self.transform = transform
        self.min_slice_coverage = min_slice_coverage
        self.track_label_status = track_label_status
        
        # Pre-load all slices and cache
        self.slices = []
        self._load_all_slices()
    
    def _load_all_slices(self):
        """
        Load and cache all 2D slices from patient volumes.
        Handles errors gracefully with warnings.
        """
        loaded_count = 0
        error_count = 0
        
        for patient_id in self.patient_ids:
            try:
                # Determine if patient is labeled or unlabeled
                if self.track_label_status:
                    is_labeled = patient_id in self.discovery.labeled_patients
                else:
                    is_labeled = True
                
                # Load 3D volumes
                img_vol = self.discovery.load_image(patient_id)
                label_vol = self.discovery.load_label(patient_id)
                
                # Normalize image based on mode
                if self.mode == "ct":
                    img_vol = CTPreprocessor.apply_ct_window(img_vol)
                elif self.mode == "zscore":
                    img_vol = CTPreprocessor.normalize_zscore(img_vol)
                else:  # "minmax" or default
                    img_vol = CTPreprocessor.normalize_minmax(img_vol)
                
                # Extract 2D axial slices
                slices = SliceExtractor.extract_slices(
                    img_vol, label_vol,
                    slice_thickness=self.slice_thickness,
                    min_slice_coverage=self.min_slice_coverage
                )
                
                # Accumulate slices with metadata
                for img_2d, label_2d, z_idx in slices:
                    self.slices.append({
                        "patient_id": patient_id,
                        "image": img_2d,  # (1, H, W) float32
                        "label": label_2d,  # (H, W) int32
                        "z_index": z_idx,
                        "is_labeled": is_labeled  # Track label status for SSL
                    })
                
                loaded_count += 1
            
            except Exception as e:
                logger.warning(f"⚠ Error loading patient {patient_id}: {e}")
                error_count += 1
                continue
        
        logger.info(
            f"Dataset loaded: {len(self.slices)} slices from {loaded_count} patients. "
            f"({error_count} errors)"
        )
        
        if self.track_label_status:
            labeled_slices = sum(1 for s in self.slices if s["is_labeled"])
            logger.info(f"Labeled slices: {labeled_slices}/{len(self.slices)}")
    
    def __len__(self) -> int:
        """Return number of slices."""
        return len(self.slices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single slice with augmentation and format conversion.
        
        Args:
            idx: Index of slice to retrieve
            
        Returns:
            Dictionary containing:
            - "image": (1, H, W) float32 tensor
            - "label": (H, W) int64 tensor (for cross-entropy loss)
            - "patient_id": Patient ID string
            - "z_index": Z-index in original 3D volume
            - "is_labeled": Boolean (if track_label_status=True)
        """
        slice_data = self.slices[idx]
        
        # Copy to avoid modifying cached data
        img = slice_data["image"].copy()  # (1, H, W) float32
        label = slice_data["label"].copy()  # (H, W) int32
        
        # Apply augmentation if provided
        if self.transform is not None:
            if isinstance(self.transform, (tuple, list)):
                # Multiple transforms available - pick one randomly
                aug = self.transform[int(np.random.randint(0, len(self.transform)))]
                img = aug(img)
            else:
                # Single transform
                img = self.transform(img)
        
        # Ensure correct dtypes and convert to tensors
        img = img.astype(np.float32)
        label = label.astype(np.int32)
        
        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.from_numpy(label).long()  # long for cross-entropy
        
        result = {
            "image": img_tensor,
            "label": label_tensor,
            "patient_id": slice_data["patient_id"],
            "z_index": slice_data["z_index"]
        }
        
        if self.track_label_status:
            result["is_labeled"] = slice_data["is_labeled"]
        
        return result
    
    def get_label_distribution(self) -> Dict[int, int]:
        """
        Compute class distribution across all slices.
        
        Returns:
            Dictionary mapping class_id -> count
        """
        distribution = {}
        for slice_data in self.slices:
            label = slice_data["label"]
            unique, counts = np.unique(label, return_counts=True)
            for u, c in zip(unique, counts):
                distribution[int(u)] = distribution.get(int(u), 0) + c
        return distribution
