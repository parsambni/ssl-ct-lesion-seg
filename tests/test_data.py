"""
test_data.py: Tests for data discovery, loading, and slicing.
Includes tests for flexible directory discovery and label budgeting.
"""
import os
import tempfile
import pytest
import numpy as np
import nibabel as nib
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import DatasetDiscovery, SliceExtractor, CTPreprocessor


@pytest.fixture
def synthetic_dataset():
    """Create a minimal synthetic dataset for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure (direct children)
        images_dir = Path(tmpdir) / "imagesTr"
        labels_dir = Path(tmpdir) / "labelsTr"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # Create a small synthetic 3D volume
        for patient_id in ["patient_001", "patient_002", "patient_003"]:
            # Synthetic 3D image (10x10x5)
            img_vol = np.random.randn(10, 10, 5).astype(np.float32) * 100  # Fake HU
            label_vol = np.zeros((10, 10, 5), dtype=np.int32)
            label_vol[3:7, 3:7, 1:4] = 1  # Tumor in center
            
            # Save as NIfTI
            img_nii = nib.Nifti1Image(img_vol, np.eye(4))
            label_nii = nib.Nifti1Image(label_vol, np.eye(4))
            
            # Save without .gz suffix - nibabel will handle it
            nib.save(img_nii, str(images_dir / f"{patient_id}.nii"))
            nib.save(label_nii, str(labels_dir / f"{patient_id}.nii"))
        
        yield tmpdir


@pytest.fixture
def nested_dataset():
    """Create a nested dataset (as in MSD task structure)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure: tmpdir/Task08/imagesTr/...
        task_dir = Path(tmpdir) / "Task08_HepaticVessel"
        images_dir = task_dir / "imagesTr"
        labels_dir = task_dir / "labelsTr"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Create synthetic data
        for patient_id in ["liver_001", "liver_002"]:
            img_vol = np.random.randn(10, 10, 5).astype(np.float32) * 50
            label_vol = np.zeros((10, 10, 5), dtype=np.int32)
            label_vol[3:7, 3:7, :] = 2  # Multi-class label
            
            img_nii = nib.Nifti1Image(img_vol, np.eye(4))
            label_nii = nib.Nifti1Image(label_vol, np.eye(4))
            
            nib.save(img_nii, str(images_dir / f"{patient_id}.nii.gz"))
            nib.save(label_nii, str(labels_dir / f"{patient_id}.nii.gz"))
        
        yield tmpdir


def test_discovery_direct_children(synthetic_dataset):
    """Test dataset discovery with direct children structure."""
    discovery = DatasetDiscovery(synthetic_dataset)
    patient_ids = discovery.get_patient_ids()
    assert len(patient_ids) == 3
    assert "patient_001" in patient_ids
    assert discovery.num_classes == 2


def test_discovery_nested_structure(nested_dataset):
    """Test dataset discovery with nested directory structure."""
    discovery = DatasetDiscovery(nested_dataset)
    patient_ids = discovery.get_patient_ids()
    assert len(patient_ids) == 2
    assert "liver_001" in patient_ids
    assert "liver_002" in patient_ids


def test_label_budget_full(synthetic_dataset):
    """Test label budget with budget=1.0 (all labeled)."""
    discovery = DatasetDiscovery(synthetic_dataset, label_budget=1.0)
    labeled = discovery.get_patient_ids(labeled_only=True)
    unlabeled = discovery.get_patient_ids(unlabeled_only=True)
    
    assert len(labeled) == 3
    assert len(unlabeled) == 0


def test_label_budget_partial(synthetic_dataset):
    """Test label budget with budget=0.5."""
    discovery = DatasetDiscovery(synthetic_dataset, label_budget=0.5, budget_seed=42)
    labeled = discovery.get_patient_ids(labeled_only=True)
    unlabeled = discovery.get_patient_ids(unlabeled_only=True)
    
    # 50% of 3 = 1.5, truncated to 1
    assert len(labeled) == 1 or len(labeled) == 2
    assert len(unlabeled) > 0
    assert len(labeled) + len(unlabeled) == 3


def test_label_budget_zero(synthetic_dataset):
    """Test label budget with budget=0.0 (all unlabeled)."""
    discovery = DatasetDiscovery(synthetic_dataset, label_budget=0.0)
    labeled = discovery.get_patient_ids(labeled_only=True)
    unlabeled = discovery.get_patient_ids(unlabeled_only=True)
    
    assert len(labeled) == 0
    assert len(unlabeled) == 3


def test_label_budget_reproducibility(synthetic_dataset):
    """Test that label budget is reproducible with same seed."""
    discovery1 = DatasetDiscovery(synthetic_dataset, label_budget=0.5, budget_seed=42)
    discovery2 = DatasetDiscovery(synthetic_dataset, label_budget=0.5, budget_seed=42)
    
    labeled1 = set(discovery1.get_patient_ids(labeled_only=True))
    labeled2 = set(discovery2.get_patient_ids(labeled_only=True))
    
    assert labeled1 == labeled2


def test_load_image_label(synthetic_dataset):
    """Test loading individual image and label."""
    discovery = DatasetDiscovery(synthetic_dataset)
    
    img = discovery.load_image("patient_001")
    label = discovery.load_label("patient_001")
    
    assert img.shape == (10, 10, 5)
    assert label.shape == (10, 10, 5)
    assert img.dtype == np.float32
    assert label.dtype == np.int32


def test_slice_extraction(synthetic_dataset):
    """Test 2D slicing from 3D volume."""
    discovery = DatasetDiscovery(synthetic_dataset)
    img_vol = discovery.load_image("patient_001")
    label_vol = discovery.load_label("patient_001")
    
    slices = SliceExtractor.extract_slices(
        img_vol, label_vol, slice_thickness=1, min_slice_coverage=0.0
    )
    
    # Should get 5 slices (depth=5, thickness=1)
    assert len(slices) == 5
    
    # Each slice should be (img_2d, label_2d, slice_idx)
    for img_2d, label_2d, z_idx in slices:
        assert img_2d.shape == (1, 10, 10), f"Expected (1, 10, 10), got {img_2d.shape}"
        assert label_2d.shape == (10, 10)
        assert img_2d.dtype == np.float32
        assert label_2d.dtype == np.int32
        assert 0 <= z_idx < 5


def test_slice_thickness(synthetic_dataset):
    """Test slice thickness filter."""
    discovery = DatasetDiscovery(synthetic_dataset)
    img_vol = discovery.load_image("patient_001")
    label_vol = discovery.load_label("patient_001")
    
    # Every 2nd slice
    slices = SliceExtractor.extract_slices(
        img_vol, label_vol, slice_thickness=2, min_slice_coverage=0.0
    )
    
    # Should get 3 slices: z=0, z=2, z=4
    assert len(slices) == 3


def test_ct_windowing():
    """Test CT windowing."""
    # Synthetic HU values
    img_vol = np.array([-200, 0, 50, 100, 200], dtype=np.float32)
    
    windowed = CTPreprocessor.apply_ct_window(
        img_vol, window_center=50, window_width=400
    )
    
    # Check output is in [0, 1]
    assert windowed.min() >= 0.0
    assert windowed.max() <= 1.0
    assert windowed.dtype == np.float32


def test_minmax_normalization():
    """Test min-max normalization."""
    img_vol = np.array([0, 50, 100, 150, 200], dtype=np.float32)
    normalized = CTPreprocessor.normalize_minmax(img_vol)
    
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert normalized.dtype == np.float32


def test_zscore_normalization():
    """Test z-score normalization."""
    img_vol = np.random.randn(100).astype(np.float32) * 50 + 100
    normalized = CTPreprocessor.normalize_zscore(img_vol)
    
    # Should be in [0, 1] after clipping and rescaling
    assert 0.0 <= normalized.min()
    assert normalized.max() <= 1.0
    assert normalized.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
