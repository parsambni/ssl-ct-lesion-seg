"""Data loading and preprocessing for MSD-format medical imaging datasets."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)


class DatasetDiscovery:
    """Discover and load MSD-format datasets with flexible directory structure."""
    
    def __init__(self, data_root: str, label_budget: float = 1.0, budget_seed: int = 42):
        """
        Args:
            data_root: Root directory (searches imagesTr/labelsTr directly or nested)
            label_budget: Fraction of patients marked labeled (rest unlabeled for SSL)
            budget_seed: Seed for reproducible label selection
        """
        self.data_root = Path(data_root)
        
        # Find directories
        self.images_dir = self._find_dir("imagesTr")
        self.labels_dir = self._find_dir("labelsTr")
        
        if not self.images_dir or not self.labels_dir:
            raise FileNotFoundError(
                f"Could not find imagesTr/ and labelsTr/ in {data_root}\n"
                "Expected MSD format: data_root/imagesTr, data_root/labelsTr"
            )
        
        # Load metadata (optional)
        meta_path = self.images_dir.parent / "dataset.json"
        if not meta_path.exists():
            meta_path = self.data_root / "dataset.json"
        
        self.metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        else:
            logger.warning(f"No dataset.json found. Inferring metadata from data.")
        
        # Setup patients
        self.patient_ids = self._get_patient_ids()
        self.label_budget = label_budget
        self.budget_seed = budget_seed
        self.labeled_patients, self.unlabeled_patients = self._apply_label_budget()
        self.num_classes = self._infer_num_classes()
        
        logger.info(f"Found {len(self.patient_ids)} patients, {self.num_classes} classes")
    
    def _find_dir(self, dirname: str) -> Optional[Path]:
        """Find directory: check direct child, then one level nested."""
        d = self.data_root / dirname
        if d.exists():
            return d
        for subdir in sorted(self.data_root.iterdir()):
            if subdir.is_dir() and not subdir.name.startswith('.'):
                d = subdir / dirname
                if d.exists():
                    return d
        return None
    
    def _infer_num_classes(self) -> int:
        """Infer classes from metadata or label scan."""
        if "labels" in self.metadata:
            return max(int(k) for k in self.metadata["labels"].keys()) + 1
        
        # Scan first few labels
        label_files = list(self.labels_dir.glob("*.nii*"))[:3]
        unique_labels = set()
        for lf in label_files:
            try:
                vol = nib.load(lf).get_fdata().astype(np.int32)
                unique_labels.update(np.unique(vol).astype(int))
            except:
                pass
        
        return int(max(unique_labels)) + 1 if unique_labels else 2
    
    def _get_patient_ids(self) -> List[str]:
        """Get patients with matching image/label pairs."""
        patient_ids = []
        for img_file in sorted(self.images_dir.glob("*.nii*")):
            if img_file.name.startswith('._'):
                continue
            pid = img_file.stem if img_file.name.endswith('.gz') else img_file.stem
            pid = pid.replace('.nii', '')
            
            # Check label exists
            for ext in ['.nii.gz', '.nii']:
                if (self.labels_dir / f"{pid}{ext}").exists():
                    patient_ids.append(pid)
                    break
        return patient_ids
    
    def _apply_label_budget(self) -> Tuple[List[str], List[str]]:
        """Select labeled/unlabeled patients based on budget."""
        if self.label_budget >= 1.0:
            return self.patient_ids, []
        if self.label_budget <= 0.0:
            return [], self.patient_ids
        
        rng = np.random.RandomState(self.budget_seed)
        n_labeled = max(1, int(len(self.patient_ids) * self.label_budget))
        indices = np.arange(len(self.patient_ids))
        rng.shuffle(indices)
        
        labeled = [self.patient_ids[i] for i in indices[:n_labeled]]
        unlabeled = [self.patient_ids[i] for i in indices[n_labeled:]]
        return labeled, unlabeled
    
    def get_patient_ids(self, labeled_only: bool = False, unlabeled_only: bool = False) -> List[str]:
        """Get patient IDs with optional filtering."""
        if labeled_only:
            return self.labeled_patients
        if unlabeled_only:
            return self.unlabeled_patients
        return self.patient_ids
    
    def load_image(self, patient_id: str) -> np.ndarray:
        """Load 3D image as float32."""
        for ext in ['.nii.gz', '.nii']:
            path = self.images_dir / f"{patient_id}{ext}"
            if path.exists():
                return nib.load(path).get_fdata().astype(np.float32)
        raise FileNotFoundError(f"Image not found: {patient_id}")
    
    def load_label(self, patient_id: str) -> np.ndarray:
        """Load 3D label as int32."""
        for ext in ['.nii.gz', '.nii']:
            path = self.labels_dir / f"{patient_id}{ext}"
            if path.exists():
                return nib.load(path).get_fdata().astype(np.int32)
        raise FileNotFoundError(f"Label not found: {patient_id}")


class SliceExtractor:
    """Extract 2D axial slices from 3D volumes."""
    
    @staticmethod
    def extract_slices(
        img_vol: np.ndarray,
        label_vol: np.ndarray,
        slice_thickness: int = 1,
        min_slice_coverage: float = 0.0
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Extract 2D axial slices (H, W, D) -> list of (img_2d, label_2d, z_idx)."""
        assert img_vol.ndim == 3 and label_vol.ndim == 3
        assert img_vol.shape == label_vol.shape
        
        slices = []
        for z in range(0, img_vol.shape[2], slice_thickness):
            img_2d = img_vol[:, :, z]
            label_2d = label_vol[:, :, z]
            
            # Filter by coverage
            if min_slice_coverage > 0:
                if np.count_nonzero(label_2d) / label_2d.size < min_slice_coverage:
                    continue
            
            img_2d = img_2d[np.newaxis, ...].astype(np.float32)
            label_2d = label_2d.astype(np.int32)
            
            slices.append((img_2d, label_2d, z))
        
        return slices


class CTPreprocessor:
    """Image normalization for CT scans."""
    
    @staticmethod
    def apply_ct_window(img_vol: np.ndarray, window_center: float = 50, window_width: float = 400) -> np.ndarray:
        """Apply Hounsfield windowing (for CT in HU)."""
        win_min = window_center - window_width / 2
        win_max = window_center + window_width / 2
        img = np.clip(img_vol, win_min, win_max)
        return ((img - win_min) / (win_max - win_min + 1e-8)).astype(np.float32)
    
    @staticmethod
    def normalize_minmax(img_vol: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        v_min, v_max = img_vol.min(), img_vol.max()
        if v_max == v_min:
            return np.zeros_like(img_vol, dtype=np.float32)
        return ((img_vol - v_min) / (v_max - v_min)).astype(np.float32)
    
    @staticmethod
    def normalize_zscore(img_vol: np.ndarray) -> np.ndarray:
        """Z-score normalization (robust to multi-center variation)."""
        mean, std = img_vol.mean(), img_vol.std()
        if std < 1e-8:
            return np.zeros_like(img_vol, dtype=np.float32)
        img = np.clip((img_vol - mean) / std, -3, 3)
        return ((img + 3) / 6).astype(np.float32)
