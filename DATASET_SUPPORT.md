# MSD Task08 Dataset Support Implementation

## Overview

This document describes the comprehensive dataset support implementation for Medical Segmentation Decathlon (MSD) Task08 HepaticVessel, including flexible directory discovery, robust fallback handling, label budgeting, and multiple normalization modes.

**Date**: February 2025  
**Status**: ✓ Complete and tested (25 tests passing)

---

## 1. Flexible Directory Discovery

### Problem
Medical imaging datasets often have varying directory structures:
- Direct children: `dataset/imagesTr/`, `dataset/labelsTr/`
- Nested: `dataset/Task08/imagesTr/`, `dataset/Task08/labelsTr/`
- Deep nesting: `dataset/MSD/Task08/imagesTr/`, etc.

Users with different data layouts should not need to reorganize files.

### Solution: `DatasetDiscovery._find_image_label_dirs()`

The discovery module implements a **two-level search**:

#### Level 1: Direct Children
```python
# Check data_root/imagesTr and data_root/labelsTr
images_dir = data_root / "imagesTr"
labels_dir = data_root / "labelsTr"
if images_dir.exists() and labels_dir.exists():
    return images_dir, labels_dir
```

#### Level 2: One Level Nested
```python
# Check data_root/*/imagesTr and data_root/*/labelsTr
for subdir in sorted(data_root.iterdir()):
    if not subdir.is_dir() or subdir.name.startswith('.'):
        continue
    images_dir = subdir / "imagesTr"
    labels_dir = subdir / "labelsTr"
    if images_dir.exists() and labels_dir.exists():
        return images_dir, labels_dir
```

### Usage

```bash
# Both work automatically
python train.py --data_root dataset                    # Direct structure
python train.py --data_root dataset                    # Nested (auto-detects)
python train.py --data_root dataset/Task08_Hepatic    # Explicit nested path
```

### Test Coverage

- `test_discovery_direct_children`: ✓ Passes
- `test_discovery_nested_structure`: ✓ Passes

---

## 2. Robust Fallback for Missing dataset.json

### Problem
Not all medical datasets include `dataset.json`. Without metadata, we cannot infer:
- Number of classes
- Class names/meanings
- Data version information

### Solution: Intelligent Inference

#### Priority Chain
1. **metadata["labels"]**  
   Most reliable: direct class labels from JSON
   ```python
   if "labels" in self.metadata:
       num_classes = max(int(k) for k in labels.keys()) + 1
   ```

2. **metadata["num_classes"]**  
   Fallback in JSON if available

3. **Label Volume Scanning** (robust fallback)
   - Scan up to 3 label volumes
   - Compute unique label values across samples
   - Infer num_classes as max(unique_labels) + 1
   ```python
   unique_labels_set = set()
   for label_file in label_files[:3]:
       label_vol = nib.load(label_file).get_fdata().astype(np.int32)
       unique_labels = np.unique(label_vol)
       unique_labels_set.update(unique_labels.astype(int).tolist())
   num_classes = int(max(unique_labels_set)) + 1
   ```

#### Logging

When dataset.json is missing, the module logs loudly:
```
⚠ WARNING: dataset.json not found!
Will infer num_classes by scanning label volumes.
Expected location: /path/to/data/dataset.json
Task: MSD Task08 HepaticVessel (2 classes: background + vessel)
```

When inference succeeds:
```
✓ Scanned 3/303 label volumes
✓ Found unique labels: [0, 1, 2]
✓ Inferred 3 classes
```

### Test Coverage

- Label volume scanning works even without dataset.json
- All tests pass with synthetic datasets (no JSON)
- Real MSD data with JSON verified correct class inference (3 classes)

---

## 3. Label Budgeting for Semi-Supervised Learning

### Problem
Semi-supervised learning requires controlling what fraction of training data is labeled.

Options:
- **Random sampling per-slice**: Doesn't match real-world labeling (per-patient)
- **Random sampling per-patient**: More realistic, but complex to implement

### Solution: Patient-Level Label Budgeting

The `DatasetDiscovery` class supports label budgeting at the **patient level**:

```python
discovery = DatasetDiscovery(
    data_root="dataset/Task08_HepaticVessel",
    label_budget=0.2,    # 20% of patients labeled
    budget_seed=42       # Reproducible selection
)

labeled_patients = discovery.get_patient_ids(labeled_only=True)
unlabeled_patients = discovery.get_patient_ids(unlabeled_only=True)
```

#### Implementation: `DatasetDiscovery._apply_label_budget()`

```python
def _apply_label_budget(self) -> Tuple[List[str], List[str]]:
    """Apply label budget: randomly mark some patients as unlabeled."""
    if self.label_budget >= 1.0:
        return self.patient_ids, []
    
    if self.label_budget <= 0.0:
        return [], self.patient_ids
    
    rng = np.random.RandomState(self.budget_seed)
    n_labeled = max(1, int(len(self.patient_ids) * self.label_budget))
    
    indices = np.arange(len(self.patient_ids))
    rng.shuffle(indices)
    
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]
    
    return labeled, unlabeled
```

#### Reproducibility

With the same seed, **exactly the same patients** are selected as labeled:

```bash
# Run 1
python train.py --data_root dataset --labeled_ratio 0.2 --seed 42
# → Same 60 patients in "labeled" group

# Run 2 (different machine, different run)
python train.py --data_root dataset --labeled_ratio 0.2 --seed 42
# → Same 60 patients in "labeled" group
```

#### Usage in Training

The trained dataset tracks label status:

```python
dataset = SliceDataset(
    train_ids, discovery,
    track_label_status=(label_budget < 1.0)
)

for item in dataset:
    image = item["image"]
    label = item["label"]
    is_labeled = item["is_labeled"]  # True or False
```

SSL training uses this to:
1. Train on labeled slices with full supervision
2. Train on unlabeled slices with consistency loss only

#### Test Coverage

- `test_label_budget_full`: ✓ label_budget=1.0 → all labeled
- `test_label_budget_partial`: ✓ label_budget=0.5 → ~50% split
- `test_label_budget_zero`: ✓ label_budget=0.0 → all unlabeled
- `test_label_budget_reproducibility`: ✓ Same seed → same patients

---

## 4. Image Normalization Modes

### Problem
Medical images come from different sources:
- **CT scans**: Hounsfield Units (HU), not directly displayable
- **Generic images**: Unknown scaling
- **Multi-center data**: Different scanners, preprocessing

No single normalization works for all cases.

### Solution: Three Normalization Modes

All modes output float32 in [0.0, 1.0] range.

#### Mode 1: CT Windowing (default)
```python
CTPreprocessor.apply_ct_window(
    img_vol, 
    window_center=50,   # Liver tissue HU
    window_width=400    # Typical liver window
)
```

- **Input**: Raw CT in Hounsfield Units
- **Process**:
  1. Clip to [window_min, window_max]
  2. Linearly scale to [0.0, 1.0]
- **Output**: [0.0, 1.0] float32
- **Best for**: CT scans in HU (MSD Task08)

**Example**:
```
HU_min = 50 - 400/2 = -150
HU_max = 50 + 400/2 = 250

img_clipped = clip(img, -150, 250)
img_normalized = (img_clipped - (-150)) / 400
```

#### Mode 2: Min-Max Normalization
```python
CTPreprocessor.normalize_minmax(img_vol)
```

- **Input**: Any numeric array
- **Process**:
  1. Compute min and max
  2. Linearly scale: (x - min) / (max - min)
- **Output**: [0.0, 1.0] float32
- **Best for**: Unknown windowing, non-HU data
- **Robust**: Handles outliers by clipping

#### Mode 3: Z-Score Normalization
```python
CTPreprocessor.normalize_zscore(img_vol)
```

- **Input**: Any numeric array
- **Process**:
  1. Compute mean and std
  2. Normalize: (x - mean) / std
  3. Clip to [-3, 3], rescale to [0.0, 1.0]
- **Output**: [0.0, 1.0] float32
- **Best for**: Multi-center studies
- **Robust**: Standardizes across scanners

#### Configuration

In YAML config:
```yaml
image_norm: "ct"        # "ct", "minmax", or "zscore"
```

#### Test Coverage

- `test_ct_windowing`: ✓ Output in [0, 1]
- `test_minmax_normalization`: ✓ min=0, max=1
- `test_zscore_normalization`: ✓ Robust to outliers

---

## 5. Correct Data Types Throughout Pipeline

### Problem
Type inconsistencies can cause:
- Loss computation bugs (int32 vs float32)
- GPU memory errors
- Silent numerical errors

### Solution: Explicit Type Conversion

**Image**: float32 (normalized values)  
**Label**: int32 (class indices) → int64 for PyTorch loss functions

#### Pipeline Stages

1. **Load from disk**
   ```python
   def load_image(self, patient_id: str) -> np.ndarray:
       img_vol = nib.load(str(img_file)).get_fdata()
       return img_vol.astype(np.float32)  # ← explicit
   
   def load_label(self, patient_id: str) -> np.ndarray:
       label_vol = nib.load(str(label_file)).get_fdata()
       return label_vol.astype(np.int32)  # ← explicit
   ```

2. **Normalize**
   ```python
   def apply_ct_window(img_vol: np.ndarray) -> np.ndarray:
       # ...
       return img_windowed.astype(np.float32)  # ← explicit
   ```

3. **Extract slices**
   ```python
   img_slice = img_slice.astype(np.float32)  # ← explicit
   label_slice = label_slice.astype(np.int32)  # ← explicit
   ```

4. **PyTorch Dataset**
   ```python
   img_tensor = torch.from_numpy(img).float()      # → torch.float32
   label_tensor = torch.from_numpy(label).long()  # → torch.int64 (for CE loss)
   ```

---

## 6. Demo Script

A comprehensive demonstration script is provided: `scripts/demo_dataset.py`

### Features
- Dataset discovery with logging
- Patient distribution visualization
- Sample data loading and inspection
- Multiple normalization modes
- Slice extraction statistics
- Label distribution analysis

### Usage

```bash
# Basic exploration
python scripts/demo_dataset.py --data_root dataset/Task08_HepaticVessel

# With label budgeting
python scripts/demo_dataset.py \
  --data_root dataset/Task08_HepaticVessel \
  --label_budget 0.3 \
  --norm_mode ct

# Verbose output
python scripts/demo_dataset.py \
  --data_root dataset/Task08_HepaticVessel \
  --verbose
```

### Output Example

```
========================================================================
Dataset Discovery & Label Budgeting Demo
========================================================================

[1/4] DATASET DISCOVERY
✓ Loaded metadata from dataset.json
✓ Inferred 3 classes from dataset.json labels
Dataset discovery complete:
  Data root: dataset/Task08_HepaticVessel
  Images: Task08_HepaticVessel/imagesTr
  Labels: Task08_HepaticVessel/labelsTr
  Total patients: 303
  Labeled: 90, Unlabeled: 213
  Classes: 3

[2/4] PATIENT DISTRIBUTION
✓ Total patients: 303
  - Labeled:   90 (29.7%)
  - Unlabeled: 213 (70.3%)

[3/4] SAMPLE DATA LOADING
Loading sample patient: hepaticvessel_001
✓ Image loaded: (512, 512, 49), dtype=float32
✓ Label loaded: (512, 512, 49), dtype=int32
✓ Unique label values: [0, 1, 2]

[4/4] IMAGE PREPROCESSING
Normalization mode: ct
✓ Applied CT windowing (center=50 HU, width=400 HU)
✓ Normalized image: (512, 512, 49), dtype=float32
  - Min: 0.0000
  - Max: 1.0000
  - Mean: 0.1375
  - Std: 0.2560

Slice extraction:
✓ Extracted 49 2D slices from 3D volume
  - Slice 0: image (1, 512, 512) (float32), label (512, 512) (int32)

========================================================================
SUMMARY
========================================================================
✓ Dataset structure: direct children
✓ Dataset metadata: from JSON
✓ Classes detected: 3
✓ Label budget applied: 30% labeled, 70% unlabeled
✓ Ready for training with 90 labeled + 213 unlabeled patients

✓ Demo completed successfully!
```

---

## 7. Integration with Training Pipeline

### Training with Label Budgeting

```bash
# Supervised (100% labeled)
python train.py \
  --config configs/supervised.yaml \
  --data_root dataset/Task08_HepaticVessel

# Semi-supervised (20% labeled)
python train.py \
  --config configs/ssl_meanteacher.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --labeled_ratio 0.2
```

The training script:
1. Initializes `DatasetDiscovery` with `label_budget=args.labeled_ratio`
2. Creates `SliceDataset` with `track_label_status=True`
3. Splits slices by `is_labeled` flag
4. Trains labeled and unlabeled loaders separately for SSL

### Expected Behavior

With `--labeled_ratio 0.2`:

```python
discovery = DatasetDiscovery(
    "dataset/Task08_HepaticVessel",
    label_budget=0.2,  # ← 20% of patients
    budget_seed=42
)
# Result: ~60 labeled patients, ~243 unlabeled patients

# When SliceDataset loads ~15,000 slices total:
# → ~3,000 labeled slices (supervised loss)
# → ~12,000 unlabeled slices (consistency loss)
```

---

## 8. Test Coverage

**Total: 25 tests passing**

### Dataset Discovery Tests (12 tests)
1. `test_discovery_direct_children` - ✓
2. `test_discovery_nested_structure` - ✓
3. `test_label_budget_full` - ✓
4. `test_label_budget_partial` - ✓
5. `test_label_budget_zero` - ✓
6. `test_label_budget_reproducibility` - ✓
7. `test_load_image_label` - ✓
8. `test_slice_extraction` - ✓
9. `test_slice_thickness` - ✓
10. `test_ct_windowing` - ✓
11. `test_minmax_normalization` - ✓
12. `test_zscore_normalization` - ✓

### Model Tests (4 tests)
- U-Net initialization, forward pass, backward pass

### SSL Tests (6 tests)
- Mean Teacher, EMA updates, reliability gating

### Augmentation Tests (3 tests)
- Weak, strong, targeted augmentation

---

## 9. Key Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/data.py` | 330+ | DatasetDiscovery, SliceExtractor, CTPreprocessor |
| `src/dataset.py` | 190+ | SliceDataset with label tracking |
| `scripts/train.py` | 397 | Training with label budget support |
| `scripts/demo_dataset.py` | 196 | Dataset exploration demo |
| `tests/test_data.py` | 200+ | Comprehensive dataset tests |
| `README.md` | 500+ | Full documentation |

---

## 10. Backward Compatibility

**No breaking changes**. Existing code continues to work:

```python
# Old code (still works)
discovery = DatasetDiscovery("dataset")  # label_budget defaults to 1.0
all_patients = discovery.get_patient_ids()  # Returns all patients

# New code
discovery = DatasetDiscovery("dataset", label_budget=0.2)
labeled = discovery.get_patient_ids(labeled_only=True)
unlabeled = discovery.get_patient_ids(unlabeled_only=True)
```

---

## 11. Error Handling

### Clear Error Messages

When directories not found:
```
FileNotFoundError: Could not find imagesTr/ and labelsTr/ directories.
Searched:
  1. Direct: /path/to/data/imagesTr, /path/to/data/labelsTr
  2. Nested (1 level): /path/to/data/*/imagesTr, /path/to/data/*/labelsTr
Please ensure your data is in MSD format with imagesTr/ and labelsTr/ folders.
```

When dataset.json missing:
```
⚠ WARNING: dataset.json not found!
Will infer num_classes by scanning label volumes.
Expected location: /path/to/data/dataset.json
Task: MSD Task08 HepaticVessel (2 classes: background + vessel)
```

### Robust Fallbacks

- Missing labels: Skipped with warning, other patients still load
- Empty label volumes: Scanned for class inference
- Invalid NIfTI files: Caught, logged, training continues

---

## 12. Summary

This implementation provides:

✓ **Flexible directory discovery** (direct or nested)  
✓ **Robust fallback** for missing metadata  
✓ **Label budgeting** for semi-supervised experiments  
✓ **Multiple normalization modes** (CT, minmax, zscore)  
✓ **Correct data types** throughout pipeline  
✓ **Comprehensive tests** (12 dataset tests)  
✓ **Demo script** for exploration  
✓ **Full documentation** with examples  
✓ **Zero breaking changes** to existing code  

**Status**: Production-ready for MSD Task08 and similar medical imaging datasets.
