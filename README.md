# Semi-Supervised CT Vessel/Tumor Segmentation (MSD Task08)

A minimal, academic research repository implementing **semi-supervised learning with Mean Teacher** and **reliability-aware targeted augmentation** for 2D medical image segmentation.

## Overview

This repository provides a complete pipeline for:
1. **Automated data discovery** from Medical Segmentation Decathlon Task08 (Hepatic Vessel)
2. **3D-to-2D axial slicing** for efficient 2D U-Net training
3. **Supervised baseline** - standard 2D U-Net training
4. **Semi-supervised Mean Teacher** - with EMA teacher network and consistency loss
5. **Novelty: Reliability-aware targeted augmentation**:
   - Computes teacher confidence (entropy or max-probability)
   - Uses reliability to gate pseudo-label usage
   - Applies spatially-varying augmentation focused on uncertain tumor boundaries

## Repository Structure

```
.
├── configs/                    # YAML configuration files
│   ├── base.yaml              # Base config (all experiments)
│   ├── supervised.yaml        # Supervised baseline
│   ├── ssl_meanteacher.yaml   # Mean Teacher SSL
│   └── ssl_meanteacher_targetaug.yaml  # With targeted augmentation
├── src/                        # Core modules (~500 LOC each)
│   ├── utils.py               # Config, logging, seeding
│   ├── data.py                # Data discovery, loading, slicing
│   ├── dataset.py             # PyTorch Dataset class for 2D slices
│   ├── transforms.py          # Weak/strong augmentation, targeted boundary aug
│   ├── models.py              # 2D U-Net architecture
│   ├── losses.py              # Dice, cross-entropy, confidence-gated pseudo-label
│   ├── metrics.py             # Segmentation metrics (Dice, Jaccard, etc.)
│   ├── ssl.py                 # Mean Teacher, reliability gating, pseudo-labels
│   └── train_engine.py        # Training loop, checkpointing, metrics logging
├── scripts/                    # Executable scripts
│   ├── train.py               # Main training script (supervised & SSL)
│   ├── make_splits.py         # Create patient-level train/val/test splits
│   ├── eval.py                # Evaluation on test set
│   ├── sanity_viz.py          # Visualization of predictions
│   └── run_compare.py         # Multi-experiment comparison
├── tests/                      # Pytest unit tests (19 tests)
│   ├── test_data.py           # Data loading, slicing
│   ├── test_transforms.py     # Augmentation pipelines
│   ├── test_models.py         # U-Net forward/backward
│   ├── test_ssl.py            # Mean Teacher, reliability gating
│   └── conftest.py            # Pytest configuration
├── pyproject.toml             # Project metadata
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

```bash
# Clone and install
cd ssl-ct-lesion-seg
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

```bash
# Create train/val/test splits
python scripts/make_splits.py --data_root dataset/Task08_HepaticVessel --seed 42
```

### 2. Run Supervised Baseline

```bash
# Full training (with real dataset)
python scripts/train.py \
  --config configs/supervised.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --seed 42 \
  --num_epochs 50

# Smoke test mode (2 epochs, minimal data)
python scripts/train.py \
  --config configs/supervised.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --smoke --smoke_samples 2
```

### 3. Run Semi-Supervised Mean Teacher

```bash
# Train with 20% labeled data
python scripts/train.py \
  --config configs/ssl_meanteacher.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --labeled_ratio 0.2 \
  --seed 42
```

### 4. Run with Reliability-Aware Targeted Augmentation

```bash
# Full SSL with novelty method
python scripts/train.py \
  --config configs/ssl_meanteacher_targetaug.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --labeled_ratio 0.2 \
  --seed 42
```

### 5. Evaluate & Visualize

```bash
# Evaluate best checkpoint
python scripts/eval.py \
  --checkpoint runs/baseline/best.pt \
  --data_root dataset/Task08_HepaticVessel \
  --splits_file splits/splits.json

# Visualize predictions
python scripts/sanity_viz.py \
  --checkpoint runs/baseline/best.pt \
  --data_root dataset/Task08_HepaticVessel \
  --output_dir runs/baseline/viz

# Compare multiple experiments
python scripts/run_compare.py \
  --configs configs/supervised.yaml configs/ssl_meanteacher.yaml \
  --labeled_ratios 0.2 0.5 1.0 \
  --seeds 42 123
```

## Testing

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_data.py -v

# Run with coverage
pytest --cov=src tests/ -v
```

All 19 tests pass and validate:
- Data discovery and 3D-to-2D slicing
- U-Net forward/backward passes
- Augmentation pipelines (weak, strong, targeted)
- Mean Teacher EMA updates
- Reliability gating
- SSL loss computation

## Key Features

### 1. Automatic Data Discovery
- Searches for `imagesTr/`, `labelsTr/` under `--data_root`
- Parses `dataset.json` for label information (or infers from data)
- Supports both `.nii` and `.nii.gz` formats
- Ignores metadata files (e.g., `._*`)

### 2. 3D-to-2D Slicing
- Extracts axial slices from 3D CT volumes
- Configurable slice thickness (every Nth slice)
- Optional min-coverage filter for tumor-containing slices
- Automatic CT windowing (Hounsfield normalization) or minmax

### 3. Semi-Supervised Learning
- **Mean Teacher**: Student model + EMA teacher
- **Consistency Loss**: MSE/KL between student and teacher predictions
- **Pseudo-labels**: Teacher-generated labels gated by confidence
- **Confidence Threshold**: Only use high-confidence pseudo-labels

### 4. Reliability-Aware Targeted Augmentation (Novel)
- **Reliability Computation**: From teacher entropy or max-probability
- **Boundary Detection**: Morphological operations to find tumor boundaries
- **Spatial Augmentation**: Stronger augmentation in uncertain boundary regions
- **Deterministic & Reproducible**: All random seeds controlled

### 5. Full Reproducibility
- Every run saves:
  - Config snapshot (`config.yaml`)
  - Training metrics (`metrics.csv`)
  - Best checkpoint (`best.pt`)
  - Periodic checkpoints (`epoch_*.pt`)
- Configurable deterministic mode (CuDNN, PyTorch)
- Seed control across numpy, torch, random

## Configuration

Each config file overrides the base config with method-specific settings:

```yaml
# Key configuration parameters
data_root: "dataset/Task08_HepaticVessel"  # Data directory
seed: 42                                    # Random seed
num_epochs: 50                              # Training epochs
batch_size: 8                               # Batch size
learning_rate: 1.0e-4                      # Adam learning rate
patch_size: [256, 256]                      # 2D patch size

# SSL-specific
ema_decay: 0.99                             # Teacher EMA decay
consistency_lambda: 0.1                     # Consistency loss weight
pseudo_label_threshold: 0.9                 # Confidence threshold

# Reliability-aware augmentation
use_targeted_augmentation: true
reliability_method: "entropy"               # "entropy", "maxprob", "combined"
min_augmentation_strength: 0.3
max_augmentation_strength: 1.0
```

## Dataset Support

### Flexible Directory Discovery

The dataset discovery module automatically finds `imagesTr/` and `labelsTr/` directories:

1. **Direct children** (most common):
   ```
   dataset/
   ├── imagesTr/
   ├── labelsTr/
   └── dataset.json
   ```

2. **One level nested** (MSD-style):
   ```
   dataset/
   └── Task08_HepaticVessel/
       ├── imagesTr/
       ├── labelsTr/
       └── dataset.json
   ```

Pass the root directory to `--data_root`; the discovery module handles both structures automatically:

```bash
python scripts/train.py --data_root dataset                         # Direct children
python scripts/train.py --data_root dataset                         # Nested (auto-detected)
python scripts/train.py --data_root dataset/Task08_HepaticVessel    # Explicit nested path
```

### Dataset JSON & Fallback Inference

- **With `dataset.json`**: Parses label metadata to infer num_classes
- **Without `dataset.json`**: Scans 1-3 label volumes to infer classes (logs warning)

Example `dataset.json`:
```json
{
  "description": "Hepatic Vessel Segmentation",
  "name": "MSD Task08",
  "labels": {
    "0": "background",
    "1": "vessel",
    "2": "tumor"
  },
  "numTraining": 303
}
```

If missing, the module logs:
```
⚠ WARNING: dataset.json not found!
Will infer num_classes by scanning label volumes.
```

### Label Budgeting for Semi-Supervised Learning

**Label budgeting** randomly selects a fraction of training patients to keep labeled; the rest are unlabeled for SSL experiments.

#### Usage

```bash
# 100% labeled (default, supervised)
python scripts/train.py --config configs/supervised.yaml \
  --data_root dataset/Task08_HepaticVessel

# 20% labeled, 80% unlabeled for SSL
python scripts/train.py --config configs/ssl_meanteacher.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --labeled_ratio 0.2

# 50% labeled experiment
python scripts/train.py --config configs/ssl_meanteacher.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --labeled_ratio 0.5
```

#### Reproducibility

Label budgeting is deterministic with seed control:

```python
discovery = DatasetDiscovery(
    data_root="dataset/Task08_HepaticVessel",
    label_budget=0.2,      # 20% labeled
    budget_seed=42         # Reproducible selection
)

labeled = discovery.get_patient_ids(labeled_only=True)
unlabeled = discovery.get_patient_ids(unlabeled_only=True)
```

With the same seed, the same patients are always selected as labeled/unlabeled.

### Image Normalization

Three normalization modes available:

#### 1. CT Windowing (default for hepatic vessel)
```bash
python scripts/train.py ... --config configs/supervised.yaml
```
- **Window center**: 50 HU (liver tissue)
- **Window width**: 400 HU
- Output: [0.0, 1.0]
- Best for: CT scans in Hounsfield Units

#### 2. Min-Max Normalization
- Scales pixel values to [0.0, 1.0]
- Robust to outliers: clips to min/max
- Best for: Unknown windowing or non-HU data

#### 3. Z-Score Normalization
- Standardizes to zero mean, unit variance
- Clips to ±3σ, then rescales to [0.0, 1.0]
- Best for: Multi-center studies with varying scanners

Configure in YAML:
```yaml
image_norm: "ct"        # "ct", "minmax", or "zscore"
```

### Demo: Explore Dataset

```bash
# Examine dataset structure and label distribution
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

Example output:
```
✓ Total patients: 303
  - Labeled:   90 (29.7%)
  - Unlabeled: 213 (70.3%)
  - Classes: 3 (background, vessel, tumor)
✓ Extracted 49 2D slices from 3D volume
✓ Ready for training with 90 labeled + 213 unlabeled patients
```

## Dataset Format

Expected structure (Medical Segmentation Decathlon Task08):

```
dataset/Task08_HepaticVessel/
├── dataset.json          # Metadata (optional, auto-inferred if missing)
├── imagesTr/             # Training images (3D NIfTI)
│   ├── hepaticvessel_001.nii.gz
│   ├── hepaticvessel_002.nii.gz
│   └── ...
├── labelsTr/             # Training labels (3D NIfTI)
│   ├── hepaticvessel_001.nii.gz
│   └── ...
└── imagesTs/             # Test images (optional)
```

## Methods

### U-Net 2D Architecture
- 4 encoder levels + bottleneck + 4 decoder levels
- Skip connections between corresponding levels
- ~7-8M parameters with base_channels=32

### Loss Functions
- **Dice Loss**: Region-based (ignores background)
- **Weighted Cross-Entropy**: Per-pixel classification loss
- **Confidence-Gated Pseudo-Label Loss**: Only optimize high-confidence regions
- **Consistency Loss**: MSE between student and teacher predictions

### Metrics
- **Dice Score** (F1)
- **Jaccard Score** (IoU)
- **Sensitivity** (recall)
- **Specificity**
- **Hausdorff Distance** (optional, slow)

### Optimization
- **Optimizer**: Adam (default) or SGD
- **Scheduler**: Cosine annealing (default) or constant
- **Batch Size**: 8 (configurable)
- **Learning Rate**: 1e-4 (configurable)

## Parameters

### Data Processing
- `image_norm`: "ct" (windowing) or "minmax"
- `slice_thickness`: Load every Nth slice
- `patch_size`: 2D spatial dimensions

### Model
- `in_channels`: 1 (grayscale)
- `out_channels`: Inferred from dataset (usually 2 = background + tumor)
- `base_channels`: Starting channel count (doubled at each encoder level)

### Training
- `num_epochs`: Total training epochs
- `batch_size`: Batch size
- `learning_rate`: Adam learning rate
- `weight_decay`: L2 regularization
- `seed`: Random seed for reproducibility
- `deterministic`: Enable deterministic algorithms

### SSL
- `ema_decay`: EMA decay for teacher (0.99 = slow update)
- `consistency_lambda`: Weight for consistency loss
- `consistency_rampup_epochs`: Warm-up period for consistency loss
- `pseudo_label_threshold`: Min confidence to use pseudo-label

### Reliability Gating (Targeted Augmentation)
- `use_targeted_augmentation`: Enable novelty method
- `reliability_method`: How to compute confidence
- `confidence_threshold`: Min confidence to gate pseudo-labels
- `min_augmentation_strength`: Minimum augmentation in high-confidence regions
- `max_augmentation_strength`: Maximum augmentation in low-confidence regions

## Output Structure

Each training run creates:

```
runs/
└── {experiment_name}/
    ├── config.yaml              # Saved configuration
    ├── train.log                # Training logs
    ├── metrics.csv              # Training metrics per epoch
    ├── best.pt                  # Best checkpoint (by val Dice)
    ├── epoch_5.pt               # Periodic checkpoints
    └── epoch_10.pt
```

## Code Style & Documentation

- **Minimal codebase**: ~500-600 lines per module
- **Type hints**: All functions documented with types
- **Docstrings**: Every class/function has clear docstring
- **Logging**: Comprehensive logging at INFO level
- **No unnecessary complexity**: Direct implementations over MONAI/advanced frameworks

## References

- **Mean Teacher**: Tarvainen & Valpola (2017)
- **Medical Segmentation Decathlon**: Simpson et al. (2019)
- **Targeted Augmentation**: Novel contribution (reliability + boundary)

## Limitations & Future Work

1. **GPU-only**: Designed for GPU training (CPU very slow)
2. **Single GPU**: No distributed training support
3. **2D only**: 3D U-Net not implemented (extend to 3D slices easily)
4. **Fixed patch size**: No adaptive patching
5. **Batch normalization**: Could replace with instance norm for small batches

## Contributing

This is an academic research codebase. Contributions welcome for:
- 3D U-Net architecture
- Distributed training support
- Additional augmentation strategies
- Performance optimizations

## License

This code is provided as-is for academic research purposes.

## Contact

For questions or issues, open an issue on the repository.

---

**Note**: This repository prioritizes **clarity and reproducibility** over performance. Every design choice favors readable, understandable code suitable for academic publication.
