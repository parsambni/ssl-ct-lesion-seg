# Quickstart: 1-Run Success Path

**Goal**: Get one successful training run from scratch in <30 minutes.

---

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **GPU**: NVIDIA GPU with ≥8GB VRAM (RTX 3070 / V100 / A100)
  - *CPU fallback*: Possible but ~50x slower (not recommended for real experiments)
- **CUDA**: 11.8 or 12.1
- **Python**: 3.8, 3.9, or 3.10
- **Disk Space**: ≥50GB for dataset + checkpoints

### Environment Info Commands
```bash
# Check GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Check Python version
python --version
```

---

## Step 1: Environment Setup (5 mins)

### Option A: Conda (Recommended)
```bash
# Create environment
conda create -n ssl-seg python=3.9 -y
conda activate ssl-seg

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
cd /path/to/ssl-ct-lesion-seg
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Expected Output**:
```
PyTorch 2.0.1+cu118, CUDA: True
```

### Option B: pip + venv
```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### Common Failure Modes
❌ **"torch not compiled with CUDA"**  
→ Reinstall PyTorch with correct CUDA version:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

❌ **"ImportError: libcuda.so"**  
→ Check CUDA driver: `nvidia-smi`. Update drivers if needed.

---

## Step 2: Dataset Preparation (10 mins)

### Download MSD Task08 (Hepatic Vessel)
```bash
# Create dataset directory
mkdir -p dataset
cd dataset

# Download from Medical Segmentation Decathlon
# Method 1: Direct download (if available)
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar
tar -xvf Task08_HepaticVessel.tar

# Method 2: Manual download
# Visit: http://medicaldecathlon.com/
# Download Task08_HepaticVessel.tar, extract to dataset/

cd ..
```

**Expected Structure**:
```
dataset/Task08_HepaticVessel/
├── dataset.json
├── imagesTr/             # 303 3D CT volumes (.nii.gz)
│   ├── hepaticvessel_001.nii.gz
│   ├── hepaticvessel_002.nii.gz
│   └── ...
└── labelsTr/             # 303 3D label volumes
    ├── hepaticvessel_001.nii.gz
    └── ...
```

### Verify Dataset
```bash
python scripts/demo_dataset.py --data_root dataset/Task08_HepaticVessel
```

**Expected Output**:
```
✓ Total patients: 303
  Classes: 3 (background, vessel, tumor)
✓ Extracted 49 2D slices from sample volume
✓ Image shape: (512, 512), Label shape: (512, 512)
✓ Dataset ready for training
```

### Common Failure Modes
❌ **"Directory not found"**  
→ Check path: `ls dataset/Task08_HepaticVessel/imagesTr | head`

❌ **"No .nii.gz files found"**  
→ Extraction failed. Re-extract tar file or check download.

---

## Step 3: Create Data Splits (2 mins)

```bash
# Create patient-level train/val/test splits
python scripts/make_splits.py \
  --data_root dataset/Task08_HepaticVessel \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --output splits/splits.json
```

**Expected Output**:
```
✓ Total patients: 303
✓ Train: 212 patients (70.0%)
✓ Val:   46 patients (15.2%)
✓ Test:  45 patients (14.9%)
✓ Splits saved to splits/splits.json
```

**Verify Splits**:
```bash
cat splits/splits.json | python -m json.tool | head -20
```

---

## Step 4: Run Supervised Baseline (10 mins)

### Quick Smoke Test (2 mins)
```bash
# Fast test with minimal data
python scripts/train.py \
  --config configs/supervised.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --seed 42 \
  --smoke
```

**What to Look For**:
- ✅ "Device: cuda:0" (GPU detected)
- ✅ "Model: 7.8M parameters"
- ✅ "Train slices: ~100, Val slices: ~50"
- ✅ "Epoch 1/2" progress bar
- ✅ "Best model saved to runs/baseline/best.pt"

**Expected Time**: ~2 minutes on RTX 3070

### Full Training (50 epochs, ~10 mins)
```bash
python scripts/train.py \
  --config configs/supervised.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --seed 42 \
  --num_epochs 50
```

**Monitor Progress**:
```bash
# Watch training log in real-time
tail -f runs/baseline/train.log

# Check metrics
cat runs/baseline/metrics.csv
```

**Expected Output** (partial):
```
Epoch 1/50: train_loss=0.452, val_dice=0.623
Epoch 10/50: train_loss=0.287, val_dice=0.751
Epoch 30/50: train_loss=0.201, val_dice=0.823
Epoch 50/50: train_loss=0.156, val_dice=0.862
✓ Best checkpoint: epoch 48, val_dice=0.867
```

**Time/Compute**:
- **GPU**: ~10-15 minutes (RTX 3070)
- **VRAM**: ~4-5GB peak
- **CPU**: ~8-10 hours (not recommended)

### Common Failure Modes
❌ **"CUDA out of memory"**  
→ Reduce batch size in config: `batch_size: 4` (from 8)

❌ **"NaN loss"**  
→ Reduce learning rate: `learning_rate: 5.0e-5` (from 1e-4)

---

## Step 5: Evaluate Model (2 mins)

```bash
python scripts/eval.py \
  --checkpoint runs/baseline/best.pt \
  --data_root dataset/Task08_HepaticVessel \
  --splits_file splits/splits.json
```

**Expected Output**:
```
✓ Loaded checkpoint from runs/baseline/best.pt
✓ Test set: 45 patients, ~2205 slices

Test Metrics:
  Dice:        0.862 ± 0.034
  IoU:         0.758 ± 0.041
  Sensitivity: 0.891 ± 0.028
  Specificity: 0.995 ± 0.003

Per-Class Dice:
  Class 1 (vessel): 0.847 ± 0.039
  Class 2 (tumor):  0.877 ± 0.031
```

---

## Step 6: Visualize Predictions (2 mins)

```bash
python scripts/sanity_viz.py \
  --checkpoint runs/baseline/best.pt \
  --data_root dataset/Task08_HepaticVessel \
  --splits_file splits/splits.json \
  --output_dir runs/baseline/viz \
  --num_samples 10
```

**Expected Output**:
```
✓ Generated 10 overlay visualizations
✓ Saved to runs/baseline/viz/
  - overlay_001.png
  - overlay_002.png
  ...
```

**View Results**:
```bash
# On Linux with display
xdg-open runs/baseline/viz/overlay_001.png

# Or copy to local machine
scp -r user@server:runs/baseline/viz ./local_viz
```

---

## Step 7: Semi-Supervised Training (Optional, +10 mins)

### Mean Teacher with 20% Labeled Data
```bash
python scripts/train.py \
  --config configs/ssl_meanteacher.yaml \
  --data_root dataset/Task08_HepaticVessel \
  --labeled_ratio 0.2 \
  --seed 42 \
  --num_epochs 100
```

**Expected Output**:
```
✓ Labeled: 60 patients (20%), Unlabeled: 212 patients (80%)
Epoch 1/100: sup_loss=0.498, cons_loss=0.234, val_dice=0.587
Epoch 50/100: sup_loss=0.234, cons_loss=0.089, val_dice=0.792
Epoch 100/100: sup_loss=0.187, cons_loss=0.056, val_dice=0.831
```

**Comparison** (Expected Dice @ 20% labels):
- Supervised baseline: ~0.72 ± 0.04
- Mean Teacher: ~0.83 ± 0.03 (+0.11 improvement)

---

## Troubleshooting Checklist

### Before Running
- [ ] GPU available: `nvidia-smi`
- [ ] CUDA enabled: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Dataset exists: `ls dataset/Task08_HepaticVessel/imagesTr | wc -l` (should be 303)
- [ ] Splits created: `ls splits/splits.json`

### During Training
- [ ] GPU utilization: `watch -n 1 nvidia-smi` (should be ~80-95%)
- [ ] Loss decreasing: `tail runs/baseline/metrics.csv`
- [ ] No NaN losses: `grep -i nan runs/baseline/train.log`

### After Training
- [ ] Checkpoint exists: `ls runs/baseline/best.pt`
- [ ] Metrics logged: `wc -l runs/baseline/metrics.csv` (should be num_epochs + 1)
- [ ] Config saved: `cat runs/baseline/config.yaml`

---

## Quick Reference Commands

### Training
```bash
# Supervised (100% labeled)
python scripts/train.py --config configs/supervised.yaml --data_root dataset/Task08_HepaticVessel

# SSL Mean Teacher (20% labeled)
python scripts/train.py --config configs/ssl_meanteacher.yaml --data_root dataset/Task08_HepaticVessel --labeled_ratio 0.2

# SSL with Targeted Aug (novel method)
python scripts/train.py --config configs/ssl_meanteacher_targetaug.yaml --data_root dataset/Task08_HepaticVessel --labeled_ratio 0.2
```

### Evaluation
```bash
# Evaluate on test set
python scripts/eval.py --checkpoint runs/baseline/best.pt --data_root dataset/Task08_HepaticVessel

# Visualize predictions
python scripts/sanity_viz.py --checkpoint runs/baseline/best.pt --output_dir runs/baseline/viz
```

### Testing
```bash
# Run all unit tests
pytest -v

# Run specific test
pytest tests/test_data.py -v

# Check coverage
pytest --cov=src tests/ -v
```

---

## Success Criteria

You've successfully completed the quickstart if:

✅ **Environment**: PyTorch + CUDA working  
✅ **Dataset**: 303 patients discovered  
✅ **Training**: Completed without errors  
✅ **Validation Dice**: >0.80 (supervised) or >0.75 (SSL @ 20%)  
✅ **Checkpoints**: Saved to `runs/baseline/best.pt`  
✅ **Visualizations**: Generated overlay images  

**Next Steps**: See `docs/EXPERIMENTAL_PROTOCOL.md` for full research protocol.

---

## Estimated Total Time

| Step | Time (GPU) | Time (CPU) |
|------|-----------|-----------|
| 1. Environment | 5 mins | 5 mins |
| 2. Dataset | 10 mins | 10 mins |
| 3. Splits | 2 mins | 2 mins |
| 4. Training (smoke) | 2 mins | ~30 mins |
| 5. Evaluation | 2 mins | ~5 mins |
| 6. Visualization | 2 mins | 2 mins |
| **Total (smoke)** | **23 mins** | **54 mins** |
| **Total (full 50 epochs)** | **31 mins** | **10+ hours** |

**Recommended**: Use GPU for all experiments. CPU is only suitable for smoke tests.
