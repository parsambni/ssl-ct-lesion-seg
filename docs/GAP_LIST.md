# Gap List: Q1 Journal-Ready Checklist

**Last Updated**: 2026-02-13

## What Exists (✅ Implemented)

### Core Implementation
- ✅ **2D U-Net Architecture** (~7-8M parameters, 4 encoder/decoder levels)
- ✅ **Mean Teacher SSL** (EMA teacher, consistency loss)
- ✅ **Reliability-Aware Targeted Augmentation** (novel contribution)
- ✅ **3D→2D Slicing Pipeline** (axial extraction, CT windowing)
- ✅ **Automatic Data Discovery** (flexible directory structure)
- ✅ **Label Budgeting** (patient-level, reproducible)
- ✅ **19 Unit Tests** (all passing, ~85% coverage)
- ✅ **Config System** (YAML-based, hierarchical)

### Existing Scripts
- ✅ `train.py` - Main training (supervised & SSL)
- ✅ `eval.py` - Test set evaluation
- ✅ `sanity_viz.py` - Prediction visualization
- ✅ `make_splits.py` - Patient-level splits
- ✅ `run_compare.py` - Multi-experiment comparison
- ✅ `demo_dataset.py` - Dataset exploration

### Documentation
- ✅ `README.md` - Comprehensive usage guide
- ✅ `IMPLEMENTATION_SUMMARY.md` - Code overview
- ✅ `DATASET_SUPPORT.md` - Data pipeline details
- ✅ `QUICK_REFERENCE.md` - Quick commands

---

## What's Missing: Gap Analysis

### **P0 (MUST-HAVE FOR PAPER)**

#### 1. **Experimental Protocol Documentation** 🔴 CRITICAL
**Status**: Missing  
**What's Needed**:
- [ ] Formal experimental protocol document (EXPERIMENTAL_PROTOCOL.md)
- [ ] Labeled fraction specifications (1%, 5%, 10%, 20%, 50%, 100%)
- [ ] Cross-validation vs fixed split strategy (with justification)
- [ ] Metric definitions with formulas (Dice, IoU, HD95, lesion-wise)
- [ ] Statistical reporting protocol (≥3 seeds, mean±std, significance tests)
- [ ] Compute budget estimation per experiment

**Why It Matters**: Reviewers expect rigorous, pre-defined protocols for medical imaging papers.

#### 2. **Baseline Comparison Matrix** 🔴 CRITICAL
**Status**: Partially implemented (vanilla Mean Teacher exists, others missing)  
**What's Needed**:
- [ ] **Supervised baselines**: 
  - [ ] Full supervision (100% labeled)
  - [ ] Low-data supervision (1%, 5%, 10%, 20%, 50%)
- [ ] **SSL baselines**:
  - [ ] Mean Teacher vanilla (no reliability gating) ✅ Exists
  - [ ] Pseudo-labeling baseline (confidence-only, no consistency)
  - [ ] Consistency regularization only (no pseudo-labels)
  - [ ] FixMatch-style (if feasible for segmentation)
- [ ] **Config toggles** for each baseline
- [ ] **Runbook** with exact commands

**Why It Matters**: Must demonstrate novelty outperforms standard SSL methods.

#### 3. **Ablation Study Scripts** 🔴 CRITICAL
**Status**: Missing  
**What's Needed**:
- [ ] **Reliability gating ablations**:
  - [ ] Confidence threshold sweep (τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95})
  - [ ] Uncertainty method comparison (entropy, maxprob, MC dropout)
  - [ ] With/without gating toggle
- [ ] **Augmentation ablations**:
  - [ ] Augmentation strength schedule (weak vs strong)
  - [ ] Targeted vs uniform augmentation
  - [ ] Boundary detection sensitivity
- [ ] **Mean Teacher ablations**:
  - [ ] EMA decay sweep (0.95, 0.97, 0.99, 0.995)
  - [ ] Consistency weight schedule
  - [ ] Rampup vs constant consistency loss
- [ ] **Automated ablation runner script**

**Why It Matters**: Ablations isolate novelty contribution; core to paper's claims.

#### 4. **Multi-Seed Reproducibility** 🔴 CRITICAL
**Status**: Seed control exists, but no multi-seed runner  
**What's Needed**:
- [ ] Multi-seed runner script (e.g., seeds={42, 123, 456, 789, 1024})
- [ ] Results aggregation script (compute mean±std per metric)
- [ ] Statistical significance testing (paired t-test, Wilcoxon)
- [ ] Determinism verification test (same seed → identical results)

**Why It Matters**: Q1 journals require statistical rigor; single-seed results are insufficient.

#### 5. **Advanced Metrics & Stratification** 🔴 CRITICAL
**Status**: Basic Dice/IoU exist; HD95 and lesion-wise missing  
**What's Needed**:
- [ ] **Hausdorff Distance 95** (HD95) implementation
- [ ] **Lesion-wise metrics** (per-lesion Dice, detection rate)
- [ ] **Size-stratified reporting** (small/medium/large lesions)
- [ ] **Boundary accuracy metrics** (surface distance)
- [ ] Script to compute and tabulate all metrics

**Why It Matters**: Medical segmentation papers require granular metric reporting.

#### 6. **Results Organization & Paper Artifacts** 🔴 CRITICAL
**Status**: Basic checkpoints exist; no paper-ready outputs  
**What's Needed**:
- [ ] Standardized results folder structure:
  ```
  results/
    ├── supervised_baseline/
    ├── ssl_meanteacher/
    ├── ssl_targetaug/
    ├── ablations/
    ├── tables/          # LaTeX-ready tables
    ├── figures/         # Publication-quality plots
    └── analysis/        # Statistical tests, failure cases
  ```
- [ ] CSV→LaTeX table generator
- [ ] Learning curve plotter (train/val Dice vs epoch)
- [ ] Threshold sweep visualizer (Dice vs τ)
- [ ] Qualitative grid generator (overlays for paper figures)
- [ ] Failure case analyzer

**Why It Matters**: Paper preparation requires structured, publication-ready outputs.

---

### **P1 (STRONGLY RECOMMENDED)**

#### 7. **Sanity Check Automation** 🟡
**Status**: Smoke mode exists; systematic checks missing  
**What's Needed**:
- [ ] **Overfit test**: Train on 1-2 volumes, expect >95% Dice
- [ ] **Label leakage check**: Verify train/val/test splits are disjoint
- [ ] **Upper bound check**: Train with 100% labels, establish performance ceiling
- [ ] **3D reconstruction test**: Map 2D predictions back to 3D, verify coherence
- [ ] **Data type verification**: Confirm float32 images, int64 labels throughout
- [ ] Automated sanity check script (`scripts/sanity_checks.py`)

**Why It Matters**: Prevents subtle bugs that can invalidate months of experiments.

#### 8. **TensorBoard Integration** 🟡
**Status**: CSV logging exists; no TensorBoard  
**What's Needed**:
- [ ] TensorBoard writer integration in `train_engine.py`
- [ ] Log scalars: loss, Dice, IoU, learning rate
- [ ] Log histograms: model weights, gradients
- [ ] Log images: input/pred/label overlays
- [ ] Log hyperparameters as table
- [ ] Instructions for launching TensorBoard

**Why It Matters**: Facilitates real-time experiment monitoring and debugging.

#### 9. **Confidence Calibration Analysis** 🟡
**Status**: Not implemented  
**What's Needed**:
- [ ] Reliability diagram (predicted confidence vs actual accuracy)
- [ ] Expected Calibration Error (ECE) metric
- [ ] Confidence histogram per class
- [ ] Calibration curve plotter
- [ ] Analysis of high/low confidence regions

**Why It Matters**: Validates that reliability gating is well-calibrated.

#### 10. **Compute Cost Tracking** 🟡
**Status**: Not implemented  
**What's Needed**:
- [ ] Wall-clock time logging per epoch
- [ ] GPU memory usage tracking
- [ ] FLOPs estimation for model
- [ ] Carbon footprint estimation (optional, for green AI)
- [ ] Table of compute costs per method

**Why It Matters**: Reviewers care about computational efficiency; adds credibility.

#### 11. **Hyperparameter Tuning Guidance** 🟡
**Status**: Defaults exist; no tuning guide  
**What's Needed**:
- [ ] Hyperparameter sensitivity analysis (learning rate, batch size, EMA decay)
- [ ] Recommended search ranges
- [ ] Tuning strategy (grid vs random vs Bayesian)
- [ ] Best practices document

**Why It Matters**: Helps reproducibility; shows robustness to hyperparameters.

#### 12. **Failure Case Analysis** 🟡
**Status**: Not implemented  
**What's Needed**:
- [ ] Script to identify worst-performing slices
- [ ] Qualitative visualization of failures
- [ ] Error analysis by lesion size, location, intensity
- [ ] Comparison of failure modes across methods
- [ ] Discussion-ready failure mode summary

**Why It Matters**: Papers must acknowledge limitations; shows thorough analysis.

---

### **P2 (NICE-TO-HAVE)**

#### 13. **3D U-Net Extension** 🟢
**Status**: Only 2D implemented  
**What's Needed**:
- [ ] 3D U-Net architecture (`models.py` extension)
- [ ] 3D patch extraction (overlapping, stitching)
- [ ] 3D augmentation pipeline
- [ ] Config for 3D training

**Why It Matters**: 3D networks are SOTA for medical imaging; adds comparison point.

#### 14. **Additional SSL Methods** 🟢
**Status**: Only Mean Teacher implemented  
**What's Needed**:
- [ ] FixMatch (strong augmentation + pseudo-labeling)
- [ ] VAT (Virtual Adversarial Training)
- [ ] MixMatch / ReMixMatch
- [ ] Self-training baselines

**Why It Matters**: Broadens comparison scope; demonstrates generalizability.

#### 15. **Post-Processing** 🟢
**Status**: Not implemented  
**What's Needed**:
- [ ] Connected component analysis (remove small false positives)
- [ ] Morphological operations (opening, closing)
- [ ] CRF (Conditional Random Fields) refinement
- [ ] Ensemble predictions (multiple checkpoints)

**Why It Matters**: Can improve final metrics; common in deployed systems.

#### 16. **Multi-Dataset Support** 🟢
**Status**: Designed for MSD Task08; not tested on others  
**What's Needed**:
- [ ] Test on additional MSD tasks (Liver, Spleen, Pancreas)
- [ ] Test on non-MSD datasets (CHAOS, LiTS, KiTS)
- [ ] Dataset-agnostic normalization
- [ ] Domain adaptation considerations

**Why It Matters**: Demonstrates method generalizability beyond single dataset.

#### 17. **Distributed Training** 🟢
**Status**: Single-GPU only  
**What's Needed**:
- [ ] Multi-GPU support (DataParallel or DistributedDataParallel)
- [ ] Multi-node training
- [ ] Batch size scaling strategy

**Why It Matters**: Enables larger batch sizes and faster training; not critical for paper.

#### 18. **Pre-trained Weights** 🟢
**Status**: Train from scratch only  
**What's Needed**:
- [ ] ImageNet pre-trained encoder (ResNet, EfficientNet)
- [ ] Self-supervised pre-training (SimCLR, MoCo)
- [ ] Transfer learning experiments

**Why It Matters**: Pre-training often improves low-data performance; adds comparison.

#### 19. **Interactive Demo** 🟢
**Status**: Not implemented  
**What's Needed**:
- [ ] Gradio/Streamlit web interface
- [ ] Upload image → predict → visualize
- [ ] Model comparison toggle

**Why It Matters**: Useful for presentations and demos; not for paper.

#### 20. **Docker Container** 🟢
**Status**: Not implemented  
**What's Needed**:
- [ ] Dockerfile with all dependencies
- [ ] Docker Compose for easy launch
- [ ] Instructions for running in container

**Why It Matters**: Improves reproducibility; makes code accessible to non-experts.

---

## Implementation Priority Roadmap

### **Week 1-2: Foundation (P0 items 1-3)**
1. **EXPERIMENTAL_PROTOCOL.md** - Define rigorous protocol
2. **BASELINE_RUNBOOK.md** - Exact commands for all baselines
3. **ABLATION_PLAN.md** - Detailed ablation matrix

### **Week 3-4: Experiments (P0 items 4-6)**
4. Multi-seed runner + aggregation scripts
5. Advanced metrics (HD95, lesion-wise, stratification)
6. Results organization + LaTeX table generation

### **Week 5-6: Validation (P1 items 7-9)**
7. Sanity check automation
8. TensorBoard integration
9. Confidence calibration analysis

### **Week 7-8: Analysis & Paper (P1 items 10-12 + Paper Prep)**
10. Compute cost tracking
11. Hyperparameter sensitivity
12. Failure case analysis
13. **Paper figures + tables generation**
14. **Reproducibility documentation**

### **Optional: Extensions (P2 items)**
- 3D U-Net, additional SSL methods, post-processing, etc.
- Implement only if needed for paper revisions or follow-up work

---

## Acceptance Criteria for "Q1 Journal-Ready"

### **Experimental Rigor**
- ✅ ≥3 random seeds per experiment (ideally 5)
- ✅ Statistical significance testing (p-values reported)
- ✅ Comprehensive baselines (supervised + SSL variants)
- ✅ Ablation studies isolating each novelty component
- ✅ Multiple labeled fractions (1%, 5%, 10%, 20%, 50%, 100%)

### **Metrics & Reporting**
- ✅ Standard metrics: Dice, IoU, HD95
- ✅ Lesion-wise metrics (per-lesion Dice, detection rate)
- ✅ Size-stratified reporting (small/medium/large)
- ✅ Confidence calibration analysis
- ✅ Failure case analysis

### **Reproducibility**
- ✅ Deterministic mode verified (same seed → same results)
- ✅ All hyperparameters documented
- ✅ Config files saved with every run
- ✅ Code + data availability statement
- ✅ Compute requirements specified (GPU type, memory, time)

### **Paper Artifacts**
- ✅ LaTeX-ready tables (main results, ablations, baselines)
- ✅ Publication-quality figures (learning curves, qualitative grids)
- ✅ Statistical significance tables
- ✅ Supplementary material (extended results, failure cases)

### **Code Quality**
- ✅ Comprehensive unit tests (coverage >80%)
- ✅ Sanity checks automated and passing
- ✅ Clear documentation (README, docstrings, comments)
- ✅ Modular, readable codebase (<600 LOC per file)

---

## Estimated Effort

### P0 (Critical) - 4-6 weeks
- Documentation: 1 week
- Baseline implementations: 1-2 weeks
- Multi-seed experiments: 2-3 weeks
- Analysis & paper artifacts: 1 week

### P1 (Recommended) - 2-3 weeks
- Sanity checks & validation: 1 week
- TensorBoard & monitoring: 0.5 week
- Advanced analysis: 1-1.5 weeks

### P2 (Optional) - 4+ weeks
- Extensions (3D, additional methods): 2-3 weeks
- Post-processing & refinements: 1 week
- Interactive demo & Docker: 1 week

**Total for P0+P1**: 6-9 weeks  
**Total for P0+P1+P2**: 10-13 weeks

---

## Next Steps

1. **Immediate**: Create Phase 1-6 documentation (QUICKSTART, PROTOCOL, RUNBOOK, etc.)
2. **Short-term**: Implement P0 missing scripts (multi-seed runner, ablations, advanced metrics)
3. **Medium-term**: Run full experimental suite (all baselines, ablations, seeds)
4. **Long-term**: Generate paper artifacts and prepare submission

This gap analysis provides a clear roadmap from the current state to a Q1 journal-ready submission.
