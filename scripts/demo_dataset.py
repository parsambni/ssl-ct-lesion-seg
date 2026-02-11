#!/usr/bin/env python3
"""
demo_dataset.py: Demonstrate dataset discovery and label budgeting.

This script shows how to use the flexible dataset discovery system
for MSD Task08 HepaticVessel with:
- Automatic directory detection (direct or nested structure)
- Dataset.json parsing with fallback inference
- Label budgeting for semi-supervised learning
- Multiple normalization modes

Usage:
    python scripts/demo_dataset.py --data_root dataset
    python scripts/demo_dataset.py --data_root dataset --label_budget 0.2
    python scripts/demo_dataset.py --data_root dataset --norm_mode zscore
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DatasetDiscovery, SliceExtractor, CTPreprocessor
from src.dataset import SliceDataset


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate dataset discovery and label budgeting"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root directory containing imagesTr/labelsTr (direct or nested)"
    )
    parser.add_argument(
        "--label_budget",
        type=float,
        default=1.0,
        help="Fraction of training patients to mark as labeled [0.0, 1.0] (default: 1.0)"
    )
    parser.add_argument(
        "--norm_mode",
        choices=["ct", "minmax", "zscore"],
        default="ct",
        help="Image normalization mode (default: ct - Hounsfield windowing for liver)"
    )
    parser.add_argument(
        "--budget_seed",
        type=int,
        default=42,
        help="Random seed for reproducible label budget selection (default: 42)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("Dataset Discovery & Label Budgeting Demo")
    logger.info("="*70)
    
    # ========== PART 1: Dataset Discovery ==========
    logger.info("\n[1/4] DATASET DISCOVERY")
    logger.info(f"Searching for dataset in: {args.data_root}")
    
    try:
        discovery = DatasetDiscovery(
            args.data_root,
            label_budget=args.label_budget,
            budget_seed=args.budget_seed
        )
    except FileNotFoundError as e:
        logger.error(f"✗ Discovery failed: {e}")
        return 1
    
    logger.info(f"✓ Dataset discovery successful")
    logger.info(f"  - Image directory: {discovery.images_dir.relative_to(discovery.data_root.parent)}")
    logger.info(f"  - Label directory: {discovery.labels_dir.relative_to(discovery.data_root.parent)}")
    logger.info(f"  - Dataset JSON: {discovery.dataset_json}")
    logger.info(f"  - Number of classes: {discovery.num_classes}")
    
    # ========== PART 2: Patient Distribution ==========
    logger.info("\n[2/4] PATIENT DISTRIBUTION")
    
    all_patients = discovery.get_patient_ids()
    labeled_patients = discovery.get_patient_ids(labeled_only=True)
    unlabeled_patients = discovery.get_patient_ids(unlabeled_only=True)
    
    logger.info(f"✓ Total patients: {len(all_patients)}")
    logger.info(f"  - Labeled:   {len(labeled_patients)} ({100*len(labeled_patients)/max(1, len(all_patients)):.1f}%)")
    logger.info(f"  - Unlabeled: {len(unlabeled_patients)} ({100*len(unlabeled_patients)/max(1, len(all_patients)):.1f}%)")
    
    if labeled_patients:
        logger.info(f"  - Labeled patient IDs: {labeled_patients[:3]} {...}")
    if unlabeled_patients:
        logger.info(f"  - Unlabeled patient IDs: {unlabeled_patients[:3]} {...}")
    
    # ========== PART 3: Load Sample Patient ==========
    logger.info("\n[3/4] SAMPLE DATA LOADING")
    
    sample_patient = all_patients[0]
    logger.info(f"Loading sample patient: {sample_patient}")
    
    try:
        img_vol = discovery.load_image(sample_patient)
        label_vol = discovery.load_label(sample_patient)
        logger.info(f"✓ Image loaded: {img_vol.shape}, dtype={img_vol.dtype}")
        logger.info(f"✓ Label loaded: {label_vol.shape}, dtype={label_vol.dtype}")
        
        # Show label classes in this volume
        unique_labels = sorted(set(label_vol.flatten().tolist()))
        logger.info(f"✓ Unique label values: {unique_labels}")
    except Exception as e:
        logger.error(f"✗ Error loading sample: {e}")
        return 1
    
    # ========== PART 4: Image Preprocessing ==========
    logger.info("\n[4/4] IMAGE PREPROCESSING")
    logger.info(f"Normalization mode: {args.norm_mode}")
    
    if args.norm_mode == "ct":
        img_normalized = CTPreprocessor.apply_ct_window(img_vol)
        logger.info(f"✓ Applied CT windowing (center=50 HU, width=400 HU)")
    elif args.norm_mode == "minmax":
        img_normalized = CTPreprocessor.normalize_minmax(img_vol)
        logger.info(f"✓ Applied min-max normalization")
    elif args.norm_mode == "zscore":
        img_normalized = CTPreprocessor.normalize_zscore(img_vol)
        logger.info(f"✓ Applied z-score normalization")
    
    logger.info(f"✓ Normalized image: {img_normalized.shape}, dtype={img_normalized.dtype}")
    logger.info(f"  - Min: {img_normalized.min():.4f}")
    logger.info(f"  - Max: {img_normalized.max():.4f}")
    logger.info(f"  - Mean: {img_normalized.mean():.4f}")
    logger.info(f"  - Std: {img_normalized.std():.4f}")
    
    # Extract 2D slices
    logger.info("\nSlice extraction:")
    slices = SliceExtractor.extract_slices(
        img_normalized, label_vol,
        slice_thickness=1,
        min_slice_coverage=0.0
    )
    logger.info(f"✓ Extracted {len(slices)} 2D slices from 3D volume")
    if slices:
        img_2d, label_2d, z_idx = slices[0]
        logger.info(f"  - Slice 0: image {img_2d.shape} (float32), label {label_2d.shape} (int32)")
    
    # ========== Summary ==========
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"✓ Dataset structure: {'direct children' if discovery.images_dir.parent == discovery.data_root else 'nested'}")
    logger.info(f"✓ Dataset metadata: {'from JSON' if discovery.dataset_json else 'inferred from data'}")
    logger.info(f"✓ Classes detected: {discovery.num_classes}")
    logger.info(f"✓ Label budget applied: {args.label_budget*100:.0f}% labeled, {100-args.label_budget*100:.0f}% unlabeled")
    logger.info(f"✓ Ready for training with {len(labeled_patients)} labeled + {len(unlabeled_patients)} unlabeled patients")
    
    # Optional: Create a SliceDataset
    if args.verbose and len(all_patients) > 0:
        logger.info("\n[BONUS] Creating SliceDataset for first 2 patients...")
        try:
            dataset = SliceDataset(
                all_patients[:min(2, len(all_patients))],
                discovery,
                mode=args.norm_mode,
                track_label_status=(args.label_budget < 1.0)
            )
            logger.info(f"✓ SliceDataset created with {len(dataset)} slices")
            
            # Show label distribution
            dist = dataset.get_label_distribution()
            logger.info(f"✓ Label distribution:")
            for class_id in sorted(dist.keys()):
                logger.info(f"  - Class {class_id}: {dist[class_id]} pixels")
        except Exception as e:
            logger.warning(f"Could not create SliceDataset: {e}")
    
    logger.info("\n✓ Demo completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
