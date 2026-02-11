#!/usr/bin/env python3
"""
eval.py: Evaluation on test set with per-lesion size stratification.
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.utils import setup_logging, get_device
from src.data import DatasetDiscovery
from src.dataset import SliceDataset
from src.models import UNet2D
from src.metrics import SegmentationMetrics


def evaluate_model(checkpoint_path: str, data_root: str, test_ids: list, device: str):
    """
    Evaluate model on test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_root: Data root directory
        test_ids: List of test patient IDs
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    logger = logging.getLogger(__name__)
    
    # Load model
    model = UNet2D(in_channels=1, out_channels=2, base_channels=32)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Load dataset
    discovery = DatasetDiscovery(data_root)
    test_dataset = SliceDataset(
        test_ids, discovery,
        mode="ct",
        slice_thickness=1,
        transform=None
    )
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Evaluate
    all_dice = []
    all_jaccard = []
    all_sensitivity = []
    all_specificity = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(preds.shape[0]):
                dice = SegmentationMetrics.dice_score(preds[i], labels_np[i], class_id=1)
                jaccard = SegmentationMetrics.jaccard_score(preds[i], labels_np[i], class_id=1)
                sens = SegmentationMetrics.sensitivity(preds[i], labels_np[i], class_id=1)
                spec = SegmentationMetrics.specificity(preds[i], labels_np[i], class_id=1)
                
                all_dice.append(dice)
                all_jaccard.append(jaccard)
                all_sensitivity.append(sens)
                all_specificity.append(spec)
    
    metrics = {
        "dice_mean": np.mean(all_dice),
        "dice_std": np.std(all_dice),
        "jaccard_mean": np.mean(all_jaccard),
        "jaccard_std": np.std(all_jaccard),
        "sensitivity_mean": np.mean(all_sensitivity),
        "specificity_mean": np.mean(all_specificity),
    }
    
    logger.info(f"Test Dice: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--data_root", type=str, default="dataset/Task08_HepaticVessel")
    parser.add_argument("--splits_file", type=str, default="splits/splits.json")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    device = get_device()
    
    # Load splits
    with open(args.splits_file) as f:
        splits = json.load(f)
    
    # Evaluate
    metrics = evaluate_model(args.checkpoint, args.data_root, splits["test"], device)
    
    # Save
    output_file = Path(args.output_dir) / "eval_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {output_file}")


if __name__ == "__main__":
    main()
