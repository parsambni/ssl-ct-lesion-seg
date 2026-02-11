#!/usr/bin/env python3
"""
sanity_viz.py: Save overlay images for quick sanity checking.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from src.utils import setup_logging, get_device
from src.data import DatasetDiscovery
from src.dataset import SliceDataset
from src.models import UNet2D


def save_overlay_image(image: np.ndarray, pred: np.ndarray, label: np.ndarray, output_path: str):
    """
    Save overlay image (if cv2 available, use colored overlay; otherwise save separately).
    
    Args:
        image: Input image (H, W) in [0, 1]
        pred: Predicted segmentation (H, W)
        label: Ground truth segmentation (H, W)
        output_path: Path to save image
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Normalize image to 0-255
    img_vis = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    if HAS_CV2:
        # Create colored overlay
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
        
        # Green for correct predictions, red for false positives, blue for false negatives
        pred_binary = pred > 0
        label_binary = label > 0
        
        tp = pred_binary & label_binary
        fp = pred_binary & ~label_binary
        fn = ~pred_binary & label_binary
        
        img_rgb[tp] = [0, 255, 0]  # Green: TP
        img_rgb[fp] = [0, 0, 255]  # Red: FP
        img_rgb[fn] = [255, 0, 0]  # Blue: FN
        
        cv2.imwrite(output_path, img_rgb)
    else:
        # Save grayscale if cv2 not available
        np.save(output_path.replace('.png', '.npy'), img_vis)


def visualize_predictions(checkpoint_path: str, data_root: str, test_ids: list, device: str, output_dir: str, num_samples: int = 10):
    """
    Visualize predictions on test set.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_root: Data root directory
        test_ids: List of test patient IDs
        device: Device to use
        output_dir: Output directory for images
        num_samples: Number of slices to visualize
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
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Visualize
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            if count >= num_samples:
                break
            
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images)
            preds = logits.argmax(dim=1)
            
            # Denormalize image
            img_np = images[0, 0].cpu().numpy()
            img_np = np.clip((img_np + 1) / 2, 0, 1)  # Rough denormalization
            
            pred_np = preds[0].cpu().numpy()
            label_np = labels[0].cpu().numpy()
            
            patient_id = batch["patient_id"][0]
            z_idx = batch["z_index"][0].item()
            
            output_path = Path(output_dir) / f"{patient_id}_z{z_idx:03d}_overlay.png"
            save_overlay_image(img_np, pred_np, label_np, str(output_path))
            
            logger.info(f"Saved {output_path}")
            count += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--data_root", type=str, default="dataset/Task08_HepaticVessel")
    parser.add_argument("--splits_file", type=str, default="splits/splits.json")
    parser.add_argument("--output_dir", type=str, default="runs/viz", help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    
    device = get_device()
    
    # Load splits
    import json
    with open(args.splits_file) as f:
        splits = json.load(f)
    
    # Visualize
    logger.info(f"Saving visualization to {args.output_dir}")
    visualize_predictions(args.checkpoint, args.data_root, splits["test"], device, args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()
