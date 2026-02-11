"""
make_splits.py: Create patient-level train/val/test splits.
"""
import os
import json
import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import DatasetDiscovery
from src.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Create patient-level splits")
    parser.add_argument("--data_root", type=str, default="dataset", help="Root data directory")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="splits")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    logger.info(f"Creating splits: data_root={args.data_root}")
    
    # Discover dataset
    discovery = DatasetDiscovery(args.data_root)
    patient_ids = discovery.get_patient_ids()
    
    # Split: train+val / test first
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    train_val, test = train_test_split(
        patient_ids,
        test_size=test_ratio,
        random_state=args.seed
    )
    
    # Split train+val into train / val
    val_size_of_trainval = args.val_ratio / (args.train_ratio + args.val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_size_of_trainval,
        random_state=args.seed
    )
    
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Save splits
    splits = {
        "train": train,
        "val": val,
        "test": test,
        "seed": args.seed,
        "num_classes": discovery.num_classes
    }
    
    output_file = os.path.join(args.output_dir, "splits.json")
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    logger.info(f"Splits saved to {output_file}")


if __name__ == "__main__":
    main()
