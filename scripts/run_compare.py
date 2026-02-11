#!/usr/bin/env python3
"""
run_compare.py: Run and compare training across different configs and seeds.
"""
import os
import sys
import argparse
import subprocess
import json
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging


def run_experiment(config: str, labeled_ratio: float, seed: int, output_base: str, data_root: str) -> dict:
    """
    Run a single training experiment.
    
    Args:
        config: Config file path
        labeled_ratio: Labeled data ratio (for SSL)
        seed: Random seed
        output_base: Base output directory
        data_root: Data root directory
        
    Returns:
        Dictionary with results
    """
    experiment_name = f"{Path(config).stem}_lr{labeled_ratio:.2f}_s{seed}"
    output_dir = Path(output_base) / experiment_name
    
    cmd = [
        "python", "scripts/train.py",
        "--config", config,
        "--data_root", data_root,
        "--seed", str(seed),
        "--labeled_ratio", str(labeled_ratio),
        "--output_dir", str(output_base)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    
    # Load metrics
    metrics_file = output_dir / "metrics.csv"
    results = {
        "config": config,
        "labeled_ratio": labeled_ratio,
        "seed": seed,
        "success": result.returncode == 0,
        "metrics_file": str(metrics_file)
    }
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                results["final_loss"] = float(last_row.get("train_loss", 0))
                results["final_val_loss"] = float(last_row.get("val_loss", 0))
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, nargs='+', default=["configs/supervised.yaml"], help="Config files to test")
    parser.add_argument("--labeled_ratios", type=float, nargs='+', default=[1.0], help="Labeled ratios for SSL")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42], help="Random seeds")
    parser.add_argument("--data_root", type=str, default="dataset/Task08_HepaticVessel")
    parser.add_argument("--output_base", type=str, default="runs", help="Base output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_base, exist_ok=True)
    logger = setup_logging(args.output_base)
    
    # Run experiments
    results = []
    for config in args.configs:
        for labeled_ratio in args.labeled_ratios:
            for seed in args.seeds:
                result = run_experiment(config, labeled_ratio, seed, args.output_base, args.data_root)
                results.append(result)
    
    # Save results summary
    summary_file = Path(args.output_base) / "run_compare_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Summary saved to {summary_file}")
    
    # Print summary
    logger.info("\nComparison Results:")
    logger.info("-" * 80)
    for r in results:
        status = "✓" if r["success"] else "✗"
        logger.info(f"{status} {r['config']} lr={r['labeled_ratio']} seed={r['seed']}")
        if "final_val_loss" in r:
            logger.info(f"   Val Loss: {r['final_val_loss']:.6f}")


if __name__ == "__main__":
    main()
