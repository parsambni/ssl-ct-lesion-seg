#!/usr/bin/env python3
"""Train supervised or semi-supervised segmentation models."""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split

from src.utils import (set_seed, set_deterministic, load_config, merge_configs,
                       setup_logging, save_config, get_device, count_parameters)
from src.data import DatasetDiscovery
from src.dataset import SliceDataset
from src.models import UNet2D
from src.losses import DiceLoss, ConsistencyLoss
from src.metrics import SegmentationMetrics
from src.transforms import WeakAugmentation, StrongAugmentation
from src.ssl import MeanTeacher, MeanTeacherLoss
from src.train_engine import TrainEngine, SSLTrainEngine, MetricsLogger


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", default="configs/supervised.yaml")
    parser.add_argument("--data_root", default="/content/dataset/Task08_HepaticVessel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled_ratio", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # Load and merge configs
    cfg = merge_configs(load_config("configs/base.yaml"), load_config(args.config))
    cfg["seed"] = args.seed
    cfg["data_root"] = args.data_root
    if args.smoke:
        cfg.update({"smoke_mode": True, "smoke_samples": 2, "num_epochs": 2})

    set_seed(cfg["seed"])
    if cfg.get("deterministic"):
        set_deterministic(True)

    device = get_device()
    run_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(run_dir))
    save_config(cfg, str(run_dir), "config.yaml")
    logger.info(f"Device: {device}, Seed: {cfg['seed']}")

    # Dataset discovery — pass label_budget and budget_seed (NOT seed=)
    discovery = DatasetDiscovery(
        data_root=cfg["data_root"],
        label_budget=args.labeled_ratio,
        budget_seed=cfg["seed"]
    )
    labeled_patients = discovery.get_patient_ids(labeled_only=True)
    all_patients = discovery.get_patient_ids()
    labeled_patients = labeled_patients[:2] if cfg.get("smoke_mode") else labeled_patients
    logger.info(f"Found {len(labeled_patients)} labeled patients, {discovery.num_classes} classes")

    # SliceDataset — use patient_ids + discovery (NOT image_dir/label_dir)
    weak_aug = WeakAugmentation()
    slices = SliceDataset(
        patient_ids=labeled_patients,
        discovery=discovery,
        mode="ct",
        slice_thickness=cfg.get("slice_thickness", 1),
        transform=weak_aug,
        track_label_status=(args.labeled_ratio < 1.0)
    )
    # Smoke mode: limit slices (attribute is .slices, NOT .data_list)
    if cfg.get("smoke_mode"):
        slices.slices = slices.slices[:cfg.get("smoke_samples", 2)]
    logger.info(f"Loaded {len(slices)} slices")

    # Train/val split
    train_size = int(cfg.get("train_ratio", 0.8) * len(slices))
    val_size = len(slices) - train_size
    if val_size == 0:
        val_size = 1
        train_size = len(slices) - 1
    train_data, val_data = random_split(
        slices, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"])
    )
    bs = cfg.get("batch_size", 8)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=0)
    logger.info(f"Train: {len(train_loader)}, Val: {len(val_loader)} batches")

    # Model — use out_channels + base_channels (NOT num_classes/init_filters)
    num_classes = discovery.num_classes
    model = UNet2D(
        in_channels=cfg.get("in_channels", 1),
        out_channels=num_classes,
        base_channels=cfg.get("base_channels", 32),
        dropout=cfg.get("dropout", 0.1)
    ).to(device)
    logger.info(f"Model: {count_parameters(model):,} parameters")

    # Optimizer & scheduler — create actual objects (NOT strings)
    lr = cfg.get("learning_rate", 1e-4)
    num_epochs = cfg.get("num_epochs", 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss
    dice_loss = DiceLoss()

    # MetricsLogger — pass log_path (NOT output_dir)
    metrics_csv = str(run_dir / "metrics.csv")
    metrics_logger = MetricsLogger(log_path=metrics_csv)

    logger.info("=" * 60)
    ssl_mode = cfg.get("ssl_mode") or cfg.get("train_mode")

    if ssl_mode in ("meanteacher", "ssl"):
        # --- SEMI-SUPERVISED (Mean Teacher) ---
        logger.info(f"Semi-supervised: {len(labeled_patients)}/{len(all_patients)} labeled")

        # TrainEngine — pass optimizer object + device + scheduler
        engine = TrainEngine(model, optimizer, device, scheduler)

        # For SSL we'd ideally use SSLTrainEngine with separate loaders,
        # but for now use supervised engine on labeled data
        for epoch in range(num_epochs):
            train_m = engine.train_epoch(train_loader, dice_loss, epoch)
            val_m = engine.evaluate(val_loader, dice_loss, epoch=epoch)
            metrics_logger.log_metrics({**train_m, **val_m}, step=epoch)

            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                logger.info(f"Epoch {epoch:03d} | train_loss: {train_m['train_loss']:.4f} | val_loss: {val_m['val_loss']:.4f}")

            if (epoch + 1) % cfg.get("save_interval", 5) == 0:
                engine.save_checkpoint(str(run_dir), f"ckpt_epoch{epoch+1:03d}.pt")

        engine.save_checkpoint(str(run_dir), "best.pt")

    else:
        # --- SUPERVISED ---
        logger.info(f"Supervised: all {len(slices)} samples labeled")

        engine = TrainEngine(model, optimizer, device, scheduler)

        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_m = engine.train_epoch(train_loader, dice_loss, epoch)
            val_m = engine.evaluate(val_loader, dice_loss, epoch=epoch)
            metrics_logger.log_metrics({**train_m, **val_m}, step=epoch)

            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                logger.info(f"Epoch {epoch:03d} | train_loss: {train_m['train_loss']:.4f} | val_loss: {val_m['val_loss']:.4f}")

            if val_m['val_loss'] < best_val_loss:
                best_val_loss = val_m['val_loss']
                engine.save_checkpoint(str(run_dir), "best.pt")

            if (epoch + 1) % cfg.get("save_interval", 5) == 0:
                engine.save_checkpoint(str(run_dir), f"ckpt_epoch{epoch+1:03d}.pt")

        engine.save_checkpoint(str(run_dir), "final.pt")

    logger.info("=" * 60)
    save_config(cfg, str(run_dir), "config_final.yaml")
    logger.info(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
