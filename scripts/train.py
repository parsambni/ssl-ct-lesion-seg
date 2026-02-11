#!/usr/bin/env python3
"""Train supervised or semi-supervised segmentation models."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split

from src.utils import set_seed, set_deterministic, load_config, merge_configs, setup_logging, save_config, get_device, count_parameters
from src.data import DatasetDiscovery
from src.dataset import SliceDataset
from src.models import UNet2D
from src.losses import DiceLoss
from src.metrics import SegmentationMetrics
from src.transforms import WeakAugmentation, StrongAugmentation
from src.ssl import MeanTeacherLoss
from src.train_engine import TrainEngine, MetricsLogger


def main():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--config", default="configs/supervised.yaml")
    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled_ratio", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    cfg = merge_configs(load_config("configs/base.yaml"), load_config(args.config))
    if args.seed:
        cfg["seed"] = args.seed
    if args.data_root:
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
    save_config(cfg, run_dir / "config.yaml")
    logger.info(f"Device: {device}, Seed: {cfg['seed']}")
    
    discovery = DatasetDiscovery(data_root=cfg["data_root"], seed=cfg["seed"])
    labeled_patients = discovery.get_patients(label_budget=min(args.labeled_ratio, 1.0))
    labeled_patients = labeled_patients[:2] if cfg.get("smoke_mode") else labeled_patients
    logger.info(f"Found {len(labeled_patients)} labeled patients, {discovery.num_classes} classes")
    
    slices = SliceDataset(
        image_dir=discovery.image_dir, label_dir=discovery.label_dir,
        labeled_patients=labeled_patients,
        weak_transforms=WeakAugmentation().get_transforms(),
        strong_transforms=StrongAugmentation().get_transforms(),
        normalization_mode=cfg.get("normalization_mode", "ct_windowing"),
        track_label_status=(args.labeled_ratio < 1.0)
    )
    slices.data_list = slices.data_list[:cfg.get("smoke_samples", 2)] if cfg.get("smoke_mode") else slices.data_list
    logger.info(f"Loaded {len(slices)} slices")
    
    train_size = int(cfg.get("train_split", 0.8) * len(slices))
    train_data, val_data = random_split(slices, [train_size, len(slices) - train_size], 
                                       generator=torch.Generator().manual_seed(cfg["seed"]))
    bs = cfg.get("batch_size", 8)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=0)
    logger.info(f"Train: {len(train_loader)}, Val: {len(val_loader)} batches")
    
    model = UNet2D(in_channels=cfg.get("in_channels", 1), num_classes=discovery.num_classes,
                  init_filters=cfg.get("init_filters", 32), dropout=cfg.get("dropout", 0.1)).to(device)
    logger.info(f"Model: {count_parameters(model):,} parameters")
    
    dice_loss = DiceLoss()
    metrics = SegmentationMetrics(num_classes=discovery.num_classes)
    trainer = TrainEngine(model, cfg["optimizer"], cfg.get("scheduler", "cosine"), device)
    metrics_logger = MetricsLogger(output_dir=run_dir)
    n_labeled = sum(1 for s in slices.data_list if s.get("is_labeled", True))
    
    logger.info("=" * 60)
    if cfg.get("ssl_mode") == "meanteacher":
        logger.info(f"Semi-supervised: {n_labeled}/{len(slices)} labeled")
        ema_model = UNet2D(cfg.get("in_channels", 1), discovery.num_classes, 
                         cfg.get("init_filters", 32), cfg.get("dropout", 0.1)).to(device)
        mt_loss = MeanTeacherLoss(consistency_weight=cfg.get("consistency_weight", 1.0), device=device)
        for epoch in range(cfg["num_epochs"]):
            train_m = trainer.train_ssl(train_loader, ema_model, mt_loss, dice_loss, epoch, cfg["num_epochs"], discovery.num_classes, device)
            val_m = trainer.evaluate(val_loader, metrics, epoch, cfg["num_epochs"], discovery.num_classes, device)
            metrics_logger.log_epoch(epoch, train_m, val_m)
            for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(cfg.get("ema_decay", 0.999)).add_(p.data, alpha=1-cfg.get("ema_decay", 0.999))
            if (epoch+1) % cfg.get("save_interval", 10) == 0:
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'ema_model': ema_model.state_dict()}, 
                          run_dir/f"ckpt_epoch{epoch+1:03d}.pt")
    else:
        logger.info(f"Supervised: all {len(slices)} samples labeled")
        for epoch in range(cfg["num_epochs"]):
            train_m = trainer.train_supervised(train_loader, dice_loss, epoch, cfg["num_epochs"], device)
            val_m = trainer.evaluate(val_loader, metrics, epoch, cfg["num_epochs"], discovery.num_classes, device)
            metrics_logger.log_epoch(epoch, train_m, val_m)
            if (epoch+1) % cfg.get("save_interval", 10) == 0:
                torch.save({'epoch': epoch, 'model': model.state_dict()}, run_dir/f"ckpt_epoch{epoch+1:03d}.pt")
    
    logger.info("=" * 60)
    save_config(cfg, run_dir/"config_final.yaml")


if __name__ == "__main__":
    main()
