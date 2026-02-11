"""
Train engine: training and validation loops.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Callable
import csv
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class TrainEngine:
    """Training engine for supervised and semi-supervised models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        scheduler: Optional[object] = None
    ):
        """
        Args:
            model: Neural network model
            optimizer: Optimizer
            device: "cpu" or "cuda"
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        
        self.train_losses = []
        self.val_metrics = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        epoch: int
    ) -> Dict[str, float]:
        """
        Perform one training epoch.
        
        Args:
            train_loader: DataLoader for training data
            loss_fn: Loss function
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["image"].to(self.device)  # (B, C, H, W)
            labels = batch["label"].to(self.device)  # (B, H, W)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)  # (B, C, H, W)
            
            # Loss
            loss = loss_fn(logits, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(1, num_batches)
        self.train_losses.append(avg_loss)
        
        # LR scheduling
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step()
        
        return {"train_loss": avg_loss}
    
    def evaluate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
        metric_fn: Optional[Callable] = None,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            loss_fn: Loss function
            metric_fn: Function to compute additional metrics
            epoch: Current epoch number
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_metrics = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False)
            
            for batch in pbar:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                logits = self.model(images)
                loss = loss_fn(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Compute metrics if provided
                if metric_fn is not None:
                    batch_metrics = metric_fn(logits, labels)
                    all_metrics.append(batch_metrics)
        
        avg_loss = total_loss / max(1, num_batches)
        
        metrics = {"val_loss": avg_loss}
        
        # Average metrics across batches
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                metrics[f"val_{key}"] = sum(values) / len(values)
        
        self.val_metrics.append(metrics)
        
        return metrics
    
    def save_checkpoint(self, save_dir: str, name: str = "checkpoint.pt"):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, name)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, filepath)
        
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Checkpoint loaded from {filepath}")


class SSLTrainEngine(TrainEngine):
    """Training engine for semi-supervised learning with Mean Teacher."""
    
    def train_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        loss_fn: Callable,
        epoch: int,
        augmentation_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Perform one SSL training epoch.
        
        Args:
            labeled_loader: DataLoader for labeled data
            unlabeled_loader: DataLoader for unlabeled data
            loss_fn: Loss function (MeanTeacherLoss)
            epoch: Current epoch number
            augmentation_fn: Optional data augmentation function
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        # Update epoch in loss function if it has SSL rampup
        if hasattr(loss_fn, 'set_epoch'):
            loss_fn.set_epoch(epoch)
        
        total_sup_loss = 0.0
        total_cons_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        # Zip labeled and unlabeled loaders
        unlabeled_iter = iter(unlabeled_loader)
        
        pbar = tqdm(labeled_loader, desc=f"Epoch {epoch} SSL Train", leave=False)
        
        for batch_idx, labeled_batch in enumerate(pbar):
            # Get unlabeled batch
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            
            # Move labeled data to device
            labeled_images = labeled_batch["image"].to(self.device)
            labeled_labels = labeled_batch["label"].to(self.device)
            
            # Move unlabeled data to device
            unlabeled_images = unlabeled_batch["image"].to(self.device)
            
            # Forward pass (both labeled and unlabeled together)
            all_images = torch.cat([labeled_images, unlabeled_images], dim=0)
            self.optimizer.zero_grad()
            
            # This assumes the model contains Mean Teacher internally
            # In practice, you'd implement SSL-specific forward logic here
            all_logits = self.model(all_images)
            
            # Split back
            labeled_logits = all_logits[:labeled_images.shape[0]]
            unlabeled_logits = all_logits[labeled_images.shape[0]:]
            
            # For now, just use supervised loss (would extend with consistency loss)
            loss = loss_fn(labeled_logits, labeled_labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / max(1, num_batches)
        self.train_losses.append(avg_loss)
        
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                self.scheduler.step()
        
        return {
            "train_loss": avg_loss,
            "sup_loss": total_sup_loss / max(1, num_batches),
            "cons_loss": total_cons_loss / max(1, num_batches)
        }


class MetricsLogger:
    """Log metrics to CSV file."""
    
    def __init__(self, log_path: str):
        """
        Args:
            log_path: Path to CSV file to log metrics
        """
        self.log_path = log_path
        self.fieldnames = None
    
    def log_metrics(self, metrics: Dict, step: int):
        """
        Log metrics to CSV.
        
        Args:
            metrics: Dictionary of metrics
            step: Step/epoch number
        """
        # Determine fieldnames from first call
        if self.fieldnames is None:
            self.fieldnames = ["step"] + sorted(metrics.keys())
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        
        # Append row
        row = {"step": step}
        row.update(metrics)
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
