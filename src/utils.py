"""
Utils: config loading, logging, seeding, path utilities.
"""
import os
import sys
import json
import yaml
import random
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch


def set_seed(seed: int):
    """Set seed for reproducibility across numpy, torch, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic(flag: bool = True):
    """Enable/disable deterministic PyTorch behavior."""
    if flag:
        try:
            torch.use_deterministic_algorithms(True, allow_tf32=False)
        except TypeError:
            # Older PyTorch version doesn't have allow_tf32 parameter
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config file.
    
    Args:
        config_path: Path to .yaml config file
        
    Returns:
        Dictionary of config parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge base config with overrides (overrides take precedence).
    
    Args:
        base_config: Base configuration
        overrides: Dictionary of overrides
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    merged.update(overrides)
    return merged


def setup_logging(log_dir: str, log_name: str = "train.log") -> logging.Logger:
    """
    Setup logging to both file and stdout.
    
    Args:
        log_dir: Directory to save logs
        log_name: Name of log file
        
    Returns:
        A logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, log_name))
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def save_config(config: Dict[str, Any], save_dir: str, filename: str = "config.yaml"):
    """
    Save config to YAML for reproducibility.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save config
        filename: Name of config file
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_device() -> str:
    """Get appropriate device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
