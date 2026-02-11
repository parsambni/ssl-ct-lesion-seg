"""
test_models.py: Tests for U-Net model.
"""
import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import UNet2D


def test_unet_initialization():
    """Test U-Net initialization."""
    model = UNet2D(in_channels=1, out_channels=2, base_channels=32)
    assert model is not None


def test_unet_forward():
    """Test U-Net forward pass."""
    model = UNet2D(in_channels=1, out_channels=2, base_channels=32)
    
    # Create dummy input
    x = torch.randn(2, 1, 256, 256)  # (B, C, H, W)
    
    # Forward pass
    logits = model(x)
    
    # Check output shape
    assert logits.shape == (2, 2, 256, 256), f"Expected (2, 2, 256, 256), got {logits.shape}"
    assert logits.dtype == torch.float32


def test_unet_parameter_count():
    """Test U-Net has reasonable number of parameters."""
    model = UNet2D(in_channels=1, out_channels=2, base_channels=32)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Should be in the range of 7-10M parameters for typical U-Net
    assert 1e6 < total_params < 50e6, f"Got {total_params} parameters"


def test_unet_backward():
    """Test U-Net backward pass."""
    model = UNet2D(in_channels=1, out_channels=2, base_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    x = torch.randn(2, 1, 256, 256)
    y = torch.randint(0, 2, (2, 256, 256))
    
    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    
    loss.backward()
    optimizer.step()
    
    # Check that gradients were computed
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
