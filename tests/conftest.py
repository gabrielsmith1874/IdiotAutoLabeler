"""
Shared fixtures for the test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for deterministic tests."""
    return torch.device("cpu")


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a sample RGB image (256x256)."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_512():
    """Create a sample RGB image (512x512)."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask (256x256)."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Add a circular region as "detected" area
    center_y, center_x = 128, 128
    y, x = np.ogrid[:256, :256]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask[dist < 40] = 255
    return mask


@pytest.fixture
def empty_mask():
    """Create an empty mask (no detections)."""
    return np.zeros((256, 256), dtype=np.uint8)


@pytest.fixture
def full_mask():
    """Create a fully positive mask."""
    return np.ones((256, 256), dtype=np.uint8) * 255


@pytest.fixture
def masked_unmasked_pair():
    """Create a pair of masked/unmasked images for testing mask extraction."""
    # Base image
    unmasked = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)

    # Create masked version with magenta overlay in a region
    masked = unmasked.copy()
    # Add magenta (255, 0, 255) overlay in center region
    masked[100:150, 100:150] = [255, 0, 255]

    return unmasked, masked


# =============================================================================
# Tensor Fixtures
# =============================================================================

@pytest.fixture
def sample_pred_tensor():
    """Create a sample prediction tensor (logits, before sigmoid)."""
    # Shape: (1, 1, 256, 256)
    # Values around 0 (uncertain), some positive (confident positive), some negative
    pred = torch.randn(1, 1, 256, 256)
    # Make center region confidently positive (high logit values)
    pred[:, :, 100:150, 100:150] = 3.0  # sigmoid(3) ≈ 0.95
    return pred


@pytest.fixture
def sample_target_tensor():
    """Create a sample target/ground truth tensor."""
    target = torch.zeros(1, 1, 256, 256)
    # Ground truth positive region in center
    target[:, :, 100:150, 100:150] = 1.0
    return target


@pytest.fixture
def perfect_prediction():
    """Create a prediction that perfectly matches the target."""
    target = torch.zeros(1, 1, 256, 256)
    target[:, :, 100:150, 100:150] = 1.0

    # Prediction with very high confidence where target is 1
    pred = torch.full((1, 1, 256, 256), -5.0)  # sigmoid(-5) ≈ 0
    pred[:, :, 100:150, 100:150] = 5.0  # sigmoid(5) ≈ 1

    return pred, target


@pytest.fixture
def all_negative_prediction():
    """Create a prediction that predicts all negative."""
    target = torch.zeros(1, 1, 256, 256)
    target[:, :, 100:150, 100:150] = 1.0

    pred = torch.full((1, 1, 256, 256), -5.0)  # All negative predictions

    return pred, target


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def simple_conv_model():
    """Create a simple conv model for testing (fast, no pretrained weights)."""
    import torch.nn as nn

    class SimpleSegModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            return x

    return SimpleSegModel()


# =============================================================================
# Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir()

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()

    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    return tmp_path


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def seed_everything():
    """Seed all random number generators for reproducibility."""
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
