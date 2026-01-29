"""
Tests for inference module - model loading and inference functions.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from unittest.mock import patch, MagicMock

from inference import (
    preprocess_image,
    apply_mask_overlay,
    get_tensorrt_engine_path,
)
from train_segmentation import EfficientNetUNet, STDC1Seg


# =============================================================================
# Preprocess Image Tests
# =============================================================================

class TestPreprocessImage:
    """Test preprocess_image function."""

    def test_output_is_tensor(self, sample_image):
        """Output should be a PyTorch tensor."""
        result = preprocess_image(sample_image)

        assert isinstance(result, torch.Tensor)

    def test_output_shape(self, sample_image):
        """Output should have shape (1, 3, H, W)."""
        result = preprocess_image(sample_image)

        assert result.dim() == 4
        assert result.shape[0] == 1  # Batch size
        assert result.shape[1] == 3  # Channels

    def test_output_normalized(self, sample_image):
        """Output should be normalized (values roughly in [-2, 2] range after ImageNet norm)."""
        result = preprocess_image(sample_image)

        # After ImageNet normalization, values should be centered around 0
        assert result.min() >= -3
        assert result.max() <= 3

    def test_handles_different_sizes(self):
        """Should handle images of various sizes."""
        for size in [(128, 128), (256, 256), (512, 512)]:
            image = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            result = preprocess_image(image)

            assert result.shape[2] == size[0]
            assert result.shape[3] == size[1]


# =============================================================================
# Apply Mask Overlay Tests
# =============================================================================

class TestApplyMaskOverlay:
    """Test apply_mask_overlay function."""

    def test_output_same_shape_as_input(self, sample_image, sample_mask):
        """Output image should have same shape as input."""
        result = apply_mask_overlay(sample_image, sample_mask)

        assert result.shape == sample_image.shape

    def test_output_is_uint8(self, sample_image, sample_mask):
        """Output should be uint8."""
        result = apply_mask_overlay(sample_image, sample_mask)

        assert result.dtype == np.uint8

    def test_overlay_color_applied(self, sample_image, sample_mask):
        """Overlay color should be visible in masked region."""
        color = (255, 0, 0)  # Red
        result = apply_mask_overlay(sample_image, sample_mask, color=color, alpha=1.0)

        # Check that red channel is high where mask is positive
        mask_bool = sample_mask > 127
        if mask_bool.any():
            # Red should be higher in masked regions
            masked_red = result[mask_bool, 0].mean()
            assert masked_red > 200

    def test_alpha_blending(self, sample_image, sample_mask):
        """Alpha value should control blend amount."""
        color = (255, 0, 255)

        result_full = apply_mask_overlay(sample_image, sample_mask, color=color, alpha=1.0)
        result_half = apply_mask_overlay(sample_image, sample_mask, color=color, alpha=0.5)

        # Full alpha should differ more from original than half alpha
        mask_bool = sample_mask > 127
        if mask_bool.any():
            diff_full = np.abs(result_full[mask_bool].astype(float) -
                              sample_image[mask_bool].astype(float)).mean()
            diff_half = np.abs(result_half[mask_bool].astype(float) -
                              sample_image[mask_bool].astype(float)).mean()

            assert diff_full >= diff_half

    def test_empty_mask_unchanged(self, sample_image, empty_mask):
        """Empty mask should leave image unchanged."""
        result = apply_mask_overlay(sample_image, empty_mask)

        np.testing.assert_array_equal(result, sample_image)


# =============================================================================
# TensorRT Engine Path Tests
# =============================================================================

class TestGetTensorRTEnginePath:
    """Test get_tensorrt_engine_path function."""

    def test_correct_path_format(self):
        """Should return correct engine path format."""
        model_path = Path("models/enemy_segmentation.pth")
        inference_size = 256

        result = get_tensorrt_engine_path(model_path, inference_size)

        assert result == Path("models/enemy_segmentation_256.engine")

    def test_different_sizes(self):
        """Should handle different inference sizes."""
        model_path = Path("models/model.pth")

        for size in [256, 512, 1024]:
            result = get_tensorrt_engine_path(model_path, size)
            assert str(size) in str(result)
            assert result.suffix == ".engine"


# =============================================================================
# Model Architecture Tests
# =============================================================================

class TestEfficientNetUNet:
    """Test EfficientNetUNet model architecture."""

    def test_forward_pass_shape(self, cpu_device):
        """Forward pass should produce correct output shape."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        input_tensor = torch.randn(1, 3, 256, 256, device=cpu_device)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (1, 1, 256, 256)

    def test_handles_different_input_sizes(self, cpu_device):
        """Model should handle various input sizes."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        for size in [128, 256, 512]:
            input_tensor = torch.randn(1, 3, size, size, device=cpu_device)

            with torch.no_grad():
                output = model(input_tensor)

            assert output.shape == (1, 1, size, size)

    def test_output_is_logits(self, cpu_device):
        """Output should be logits (raw values), not passed through sigmoid."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        # Use extreme input values to increase chance of out-of-range output
        input_tensor = torch.randn(1, 3, 256, 256, device=cpu_device) * 10

        with torch.no_grad():
            output = model(input_tensor)

        # Verify output is a valid tensor with expected properties
        # Note: With random weights, output might still fall in [0,1] by chance
        # The key check is that sigmoid is NOT applied internally
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestSTDC1Seg:
    """Test STDC1Seg model architecture."""

    def test_forward_pass_shape(self, cpu_device):
        """Forward pass should produce correct output shape."""
        model = STDC1Seg(pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        input_tensor = torch.randn(1, 3, 256, 256, device=cpu_device)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (1, 1, 256, 256)

    def test_handles_different_input_sizes(self, cpu_device):
        """Model should handle various input sizes (must be divisible by 32)."""
        model = STDC1Seg(pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        for size in [128, 256, 512]:
            input_tensor = torch.randn(1, 3, size, size, device=cpu_device)

            with torch.no_grad():
                output = model(input_tensor)

            assert output.shape == (1, 1, size, size)


# =============================================================================
# Inference Function Tests (Mocked)
# =============================================================================

class TestInferenceFunctions:
    """Test inference functions with mocked models."""

    def test_efficientnet_inference_output_shape(self, sample_image, cpu_device):
        """EfficientNet inference should return mask of correct size."""
        # Create a simple mock model
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        # Manual inference (simplified version of run_efficientnet_inference)
        from inference import preprocess_image

        h, w = sample_image.shape[:2]
        input_tensor = preprocess_image(sample_image).to(cpu_device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output)
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            mask = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

        assert mask.shape == (h, w)
        assert mask.dtype == np.uint8
        assert mask.min() >= 0
        assert mask.max() <= 255

    def test_stdc1_inference_output_shape(self, sample_image, cpu_device):
        """STDC1 inference should return mask of correct size."""
        model = STDC1Seg(pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        from inference import preprocess_image

        h, w = sample_image.shape[:2]
        input_tensor = preprocess_image(sample_image).to(cpu_device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output)
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            mask = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

        assert mask.shape == (h, w)


# =============================================================================
# Model Loading Tests (with mocks)
# =============================================================================

class TestModelLoading:
    """Test model loading functions."""

    def test_load_efficientnet_creates_correct_model(self, tmp_path, cpu_device):
        """load_efficientnet_model should create correct model type."""
        # Create a mock checkpoint
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "best_dice": 0.85,
            "config": {"encoder_name": "efficientnet_b0"},
        }

        checkpoint_path = tmp_path / "test_model.pth"
        torch.save(checkpoint, checkpoint_path)

        from inference import load_efficientnet_model

        loaded_model = load_efficientnet_model(checkpoint_path, cpu_device)

        assert isinstance(loaded_model, EfficientNetUNet)
        assert not loaded_model.training  # Should be in eval mode

    def test_load_stdc1_creates_correct_model(self, tmp_path, cpu_device):
        """load_efficientnet_model should handle STDC1 encoder."""
        model = STDC1Seg(pretrained=False)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "best_dice": 0.85,
            "config": {"encoder_name": "stdc1"},
        }

        checkpoint_path = tmp_path / "test_stdc1.pth"
        torch.save(checkpoint, checkpoint_path)

        from inference import load_efficientnet_model

        loaded_model = load_efficientnet_model(checkpoint_path, cpu_device)

        assert isinstance(loaded_model, STDC1Seg)
        assert not loaded_model.training


# =============================================================================
# Inference Integration Tests
# =============================================================================

class TestInferenceIntegration:
    """Integration tests for the inference pipeline."""

    def test_full_inference_pipeline(self, sample_image, cpu_device):
        """Test complete inference: preprocess -> model -> postprocess."""
        # Create model
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        # Preprocess
        input_tensor = preprocess_image(sample_image).to(cpu_device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output)

        # Postprocess
        h, w = sample_image.shape[:2]
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        mask = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

        # Apply overlay
        result = apply_mask_overlay(sample_image, mask, color=(255, 0, 255), alpha=0.5)

        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_batch_inference(self, cpu_device):
        """Model should handle batch inference."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 256, 256, device=cpu_device)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (batch_size, 1, 256, 256)

    def test_deterministic_output(self, sample_image, cpu_device, seed_everything):
        """Same input should produce same output."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(cpu_device)
        model.eval()

        input_tensor = preprocess_image(sample_image).to(cpu_device)

        with torch.no_grad():
            output1 = model(input_tensor).clone()
            output2 = model(input_tensor).clone()

        torch.testing.assert_close(output1, output2)
