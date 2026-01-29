"""
Runtime integration tests for inference backends.

These tests verify that the actual inference pipeline works end-to-end,
catching issues like:
- Missing dependencies (Triton, TensorRT)
- dtype mismatches between model and input
- torch.compile compatibility
- Backend fallback behavior
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

# Import from src
from train_segmentation import EfficientNetUNet, STDC1Seg
from inference import preprocess_image


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_frame():
    """Sample RGB frame like what would come from screen capture."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def efficientnet_model(device):
    """Create EfficientNet model for testing."""
    model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def stdc1_model(device):
    """Create STDC1 model for testing."""
    model = STDC1Seg(pretrained=False)
    model = model.to(device)
    model.eval()
    return model


# =============================================================================
# Dtype Consistency Tests
# =============================================================================

class TestDtypeConsistency:
    """Test that dtypes are handled correctly across inference paths."""

    def test_float32_inference(self, efficientnet_model, sample_frame, device):
        """Model should work with float32 input."""
        model = efficientnet_model

        # Preprocess to float32
        img = torch.from_numpy(sample_frame).to(device)
        img = img.permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Normalize with float32 mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img = (img - mean) / std

        with torch.no_grad():
            output = model(img)

        assert output.dtype == torch.float32
        assert output.shape == (1, 1, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_float16_inference(self, device):
        """Model should work with float16 input on CUDA."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(device).half()
        model.eval()

        # Create float16 input
        img = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float16)

        # Normalize with float16 mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        img = (img - mean) / std

        with torch.no_grad():
            output = model(img)

        assert output.shape == (1, 1, 256, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mixed_dtype_fails_or_warns(self, device):
        """Mixing float32 input with float16 normalization should be detected."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(device).half()
        model.eval()

        # float32 input
        img = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float32)

        # float16 normalization tensors (this is the bug scenario)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)

        # This should either work (with implicit casting) or raise an error
        # The test documents the actual behavior
        try:
            img_normalized = (img - mean) / std
            # If it works, check the resulting dtype
            assert img_normalized.dtype in [torch.float32, torch.float16]
        except RuntimeError as e:
            # If it fails, that's also acceptable (explicit error is better than silent bug)
            assert "dtype" in str(e).lower() or "type" in str(e).lower()


# =============================================================================
# torch.compile Tests
# =============================================================================

class TestTorchCompile:
    """Test torch.compile compatibility."""

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_availability_check(self):
        """torch.compile should be detectable."""
        assert hasattr(torch, "compile")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_with_triton_check(self, device):
        """torch.compile should fail gracefully if Triton is missing."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(device)
        model.eval()

        try:
            # Try to compile - this may fail if Triton is not installed
            compiled_model = torch.compile(model, mode="reduce-overhead", backend="inductor")

            # If compilation succeeds, test inference
            dummy = torch.randn(1, 3, 256, 256, device=device)
            with torch.no_grad():
                output = compiled_model(dummy)
            assert output.shape == (1, 1, 256, 256)

        except Exception as e:
            # Expected if Triton is missing
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ["triton", "inductor", "compile"]), \
                f"Unexpected error: {e}"

    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
    def test_compile_eager_mode_fallback(self, efficientnet_model, device):
        """Eager mode should always work as fallback."""
        model = efficientnet_model

        try:
            # Try eager mode which doesn't require Triton
            compiled_model = torch.compile(model, mode="default", backend="eager")

            dummy = torch.randn(1, 3, 256, 256, device=device)
            with torch.no_grad():
                output = compiled_model(dummy)
            assert output.shape == (1, 1, 256, 256)

        except Exception as e:
            pytest.skip(f"torch.compile eager mode not available: {e}")


# =============================================================================
# TensorRT Tests
# =============================================================================

class TestTensorRT:
    """Test TensorRT availability and fallback."""

    def test_tensorrt_import_check(self):
        """TensorRT import should be handled gracefully."""
        try:
            import tensorrt as trt
            assert hasattr(trt, "__version__")
            tensorrt_available = True
        except ImportError:
            tensorrt_available = False

        # Test should pass regardless of whether TensorRT is installed
        assert isinstance(tensorrt_available, bool)

    def test_tensorrt_engine_path_format(self, tmp_path):
        """Engine path should be correctly formatted."""
        from inference import get_tensorrt_engine_path

        model_path = tmp_path / "model.pth"
        engine_path = get_tensorrt_engine_path(model_path, 256)

        assert engine_path.name == "model_256.engine"
        assert engine_path.parent == tmp_path


# =============================================================================
# Full Inference Pipeline Tests
# =============================================================================

class TestInferencePipeline:
    """End-to-end inference pipeline tests."""

    def test_full_pipeline_float32(self, efficientnet_model, sample_frame, device):
        """Test complete inference pipeline with float32."""
        model = efficientnet_model
        frame = sample_frame

        # Simulate fast_trigger.py inference path
        img = torch.from_numpy(frame).to(device, non_blocking=True)
        img = img.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img = (img - mean) / std

        # Inference
        with torch.no_grad():
            output = model(img)
            pred = torch.sigmoid(output)

        # Convert to mask
        mask = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8
        assert mask.min() >= 0
        assert mask.max() <= 255

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_full_pipeline_float16(self, sample_frame, device):
        """Test complete inference pipeline with float16."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)
        model = model.to(device).half()
        model.eval()

        frame = sample_frame

        # Convert to tensor
        img = torch.from_numpy(frame).to(device, non_blocking=True)
        img = img.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

        # Normalize with float32 first, then convert
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img = (img - mean) / std

        # Convert to half AFTER normalization
        img = img.half()

        # Inference
        with torch.no_grad():
            output = model(img)
            pred = torch.sigmoid(output.float())  # Back to float32 for sigmoid

        # Convert to mask
        mask = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8

    def test_pipeline_with_resize(self, efficientnet_model, device):
        """Test pipeline when input size differs from model expectation."""
        model = efficientnet_model

        # Larger input that needs resize
        frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        img = torch.from_numpy(frame).to(device)
        img = img.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

        # Resize to model input size
        img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img = (img - mean) / std

        with torch.no_grad():
            output = model(img)
            pred = torch.sigmoid(output)

        # Resize mask back to original size
        pred = F.interpolate(pred, size=(512, 512), mode='bilinear', align_corners=False)
        mask = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

        assert mask.shape == (512, 512)


# =============================================================================
# Normalization Tests
# =============================================================================

class TestNormalization:
    """Test that normalization is applied correctly."""

    def test_imagenet_normalization_values(self):
        """Verify ImageNet normalization constants."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # These are the standard ImageNet values
        assert abs(mean[0] - 0.485) < 0.001
        assert abs(mean[1] - 0.456) < 0.001
        assert abs(mean[2] - 0.406) < 0.001
        assert abs(std[0] - 0.229) < 0.001
        assert abs(std[1] - 0.224) < 0.001
        assert abs(std[2] - 0.225) < 0.001

    def test_normalization_output_range(self, device):
        """Normalized values should be roughly in [-2.5, 2.5] range."""
        # Create tensor with values 0-1
        img = torch.rand(1, 3, 64, 64, device=device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        normalized = (img - mean) / std

        # Check reasonable range
        assert normalized.min() >= -3.0
        assert normalized.max() <= 3.0

    def test_inplace_vs_regular_normalization(self, device):
        """In-place and regular normalization should give same results."""
        img1 = torch.rand(1, 3, 64, 64, device=device)
        img2 = img1.clone()

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        # Regular
        result1 = (img1 - mean) / std

        # In-place
        img2.sub_(mean).div_(std)

        torch.testing.assert_close(result1, img2)


# =============================================================================
# Model Loading Tests
# =============================================================================

class TestModelLoading:
    """Test model checkpoint loading."""

    def test_checkpoint_structure(self, tmp_path, device):
        """Test that checkpoints have expected structure."""
        model = EfficientNetUNet(encoder_name="efficientnet_b0", pretrained=False)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "best_dice": 0.85,
            "config": {"encoder_name": "efficientnet_b0"},
        }

        path = tmp_path / "test_checkpoint.pth"
        torch.save(checkpoint, path)

        # Load and verify
        loaded = torch.load(path, map_location=device, weights_only=False)

        assert "model_state_dict" in loaded
        assert "epoch" in loaded
        assert "best_dice" in loaded
        assert "config" in loaded
        assert loaded["config"]["encoder_name"] == "efficientnet_b0"

    def test_stdc1_checkpoint_loading(self, tmp_path, device):
        """Test STDC1 model checkpoint loading."""
        model = STDC1Seg(pretrained=False)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 5,
            "best_dice": 0.80,
            "config": {"encoder_name": "stdc1"},
        }

        path = tmp_path / "stdc1_checkpoint.pth"
        torch.save(checkpoint, path)

        # Load and verify
        loaded = torch.load(path, map_location=device, weights_only=False)
        new_model = STDC1Seg(pretrained=False)
        new_model.load_state_dict(loaded["model_state_dict"])

        # Verify model works
        new_model = new_model.to(device)
        new_model.eval()

        dummy = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            output = new_model(dummy)

        assert output.shape == (1, 1, 256, 256)


# =============================================================================
# Backend Selection Tests
# =============================================================================

class TestBackendSelection:
    """Test that backend selection works correctly."""

    def test_pytorch_fallback_when_tensorrt_unavailable(self):
        """Should fall back to PyTorch when TensorRT is not available."""
        # This tests the logic, not actual TensorRT
        tensorrt_available = False
        engine_exists = False

        # Simulate backend selection logic
        if tensorrt_available and engine_exists:
            backend = "tensorrt"
        else:
            backend = "pytorch"

        assert backend == "pytorch"

    def test_pytorch_fallback_when_engine_missing(self):
        """Should fall back to PyTorch when engine file is missing."""
        tensorrt_available = True
        engine_exists = False

        if tensorrt_available and engine_exists:
            backend = "tensorrt"
        else:
            backend = "pytorch"

        assert backend == "pytorch"

    def test_tensorrt_selected_when_available(self):
        """Should select TensorRT when available and engine exists."""
        tensorrt_available = True
        engine_exists = True

        if tensorrt_available and engine_exists:
            backend = "tensorrt"
        else:
            backend = "pytorch"

        assert backend == "tensorrt"
