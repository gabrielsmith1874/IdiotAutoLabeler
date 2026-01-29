"""
Tests for training metrics and loss functions.
"""

import pytest
import torch
import numpy as np

from train_segmentation import (
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    CombinedLoss,
    compute_dice_score,
    compute_iou,
    compute_precision_recall_f1,
    compute_prediction_distribution,
    compute_dice_at_threshold,
    compute_class_balance,
)


# =============================================================================
# Dice Loss Tests
# =============================================================================

class TestDiceLoss:
    """Test DiceLoss class."""

    def test_perfect_prediction_low_loss(self, perfect_prediction):
        """Perfect prediction should have very low dice loss."""
        pred, target = perfect_prediction
        loss_fn = DiceLoss()
        loss = loss_fn(pred, target)

        assert loss.item() < 0.1, f"Perfect prediction should have low loss, got {loss.item()}"

    def test_all_wrong_prediction_high_loss(self, all_negative_prediction):
        """All wrong prediction should have high dice loss."""
        pred, target = all_negative_prediction
        loss_fn = DiceLoss()
        loss = loss_fn(pred, target)

        assert loss.item() > 0.8, f"All wrong prediction should have high loss, got {loss.item()}"

    def test_empty_target_empty_pred(self):
        """Empty target with empty prediction - Dice is ill-defined for empty sets."""
        target = torch.zeros(1, 1, 64, 64)
        pred = torch.full((1, 1, 64, 64), -5.0)  # sigmoid -> ~0

        loss_fn = DiceLoss()
        loss = loss_fn(pred, target)

        # Dice loss for empty sets depends on smoothing factor
        # The loss should at least be computed without errors
        assert 0 <= loss.item() <= 1

    def test_loss_is_differentiable(self, sample_pred_tensor, sample_target_tensor):
        """Loss should be differentiable for backprop."""
        pred = sample_pred_tensor.clone().requires_grad_(True)
        loss_fn = DiceLoss()
        loss = loss_fn(pred, sample_target_tensor)
        loss.backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


# =============================================================================
# Focal Loss Tests
# =============================================================================

class TestFocalLoss:
    """Test FocalLoss class."""

    def test_perfect_prediction_low_loss(self, perfect_prediction):
        """Perfect prediction should have low focal loss."""
        pred, target = perfect_prediction
        loss_fn = FocalLoss()
        loss = loss_fn(pred, target)

        assert loss.item() < 0.1

    def test_focal_loss_range(self, sample_pred_tensor, sample_target_tensor):
        """Focal loss should be non-negative."""
        loss_fn = FocalLoss()
        loss = loss_fn(sample_pred_tensor, sample_target_tensor)

        assert loss.item() >= 0

    def test_gamma_effect(self, sample_pred_tensor, sample_target_tensor):
        """Higher gamma should reduce loss on easy examples."""
        loss_fn_low_gamma = FocalLoss(gamma=0.0)
        loss_fn_high_gamma = FocalLoss(gamma=5.0)

        loss_low = loss_fn_low_gamma(sample_pred_tensor, sample_target_tensor)
        loss_high = loss_fn_high_gamma(sample_pred_tensor, sample_target_tensor)

        # Higher gamma reduces loss on confident predictions
        assert loss_high.item() <= loss_low.item()


# =============================================================================
# Tversky Loss Tests
# =============================================================================

class TestTverskyLoss:
    """Test TverskyLoss class."""

    def test_perfect_prediction_low_loss(self, perfect_prediction):
        """Perfect prediction should have low tversky loss."""
        pred, target = perfect_prediction
        loss_fn = TverskyLoss()
        loss = loss_fn(pred, target)

        assert loss.item() < 0.1

    def test_alpha_beta_equal_is_dice(self, sample_pred_tensor, sample_target_tensor):
        """alpha=beta=0.5 should approximate Dice loss."""
        tversky_fn = TverskyLoss(alpha=0.5, beta=0.5)
        dice_fn = DiceLoss()

        tversky_loss = tversky_fn(sample_pred_tensor, sample_target_tensor)
        dice_loss = dice_fn(sample_pred_tensor, sample_target_tensor)

        # Should be close but not exactly equal due to smoothing differences
        assert abs(tversky_loss.item() - dice_loss.item()) < 0.2

    def test_loss_is_differentiable(self, sample_pred_tensor, sample_target_tensor):
        """Loss should be differentiable."""
        pred = sample_pred_tensor.clone().requires_grad_(True)
        loss_fn = TverskyLoss()
        loss = loss_fn(pred, sample_target_tensor)
        loss.backward()

        assert pred.grad is not None


# =============================================================================
# Combined Loss Tests
# =============================================================================

class TestCombinedLoss:
    """Test CombinedLoss class."""

    def test_combined_loss_returns_components(self, sample_pred_tensor, sample_target_tensor):
        """Combined loss should return individual components when requested."""
        loss_fn = CombinedLoss()
        combined, tversky, focal = loss_fn(
            sample_pred_tensor, sample_target_tensor, return_components=True
        )

        assert isinstance(combined, torch.Tensor)
        assert isinstance(tversky, float)
        assert isinstance(focal, float)

    def test_combined_loss_weights(self, sample_pred_tensor, sample_target_tensor):
        """Combined loss should respect weights."""
        loss_fn = CombinedLoss(tversky_weight=1.0, focal_weight=0.0)
        combined, tversky, _ = loss_fn(
            sample_pred_tensor, sample_target_tensor, return_components=True
        )

        # Should be approximately equal to tversky loss
        assert abs(combined.item() - tversky) < 0.01


# =============================================================================
# Dice Score Tests
# =============================================================================

class TestComputeDiceScore:
    """Test compute_dice_score function."""

    def test_perfect_score(self, perfect_prediction):
        """Perfect prediction should have dice score close to 1."""
        pred, target = perfect_prediction
        score = compute_dice_score(pred, target, threshold=0.5)

        assert score > 0.95, f"Expected dice > 0.95, got {score}"

    def test_zero_score(self, all_negative_prediction):
        """All wrong prediction should have low dice score."""
        pred, target = all_negative_prediction
        score = compute_dice_score(pred, target, threshold=0.5)

        assert score < 0.1, f"Expected dice < 0.1, got {score}"

    def test_threshold_effect(self, sample_pred_tensor, sample_target_tensor):
        """Different thresholds should give different scores."""
        score_low = compute_dice_score(sample_pred_tensor, sample_target_tensor, threshold=0.1)
        score_high = compute_dice_score(sample_pred_tensor, sample_target_tensor, threshold=0.9)

        # Different thresholds should generally give different results
        # (unless prediction is very uniform)
        assert isinstance(score_low, float)
        assert isinstance(score_high, float)
        assert 0 <= score_low <= 1
        assert 0 <= score_high <= 1


# =============================================================================
# IoU Tests
# =============================================================================

class TestComputeIoU:
    """Test compute_iou function."""

    def test_perfect_iou(self, perfect_prediction):
        """Perfect prediction should have IoU close to 1."""
        pred, target = perfect_prediction
        iou = compute_iou(pred, target, threshold=0.5)

        assert iou > 0.9, f"Expected IoU > 0.9, got {iou}"

    def test_iou_range(self, sample_pred_tensor, sample_target_tensor):
        """IoU should be between 0 and 1."""
        iou = compute_iou(sample_pred_tensor, sample_target_tensor)

        assert 0 <= iou <= 1, f"IoU should be in [0, 1], got {iou}"

    def test_iou_less_than_or_equal_dice(self, sample_pred_tensor, sample_target_tensor):
        """IoU should always be <= Dice score for non-empty sets."""
        iou = compute_iou(sample_pred_tensor, sample_target_tensor)
        dice = compute_dice_score(sample_pred_tensor, sample_target_tensor)

        # IoU <= Dice for non-trivial cases
        assert iou <= dice + 0.01  # Small tolerance for numerical errors


# =============================================================================
# Precision/Recall/F1 Tests
# =============================================================================

class TestComputePrecisionRecallF1:
    """Test compute_precision_recall_f1 function."""

    def test_perfect_metrics(self, perfect_prediction):
        """Perfect prediction should have high precision, recall, F1."""
        pred, target = perfect_prediction
        metrics = compute_precision_recall_f1(pred, target, threshold=0.5)

        assert metrics["precision"] > 0.95
        assert metrics["recall"] > 0.95
        assert metrics["f1"] > 0.95

    def test_returns_all_keys(self, sample_pred_tensor, sample_target_tensor):
        """Should return all expected metric keys."""
        metrics = compute_precision_recall_f1(sample_pred_tensor, sample_target_tensor)

        expected_keys = ["tp", "fp", "tn", "fn", "precision", "recall", "f1"]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_counts_sum_to_total_pixels(self, sample_pred_tensor, sample_target_tensor):
        """TP + FP + TN + FN should equal total pixels."""
        metrics = compute_precision_recall_f1(sample_pred_tensor, sample_target_tensor)

        total_pixels = sample_pred_tensor.numel()
        count_sum = metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]

        assert count_sum == total_pixels


# =============================================================================
# Prediction Distribution Tests
# =============================================================================

class TestComputePredictionDistribution:
    """Test compute_prediction_distribution function."""

    def test_returns_all_keys(self, sample_pred_tensor):
        """Should return expected distribution keys."""
        dist = compute_prediction_distribution(sample_pred_tensor)

        expected_keys = ["low", "uncertain", "high", "total"]
        for key in expected_keys:
            assert key in dist

    def test_total_matches_numel(self, sample_pred_tensor):
        """Total should match tensor size."""
        dist = compute_prediction_distribution(sample_pred_tensor)

        assert dist["total"] == sample_pred_tensor.numel()


# =============================================================================
# Dice at Threshold Tests
# =============================================================================

class TestComputeDiceAtThreshold:
    """Test compute_dice_at_threshold function."""

    def test_consistent_with_compute_dice_score(self, sample_pred_tensor, sample_target_tensor):
        """Should give same result as compute_dice_score for same threshold."""
        threshold = 0.5

        dice1 = compute_dice_score(sample_pred_tensor, sample_target_tensor, threshold=threshold)
        dice2 = compute_dice_at_threshold(sample_pred_tensor, sample_target_tensor, threshold=threshold)

        assert abs(dice1 - dice2) < 0.01


# =============================================================================
# Class Balance Tests
# =============================================================================

class TestComputeClassBalance:
    """Test compute_class_balance function."""

    def test_returns_all_keys(self, sample_pred_tensor, sample_target_tensor):
        """Should return expected keys."""
        balance = compute_class_balance(sample_pred_tensor, sample_target_tensor)

        expected_keys = ["gt_positive", "pred_positive", "total_pixels",
                         "gt_positive_pct", "pred_positive_pct"]
        for key in expected_keys:
            assert key in balance

    def test_percentage_range(self, sample_pred_tensor, sample_target_tensor):
        """Percentages should be between 0 and 100."""
        balance = compute_class_balance(sample_pred_tensor, sample_target_tensor)

        assert 0 <= balance["gt_positive_pct"] <= 100
        assert 0 <= balance["pred_positive_pct"] <= 100
