"""
Tests for image preprocessing and mask extraction functions.
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from train_segmentation import extract_mask, find_image_pairs
from preprocess_tiles import (
    extract_mask as preprocess_extract_mask,
    extract_tiles_from_image,
    find_image_pairs as preprocess_find_pairs,
)


# =============================================================================
# Mask Extraction Tests
# =============================================================================

class TestExtractMask:
    """Test extract_mask function from train_segmentation."""

    def test_extracts_difference_region(self, masked_unmasked_pair):
        """Should extract mask where images differ."""
        unmasked, masked = masked_unmasked_pair
        mask = extract_mask(masked, unmasked, threshold=30)

        # Should have some positive pixels
        assert mask.sum() > 0, "Mask should have positive pixels"

        # Center region should be positive (where we added magenta)
        center_region = mask[100:150, 100:150]
        assert center_region.sum() > 0, "Center region should be detected"

    def test_identical_images_empty_mask(self, sample_image):
        """Identical images should produce empty mask."""
        mask = extract_mask(sample_image, sample_image, threshold=30)

        assert mask.sum() == 0, "Identical images should have empty mask"

    def test_threshold_affects_detection(self):
        """Higher threshold should reduce detected pixels."""
        unmasked = np.full((64, 64, 3), 100, dtype=np.uint8)
        masked = unmasked.copy()
        # Add small difference
        masked[20:40, 20:40] = [120, 100, 100]  # Small diff of 20 in R channel

        mask_low_thresh = extract_mask(masked, unmasked, threshold=10)
        mask_high_thresh = extract_mask(masked, unmasked, threshold=50)

        assert mask_low_thresh.sum() > mask_high_thresh.sum()

    def test_output_binary(self, masked_unmasked_pair):
        """Output mask should be binary (0 or 1)."""
        unmasked, masked = masked_unmasked_pair
        mask = extract_mask(masked, unmasked, threshold=30)

        unique_values = np.unique(mask)
        assert all(v in [0, 1] for v in unique_values)

    def test_output_shape_matches_input(self, masked_unmasked_pair):
        """Output mask should have same HxW as input images."""
        unmasked, masked = masked_unmasked_pair
        mask = extract_mask(masked, unmasked, threshold=30)

        assert mask.shape == unmasked.shape[:2]


class TestPreprocessExtractMask:
    """Test extract_mask from preprocess_tiles (uses max instead of sum)."""

    def test_extracts_difference_region(self, masked_unmasked_pair):
        """Should extract mask where images differ."""
        unmasked, masked = masked_unmasked_pair
        mask = preprocess_extract_mask(masked, unmasked, threshold=30)

        assert mask.sum() > 0

    def test_max_vs_sum_difference(self):
        """Max-based extraction may give different results than sum-based."""
        unmasked = np.full((64, 64, 3), 100, dtype=np.uint8)
        masked = unmasked.copy()
        # Small diff in one channel only
        masked[20:40, 20:40, 0] = 140  # +40 in R only

        mask_preprocess = preprocess_extract_mask(masked, unmasked, threshold=30)
        mask_train = extract_mask(masked, unmasked, threshold=30)

        # Both should detect the region
        assert mask_preprocess.sum() > 0
        assert mask_train.sum() > 0


# =============================================================================
# Tile Extraction Tests
# =============================================================================

class TestExtractTilesFromImage:
    """Test extract_tiles_from_image function."""

    def test_extracts_correct_number_of_tiles(self, sample_image, sample_mask):
        """Should extract expected number of tiles based on stride."""
        tile_size = 128
        stride = 64
        h, w = sample_image.shape[:2]

        tiles, masks, is_positive = extract_tiles_from_image(
            sample_image, sample_mask, tile_size=tile_size, stride=stride
        )

        expected_tiles_h = (h - tile_size) // stride + 1
        expected_tiles_w = (w - tile_size) // stride + 1
        expected_count = expected_tiles_h * expected_tiles_w

        assert len(tiles) == expected_count
        assert len(masks) == expected_count
        assert len(is_positive) == expected_count

    def test_tile_size_correct(self, sample_image, sample_mask):
        """Extracted tiles should have correct dimensions."""
        tile_size = 128
        tiles, masks, _ = extract_tiles_from_image(
            sample_image, sample_mask, tile_size=tile_size, stride=64
        )

        for tile in tiles:
            assert tile.shape == (tile_size, tile_size, 3)
        for mask in masks:
            assert mask.shape == (tile_size, tile_size)

    def test_positive_detection_correct(self, sample_image, sample_mask):
        """is_positive should correctly identify tiles with mask content."""
        tiles, masks, is_positive = extract_tiles_from_image(
            sample_image, sample_mask, tile_size=64, stride=64
        )

        for mask_tile, positive in zip(masks, is_positive):
            if positive:
                assert mask_tile.sum() > 0
            else:
                assert mask_tile.sum() == 0

    def test_empty_mask_all_negative(self, sample_image, empty_mask):
        """All tiles from empty mask should be negative."""
        _, _, is_positive = extract_tiles_from_image(
            sample_image, empty_mask, tile_size=64, stride=64
        )

        assert all(not p for p in is_positive)

    def test_full_mask_all_positive(self, sample_image, full_mask):
        """All tiles from full mask should be positive."""
        # Resize full_mask to match sample_image
        full = np.ones((sample_image.shape[0], sample_image.shape[1]), dtype=np.uint8)

        _, _, is_positive = extract_tiles_from_image(
            sample_image, full, tile_size=64, stride=64
        )

        assert all(p for p in is_positive)


# =============================================================================
# Find Image Pairs Tests
# =============================================================================

class TestFindImagePairs:
    """Test find_image_pairs function."""

    def test_finds_valid_pairs(self, temp_data_dir):
        """Should find matching masked/unmasked pairs."""
        labeled_dir = temp_data_dir / "labeled"

        # Create test image pairs
        for i in range(3):
            unmasked = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            masked = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

            unmasked.save(labeled_dir / f"test_{i}_unmasked.png")
            masked.save(labeled_dir / f"test_{i}_masked.png")

        pairs = find_image_pairs(labeled_dir)

        assert len(pairs) == 3

    def test_skips_incomplete_pairs(self, temp_data_dir):
        """Should skip files without matching pair."""
        labeled_dir = temp_data_dir / "labeled"

        # Create only unmasked file (no matching masked)
        unmasked = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        unmasked.save(labeled_dir / "orphan_unmasked.png")

        pairs = find_image_pairs(labeled_dir)

        assert len(pairs) == 0

    def test_empty_directory(self, temp_data_dir):
        """Empty directory should return empty list."""
        labeled_dir = temp_data_dir / "labeled"
        pairs = find_image_pairs(labeled_dir)

        assert pairs == []

    def test_returns_correct_tuple_format(self, temp_data_dir):
        """Should return list of (unmasked_path, masked_path) tuples."""
        labeled_dir = temp_data_dir / "labeled"

        unmasked = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        masked = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

        unmasked.save(labeled_dir / "test_unmasked.png")
        masked.save(labeled_dir / "test_masked.png")

        pairs = find_image_pairs(labeled_dir)

        assert len(pairs) == 1
        unmasked_path, masked_path = pairs[0]
        assert "unmasked" in unmasked_path.name
        assert "masked" in masked_path.name


class TestPreprocessFindPairs:
    """Test find_image_pairs from preprocess_tiles module."""

    def test_finds_pairs(self, temp_data_dir):
        """Should find image pairs."""
        labeled_dir = temp_data_dir / "labeled"

        for i in range(2):
            unmasked = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            masked = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

            unmasked.save(labeled_dir / f"img_{i}_unmasked.png")
            masked.save(labeled_dir / f"img_{i}_masked.png")

        pairs = preprocess_find_pairs(labeled_dir)

        assert len(pairs) == 2


# =============================================================================
# Image Preprocessing Pipeline Tests
# =============================================================================

class TestPreprocessingPipeline:
    """Integration tests for the preprocessing pipeline."""

    def test_full_pipeline(self, temp_data_dir):
        """Test complete preprocessing: pairs -> masks -> tiles."""
        labeled_dir = temp_data_dir / "labeled"

        # Create a larger test image pair with known mask region in one corner
        # so we get some fully negative tiles
        unmasked_arr = np.full((512, 512, 3), 100, dtype=np.uint8)
        masked_arr = unmasked_arr.copy()
        # Put mask in bottom-right corner only
        masked_arr[400:450, 400:450] = [200, 50, 200]  # Magenta-ish overlay

        unmasked = Image.fromarray(unmasked_arr)
        masked = Image.fromarray(masked_arr)

        unmasked.save(labeled_dir / "pipeline_test_unmasked.png")
        masked.save(labeled_dir / "pipeline_test_masked.png")

        # Find pairs
        pairs = find_image_pairs(labeled_dir)
        assert len(pairs) == 1

        # Extract mask
        unmasked_path, masked_path = pairs[0]
        unmasked_loaded = np.array(Image.open(unmasked_path))
        masked_loaded = np.array(Image.open(masked_path))
        mask = extract_mask(masked_loaded, unmasked_loaded, threshold=30)

        assert mask.sum() > 0

        # Extract tiles with smaller stride to get more coverage
        tiles, tile_masks, is_positive = extract_tiles_from_image(
            unmasked_loaded, mask, tile_size=128, stride=128
        )

        assert len(tiles) > 0
        # Some tiles should be positive (contain the mask region)
        assert any(is_positive), "Expected at least one positive tile"
        # Some tiles should be negative (outside mask region)
        assert any(not p for p in is_positive), "Expected at least one negative tile"
