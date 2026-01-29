"""
Tests for trigger and aimbot feature logic.

Note: This module tests the core logic independently by recreating
key classes to avoid Windows-specific imports from unified_app.
"""

import pytest
import numpy as np
import time


# Recreate Config for testing
class Config:
    TRIGGER_DELAY = 0.05
    HOLD_DURATION = 0.005
    TRIGGER_MARGIN = 5
    TRIGGER_HOTKEY = ('mouse5', 6)
    AIM_HOTKEY = ('mouse4', 5)
    AIM_SPEED = 0.5
    AIMBOT_DEADZONE = 10


# Recreate TriggerFeature logic for testing (simplified, no Windows deps)
class TriggerFeature:
    """Simplified trigger feature for testing."""

    def __init__(self):
        self.last_trigger = 0
        self.trigger_count = 0
        self.app_hwnd = None

    def _is_app_focused(self) -> bool:
        # Always return False for tests
        return False

    def process(self, mask: np.ndarray, threshold: float) -> bool:
        """Process mask and return True if should trigger."""
        if self._is_app_focused():
            return False

        h, w = mask.shape
        cy, cx = h // 2, w // 2
        margin = Config.TRIGGER_MARGIN

        y1 = max(0, cy - margin)
        y2 = min(h, cy + margin + 1)
        x1 = max(0, cx - margin)
        x2 = min(w, cx + margin + 1)

        region = mask[y1:y2, x1:x2]
        if region.size == 0:
            return False

        min_val = region.min() / 255.0

        if min_val > threshold:
            now = time.perf_counter()
            if (now - self.last_trigger) > Config.TRIGGER_DELAY:
                # Would click here, but we're testing
                self.last_trigger = now
                self.trigger_count += 1
                return True
        return False


# =============================================================================
# Trigger Feature Tests
# =============================================================================

class TestTriggerFeature:
    """Test TriggerFeature class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.trigger = TriggerFeature()

    def test_initialization(self):
        """Trigger should initialize with correct defaults."""
        assert self.trigger.last_trigger == 0
        assert self.trigger.trigger_count == 0
        assert self.trigger.app_hwnd is None

    def test_process_empty_mask_returns_false(self, empty_mask):
        """Empty mask should not trigger."""
        result = self.trigger.process(empty_mask, threshold=0.5)

        assert result is False
        assert self.trigger.trigger_count == 0

    def test_process_center_below_threshold_returns_false(self, sample_mask):
        """If center is below threshold, should not trigger."""
        # Create mask with hole in center
        mask = sample_mask.copy()
        h, w = mask.shape
        cy, cx = h // 2, w // 2
        margin = 10
        mask[cy-margin:cy+margin, cx-margin:cx+margin] = 0

        result = self.trigger.process(mask, threshold=0.5)

        assert result is False

    def test_full_mask_triggers(self):
        """Full mask should trigger."""
        trigger = TriggerFeature()
        mask = np.full((256, 256), 255, dtype=np.uint8)

        result = trigger.process(mask, threshold=0.5)

        assert result is True
        assert trigger.trigger_count == 1

    def test_process_respects_trigger_margin(self):
        """Trigger margin should require pixels inside edge."""
        trigger = TriggerFeature()

        mask = np.zeros((256, 256), dtype=np.uint8)
        h, w = mask.shape
        cy, cx = h // 2, w // 2

        # Only center pixel is high
        mask[cy, cx] = 255

        # Should not trigger because margin requires surrounding pixels
        result = trigger.process(mask, threshold=0.5)
        assert result is False

        # Fill the margin area
        margin = Config.TRIGGER_MARGIN
        mask[cy-margin:cy+margin+1, cx-margin:cx+margin+1] = 255

        # Should now trigger
        result = trigger.process(mask, threshold=0.5)
        assert result is True

    def test_process_respects_trigger_delay(self):
        """Should not trigger more often than TRIGGER_DELAY."""
        trigger = TriggerFeature()
        mask = np.full((256, 256), 255, dtype=np.uint8)

        original_delay = Config.TRIGGER_DELAY
        Config.TRIGGER_DELAY = 0.1

        try:
            result1 = trigger.process(mask, threshold=0.5)
            assert result1 is True

            # Immediate second call should be blocked
            result2 = trigger.process(mask, threshold=0.5)
            assert result2 is False

            # Wait and try again
            time.sleep(0.15)
            result3 = trigger.process(mask, threshold=0.5)
            assert result3 is True

        finally:
            Config.TRIGGER_DELAY = original_delay

    def test_trigger_count_increments(self):
        """Trigger count should increment on successful trigger."""
        trigger = TriggerFeature()
        mask = np.full((256, 256), 255, dtype=np.uint8)

        assert trigger.trigger_count == 0

        trigger.process(mask, threshold=0.5)
        assert trigger.trigger_count == 1

        # Wait for delay
        time.sleep(Config.TRIGGER_DELAY + 0.01)

        trigger.process(mask, threshold=0.5)
        assert trigger.trigger_count == 2


# =============================================================================
# Aimbot Logic Tests
# =============================================================================

class TestAimbotLogic:
    """Test aimbot targeting logic."""

    def test_find_targets_from_mask(self):
        """Should find target centroids from mask."""
        from scipy import ndimage

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:150, 100:150] = 255

        labeled, num_features = ndimage.label(mask > 127)

        if num_features > 0:
            centroids = ndimage.center_of_mass(mask, labeled, range(1, num_features + 1))
            assert len(centroids) == 1
            cy, cx = centroids[0]
            assert 100 <= cy <= 150
            assert 100 <= cx <= 150

    def test_find_multiple_targets(self):
        """Should find multiple target centroids."""
        from scipy import ndimage

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:70, 50:70] = 255
        mask[180:200, 180:200] = 255

        labeled, num_features = ndimage.label(mask > 127)

        assert num_features == 2

    def test_no_targets_in_empty_mask(self):
        """Empty mask should have no targets."""
        from scipy import ndimage

        mask = np.zeros((256, 256), dtype=np.uint8)

        labeled, num_features = ndimage.label(mask > 127)

        assert num_features == 0

    def test_target_area_calculation(self):
        """Should correctly calculate target areas."""
        from scipy import ndimage

        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:150, 100:150] = 255

        labeled, num_features = ndimage.label(mask > 127)

        if num_features > 0:
            areas = ndimage.sum(mask > 127, labeled, range(1, num_features + 1))
            assert areas[0] == 2500  # 50x50


# =============================================================================
# Target Selection Tests
# =============================================================================

class TestTargetSelection:
    """Test target selection logic."""

    def test_select_closest_to_crosshair(self):
        """Should select target closest to crosshair (center)."""
        center_x, center_y = 128, 128

        targets = [
            (50, 50, 100),
            (120, 120, 100),
            (200, 200, 100),
        ]

        closest = min(targets, key=lambda t: (t[0] - center_x)**2 + (t[1] - center_y)**2)

        assert closest == (120, 120, 100)

    def test_select_largest_when_equidistant(self):
        """When equidistant, should prefer larger target."""
        center_x, center_y = 128, 128

        targets = [
            (138, 128, 50),
            (118, 128, 200),
        ]

        min_dist = min((t[0] - center_x)**2 + (t[1] - center_y)**2 for t in targets)
        closest = [t for t in targets
                   if (t[0] - center_x)**2 + (t[1] - center_y)**2 == min_dist]

        largest = max(closest, key=lambda t: t[2])

        assert largest[2] == 200


# =============================================================================
# Mouse Movement Calculation Tests
# =============================================================================

class TestMouseMovementCalculation:
    """Test mouse movement vector calculations."""

    def test_movement_toward_target(self):
        """Movement should be toward target."""
        crosshair = (128, 128)
        target = (150, 100)

        dx = target[0] - crosshair[0]
        dy = target[1] - crosshair[1]

        assert dx > 0
        assert dy < 0

    def test_movement_magnitude_clamping(self):
        """Movement should be clamped to max step."""
        crosshair = (0, 0)
        target = (1000, 1000)

        max_step = 35

        dx = target[0] - crosshair[0]
        dy = target[1] - crosshair[1]

        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > max_step:
            scale = max_step / magnitude
            dx = int(dx * scale)
            dy = int(dy * scale)

        assert abs(dx) <= max_step
        assert abs(dy) <= max_step

    def test_deadzone_prevents_small_movements(self):
        """Movements within deadzone should be ignored."""
        crosshair = (128, 128)
        target = (130, 130)

        deadzone = 10

        dx = target[0] - crosshair[0]
        dy = target[1] - crosshair[1]

        distance = np.sqrt(dx**2 + dy**2)

        should_move = distance > deadzone

        assert not should_move  # Distance ~2.8 < 10


# =============================================================================
# Aim Smoothing Tests
# =============================================================================

class TestAimSmoothing:
    """Test aim smoothing/interpolation."""

    def test_aim_speed_scales_movement(self):
        """AIM_SPEED should scale movement amount."""
        raw_dx, raw_dy = 20, 10

        for speed in [0.25, 0.5, 0.75, 1.0]:
            scaled_dx = int(raw_dx * speed)
            scaled_dy = int(raw_dy * speed)

            assert scaled_dx <= raw_dx
            assert scaled_dy <= raw_dy
            assert scaled_dx >= 0
            assert scaled_dy >= 0

    def test_low_speed_small_movement(self):
        """Low speed should result in smaller movements."""
        raw_dx, raw_dy = 100, 100

        high_speed = 0.8
        low_speed = 0.2

        high_dx = int(raw_dx * high_speed)
        low_dx = int(raw_dx * low_speed)

        assert high_dx > low_dx
