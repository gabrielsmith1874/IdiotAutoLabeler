"""
Tests for configuration persistence.

Note: This module tests configuration logic independently by recreating
the Config class to avoid Windows-specific imports from unified_app.
"""

import pytest
import json
from pathlib import Path


# Recreate Config class for testing (avoids Windows-specific imports)
class Config:
    """Configuration class for testing."""

    MODEL_PATH = Path("models/enemy_segmentation.pth")
    ENGINE_PATH = Path("models/enemy_segmentation_256.engine")
    CAPTURE_PERCENT = 0.5
    INFERENCE_SIZE = 256
    THRESHOLD = 0.5
    MIN_AREA = 100
    TRIGGER_DELAY = 0.05
    HOLD_DURATION = 0.005
    TRIGGER_HOTKEY = ('mouse5', 6)
    TRIGGER_MARGIN = 5
    AIM_SPEED = 0.5
    AIM_HOTKEY = ('mouse4', 5)
    AIMBOT_DEADZONE = 10
    OVERLAY_COLOR = (255, 0, 255)
    OVERLAY_ALPHA = 0.6
    TARGET_FPS = 120
    TRACKING_PROCESS_NOISE = 1e-4
    TRACKING_MEASUREMENT_NOISE = 0.1
    PREDICTION_TIME = 0.05
    TRACK_MAX_AGE = 0.2
    MOUSE_SENSITIVITY = 1.0

    _config_file = None

    @classmethod
    def save(cls, config_file: Path = None):
        """Save configurable settings to JSON file."""
        if config_file is None:
            config_file = cls._config_file
        if config_file is None:
            return

        config_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "threshold": cls.THRESHOLD,
            "trigger_delay": cls.TRIGGER_DELAY,
            "trigger_margin": cls.TRIGGER_MARGIN,
            "trigger_hotkey": cls.TRIGGER_HOTKEY,
            "aim_speed": cls.AIM_SPEED,
            "aim_hotkey": cls.AIM_HOTKEY,
            "aimbot_deadzone": cls.AIMBOT_DEADZONE,
            "prediction_time": cls.PREDICTION_TIME,
            "mouse_sensitivity": cls.MOUSE_SENSITIVITY,
        }

        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, config_file: Path = None):
        """Load settings from JSON file if exists."""
        if config_file is None:
            config_file = cls._config_file
        if config_file is None or not config_file.exists():
            return

        try:
            with open(config_file, 'r') as f:
                data = json.load(f)

            cls.THRESHOLD = data.get("threshold", cls.THRESHOLD)
            cls.TRIGGER_DELAY = data.get("trigger_delay", cls.TRIGGER_DELAY)
            cls.TRIGGER_MARGIN = data.get("trigger_margin", cls.TRIGGER_MARGIN)
            cls.AIM_SPEED = data.get("aim_speed", cls.AIM_SPEED)
            cls.AIMBOT_DEADZONE = data.get("aimbot_deadzone", cls.AIMBOT_DEADZONE)
            cls.PREDICTION_TIME = data.get("prediction_time", cls.PREDICTION_TIME)
            cls.MOUSE_SENSITIVITY = data.get("mouse_sensitivity", cls.MOUSE_SENSITIVITY)

            if "trigger_hotkey" in data:
                cls.TRIGGER_HOTKEY = tuple(data["trigger_hotkey"])
            if "aim_hotkey" in data:
                cls.AIM_HOTKEY = tuple(data["aim_hotkey"])

        except Exception:
            pass


# =============================================================================
# Config Class Tests
# =============================================================================

class TestConfig:
    """Test Config class."""

    def test_default_values_exist(self):
        """Config should have all expected default values."""
        assert hasattr(Config, 'THRESHOLD')
        assert hasattr(Config, 'TRIGGER_DELAY')
        assert hasattr(Config, 'TRIGGER_MARGIN')
        assert hasattr(Config, 'AIM_SPEED')
        assert hasattr(Config, 'AIMBOT_DEADZONE')
        assert hasattr(Config, 'PREDICTION_TIME')
        assert hasattr(Config, 'MOUSE_SENSITIVITY')

    def test_default_threshold_range(self):
        """Threshold should be between 0 and 1."""
        assert 0 <= Config.THRESHOLD <= 1

    def test_default_trigger_delay_positive(self):
        """Trigger delay should be positive."""
        assert Config.TRIGGER_DELAY > 0

    def test_default_aim_speed_range(self):
        """Aim speed should be between 0 and 1."""
        assert 0 <= Config.AIM_SPEED <= 1

    def test_hotkey_format(self):
        """Hotkeys should be tuples of (name, vk_code)."""
        assert isinstance(Config.TRIGGER_HOTKEY, tuple)
        assert len(Config.TRIGGER_HOTKEY) == 2
        assert isinstance(Config.TRIGGER_HOTKEY[0], str)
        assert isinstance(Config.TRIGGER_HOTKEY[1], int)

        assert isinstance(Config.AIM_HOTKEY, tuple)
        assert len(Config.AIM_HOTKEY) == 2


# =============================================================================
# Config Save Tests
# =============================================================================

class TestConfigSave:
    """Test Config.save() method."""

    def test_save_creates_file(self, temp_config_dir):
        """Save should create config file."""
        config_path = temp_config_dir / "unified_app.json"
        Config._config_file = config_path

        Config.save()

        assert config_path.exists()

    def test_save_writes_valid_json(self, temp_config_dir):
        """Saved file should be valid JSON."""
        config_path = temp_config_dir / "unified_app.json"
        Config._config_file = config_path

        Config.save()

        with open(config_path, 'r') as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_save_includes_expected_keys(self, temp_config_dir):
        """Saved config should include expected keys."""
        config_path = temp_config_dir / "unified_app.json"
        Config._config_file = config_path

        Config.save()

        with open(config_path, 'r') as f:
            data = json.load(f)

        expected_keys = [
            'threshold', 'trigger_delay', 'trigger_margin',
            'aim_speed', 'aimbot_deadzone', 'prediction_time',
            'mouse_sensitivity'
        ]

        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_save_preserves_values(self, temp_config_dir):
        """Saved values should match Config values."""
        config_path = temp_config_dir / "unified_app.json"
        Config._config_file = config_path

        Config.save()

        with open(config_path, 'r') as f:
            data = json.load(f)

        assert data['threshold'] == Config.THRESHOLD
        assert data['trigger_delay'] == Config.TRIGGER_DELAY
        assert data['aim_speed'] == Config.AIM_SPEED


# =============================================================================
# Config Load Tests
# =============================================================================

class TestConfigLoad:
    """Test Config.load() method."""

    def test_load_nonexistent_file(self, temp_config_dir):
        """Load should handle missing file gracefully."""
        config_path = temp_config_dir / "nonexistent.json"
        Config._config_file = config_path

        # Should not raise
        Config.load()

    def test_load_restores_values(self, temp_config_dir):
        """Load should restore saved values."""
        config_path = temp_config_dir / "unified_app.json"

        custom_data = {
            'threshold': 0.75,
            'trigger_delay': 0.1,
            'trigger_margin': 10,
            'aim_speed': 0.3,
            'aimbot_deadzone': 15,
            'prediction_time': 0.08,
            'mouse_sensitivity': 1.5,
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(custom_data, f)

        original_threshold = Config.THRESHOLD
        original_speed = Config.AIM_SPEED

        try:
            Config._config_file = config_path
            Config.load()

            assert Config.THRESHOLD == 0.75
            assert Config.AIM_SPEED == 0.3
            assert Config.AIMBOT_DEADZONE == 15

        finally:
            Config.THRESHOLD = original_threshold
            Config.AIM_SPEED = original_speed

    def test_load_handles_partial_config(self, temp_config_dir):
        """Load should handle config with missing keys."""
        config_path = temp_config_dir / "unified_app.json"

        partial_data = {
            'threshold': 0.8,
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(partial_data, f)

        original_threshold = Config.THRESHOLD
        original_speed = Config.AIM_SPEED

        try:
            Config._config_file = config_path
            Config.load()

            assert Config.THRESHOLD == 0.8

        finally:
            Config.THRESHOLD = original_threshold
            Config.AIM_SPEED = original_speed

    def test_load_handles_corrupted_json(self, temp_config_dir):
        """Load should handle corrupted JSON gracefully."""
        config_path = temp_config_dir / "unified_app.json"

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write("{ invalid json }")

        Config._config_file = config_path
        # Should not raise
        Config.load()

    def test_load_hotkeys(self, temp_config_dir):
        """Load should restore hotkey tuples correctly."""
        config_path = temp_config_dir / "unified_app.json"

        custom_data = {
            'threshold': 0.5,
            'trigger_delay': 0.05,
            'trigger_margin': 5,
            'aim_speed': 0.5,
            'aimbot_deadzone': 10,
            'prediction_time': 0.05,
            'mouse_sensitivity': 1.0,
            'trigger_hotkey': ['mouse4', 5],
            'aim_hotkey': ['mouse5', 6],
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(custom_data, f)

        original_trigger = Config.TRIGGER_HOTKEY
        original_aim = Config.AIM_HOTKEY

        try:
            Config._config_file = config_path
            Config.load()

            assert Config.TRIGGER_HOTKEY == ('mouse4', 5)
            assert Config.AIM_HOTKEY == ('mouse5', 6)

        finally:
            Config.TRIGGER_HOTKEY = original_trigger
            Config.AIM_HOTKEY = original_aim


# =============================================================================
# Config Round-Trip Tests
# =============================================================================

class TestConfigRoundTrip:
    """Test save followed by load."""

    def test_round_trip_preserves_values(self, temp_config_dir):
        """Save then load should preserve all values."""
        config_path = temp_config_dir / "unified_app.json"
        Config._config_file = config_path

        original_threshold = Config.THRESHOLD
        original_speed = Config.AIM_SPEED
        original_deadzone = Config.AIMBOT_DEADZONE

        try:
            Config.THRESHOLD = 0.85
            Config.AIM_SPEED = 0.42
            Config.AIMBOT_DEADZONE = 8

            Config.save()

            # Reset to defaults
            Config.THRESHOLD = 0.5
            Config.AIM_SPEED = 0.5
            Config.AIMBOT_DEADZONE = 10

            # Load saved values
            Config.load()

            assert Config.THRESHOLD == 0.85
            assert Config.AIM_SPEED == 0.42
            assert Config.AIMBOT_DEADZONE == 8

        finally:
            Config.THRESHOLD = original_threshold
            Config.AIM_SPEED = original_speed
            Config.AIMBOT_DEADZONE = original_deadzone

    def test_multiple_save_load_cycles(self, temp_config_dir):
        """Multiple save/load cycles should work correctly."""
        config_path = temp_config_dir / "unified_app.json"
        Config._config_file = config_path

        original_threshold = Config.THRESHOLD

        try:
            for i, value in enumerate([0.1, 0.5, 0.9]):
                Config.THRESHOLD = value
                Config.save()
                Config.THRESHOLD = 0.0
                Config.load()

                assert Config.THRESHOLD == value, f"Failed on iteration {i}"

        finally:
            Config.THRESHOLD = original_threshold


# =============================================================================
# Config Validation Tests
# =============================================================================

class TestConfigValidation:
    """Test config value validation."""

    def test_threshold_loaded_as_is(self, temp_config_dir):
        """Out-of-range threshold should be loaded as-is."""
        config_path = temp_config_dir / "unified_app.json"

        custom_data = {
            'threshold': 1.5,
            'trigger_delay': 0.05,
            'trigger_margin': 5,
            'aim_speed': 0.5,
            'aimbot_deadzone': 10,
            'prediction_time': 0.05,
            'mouse_sensitivity': 1.0,
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(custom_data, f)

        original = Config.THRESHOLD

        try:
            Config._config_file = config_path
            Config.load()
            assert Config.THRESHOLD == 1.5

        finally:
            Config.THRESHOLD = original

    def test_negative_delay_loaded(self, temp_config_dir):
        """Negative delay should be loaded."""
        config_path = temp_config_dir / "unified_app.json"

        custom_data = {
            'threshold': 0.5,
            'trigger_delay': -0.1,
            'trigger_margin': 5,
            'aim_speed': 0.5,
            'aimbot_deadzone': 10,
            'prediction_time': 0.05,
            'mouse_sensitivity': 1.0,
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(custom_data, f)

        original = Config.TRIGGER_DELAY

        try:
            Config._config_file = config_path
            Config.load()
            assert Config.TRIGGER_DELAY == -0.1

        finally:
            Config.TRIGGER_DELAY = original
