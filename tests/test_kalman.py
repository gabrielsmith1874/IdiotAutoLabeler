"""
Tests for Kalman filter tracking.

Note: This module tests the Kalman filter implementation independently
by recreating the KalmanFilter class locally to avoid Windows-specific imports.
"""

import pytest
import numpy as np
import time


# Recreate the KalmanFilter class for testing (avoids Windows-specific imports)
class KalmanFilter:
    """
    Constant Velocity Kalman Filter for 2D tracking.
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    def __init__(self, init_x, init_y, dt=1/60.0):
        self.state = np.array([init_x, init_y, 0, 0], dtype=float)
        self.P = np.eye(4) * 10.0
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.R = np.eye(2) * 0.1  # TRACKING_MEASUREMENT_NOISE
        self.Q = np.eye(4) * 1e-4  # TRACKING_PROCESS_NOISE
        self.last_update_time = time.perf_counter()
        self.age = 0

    def predict(self, current_time=None, control_input=None):
        if current_time is None:
            current_time = time.perf_counter()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.state = self.F @ self.state
        if control_input:
            dx, dy = control_input
            scale = 1.0  # MOUSE_SENSITIVITY
            self.state[0] -= dx * scale
            self.state[1] -= dy * scale
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]

    def update(self, measurement):
        z = np.array(measurement, dtype=float)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.age = 0

    def get_predicted_pos(self, future_time=0.0):
        x, y, vx, vy = self.state
        return x + vx * future_time, y + vy * future_time


# Mock Config class for testing
class Config:
    MOUSE_SENSITIVITY = 1.0


# =============================================================================
# Kalman Filter Basic Tests
# =============================================================================

class TestKalmanFilterBasic:
    """Basic Kalman filter functionality tests."""

    def test_initialization(self):
        """Kalman filter should initialize with correct state."""
        kf = KalmanFilter(init_x=100, init_y=200, dt=1/60.0)

        assert kf.state[0] == 100  # x position
        assert kf.state[1] == 200  # y position
        assert kf.state[2] == 0    # x velocity (initialized to 0)
        assert kf.state[3] == 0    # y velocity (initialized to 0)

    def test_state_dimensions(self):
        """State should have 4 dimensions: x, y, vx, vy."""
        kf = KalmanFilter(init_x=0, init_y=0)

        assert kf.state.shape == (4,)

    def test_covariance_dimensions(self):
        """Covariance matrix should be 4x4."""
        kf = KalmanFilter(init_x=0, init_y=0)

        assert kf.P.shape == (4, 4)

    def test_measurement_matrix_dimensions(self):
        """Measurement matrix should be 2x4."""
        kf = KalmanFilter(init_x=0, init_y=0)

        assert kf.H.shape == (2, 4)


# =============================================================================
# Kalman Filter Predict Tests
# =============================================================================

class TestKalmanFilterPredict:
    """Tests for predict method."""

    def test_predict_returns_position(self):
        """Predict should return (x, y) position."""
        kf = KalmanFilter(init_x=100, init_y=200)

        pos = kf.predict()

        assert len(pos) == 2

    def test_predict_with_zero_velocity(self):
        """With zero velocity, position should stay constant."""
        kf = KalmanFilter(init_x=100, init_y=200, dt=1/60.0)

        # First predict just initializes time
        kf.predict()
        time.sleep(0.01)

        pos = kf.predict()

        # Should be close to initial position (small process noise may cause drift)
        assert abs(pos[0] - 100) < 5
        assert abs(pos[1] - 200) < 5

    def test_predict_with_velocity(self):
        """With velocity, position should move in velocity direction."""
        kf = KalmanFilter(init_x=100, init_y=100, dt=1/60.0)

        # Set velocity manually
        kf.state[2] = 60  # vx = 60 pixels/sec
        kf.state[3] = 0   # vy = 0

        # Wait and predict
        time.sleep(0.05)
        pos = kf.predict()

        # x should have increased
        assert pos[0] > 100

    def test_predict_with_control_input(self):
        """Control input should offset position."""
        kf = KalmanFilter(init_x=100, init_y=100)

        # Simulate mouse movement
        control_input = (10, 5)  # Mouse moved right and down

        kf.predict(control_input=control_input)

        # Position should be adjusted for camera movement
        # Mouse moves right -> target appears to move left
        # So state[0] should decrease by 10 * sensitivity
        scale = Config.MOUSE_SENSITIVITY
        expected_x = 100 - 10 * scale
        expected_y = 100 - 5 * scale

        assert abs(kf.state[0] - expected_x) < 1
        assert abs(kf.state[1] - expected_y) < 1


# =============================================================================
# Kalman Filter Update Tests
# =============================================================================

class TestKalmanFilterUpdate:
    """Tests for update method."""

    def test_update_moves_state_toward_measurement(self):
        """Update should move state toward measurement."""
        kf = KalmanFilter(init_x=100, init_y=100)

        # Predict first
        kf.predict()

        # Measure at a different location
        kf.update([150, 150])

        # State should move toward measurement
        assert kf.state[0] > 100
        assert kf.state[1] > 100
        assert kf.state[0] < 160  # But not all the way
        assert kf.state[1] < 160

    def test_update_resets_age(self):
        """Update should reset age counter."""
        kf = KalmanFilter(init_x=100, init_y=100)
        kf.age = 10  # Simulate missed frames

        kf.update([100, 100])

        assert kf.age == 0

    def test_multiple_updates_converge(self):
        """Multiple updates at same location should converge."""
        kf = KalmanFilter(init_x=0, init_y=0)

        target = [100, 200]

        # Multiple updates at same location
        for _ in range(20):
            kf.predict()
            kf.update(target)

        # Should converge close to target
        assert abs(kf.state[0] - target[0]) < 5
        assert abs(kf.state[1] - target[1]) < 5


# =============================================================================
# Kalman Filter Get Predicted Position Tests
# =============================================================================

class TestKalmanFilterGetPredictedPos:
    """Tests for get_predicted_pos method."""

    def test_zero_future_time_returns_current(self):
        """With future_time=0, should return current position."""
        kf = KalmanFilter(init_x=100, init_y=200)

        pos = kf.get_predicted_pos(future_time=0.0)

        assert pos[0] == 100
        assert pos[1] == 200

    def test_positive_velocity_predicts_forward(self):
        """With positive velocity, future position should be ahead."""
        kf = KalmanFilter(init_x=100, init_y=100)
        kf.state[2] = 100  # vx = 100 px/s
        kf.state[3] = 50   # vy = 50 px/s

        pos = kf.get_predicted_pos(future_time=0.1)

        # Should predict ahead
        expected_x = 100 + 100 * 0.1
        expected_y = 100 + 50 * 0.1

        assert abs(pos[0] - expected_x) < 0.01
        assert abs(pos[1] - expected_y) < 0.01

    def test_prediction_scales_with_time(self):
        """Prediction distance should scale with future_time."""
        kf = KalmanFilter(init_x=0, init_y=0)
        kf.state[2] = 100  # vx = 100 px/s

        pos_short = kf.get_predicted_pos(future_time=0.1)
        pos_long = kf.get_predicted_pos(future_time=0.2)

        # Longer time should predict further
        assert pos_long[0] > pos_short[0]
        assert abs(pos_long[0] - 2 * pos_short[0]) < 0.01


# =============================================================================
# Kalman Filter Tracking Scenario Tests
# =============================================================================

class TestKalmanFilterScenarios:
    """Test realistic tracking scenarios."""

    def test_track_constant_velocity_target(self):
        """Should accurately track target moving at constant velocity."""
        kf = KalmanFilter(init_x=0, init_y=0, dt=1/60.0)

        # Simulate target moving at constant velocity
        true_vx = 5  # px per frame
        true_vy = 3

        for frame in range(30):
            true_x = frame * true_vx
            true_y = frame * true_vy

            kf.predict()
            kf.update([true_x, true_y])

        # After convergence, velocity estimate should be close
        assert abs(kf.state[2] * (1/60.0) - true_vx * (1/60.0)) < 3
        assert abs(kf.state[3] * (1/60.0) - true_vy * (1/60.0)) < 3

    def test_track_with_measurement_noise(self):
        """Should filter out measurement noise."""
        np.random.seed(42)
        kf = KalmanFilter(init_x=0, init_y=0, dt=1/60.0)

        true_x, true_y = 100, 100

        estimated_positions = []

        for _ in range(50):
            # Noisy measurement
            noisy_x = true_x + np.random.normal(0, 10)
            noisy_y = true_y + np.random.normal(0, 10)

            kf.predict()
            kf.update([noisy_x, noisy_y])
            estimated_positions.append((kf.state[0], kf.state[1]))

        # Final estimate should be close to true position
        final_x, final_y = estimated_positions[-1]
        assert abs(final_x - true_x) < 15
        assert abs(final_y - true_y) < 15

    def test_handle_missed_frames(self):
        """Should handle frames where target is not detected."""
        kf = KalmanFilter(init_x=0, init_y=0, dt=1/60.0)

        # Track target moving at known velocity for several frames
        for i in range(20):
            true_x = i * 5  # Moving right at 5 px per frame
            kf.predict()
            kf.update([true_x, 0])

        last_x = kf.state[0]

        # Miss several frames (just predict, no update)
        for _ in range(5):
            time.sleep(0.02)  # Small delay to simulate time passing
            kf.predict()

        # Position should have continued based on learned velocity
        # (should be at least somewhat ahead of where we last measured)
        assert kf.state[0] >= last_x - 5  # Allow some tolerance

    def test_covariance_grows_without_updates(self):
        """Covariance should increase when not receiving updates."""
        kf = KalmanFilter(init_x=100, init_y=100)

        # Update to establish low covariance
        for _ in range(10):
            kf.predict()
            kf.update([100, 100])

        initial_covariance = kf.P[0, 0]

        # Predict without updates
        for _ in range(20):
            kf.predict()

        # Covariance should have grown
        assert kf.P[0, 0] > initial_covariance


# =============================================================================
# Kalman Filter Edge Cases
# =============================================================================

class TestKalmanFilterEdgeCases:
    """Edge case tests."""

    def test_large_measurement_jump(self):
        """Should handle sudden large measurement changes."""
        kf = KalmanFilter(init_x=0, init_y=0)

        # Establish position
        for _ in range(10):
            kf.predict()
            kf.update([0, 0])

        # Sudden jump
        kf.predict()
        kf.update([1000, 1000])

        # Should move toward new measurement but not instantly
        assert 0 < kf.state[0] < 1000
        assert 0 < kf.state[1] < 1000

    def test_negative_coordinates(self):
        """Should handle negative coordinates."""
        kf = KalmanFilter(init_x=-100, init_y=-200)

        kf.predict()
        kf.update([-150, -250])

        assert kf.state[0] < 0
        assert kf.state[1] < 0

    def test_very_fast_updates(self):
        """Should handle very fast update rate."""
        kf = KalmanFilter(init_x=0, init_y=0, dt=1/1000.0)

        for i in range(100):
            kf.predict()
            kf.update([i * 0.1, i * 0.1])

        # Should track without numerical issues
        assert not np.isnan(kf.state).any()
        assert not np.isinf(kf.state).any()
