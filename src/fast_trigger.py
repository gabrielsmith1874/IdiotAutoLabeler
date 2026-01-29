"""
Fast Trigger Bot
================
Minimal, high-performance trigger bot using DXCam and enemy_segmentation model.
Captures a small 128x128 region around screen center for maximum speed.

Usage:
    python src/fast_trigger.py
    python src/fast_trigger.py --preview   # With visual preview

Hold Mouse5 to activate - clicks when crosshair is on enemy mask.
Press Ctrl+C or Q (in preview) to exit.
"""

import argparse
import ctypes
import ctypes.wintypes
import time
from collections import defaultdict
from pathlib import Path

import cv2
import dxcam
import numpy as np
import torch
import torch.nn.functional as F

# TensorRT support
TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    trt = None


# =============================================================================
# SendInput Structures
# =============================================================================

PUL = ctypes.POINTER(ctypes.c_ulong)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT)
    ]


MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
VK_XBUTTON1 = 0x05  # Mouse4
VK_XBUTTON2 = 0x06  # Mouse5
VK_F3 = 0x72  # F3 key
VK_CAPITAL = 0x14  # Capslock

# Toggle key mapping
TOGGLE_KEY_MAP = {
    "capslock": VK_CAPITAL,
    "f3": VK_F3,
}


# =============================================================================
# Configuration
# =============================================================================

class Config:
    MODEL_PATH = Path("models/enemy_segmentation.pth")  # Change to stdc1_segmentation.pth for STDC1-Seg
    CAPTURE_SIZE = 256        # Capture region around cursor
    INFERENCE_SIZE = 256      # Must match training size (model trained on 256x256)
    THRESHOLD = 0.8          # Detection threshold
    TRIGGER_DEPTH_PERCENTAGE = 0.01  # Percentage of mask size for trigger depth (1%)
    LEAVE_BUFFER = 0.05        # Seconds to wait before releasing after leaving hitbox
    FOLLOW_CURSOR = False     # Whether to capture around cursor or screen center

    HOLD_DURATION = 0.005     # Click hold duration

    AIMBOT_ENABLED = True
    AIMBOT_GAIN = 0.45
    AIMBOT_MAX_STEP = 15
    AIMBOT_MIN_AREA = 30
    HEADSHOT_OFFSET = 0.37  # Percentage of component height to offset upward for headshots (20%)
    TOGGLE_KEY = "capslock"  # Toggle key: "capslock", "f3", or any key name


# =============================================================================
# TensorRT Inference Engine
# =============================================================================

class TensorRTEngine:
    """Wrapper for TensorRT inference engine."""

    def __init__(self, engine_path: Path, device: torch.device):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Install tensorrt package.")

        self.device = device
        self.engine_path = engine_path

        # Load engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get tensor info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        # Pre-allocate output buffer
        self.output_buffer = torch.zeros(
            self.output_shape,
            device=device,
            dtype=torch.float32
        )

        # Create CUDA stream
        self.stream = torch.cuda.Stream()

        print(f"TensorRT engine loaded successfully")
        print(f"  Input: {self.input_name} {list(self.input_shape)}")
        print(f"  Output: {self.output_name} {list(self.output_shape)}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on input tensor."""
        # Ensure input is float32 and contiguous
        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_buffer.data_ptr())

        # Execute
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return self.output_buffer

    def half(self):
        """No-op for compatibility with PyTorch model interface."""
        return self

    def eval(self):
        """No-op for compatibility with PyTorch model interface."""
        return self


# =============================================================================
# Model Loading
# =============================================================================

def get_tensorrt_engine_path(model_path: Path, inference_size: int) -> Path:
    """Get the expected TensorRT engine path for a given model."""
    return model_path.parent / f"{model_path.stem}_{inference_size}.engine"


def load_model(device: torch.device, use_tensorrt: bool = True, use_compile: bool = True):
    """Load the segmentation model.

    Priority order:
    1. TensorRT engine (if available and engine exists)
    2. torch.compile optimized PyTorch model
    3. Standard PyTorch model
    """

    # Check for TensorRT engine first
    engine_path = get_tensorrt_engine_path(Config.MODEL_PATH, Config.INFERENCE_SIZE)

    if use_tensorrt and TENSORRT_AVAILABLE and engine_path.exists():
        try:
            model = TensorRTEngine(engine_path, device)
            return model, "tensorrt"
        except Exception as e:
            print(f"Warning: Failed to load TensorRT engine: {e}")
            print("Falling back to PyTorch model...")
    elif use_tensorrt and TENSORRT_AVAILABLE and not engine_path.exists():
        print(f"TensorRT engine not found: {engine_path}")
        print(f"Run 'python src/export_tensorrt.py --size {Config.INFERENCE_SIZE}' to create it.")
        print("Falling back to PyTorch model...")
    elif use_tensorrt and not TENSORRT_AVAILABLE:
        print("TensorRT not available. Using PyTorch model.")

    # Fall back to PyTorch model
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_segmentation import EfficientNetUNet, STDC1Seg

    checkpoint = torch.load(Config.MODEL_PATH, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    encoder_name = config.get("encoder_name", "efficientnet_b3")

    if encoder_name == "stdc1":
        model = STDC1Seg(pretrained=False)
    else:
        model = EfficientNetUNet(encoder_name=encoder_name, pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Use FP16 for speed on CUDA
    if device.type == "cuda":
        model = model.half()

    backend = "pytorch"

    # Try to use torch.compile for optimization
    if use_compile and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            print("Applying torch.compile optimization...")
            model = torch.compile(model, mode="reduce-overhead", backend="inductor")
            backend = "compiled"
            print("torch.compile applied successfully")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
            print("Using standard PyTorch model...")

    print(f"Loaded PyTorch model ({encoder_name}) - Dice: {checkpoint['best_dice']:.4f}")
    return model, backend


# =============================================================================
# Trigger Bot
# =============================================================================

class Profiler:
    """Simple profiler to track timing of operations."""
    def __init__(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.current_start = None
        self.current_op = None

    def start(self, op_name):
        """Start timing an operation."""
        self.current_op = op_name
        self.current_start = time.perf_counter()

    def end(self):
        """End timing the current operation."""
        if self.current_op and self.current_start:
            elapsed = time.perf_counter() - self.current_start
            self.timings[self.current_op] += elapsed
            self.counts[self.current_op] += 1
            self.current_op = None
            self.current_start = None

    def get_stats(self, total_time):
        """Get statistics for all operations."""
        stats = []
        for op in sorted(self.timings.keys(), key=lambda x: self.timings[x], reverse=True):
            t = self.timings[op]
            c = self.counts[op]
            avg = t / c if c > 0 else 0
            pct = (t / total_time * 100) if total_time > 0 else 0
            stats.append((op, t, c, avg, pct))
        return stats

    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self.counts.clear()


class FastTrigger:
    def __init__(self, use_tensorrt: bool = True, use_compile: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Screen info
        user32 = ctypes.windll.user32
        self.screen_w = user32.GetSystemMetrics(0)
        self.screen_h = user32.GetSystemMetrics(1)

        # Capture region setup
        self.half = Config.CAPTURE_SIZE // 2
        self.center_x = self.screen_w // 2
        self.center_y = self.screen_h // 2
        # Default region at screen center (used when not following cursor)
        self.default_region = (
            self.center_x - self.half,
            self.center_y - self.half,
            self.center_x + self.half,
            self.center_y + self.half
        )

        # DXCam
        self.camera = dxcam.create(output_color="RGB")

        # Load model (TensorRT, compiled PyTorch, or standard PyTorch)
        self.model, self.backend = load_model(self.device, use_tensorrt=use_tensorrt, use_compile=use_compile)
        self.is_tensorrt = (self.backend == "tensorrt")

        # Normalization tensors (use float32 for TensorRT, half for PyTorch CUDA)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        if self.device.type == "cuda" and not self.is_tensorrt:
            self.mean = self.mean.half()
            self.std = self.std.half()

        # State
        self.active = False  # Toggle state (on/off)
        self.last_f3_state = False  # Track previous F3 state for toggle detection
        self.last_mouse5_state = False  # Track previous Mouse5 state for aimbot
        self.last_aimbot_time = 0  # Last time aimbot was run (for throttling)
        self.holding_click = False  # Whether click is being held
        self.we_clicked = False  # Whether WE sent the mouse down
        self.user_clicked_first = False  # Whether user was clicking before we synced
        self.click_cooldown = 0  # Cooldown before we can sync after user release
        self.left_hitbox_time = 0  # Timestamp when left hitbox
        self.trigger_count = 0
        self.running = True

        # Profiler
        self.profiler = Profiler()
        self.profiler_enabled = True

        # Warmup
        self._warmup()

        print(f"Screen: {self.screen_w}x{self.screen_h}")
        cursor_mode = "following cursor" if Config.FOLLOW_CURSOR else "at screen center"
        print(f"Capture: {Config.CAPTURE_SIZE}x{Config.CAPTURE_SIZE} {cursor_mode}")
        print(f"Inference: {Config.INFERENCE_SIZE}x{Config.INFERENCE_SIZE}")
        backend_display = {"tensorrt": "TensorRT", "compiled": "PyTorch (compiled)", "pytorch": "PyTorch"}
        print(f"Device: {self.device} ({backend_display.get(self.backend, self.backend)})")
        print("-" * 40)
        print("Press F3 to toggle trigger")
        print("Hold Mouse5 for aimbot")
        print("Press Ctrl+C to exit")
        print("-" * 40)
    
    def _warmup(self):
        """Warmup model and capture."""
        dummy = torch.randn(1, 3, Config.INFERENCE_SIZE, Config.INFERENCE_SIZE, device=self.device)
        # TensorRT uses float32, PyTorch CUDA uses half
        if self.device.type == "cuda" and not self.is_tensorrt:
            dummy = dummy.half()
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Warmup capture
        for _ in range(3):
            self.camera.grab(region=self.default_region)
    
    def check_toggle(self):
        """Check if toggle key was pressed and toggle active state."""
        toggle_key = TOGGLE_KEY_MAP.get(Config.TOGGLE_KEY.lower(), VK_F3)
        toggle_current = ctypes.windll.user32.GetAsyncKeyState(toggle_key) & 0x8000 != 0

        # Detect rising edge (was False, now True)
        if toggle_current and not self.last_f3_state:
            self.active = not self.active
            print(f"Toggle: {'ON' if self.active else 'OFF'} (Key: {Config.TOGGLE_KEY})")
        self.last_f3_state = toggle_current
    
    def is_active(self) -> bool:
        """Check if trigger is active (toggle state)."""
        return self.active
    
    def _find_target_centroid(self, mask: np.ndarray) -> tuple:
        """Find target centroid in mask using connected components.

        Returns (cy, cx) in mask coordinates, or None.
        """
        bin_mask = (mask > (Config.THRESHOLD * 255)).astype(np.uint8)
        if bin_mask.max() == 0:
            return None

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        if num <= 1:
            return None

        h, w = mask.shape
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        best = None
        best_score = None
        best_idx = None
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < Config.AIMBOT_MIN_AREA:
                continue
            c = centroids[i]
            # Score: prefer closer to center, break ties by larger area
            dist = float(np.linalg.norm(np.array([c[0], c[1]], dtype=np.float32) - center))
            score = dist - (area * 0.0005)
            if best_score is None or score < best_score:
                best_score = score
                best = c
                best_idx = i

        if best is None:
            return None

        cx, cy = best  # centroid is (x, y)
        
        # Adjust to aim at upper portion (head area)
        # Calculate offset from centroid based on component height
        component_height = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
        head_offset = int(component_height * Config.HEADSHOT_OFFSET)
        cy = cy - head_offset  # Move up from centroid
        
        return int(round(cy)), int(round(cx))

    def _mouse_move_relative(self, dx: int, dy: int):
        """Move mouse by dx/dy relative pixels using SendInput."""
        if dx == 0 and dy == 0:
            return
        extra = ctypes.c_ulong(0)
        ii_ = INPUT(0, MOUSEINPUT(ctypes.c_long(dx), ctypes.c_long(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra)))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))

    def check_aimbot(self, mask: np.ndarray):
        """While Mouse5 is held, continuously pull aim toward nearest target centroid."""
        if not Config.AIMBOT_ENABLED:
            return

        mouse5_current = ctypes.windll.user32.GetAsyncKeyState(VK_XBUTTON2) & 0x8000 != 0
        if not mouse5_current:
            self.last_mouse5_state = False
            return

        # Throttle aimbot updates to reduce CPU load
        now = time.perf_counter()
        if (now - self.last_aimbot_time) < 0.016:  # ~60 FPS max
            return
        self.last_aimbot_time = now

        target = self._find_target_centroid(mask)
        if target is None:
            if not self.last_mouse5_state:
                print("Aimbot: No mask found")
            self.last_mouse5_state = True
            return

        ty, tx = target
        h, w = mask.shape

        # Convert local mask coordinates to absolute screen coordinates
        region = self.get_capture_region()
        region_x1, region_y1 = region[0], region[1]
        target_screen_x = region_x1 + tx
        target_screen_y = region_y1 + ty

        # Calculate actual screen center
        screen_center_x = self.screen_w // 2
        screen_center_y = self.screen_h // 2

        # Calculate distance from actual screen center
        dist = np.sqrt((target_screen_x - screen_center_x) ** 2 + (target_screen_y - screen_center_y) ** 2)

        # Calculate deadzone based on mask size (same as trigger depth)
        rows, cols = np.where(mask > (Config.THRESHOLD * 255))
        if len(rows) > 0:
            min_y, max_y = rows.min(), rows.max()
            min_x, max_x = cols.min(), cols.max()
            mask_h = max_y - min_y + 1
            mask_w = max_x - min_x + 1
            mask_size = min(mask_h, mask_w)
            deadzone = max(1, int(mask_size * Config.TRIGGER_DEPTH_PERCENTAGE))
        else:
            deadzone = 1

        # Don't move if target is within deadzone (already in triggerable region)
        if dist < deadzone:
            return

        err_x = target_screen_x - screen_center_x
        err_y = target_screen_y - screen_center_y

        dx = int(round(err_x * Config.AIMBOT_GAIN))
        dy = int(round(err_y * Config.AIMBOT_GAIN))

        max_step = int(Config.AIMBOT_MAX_STEP)
        dx = max(-max_step, min(max_step, dx))
        dy = max(-max_step, min(max_step, dy))

        self._mouse_move_relative(dx, dy)
        self.last_mouse5_state = True
    
    def get_capture_region(self) -> tuple:
        """Get capture region - follows cursor if enabled, else screen center."""
        if Config.FOLLOW_CURSOR:
            # Get current cursor position
            point = ctypes.wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            cx, cy = point.x, point.y

            # Calculate region bounds
            x1 = cx - self.half
            y1 = cy - self.half
            x2 = cx + self.half
            y2 = cy + self.half

            # Clamp to screen dimensions
            x1 = max(0, min(x1, self.screen_w))
            y1 = max(0, min(y1, self.screen_h))
            x2 = max(0, min(x2, self.screen_w))
            y2 = max(0, min(y2, self.screen_h))

            # Validate region size. If invalid (offline/collapsed), fallback to default.
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                return self.default_region

            return (x1, y1, x2, y2)
        return self.default_region

    def capture(self) -> np.ndarray:
        """Capture screen region around cursor."""
        region = self.get_capture_region()
        frame = self.camera.grab(region=region)
        if frame is None:
            frame = self.camera.grab(region=region)
        if frame is None:
            return np.zeros((Config.CAPTURE_SIZE, Config.CAPTURE_SIZE, 3), dtype=np.uint8)
        return frame
    
    @torch.no_grad()
    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Run inference. Returns mask."""
        # Normalize in NumPy first (works correctly with TensorRT)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = frame.astype(np.float32) / 255.0
        img_norm = (img_norm - mean) / std

        # Convert to tensor: HWC -> CHW -> NCHW
        img = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)

        # Resize to model's expected input size
        if img.shape[2] != Config.INFERENCE_SIZE or img.shape[3] != Config.INFERENCE_SIZE:
            img = F.interpolate(img, size=(Config.INFERENCE_SIZE, Config.INFERENCE_SIZE),
                                mode='bilinear', align_corners=False)

        # TensorRT uses float32, PyTorch CUDA uses half
        if self.device.type == "cuda" and not self.is_tensorrt:
            img = img.half()

        # Inference
        output = self.model(img)
        pred = torch.sigmoid(output.float())

        # Resize mask back to capture size for trigger checking
        if pred.shape[2] != Config.CAPTURE_SIZE or pred.shape[3] != Config.CAPTURE_SIZE:
            pred = F.interpolate(pred, size=(Config.CAPTURE_SIZE, Config.CAPTURE_SIZE),
                                 mode='bilinear', align_corners=False)

        # Get mask
        mask = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
        return mask
    
    def check_trigger(self, mask: np.ndarray) -> bool:
        """Check if center region is deep inside mask. Returns True if in hitbox."""
        h, w = mask.shape
        cy, cx = h // 2, w // 2
        
        # Calculate mask bounding box to determine size
        rows, cols = np.where(mask > (Config.THRESHOLD * 255))
        if len(rows) > 0:
            min_y, max_y = rows.min(), rows.max()
            min_x, max_x = cols.min(), cols.max()
            mask_h = max_y - min_y + 1
            mask_w = max_x - min_x + 1
            # Use smaller dimension for percentage calculation
            mask_size = min(mask_h, mask_w)
            # Calculate depth as percentage of mask size (minimum 1 pixel)
            depth = max(1, int(mask_size * Config.TRIGGER_DEPTH_PERCENTAGE))
        else:
            # No mask detected, default to 1 pixel
            depth = 1
        
        # Check region around center - all pixels must be above threshold
        y1 = max(0, cy - depth)
        y2 = min(h, cy + depth + 1)
        x1 = max(0, cx - depth)
        x2 = min(w, cx + depth + 1)
        
        region = mask[y1:y2, x1:x2]
        if region.size == 0:
            return False
        
        # Use minimum value - all pixels must be above threshold
        min_val = region.min() / 255.0
        
        return min_val > Config.THRESHOLD
    
    def update_trigger_state(self, in_hitbox: bool):
        """Update trigger state based on whether center is in hitbox.

        Only manages clicks that WE send. If user is clicking manually, we don't interfere.
        """
        now = time.perf_counter()

        # Check if user is physically holding left mouse button
        user_holding = ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000 != 0

        # Update cooldown
        if self.click_cooldown > 0:
            self.click_cooldown -= 1

        if in_hitbox:
            # In hitbox - we should be clicking
            if not self.holding_click:
                # Not holding - decide who clicks
                if not user_holding and self.click_cooldown == 0:
                    # User not clicking and cooldown over - we click
                    self._mouse_down()
                    self.holding_click = True
                    self.we_clicked = True
                    self.user_clicked_first = False
                    self.trigger_count += 1
                elif user_holding:
                    # User clicking - sync state
                    self.holding_click = True
                    self.we_clicked = False
                    self.user_clicked_first = True
            elif not user_holding and self.user_clicked_first:
                # User released their click - set cooldown
                self.holding_click = False
                self.user_clicked_first = False
                self.click_cooldown = 10
            self.left_hitbox_time = 0
        else:
            # Not in hitbox
            if self.holding_click:
                # If WE clicked, release after buffer
                if self.we_clicked:
                    if self.left_hitbox_time == 0:
                        self.left_hitbox_time = now
                    elif (now - self.left_hitbox_time) > Config.LEAVE_BUFFER:
                        self._mouse_up()
                        self.holding_click = False
                        self.we_clicked = False
                        self.user_clicked_first = False
                        self.click_cooldown = 10  # Cooldown to prevent immediate re-clicking
                        self.left_hitbox_time = 0
                else:
                    # User was clicking - just reset (don't release, user handles it)
                    self.holding_click = False
                    self.user_clicked_first = False
                    self.left_hitbox_time = 0
    
    def _mouse_down(self):
        """Send mouse down event."""
        extra = ctypes.c_ulong(0)
        ii_ = INPUT(0, MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra)))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))
    
    def _mouse_up(self):
        """Send mouse up event."""
        extra = ctypes.c_ulong(0)
        ii_ = INPUT(0, MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra)))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))
    
    def _click(self):
        """Hold-click using SendInput."""
        extra = ctypes.c_ulong(0)
        
        # Mouse down
        ii_ = INPUT(0, MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra)))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))
        
        time.sleep(Config.HOLD_DURATION)
        
        # Mouse up
        ii_ = INPUT(0, MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra)))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_), ctypes.sizeof(ii_))
    
    def run(self):
        """Main loop (no preview)."""
        fps_times = []
        last_status = 0
        loop_count = 0

        try:
            while self.running:
                start = time.perf_counter()

                # Check toggle state
                self.profiler.start("check_toggle")
                self.check_toggle()
                self.profiler.end()

                # Check aimbot (always runs regardless of trigger state)
                self.profiler.start("capture")
                frame = self.capture()
                self.profiler.end()

                self.profiler.start("infer")
                mask = self.infer(frame)
                self.profiler.end()

                self.profiler.start("check_aimbot")
                self.check_aimbot(mask)
                self.profiler.end()

                # Only process trigger when active
                if self.is_active():
                    self.profiler.start("check_trigger")
                    in_hitbox = self.check_trigger(mask)
                    self.profiler.end()

                    self.profiler.start("update_trigger_state")
                    self.update_trigger_state(in_hitbox)
                    self.profiler.end()
                else:
                    # Sleep briefly when not active to save CPU
                    time.sleep(0.001)
                    if self.holding_click:
                        # Release click if deactivated while holding
                        self._mouse_up()
                        self.holding_click = False
                        self.left_hitbox_time = 0

                # FPS tracking
                elapsed = time.perf_counter() - start
                if elapsed > 0:
                    fps_times.append(1.0 / elapsed)
                    if len(fps_times) > 60:
                        fps_times.pop(0)

                # Status update every second
                now = time.perf_counter()
                if now - last_status > 1.0:
                    if fps_times:
                        avg_fps = sum(fps_times) / len(fps_times)
                        active = "ACTIVE" if self.is_active() else "idle"

                        # Print profiling stats every 5 seconds
                        if self.profiler_enabled and loop_count % 5 == 0:
                            total_time = sum(self.profiler.timings.values())
                            stats = self.profiler.get_stats(total_time)
                            print(f"\n{'='*60}")
                            print(f"FPS: {avg_fps:.0f} | Triggers: {self.trigger_count} | {active}")
                            print(f"{'='*60}")
                            print(f"{'Operation':<20} {'Total (ms)':<12} {'Avg (ms)':<10} {'%':<6} {'Count'}")
                            print(f"{'-'*60}")
                            for op, t, c, avg, pct in stats:
                                print(f"{op:<20} {t*1000:<12.2f} {avg*1000:<10.3f} {pct:<6.1f} {c}")
                            print(f"{'-'*60}")
                            self.profiler.reset()
                        else:
                            print(f"\rFPS: {avg_fps:.0f} | Triggers: {self.trigger_count} | {active}    ", end="", flush=True)
                    last_status = now
                    loop_count += 1

        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        finally:
            self.cleanup()
    
    def run_with_preview(self):
        """Main loop with visual preview window."""
        PREVIEW_SCALE = 2  # Reduced scale for faster rendering
        WINDOW_NAME = "Fast Trigger Preview"
        PREVIEW_SKIP = 2  # Render preview every N frames for higher FPS

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, Config.CAPTURE_SIZE * PREVIEW_SCALE, Config.CAPTURE_SIZE * PREVIEW_SCALE)

        fps_times = []
        loop_count = 0
        frame_count = 0
        last_preview = None

        # Pre-allocate overlay colors
        magenta = np.array([255, 0, 255], dtype=np.float32) * 0.6
        cyan = np.array([0, 255, 200], dtype=np.float32) * 0.6

        print("Preview window opened. Press 'Q' to quit.")

        try:
            while self.running:
                start = time.perf_counter()

                # Always capture for preview
                self.profiler.start("capture")
                frame = self.capture()
                self.profiler.end()

                self.profiler.start("infer")
                mask = self.infer(frame)
                self.profiler.end()

                # Check toggle state
                self.profiler.start("check_toggle")
                self.check_toggle()
                self.profiler.end()

                # Check trigger only when active
                in_hitbox = False
                if self.is_active():
                    self.profiler.start("check_trigger")
                    in_hitbox = self.check_trigger(mask)
                    self.profiler.end()

                    self.profiler.start("update_trigger_state")
                    self.update_trigger_state(in_hitbox)
                    self.profiler.end()

                    self.profiler.start("check_aimbot")
                    self.check_aimbot(mask)
                    self.profiler.end()
                else:
                    # Release click if deactivated while holding
                    if self.holding_click:
                        self._mouse_up()
                        self.holding_click = False
                        self.left_hitbox_time = 0

                # Create preview image (skip frames for performance)
                self.profiler.start("preview_render")
                frame_count += 1

                if frame_count % PREVIEW_SKIP == 0 or last_preview is None:
                    h, w = mask.shape
                    threshold_val = int(Config.THRESHOLD * 255)

                    # Fast mask thresholding
                    mask_bool = mask > threshold_val

                    # Simple overlay without erosion for speed
                    preview = frame.copy()
                    if mask_bool.any():
                        # Direct color blend using numpy broadcasting
                        preview[mask_bool] = (preview[mask_bool].astype(np.float32) * 0.4 + cyan).astype(np.uint8)

                    # Scale up with fast interpolation
                    preview = cv2.resize(preview, (w * PREVIEW_SCALE, h * PREVIEW_SCALE), interpolation=cv2.INTER_NEAREST)
                    last_preview = preview
                else:
                    preview = last_preview

                # FPS text
                elapsed = time.perf_counter() - start
                if elapsed > 0:
                    fps_times.append(1.0 / elapsed)
                    if len(fps_times) > 30:
                        fps_times.pop(0)

                avg_fps = sum(fps_times) / len(fps_times) if fps_times else 0
                active = "ACTIVE" if self.is_active() else "idle"
                status = f"FPS: {avg_fps:.0f} | Triggers: {self.trigger_count} | {active}"
                cv2.putText(preview, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Convert RGB to BGR for OpenCV
                preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
                cv2.imshow(WINDOW_NAME, preview_bgr)
                self.profiler.end()

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nQuit requested...")
                    break

                # Print profiling stats every 5 seconds
                if self.profiler_enabled and loop_count % 300 == 0:  # ~5 seconds at 60fps
                    total_time = sum(self.profiler.timings.values())
                    stats = self.profiler.get_stats(total_time)
                    print(f"\n{'='*60}")
                    print(f"FPS: {avg_fps:.0f} | Triggers: {self.trigger_count} | {active}")
                    print(f"{'='*60}")
                    print(f"{'Operation':<20} {'Total (ms)':<12} {'Avg (ms)':<10} {'%':<6} {'Count'}")
                    print(f"{'-'*60}")
                    for op, t, c, avg, pct in stats:
                        print(f"{op:<20} {t*1000:<12.2f} {avg*1000:<10.3f} {pct:<6.1f} {c}")
                    print(f"{'-'*60}")
                    self.profiler.reset()

                loop_count += 1

        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
        finally:
            cv2.destroyAllWindows()
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        del self.camera
        print(f"Total triggers: {self.trigger_count}")


def run_key_check():
    """Debug utility to check which keys are being pressed."""
    print("=" * 40)
    print("Mouse Button Check Mode")
    print("Press Ctrl+C to exit")
    print("=" * 40)
    
    seen_states = set()
    
    try:
        while True:
            m1 = ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000  # Left
            m2 = ctypes.windll.user32.GetAsyncKeyState(0x02) & 0x8000  # Right
            m3 = ctypes.windll.user32.GetAsyncKeyState(0x04) & 0x8000  # Middle
            m4 = ctypes.windll.user32.GetAsyncKeyState(VK_XBUTTON1) & 0x8000 # X1 / Back
            m5 = ctypes.windll.user32.GetAsyncKeyState(VK_XBUTTON2) & 0x8000 # X2 / Forward
            
            state = []
            if m1: state.append("Left")
            if m2: state.append("Right")
            if m3: state.append("Middle")
            if m4: state.append("Mouse4")
            if m5: state.append("Mouse5")
            
            if state:
                msg = " + ".join(state)
                # Only print if state changed to avoid spamming
                if msg not in seen_states:
                    print(f"Detected: {msg}")
                    seen_states.clear() # Clear so we print again if they release and press again? 
                    # actually let's just print unique states to not flood
                    seen_states.add(msg) 
                # hack to clear it after a bit so it prints again?
                # simpler: just print line with carriage return if we want realtime
                print(f"\rHeld: {msg:<30}", end="", flush=True)
            else:
                print(f"\rHeld: {'None':<30}", end="", flush=True)
                seen_states.clear()
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nExiting key check...")


def main():
    parser = argparse.ArgumentParser(description="Fast Trigger Bot")
    parser.add_argument("--preview", "-p", action="store_true", help="Show visual preview window")
    parser.add_argument("--detect-keys", "-d", action="store_true", help="Run key detection debug mode")
    parser.add_argument("--no-tensorrt", action="store_true", help="Disable TensorRT engine loading")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile optimization")
    args = parser.parse_args()

    if args.detect_keys:
        run_key_check()
        return

    print("=" * 50)
    print("Fast Trigger Bot")
    print("=" * 50)

    # Print TensorRT availability status
    if TENSORRT_AVAILABLE:
        print(f"TensorRT: Available (v{trt.__version__})")
    else:
        print("TensorRT: Not available")

    # Print torch.compile availability
    if hasattr(torch, "compile"):
        print("torch.compile: Available")
    else:
        print("torch.compile: Not available")

    if not Config.MODEL_PATH.exists():
        print(f"ERROR: Model not found: {Config.MODEL_PATH}")
        return

    trigger = FastTrigger(use_tensorrt=not args.no_tensorrt, use_compile=not args.no_compile)

    if args.preview:
        trigger.run_with_preview()
    else:
        trigger.run()


if __name__ == "__main__":
    main()

