"""
Run Inference on Unlabeled Images
=================================
Load the trained segmentation model and create masked images for raw/unlabeled images.

Supports multiple model types:
- efficientnet: EfficientNet-UNet (default)
- yolov8: YOLOv8-Seg
- sam: Segment Anything Model (SAM 1)
- sam2: Segment Anything Model 2 (Hiera backbone)

Usage:
    python src/inference.py --model efficientnet
    python src/inference.py --model yolov8
    python src/inference.py --model sam
    python src/inference.py --model sam2
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Import model from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))

# TensorRT support
TENSORRT_AVAILABLE = False
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    trt = None


# =============================================================================
# TensorRT Engine Wrapper
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
# Model Loaders
# =============================================================================

def get_tensorrt_engine_path(model_path: Path, inference_size: int) -> Path:
    """Get the expected TensorRT engine path for a given model."""
    return model_path.parent / f"{model_path.stem}_{inference_size}.engine"

def load_efficientnet_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load trained EfficientNet-UNet model from checkpoint."""
    from train_segmentation import EfficientNetUNet, STDC1Seg

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get encoder name from checkpoint config
    config = checkpoint.get("config", {})
    encoder_name = config.get("encoder_name", "efficientnet_b3")

    # Handle STDC1 model (not a timm encoder)
    if encoder_name == "stdc1":
        model = STDC1Seg(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        print(f"Loaded STDC1-Seg model from epoch {checkpoint['epoch']} with Dice: {checkpoint['best_dice']:.4f}")
        return model

    model = EfficientNetUNet(encoder_name=encoder_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded EfficientNet model ({encoder_name}) from epoch {checkpoint['epoch']} with Dice: {checkpoint['best_dice']:.4f}")
    return model


def load_efficientnet_model_with_tensorrt(
    checkpoint_path: Path, 
    device: torch.device, 
    inference_size: int = 512,
    use_tensorrt: bool = True
) -> tuple:
    """Load EfficientNet model with TensorRT priority and PyTorch fallback.
    
    Priority order:
    1. TensorRT engine (if available and use_tensorrt=True)
    2. PyTorch model
    
    Returns:
        (model, backend) where backend is "tensorrt" or "pytorch"
    """
    # Check for TensorRT engine first
    engine_path = get_tensorrt_engine_path(checkpoint_path, inference_size)
    
    if use_tensorrt and TENSORRT_AVAILABLE and engine_path.exists():
        try:
            model = TensorRTEngine(engine_path, device)
            return model, "tensorrt"
        except Exception as e:
            print(f"Warning: Failed to load TensorRT engine: {e}")
            print("Falling back to PyTorch model...")
    elif use_tensorrt and TENSORRT_AVAILABLE and not engine_path.exists():
        print(f"TensorRT engine not found: {engine_path}")
        print(f"Run 'python src/export_tensorrt.py --size {inference_size}' to create it.")
        print("Falling back to PyTorch model...")
    elif use_tensorrt and not TENSORRT_AVAILABLE:
        print("TensorRT not available. Using PyTorch model.")
    
    # Fall back to PyTorch model
    model = load_efficientnet_model(checkpoint_path, device)
    return model, "pytorch"


def load_yolov8_model(checkpoint_path: Path, device: torch.device):
    """Load trained YOLOv8-Seg model."""
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")

    model = YOLO(str(checkpoint_path))
    print(f"Loaded YOLOv8-Seg model from: {checkpoint_path}")
    return model


def load_stdc1_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load trained STDC1-Seg model from checkpoint."""
    from train_segmentation import STDC1Seg

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = STDC1Seg(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded STDC1-Seg model from epoch {checkpoint['epoch']} with Dice: {checkpoint['best_dice']:.4f}")
    return model


def load_sam_model(checkpoint_path: Path, device: torch.device):
    """Load fine-tuned SAM model."""
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        raise ImportError(
            "Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    # Load checkpoint to get model type
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "vit_b")

    # Need base SAM checkpoint
    sam_checkpoints = {
        "vit_h": "models/sam_checkpoints/sam_vit_h_4b8939.pth",
        "vit_l": "models/sam_checkpoints/sam_vit_l_0b3195.pth",
        "vit_b": "models/sam_checkpoints/sam_vit_b_01ec64.pth",
    }

    base_checkpoint = Path(sam_checkpoints[model_type])
    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM base checkpoint not found: {base_checkpoint}\n"
            f"Run training first to download it, or download manually from:\n"
            f"https://dl.fbaipublicfiles.com/segment_anything/"
        )

    # Load base SAM model
    sam = sam_model_registry[model_type](checkpoint=str(base_checkpoint))

    # Load fine-tuned mask decoder weights
    sam.mask_decoder.load_state_dict(checkpoint["model_state_dict"])
    sam = sam.to(device)
    sam.eval()

    print(f"Loaded SAM {model_type} with fine-tuned decoder, Dice: {checkpoint['best_dice']:.4f}")
    return sam


def load_sam2_model(checkpoint_path: Path, device: torch.device):
    """Load fine-tuned SAM 2 model."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "Please install SAM 2: pip install git+https://github.com/facebookresearch/sam2.git"
        )

    # Load checkpoint to get model size
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_size = checkpoint.get("model_size", "large")

    # SAM 2 configs and checkpoints
    sam2_configs = {
        "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
        "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
        "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
        "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
    }

    config_file, ckpt_name = sam2_configs[model_size]
    base_checkpoint = Path(f"models/sam2_checkpoints/{ckpt_name}")

    if not base_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM 2 base checkpoint not found: {base_checkpoint}\n"
            f"Run training first to download it."
        )

    # Build SAM 2 model
    sam2_model = build_sam2(
        config_file=config_file,
        ckpt_path=str(base_checkpoint),
        device=device,
    )

    # Load fine-tuned mask decoder weights
    sam2_model.sam_mask_decoder.load_state_dict(checkpoint["model_state_dict"])
    sam2_model.eval()

    print(f"Loaded SAM 2 {model_size} with fine-tuned decoder, Dice: {checkpoint['best_dice']:.4f}")
    return SAM2ImagePredictor(sam2_model)


# =============================================================================
# Inference Functions
# =============================================================================

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for EfficientNet inference."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = image.astype(np.float32) / 255.0
    img = (img - mean) / std

    img = torch.from_numpy(img).permute(2, 0, 1).float()

    return img.unsqueeze(0)


def apply_mask_overlay(image: np.ndarray, mask: np.ndarray, color=(255, 0, 255), alpha=0.7) -> np.ndarray:
    """Apply magenta overlay on detected regions."""
    result = image.copy()

    mask_bool = mask > 0
    for c, col_val in enumerate(color):
        result[:, :, c] = np.where(
            mask_bool,
            (alpha * col_val + (1 - alpha) * result[:, :, c]).astype(np.uint8),
            result[:, :, c]
        )

    return result


@torch.no_grad()
def run_efficientnet_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.99,
    tile_size: int = 256,
    inference_size: int = 256,
    stride: int = 128,
    backend: str = "pytorch",
) -> np.ndarray:
    """Run tiled inference with EfficientNet-UNet.
    
    Splits image into overlapping tiles (matching training's 256x256 crop),
    runs inference on each tile resized to inference_size (default 256x256),
    then stitches results back together.
    
    Args:
        model: The segmentation model (PyTorch or TensorRT)
        image: Input image (H, W, 3)
        device: Torch device
        threshold: Prediction threshold
        tile_size: Size of each tile (default 256, matches training crop)
        inference_size: Size to resize tiles for inference (default 256)
        stride: Stride between tiles (default 128 for 50% overlap)
        backend: "tensorrt" or "pytorch"
    """
    h, w = image.shape[:2]
    
    # Accumulator for predictions (float) and count for averaging overlaps
    pred_accum = np.zeros((h, w), dtype=np.float32)
    count_accum = np.zeros((h, w), dtype=np.float32)
    
    # Normalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Determine dtype based on backend
    use_half = (backend == "pytorch" and device.type == "cuda")
    
    # Slide over the image with tiles
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Calculate tile boundaries
            y1 = y
            y2 = min(y + tile_size, h)
            x1 = x
            x2 = min(x + tile_size, w)
            
            # Extract tile
            tile = image[y1:y2, x1:x2].copy()
            tile_h, tile_w = tile.shape[:2]
            
            # Pad tile if it's smaller than tile_size (edge tiles)
            if tile_h < tile_size or tile_w < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                padded[:tile_h, :tile_w] = tile
                tile = padded
            
            # Resize to inference size (512x512)
            tile_resized = cv2.resize(tile, (inference_size, inference_size), interpolation=cv2.INTER_LINEAR)
            
            # Preprocess
            tile_norm = tile_resized.astype(np.float32) / 255.0
            tile_norm = (tile_norm - mean) / std
            input_tensor = torch.from_numpy(tile_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            # Convert to half precision for PyTorch CUDA (not for TensorRT)
            if use_half:
                input_tensor = input_tensor.half()
            
            # Inference
            output = model(input_tensor)
            pred = torch.sigmoid(output.float()).squeeze().cpu().numpy()
            
            # Resize prediction back to tile_size
            pred_tile = cv2.resize(pred, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
            
            # Crop back to actual tile size (remove padding)
            pred_tile = pred_tile[:tile_h, :tile_w]
            
            # Accumulate predictions
            pred_accum[y1:y2, x1:x2] = np.maximum(pred_accum[y1:y2, x1:x2], pred_tile)
            count_accum[y1:y2, x1:x2] += 1
    
    # Threshold to get binary mask
    mask = (pred_accum > threshold).astype(np.uint8)
    
    return mask


@torch.no_grad()
def run_stdc1_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run inference with STDC1-Seg."""
    original_size = image.shape[:2]

    input_tensor = preprocess_image(image).to(device)
    output = model(input_tensor)

    pred = torch.sigmoid(output)
    pred = F.interpolate(pred, size=original_size, mode="bilinear", align_corners=False)

    mask = (pred.squeeze().cpu().numpy() > threshold).astype(np.uint8)

    return mask


@torch.no_grad()
def run_yolov8_inference(
    model,
    image: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run inference with YOLOv8-Seg."""
    h, w = image.shape[:2]

    results = model.predict(image, verbose=False, conf=threshold)

    if not results or results[0].masks is None:
        return np.zeros((h, w), dtype=np.uint8)

    # Combine all predicted masks
    pred_masks = results[0].masks.data
    if len(pred_masks) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Combine all masks and resize to original
    combined = pred_masks.max(dim=0)[0].cpu().numpy()

    # Resize to original image size
    from PIL import Image as PILImage
    mask_resized = np.array(PILImage.fromarray((combined * 255).astype(np.uint8)).resize((w, h), PILImage.NEAREST))
    mask = (mask_resized > 127).astype(np.uint8)

    return mask


@torch.no_grad()
def run_sam_inference(
    model,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run inference with SAM using automatic grid prompts."""
    from segment_anything import SamPredictor

    predictor = SamPredictor(model)
    predictor.set_image(image)

    h, w = image.shape[:2]

    # Generate a grid of point prompts
    grid_size = 8
    points = []
    for y in range(grid_size):
        for x in range(grid_size):
            px = int((x + 0.5) * w / grid_size)
            py = int((y + 0.5) * h / grid_size)
            points.append([px, py])

    points = np.array(points)
    labels = np.ones(len(points))  # All foreground

    # Predict with all points
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )

    # Take the mask with highest score
    if len(masks) > 0:
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    return mask


@torch.no_grad()
def run_sam2_inference(
    predictor,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run inference with SAM 2 using automatic grid prompts."""
    predictor.set_image(image)

    h, w = image.shape[:2]

    # Generate a grid of point prompts
    grid_size = 8
    points = []
    for y in range(grid_size):
        for x in range(grid_size):
            px = int((x + 0.5) * w / grid_size)
            py = int((y + 0.5) * h / grid_size)
            points.append([px, py])

    points = np.array(points)
    labels = np.ones(len(points))  # All foreground

    # Predict with all points
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )

    # Take the mask with highest score
    if len(masks) > 0:
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    return mask


# =============================================================================
# Main
# =============================================================================

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run segmentation inference")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "stdc1", "yolov8", "sam", "sam2"],
        help="Model type to use (default: efficientnet)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/raw",
        help="Input directory with images (default: data/raw)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/auto_labeled",
        help="Output directory (default: data/auto_labeled)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.99,
        help="Prediction threshold (default: 0.99)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (uses default if not specified)"
    )
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        help="Disable TensorRT engine loading (use PyTorch only)"
    )
    parser.add_argument(
        "--inference-size",
        type=int,
        default=256,
        help="Inference size for TensorRT engine (default: 256)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Segmentation Inference ({args.model.upper()})")
    print("=" * 60)

    # Print TensorRT availability status
    if TENSORRT_AVAILABLE:
        print(f"TensorRT: Available (v{trt.__version__})")
    else:
        print("TensorRT: Not available")

    # Default checkpoint paths
    default_checkpoints = {
        "efficientnet": Path("models/enemy_segmentation.pth"),
        "stdc1": Path("models/enemy_segmentation.pth"),  # Same as efficientnet
        "yolov8": Path("models/enemy_segmentation_yolov8.pt"),
        "sam": Path("models/enemy_segmentation_sam.pth"),
        "sam2": Path("models/enemy_segmentation_sam2.pth"),
    }

    model_path = Path(args.checkpoint) if args.checkpoint else default_checkpoints[args.model]
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Check model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model based on type
    if args.model == "efficientnet":
        model, backend = load_efficientnet_model_with_tensorrt(
            model_path, 
            device, 
            inference_size=args.inference_size,
            use_tensorrt=not args.no_tensorrt
        )
        backend_display = {"tensorrt": "TensorRT", "pytorch": "PyTorch"}
        print(f"Backend: {backend_display.get(backend, backend)}")
        inference_fn = lambda img: run_efficientnet_inference(model, img, device, args.threshold, backend=backend)
    elif args.model == "stdc1":
        model = load_stdc1_model(model_path, device)
        inference_fn = lambda img: run_stdc1_inference(model, img, device, args.threshold)
    elif args.model == "yolov8":
        model = load_yolov8_model(model_path, device)
        inference_fn = lambda img: run_yolov8_inference(model, img, args.threshold)
    elif args.model == "sam":
        model = load_sam_model(model_path, device)
        inference_fn = lambda img: run_sam_inference(model, img, device, args.threshold)
    elif args.model == "sam2":
        model = load_sam2_model(model_path, device)
        inference_fn = lambda img: run_sam2_inference(model, img, device, args.threshold)

    # Find all unmasked images
    input_files = list(input_dir.glob("*_unmasked.png"))
    if not input_files:
        # Also try without _unmasked suffix
        input_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        input_files = [f for f in input_files if "_masked" not in f.name]

    print(f"\nFound {len(input_files)} images to process")

    if len(input_files) == 0:
        print("No images found!")
        return

    # Process each image
    processed = 0
    for img_path in tqdm(input_files, desc="Processing"):
        try:
            # Load image
            image = np.array(Image.open(img_path).convert("RGB"))

            # Run inference
            mask = inference_fn(image)

            # Apply overlay
            masked_image = apply_mask_overlay(image, mask)

            # Save masked image
            if "_unmasked" in img_path.name:
                output_name = img_path.name.replace("_unmasked", "_masked")
            else:
                output_name = img_path.stem + "_masked" + img_path.suffix

            output_path = output_dir / output_name
            Image.fromarray(masked_image).save(output_path)

            # Also save the original unmasked to output dir for comparison
            unmasked_output = output_dir / img_path.name
            if not unmasked_output.exists():
                Image.fromarray(image).save(unmasked_output)

            processed += 1
        except Exception as e:
            print(f"\nSkipping {img_path.name}: {e}")

    print(f"\n✓ Processed {len(input_files)} images")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
