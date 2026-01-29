"""
EfficientNet-UNet Segmentation Training
========================================
Train a model to segment enemies from game screenshots using masked/unmasked image pairs.

Usage:
    python src/train_segmentation.py
"""

import os
import logging
import warnings

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"  # Suppress albumentations update check
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Suppress HF progress bars

# Suppress timm pretrained weights warning (it uses logging, not warnings)
class TimmWarningFilter(logging.Filter):
    def filter(self, record):
        return "Unexpected keys" not in record.getMessage() and "unauthenticated" not in record.getMessage()

logging.getLogger("timm.models._builder").addFilter(TimmWarningFilter())
logging.getLogger("huggingface_hub").addFilter(TimmWarningFilter())

import re
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Try to import timm for EfficientNet encoder
try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")

# Try to import segmentation_models_pytorch for smp.Unet option
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

import cv2


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "data_dir": Path("data/labeled"),
    "model_save_dir": Path("models"),
    "image_size": 256,  # Match capture size - no resize needed at inference
    "batch_size": 8,    # Larger batch with smaller images
    "num_workers": 2,   # Parallel data loading
    "epochs": 1000,
    "lr": 3e-5,  # Conservative LR for stable fine-tuning
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "early_stopping_patience": 150,
    "seed": 42,
    "encoder_name": "efficientnet_b0",  # EfficientNet encoder
    "magenta_threshold": 30,  # Pixel difference threshold for mask detection
    # New training stability settings
    "grad_clip_norm": 1.0,  # Gradient clipping to prevent explosion
    "warmup_epochs": 5,  # LR warmup for stable start
    "inference_threshold": 0.8,  # Lower threshold for better recall (prefer false positives over false negatives)
}


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_mask(masked_img: np.ndarray, unmasked_img: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Extract binary mask by comparing masked and unmasked images.
    The mask is where the magenta overlay was applied.
    """
    # Compute absolute difference
    diff = np.abs(masked_img.astype(np.int16) - unmasked_img.astype(np.int16))
    
    # Sum across color channels
    diff_sum = diff.sum(axis=2)
    
    # Threshold to get binary mask
    mask = (diff_sum > threshold).astype(np.uint8)
    
    return mask


def find_image_pairs(data_dir: Path, filter_negatives: bool = False) -> List[Tuple[Path, Path]]:
    """
    Find valid masked/unmasked image pairs.
    Returns list of (unmasked_path, masked_path) tuples.
    
    Args:
        data_dir: Directory containing image pairs
        filter_negatives: If True, exclude pairs where images are identical (no mask)
    """
    pairs = []
    skipped = 0
    positives = 0
    negatives = 0
    
    # Match any file ending in _unmasked.png
    for file in sorted(data_dir.glob("*_unmasked.png")):
        # Construct corresponding masked filename
        masked_name = file.name.replace("_unmasked.png", "_masked.png")
        masked_file = data_dir / masked_name
        
        if masked_file.exists():
            # Quick check: same file size usually means no change (negative sample)
            is_neg = os.path.getsize(file) == os.path.getsize(masked_file)
            
            if is_neg:
                negatives += 1
                if filter_negatives:
                    skipped += 1
                    continue
            else:
                positives += 1
            
            pairs.append((file, masked_file))
    
    print(f"    Sub-total: {positives} positives, {negatives} negatives")
    if filter_negatives and skipped > 0:
        print(f"    (Filtered out {skipped} negative samples with no mask)")
    
    return pairs


# =============================================================================
# Dataset
# =============================================================================

class EnemySegmentationDataset(Dataset):
    """Tiled dataset for enemy segmentation from game screenshots.

    Extracts multiple 256x256 tiles from each image using a sliding window.
    Tiles are used at native resolution (256x256) - no resize needed.
    This creates many more training samples and matches inference capture size.
    """
    
    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        image_size: int = 512,
        augment: bool = True,
        magenta_threshold: int = 30,
        tile_size: int = 256,
        stride: int = 128,  # 50% overlap
        max_negative_ratio: float = 0.3,  # Limit negative tiles per image
    ):
        self.image_size = image_size
        self.augment = augment
        self.magenta_threshold = magenta_threshold
        self.tile_size = tile_size
        self.stride = stride
        self.max_negative_ratio = max_negative_ratio
        
        # Pre-compute all tile locations for each image
        print("Pre-computing tile locations...")
        self.tiles = []  # List of (pair_idx, y1, x1, is_positive)
        
        for pair_idx, (unmasked_path, masked_path) in enumerate(tqdm(pairs, desc="Building tile index")):
            # Get image dimensions without loading full image
            with Image.open(unmasked_path) as img:
                w, h = img.size
            
            # Check if this is a positive or negative sample (by file size)
            is_positive_image = os.path.getsize(unmasked_path) != os.path.getsize(masked_path)
            
            # Generate all tile positions
            tile_positions = []
            for y in range(0, h - tile_size + 1, stride):
                for x in range(0, w - tile_size + 1, stride):
                    tile_positions.append((y, x))
            
            if is_positive_image:
                # For positive images, include all tiles (we'll check mask content later)
                for y, x in tile_positions:
                    self.tiles.append((pair_idx, y, x))
            else:
                # For negative images, sample a subset to avoid too many negatives
                max_tiles = max(1, int(len(tile_positions) * max_negative_ratio))
                sampled = random.sample(tile_positions, min(max_tiles, len(tile_positions)))
                for y, x in sampled:
                    self.tiles.append((pair_idx, y, x))
        
        self.pairs = pairs
        print(f"Created {len(self.tiles)} tiles from {len(pairs)} images")
        
        # Geometric augmentations only (no color changes)
        if augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-10, 10),
                    p=0.5
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pair_idx, y1, x1 = self.tiles[idx]
        unmasked_path, masked_path = self.pairs[pair_idx]
        
        # Load full image
        unmasked = np.array(Image.open(unmasked_path).convert("RGB"))
        
        # Try to load pre-computed mask from cache
        mask_dir = Path("data/masks")
        mask_name = unmasked_path.name.replace("_unmasked.png", "_mask.png")
        mask_cache_path = mask_dir / mask_name
        
        if mask_cache_path.exists():
            mask = np.array(Image.open(mask_cache_path).convert("L"))
            mask = (mask > 127).astype(np.uint8)
        else:
            masked = np.array(Image.open(masked_path).convert("RGB"))
            mask = extract_mask(masked, unmasked, self.magenta_threshold)
        
        # Extract tile
        y2 = y1 + self.tile_size
        x2 = x1 + self.tile_size
        tile_image = unmasked[y1:y2, x1:x2]
        tile_mask = mask[y1:y2, x1:x2]
        
        # Apply transforms
        transformed = self.transform(image=tile_image, mask=tile_mask)
        image = transformed["image"]
        mask = transformed["mask"].unsqueeze(0).float()
        
        return image, mask


class TileDataset(Dataset):
    """Fast dataset that loads preprocessed tiles from disk.
    
    Run preprocess_tiles.py first to generate tiles in data/tiles/.
    """
    
    def __init__(
        self,
        tiles_dir: Path = Path("data/tiles"),
        image_size: int = 512,
        augment: bool = True,
    ):
        self.tiles_dir = tiles_dir
        self.image_size = image_size
        
        # Find all tile pairs
        self.tile_paths = sorted(tiles_dir.glob("*_image.png"))
        if not self.tile_paths:
            raise ValueError(f"No tiles found in {tiles_dir}. Run preprocess_tiles.py first.")
        
        print(f"Found {len(self.tile_paths)} preprocessed tiles")
        
        if augment:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-10, 10),
                    p=0.5
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self) -> int:
        return len(self.tile_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.tile_paths[idx]
        mask_path = img_path.parent / img_path.name.replace("_image.png", "_mask.png")
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].unsqueeze(0).float()
        
        return image, mask


# =============================================================================
# Model: EfficientNet-UNet
# =============================================================================

class ConvBlock(nn.Module):
    """Double convolution block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Decoder block with skip connection."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        
        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        return self.conv(x)


class EfficientNetUNet(nn.Module):
    """EfficientNet encoder with UNet decoder for segmentation."""
    
    def __init__(self, encoder_name: str = CONFIG["encoder_name"], pretrained: bool = True):
        super().__init__()
        
        # Create EfficientNet encoder
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        
        # Get encoder channel sizes
        dummy = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            features = self.encoder(dummy)
        encoder_channels = [f.shape[1] for f in features]
        
        # Decoder
        self.decoder4 = DecoderBlock(encoder_channels[4], encoder_channels[3], 256)
        self.decoder3 = DecoderBlock(256, encoder_channels[2], 128)
        self.decoder2 = DecoderBlock(128, encoder_channels[1], 64)
        self.decoder1 = DecoderBlock(64, encoder_channels[0], 32)
        
        # Final upsample and segmentation head
        self.final_upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get input size for final resize
        input_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)
        
        # Decoder with skip connections
        x = self.decoder4(features[4], features[3])
        x = self.decoder3(x, features[2])
        x = self.decoder2(x, features[1])
        x = self.decoder1(x, features[0])
        
        # Final upsample
        x = self.final_upsample(x)
        
        # Ensure output matches input size
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        
        # Segmentation head
        x = self.seg_head(x)
        
        return x


# =============================================================================
# Model: STDC1-Seg (Real-time Segmentation)
# =============================================================================

class ConvX(nn.Module):
    """Basic Conv-BN-ReLU block for STDC."""
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, 
                              padding=kernel//2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CatBottleneck(nn.Module):
    """STDC module with concatenation - the core building block."""
    def __init__(self, in_planes, out_planes, block_num=4, stride=1):
        super().__init__()
        assert block_num > 1
        self.conv_list = nn.ModuleList()
        self.stride = stride
        
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, 
                          padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//(2**(idx)), out_planes//(2**(idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//(2**idx), out_planes//(2**idx)))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        return torch.cat(out_list, dim=1)


class STDCNet813(nn.Module):
    """STDC1 backbone network (STDCNet813)."""
    def __init__(self, base=64, layers=[2, 2, 2], block_num=4):
        super().__init__()
        block = CatBottleneck
        self.features = self._make_layers(base, layers, block_num, block)
        
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])
        
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        import math
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))
        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        return feat2, feat4, feat8, feat16, feat32


class AttentionRefinementModule(nn.Module):
    """ARM - refines features using channel attention."""
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )
        # No BatchNorm after conv_atten since it operates on 1x1 global avg pooled tensor
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid(self.conv_atten(atten))
        return feat * atten


class FeatureFusionModule(nn.Module):
    """FFM - fuses spatial and context features."""
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(out_chan, out_chan//4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_chan//4, out_chan, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid(self.conv2(self.relu(self.conv1(atten))))
        return feat * atten + feat


class STDC1Seg(nn.Module):
    """STDC1-Seg: Real-time semantic segmentation model.
    
    Uses STDCNet813 backbone with BiSeNet-style context path.
    Designed for fast inference while maintaining good accuracy.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Backbone
        self.backbone = STDCNet813()
        
        # Context path (ARM modules)
        self.arm16 = AttentionRefinementModule(512, 128)
        self.arm32 = AttentionRefinementModule(1024, 128)
        self.conv_head32 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv_head16 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv_avg = nn.Sequential(
            nn.Conv2d(1024, 128, 1, bias=True),  # Use bias since no BN (1x1 spatial)
            nn.ReLU(inplace=True),
        )
        
        # Feature fusion (spatial feat + context feat)
        # feat8 has 256 channels, context has 128 channels
        self.ffm = FeatureFusionModule(256 + 128, 256)
        
        # Segmentation head (binary)
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        H, W = x.shape[2:]
        
        # Backbone features
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        
        # Context path
        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, feat32.size()[2:], mode='nearest')
        
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, feat16.size()[2:], mode='nearest')
        feat32_up = self.conv_head32(feat32_up)
        
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, feat8.size()[2:], mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        
        # Feature fusion
        feat_fuse = self.ffm(feat8, feat16_up)
        
        # Segmentation
        out = self.seg_head(feat_fuse)
        out = F.interpolate(out, (H, W), mode='bilinear', align_corners=False)
        
        return out


# =============================================================================
# Loss Functions
# =============================================================================

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - focuses on hard examples."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Focal weight: (1 - p_t)^gamma
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting for positive class
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """Tversky loss for sparse masks - generalizes Dice with alpha/beta control.
    
    alpha > beta: penalize false negatives more (better recall)
    alpha < beta: penalize false positives more (better precision)
    alpha = beta = 0.5: equivalent to Dice loss
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Combined Tversky + Focal loss for sparse masks."""

    def __init__(self, tversky_weight: float = 0.7, focal_weight: float = 0.3):
        super().__init__()
        # Tversky with alpha=0.3, beta=0.7 penalizes false negatives more (better recall)
        # This is better for sparse masks where we want to find all positives
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
        # Focal loss to handle class imbalance
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_components: bool = False):
        tversky = self.tversky_loss(pred, target)
        focal = self.focal_loss(pred, target)
        combined = self.tversky_weight * tversky + self.focal_weight * focal
        if return_components:
            return combined, tversky.item(), focal.item()
        return combined


# =============================================================================
# Metrics
# =============================================================================

def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice score."""
    pred = torch.sigmoid(pred) > threshold
    
    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)
    
    return dice.item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute IoU (Intersection over Union)."""
    pred = torch.sigmoid(pred) > threshold

    pred_flat = pred.view(-1).float()
    target_flat = target.view(-1).float()

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + 1) / (union + 1)

    return iou.item()


def compute_precision_recall_f1(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute precision, recall, F1, and raw TP/FP/TN/FN counts."""
    pred_binary = (torch.sigmoid(pred) > threshold).view(-1).float()
    target_flat = target.view(-1).float()

    tp = (pred_binary * target_flat).sum().item()
    fp = (pred_binary * (1 - target_flat)).sum().item()
    fn = ((1 - pred_binary) * target_flat).sum().item()
    tn = ((1 - pred_binary) * (1 - target_flat)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": precision, "recall": recall, "f1": f1
    }


def compute_prediction_distribution(pred: torch.Tensor) -> dict:
    """Analyze sigmoid output distribution in buckets."""
    pred_sigmoid = torch.sigmoid(pred).view(-1)
    total = pred_sigmoid.numel()

    low = ((pred_sigmoid >= 0.0) & (pred_sigmoid < 0.1)).sum().item()
    uncertain = ((pred_sigmoid >= 0.4) & (pred_sigmoid <= 0.6)).sum().item()
    high = ((pred_sigmoid > 0.9) & (pred_sigmoid <= 1.0)).sum().item()

    return {"low": low, "uncertain": uncertain, "high": high, "total": total}


def compute_dice_at_threshold(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    """Compute Dice score at a specific threshold."""
    pred_binary = torch.sigmoid(pred) > threshold
    pred_flat = pred_binary.view(-1).float()
    target_flat = target.view(-1).float()

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + 1) / (pred_flat.sum() + target_flat.sum() + 1)

    return dice.item()


def compute_class_balance(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute % positive pixels in GT vs predictions."""
    pred_binary = (torch.sigmoid(pred) > threshold).view(-1).float()
    target_flat = target.view(-1).float()

    total_pixels = target_flat.numel()
    gt_positive = target_flat.sum().item()
    pred_positive = pred_binary.sum().item()

    return {
        "gt_positive": int(gt_positive),
        "pred_positive": int(pred_positive),
        "total_pixels": total_pixels,
        "gt_positive_pct": gt_positive / total_pixels * 100,
        "pred_positive_pct": pred_positive / total_pixels * 100,
    }


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> dict:
    """Train for one epoch and return debug statistics."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_tversky = 0.0
    total_focal = 0.0

    # Gradient statistics
    grad_norms = []

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            outputs = model(images)
            loss, tversky_val, focal_val = criterion(outputs, masks, return_components=True)

        scaler.scale(loss).backward()

        # Unscale gradients for clipping
        scaler.unscale_(optimizer)

        # Gradient clipping to prevent explosion (max was 5000+ before)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.get("grad_clip_norm", 1.0))

        # Collect gradient statistics every 10 batches (after clipping)
        if batch_idx % 10 == 0:
            batch_grad_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    batch_grad_norms.append(p.grad.norm().item())
            if batch_grad_norms:
                grad_norms.extend(batch_grad_norms)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_tversky += tversky_val
        total_focal += focal_val
        total_dice += compute_dice_score(outputs.detach(), masks)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "tversky": f"{tversky_val:.4f}", "focal": f"{focal_val:.4f}"})

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_dice = total_dice / n_batches
    avg_tversky = total_tversky / n_batches
    avg_focal = total_focal / n_batches

    # Compute gradient stats
    grad_mean = np.mean(grad_norms) if grad_norms else 0.0
    grad_max = np.max(grad_norms) if grad_norms else 0.0
    grad_min = np.min(grad_norms) if grad_norms else 0.0

    return {
        "loss": avg_loss,
        "dice": avg_dice,
        "tversky_loss": avg_tversky,
        "focal_loss": avg_focal,
        "grad_mean": grad_mean,
        "grad_max": grad_max,
        "grad_min": grad_min,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = None,
) -> dict:
    """Validate the model and return comprehensive debug statistics."""
    model.eval()
    total_loss = 0.0

    # Use configured threshold or default
    if threshold is None:
        threshold = CONFIG.get("inference_threshold", 0.5)

    # Per-batch metrics for distribution analysis
    batch_dice_scores = []
    batch_iou_scores = []

    # Per-sample tracking for identifying worst performers
    sample_metrics = []

    # Aggregate TP/FP/TN/FN across all batches
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

    # Prediction distribution buckets
    total_pred_dist = {"low": 0, "uncertain": 0, "high": 0, "total": 0}

    # Multi-threshold analysis
    dice_at_thresholds = {0.3: [], 0.5: [], 0.7: [], 0.8: []}

    # Class balance
    total_gt_positive = 0
    total_pred_positive = 0
    total_pixels = 0

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="Validating", leave=False)):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()

        # Basic metrics using configured threshold
        dice = compute_dice_score(outputs, masks, threshold=threshold)
        iou = compute_iou(outputs, masks, threshold=threshold)
        batch_dice_scores.append(dice)
        batch_iou_scores.append(iou)

        # Per-sample tracking: compute individual sample metrics within batch
        batch_size = images.shape[0]
        for i in range(batch_size):
            sample_mask = masks[i:i+1]
            sample_output = outputs[i:i+1]
            sample_dice = compute_dice_score(sample_output, sample_mask, threshold=threshold)
            mask_size_pct = (sample_mask > 0).float().mean().item() * 100
            sample_metrics.append({
                'batch_idx': batch_idx,
                'sample_idx': i,
                'dice': sample_dice,
                'mask_size_pct': mask_size_pct,
            })

        # Precision/Recall stats using configured threshold
        pr_stats = compute_precision_recall_f1(outputs, masks, threshold=threshold)
        total_tp += pr_stats["tp"]
        total_fp += pr_stats["fp"]
        total_tn += pr_stats["tn"]
        total_fn += pr_stats["fn"]

        # Prediction distribution
        pred_dist = compute_prediction_distribution(outputs)
        total_pred_dist["low"] += pred_dist["low"]
        total_pred_dist["uncertain"] += pred_dist["uncertain"]
        total_pred_dist["high"] += pred_dist["high"]
        total_pred_dist["total"] += pred_dist["total"]

        # Multi-threshold Dice
        for thresh in dice_at_thresholds.keys():
            dice_at_thresholds[thresh].append(compute_dice_at_threshold(outputs, masks, thresh))

        # Class balance using configured threshold
        cb = compute_class_balance(outputs, masks, threshold=threshold)
        total_gt_positive += cb["gt_positive"]
        total_pred_positive += cb["pred_positive"]
        total_pixels += cb["total_pixels"]

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_dice = np.mean(batch_dice_scores)
    avg_iou = np.mean(batch_iou_scores)

    # Compute overall precision/recall/F1
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Analyze worst samples and mask size correlation
    worst_samples = sorted(sample_metrics, key=lambda x: x['dice'])[:5]
    best_samples = sorted(sample_metrics, key=lambda x: x['dice'], reverse=True)[:5]

    # Mask size buckets analysis
    small_masks = [s for s in sample_metrics if s['mask_size_pct'] < 0.3]
    medium_masks = [s for s in sample_metrics if 0.3 <= s['mask_size_pct'] < 1.0]
    large_masks = [s for s in sample_metrics if s['mask_size_pct'] >= 1.0]
    empty_masks = [s for s in sample_metrics if s['mask_size_pct'] == 0]

    mask_size_analysis = {
        'empty': {
            'count': len(empty_masks),
            'avg_dice': np.mean([s['dice'] for s in empty_masks]) if empty_masks else 0.0,
        },
        'small': {
            'count': len(small_masks),
            'avg_dice': np.mean([s['dice'] for s in small_masks]) if small_masks else 0.0,
        },
        'medium': {
            'count': len(medium_masks),
            'avg_dice': np.mean([s['dice'] for s in medium_masks]) if medium_masks else 0.0,
        },
        'large': {
            'count': len(large_masks),
            'avg_dice': np.mean([s['dice'] for s in large_masks]) if large_masks else 0.0,
        },
    }

    return {
        "loss": avg_loss,
        "dice": avg_dice,
        "iou": avg_iou,
        "dice_min": min(batch_dice_scores),
        "dice_max": max(batch_dice_scores),
        "dice_std": float(np.std(batch_dice_scores)),
        "iou_min": min(batch_iou_scores),
        "iou_max": max(batch_iou_scores),
        "iou_std": float(np.std(batch_iou_scores)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp, "fp": total_fp, "tn": total_tn, "fn": total_fn,
        "pred_distribution": total_pred_dist,
        "dice_at_thresholds": {k: np.mean(v) for k, v in dice_at_thresholds.items()},
        "gt_positive_pct": total_gt_positive / max(total_pixels, 1) * 100,
        "pred_positive_pct": total_pred_positive / max(total_pixels, 1) * 100,
        "threshold_used": threshold,
        "worst_samples": worst_samples,
        "best_samples": best_samples,
        "mask_size_analysis": mask_size_analysis,
        "sample_metrics": sample_metrics,
    }


@torch.no_grad()
def save_worst_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    sample_metrics: list,
    n: int = 5,
    threshold: float = None,
):
    """Save visualizations of the worst-performing samples for debugging.

    Creates debug/epoch_{N}/ directory with:
    - worst_1.png through worst_n.png
    - Each shows: input | ground truth | prediction | difference
    """
    import matplotlib.pyplot as plt

    if threshold is None:
        threshold = CONFIG.get("inference_threshold", 0.5)

    # Create output directory
    debug_dir = Path("debug") / f"epoch_{epoch+1}"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Sort by dice to get worst samples
    worst = sorted(sample_metrics, key=lambda x: x['dice'])[:n]

    model.eval()

    # ImageNet denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # We need to re-fetch samples by batch_idx and sample_idx
    # Build a map of (batch_idx, sample_idx) -> position in worst list
    worst_map = {(s['batch_idx'], s['sample_idx']): i for i, s in enumerate(worst)}

    # Storage for worst samples
    worst_data = [None] * n

    for batch_idx, (images, masks) in enumerate(loader):
        # Check if any samples from this batch are in worst list
        batch_samples = [(s_idx, worst_map[(batch_idx, s_idx)])
                        for s_idx in range(images.shape[0])
                        if (batch_idx, s_idx) in worst_map]

        if not batch_samples:
            continue

        images_gpu = images.to(device)
        outputs = model(images_gpu)
        pred_sigmoid = torch.sigmoid(outputs).cpu()

        for sample_idx, worst_idx in batch_samples:
            img = images[sample_idx]
            mask = masks[sample_idx]
            pred = pred_sigmoid[sample_idx]

            # Denormalize image
            img_denorm = img * std + mean
            img_denorm = img_denorm.clamp(0, 1)

            worst_data[worst_idx] = {
                'image': img_denorm.permute(1, 2, 0).numpy(),
                'mask': mask.squeeze().numpy(),
                'pred': pred.squeeze().numpy(),
                'dice': worst[worst_idx]['dice'],
                'mask_size_pct': worst[worst_idx]['mask_size_pct'],
            }

        # Early exit if we have all worst samples
        if all(d is not None for d in worst_data):
            break

    # Save visualizations
    for i, data in enumerate(worst_data):
        if data is None:
            continue

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Input image
        axes[0].imshow(data['image'])
        axes[0].set_title('Input')
        axes[0].axis('off')

        # Ground truth mask
        axes[1].imshow(data['mask'], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Ground Truth\n({data["mask_size_pct"]:.3f}% positive)')
        axes[1].axis('off')

        # Prediction (continuous)
        im = axes[2].imshow(data['pred'], cmap='hot', vmin=0, vmax=1)
        axes[2].set_title(f'Prediction\n(Dice={data["dice"]:.4f})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        # Binary prediction vs GT (green=TP, red=FN, blue=FP)
        pred_binary = (data['pred'] > threshold).astype(np.float32)
        gt = data['mask']

        # Create RGB difference image
        diff_img = np.zeros((*gt.shape, 3), dtype=np.float32)
        diff_img[..., 1] = (pred_binary * gt)  # Green = True Positive
        diff_img[..., 0] = ((1 - pred_binary) * gt)  # Red = False Negative
        diff_img[..., 2] = (pred_binary * (1 - gt))  # Blue = False Positive

        axes[3].imshow(diff_img)
        axes[3].set_title(f'TP(G) FN(R) FP(B)\n@thresh={threshold}')
        axes[3].axis('off')

        plt.tight_layout()
        save_path = debug_dir / f"worst_{i+1}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved {n} worst sample visualizations to {debug_dir}/")


def main():
    """Main training function."""
    print("=" * 60)
    print("EfficientNet-UNet Segmentation Training")
    print("=" * 60)
    
    # Set seed
    set_seed(CONFIG["seed"])
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", type=str, nargs="+", default=["data/labeled"], help="List of data directories")
    parser.add_argument("--encoder", type=str, default=CONFIG["encoder_name"], help="Encoder architecture (e.g. efficientnet_b3, mobilenetv3_large_100)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--stdc1", action="store_true", help="Use STDC1-Seg model instead of EfficientNet-UNet")
    parser.add_argument("--tiles", action="store_true", help="Use preprocessed tiles from data/tiles/ (run preprocess_tiles.py first)")
    args = parser.parse_args()
    
    # Update config from args
    CONFIG["encoder_name"] = "stdc1" if args.stdc1 else args.encoder
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    
    # Set model save path based on model type
    if args.stdc1:
        CONFIG["model_save_path"] = CONFIG["model_save_dir"] / "stdc1_segmentation.pth"
        print("Using STDC1-Seg model")
    else:
        CONFIG["model_save_path"] = CONFIG["model_save_dir"] / "enemy_segmentation.pth"
        print(f"Using EfficientNet-UNet with encoder: {CONFIG['encoder_name']}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Find image pairs from all directories
    all_pairs = []
    print(f"\nSearching for data in: {args.data_dirs}")
    
    for d in args.data_dirs:
        d_path = Path(d)
        if not d_path.exists():
            print(f"Warning: Data directory not found: {d_path}")
            continue
            
        pairs = find_image_pairs(d_path)
        print(f"  Found {len(pairs)} pairs in {d_path}")
        all_pairs.extend(pairs)
        
    print(f"Total valid image pairs: {len(all_pairs)}")
    
    if len(all_pairs) == 0:
        raise ValueError("No valid image pairs found!")
    
    # Split into train/val
    random.shuffle(all_pairs)
    val_size = int(len(all_pairs) * CONFIG["val_split"])
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]
    
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")
    
    # Create datasets
    if args.tiles:
        # Use preprocessed tiles
        tiles_dir = Path("data/tiles")
        if not tiles_dir.exists() or not list(tiles_dir.glob("*_image.png")):
            raise ValueError("No tiles found! Run preprocess_tiles.py first.")
        
        # For tile-based training, we use all tiles for training (already balanced)
        # and still use original pairs for validation
        print(f"Using preprocessed tiles from {tiles_dir}")
        train_dataset = TileDataset(
            tiles_dir=tiles_dir,
            image_size=CONFIG["image_size"],
            augment=True,
        )
        # Validation still uses original images (full image pairs)
        val_dataset = EnemySegmentationDataset(
            val_pairs,
            image_size=CONFIG["image_size"],
            augment=False,
            magenta_threshold=CONFIG["magenta_threshold"],
        )
    else:
        # Original mode: use EnemySegmentationDataset for both
        train_dataset = EnemySegmentationDataset(
            train_pairs,
            image_size=CONFIG["image_size"],
            augment=True,
            magenta_threshold=CONFIG["magenta_threshold"],
        )
        val_dataset = EnemySegmentationDataset(
            val_pairs,
            image_size=CONFIG["image_size"],
            augment=False,
            magenta_threshold=CONFIG["magenta_threshold"],
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=CONFIG["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=CONFIG["num_workers"] > 0,
    )
    
    # Create model
    print(f"\nCreating {CONFIG['encoder_name']} model...")
    if CONFIG["encoder_name"] == "stdc1":
        model = STDC1Seg(pretrained=True)
    elif CONFIG["encoder_name"].startswith("smp_"):
        # Use segmentation_models_pytorch
        if not SMP_AVAILABLE:
            raise ImportError("Please install segmentation_models_pytorch: pip install segmentation-models-pytorch")
        
        # Extract backbone from encoder name (e.g., smp_efficientnet-b3 -> efficientnet-b3)
        backbone = CONFIG["encoder_name"].replace("smp_", "")
        if backbone == "unet":
            backbone = "efficientnet-b3"  # Default backbone for smp_unet
        elif backbone == "stdc1":
            backbone = "timm-mobilenetv3_small_100"  # Fast backbone similar to STDC1
        elif backbone == "stdc2":
            backbone = "timm-mobilenetv3_large_100"  # Slightly larger backbone
        
        print(f"  Using smp.Unet with backbone: {backbone}")
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    else:
        model = EfficientNetUNet(encoder_name=CONFIG["encoder_name"], pretrained=True)
    model = model.to(device)
    
    # Optimize model for speed (channels_last improves GPU cache utilization)
    model = model.to(memory_format=torch.channels_last)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = CombinedLoss(tversky_weight=0.7, focal_weight=0.3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Warmup + Cosine Annealing scheduler
    warmup_epochs = CONFIG.get("warmup_epochs", 5)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # Start at 10% of target LR
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG["epochs"] - warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    scaler = GradScaler("cuda")

    # Print training config
    print(f"\nTraining Config:")
    print(f"  Learning rate: {CONFIG['lr']} (with {warmup_epochs}-epoch warmup)")
    print(f"  Gradient clipping: {CONFIG.get('grad_clip_norm', 1.0)}")
    print(f"  Inference threshold: {CONFIG.get('inference_threshold', 0.5)}")
    
    # Training loop
    print("\nStarting training...")
    print("-" * 60)
    
    best_dice = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")

        # Train
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate
        val_stats = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Extract main metrics for compatibility
        train_loss = train_stats["loss"]
        train_dice = train_stats["dice"]
        val_loss = val_stats["loss"]
        val_dice = val_stats["dice"]
        val_iou = val_stats["iou"]

        # Print basic metrics
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # =====================================================================
        # DEBUG REPORT
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} DEBUG REPORT")
        print(f"{'='*60}")

        # Loss breakdown
        print(f"\nLoss Breakdown:")
        print(f"  Tversky: {train_stats['tversky_loss']:.4f}")
        print(f"  Focal:   {train_stats['focal_loss']:.4f}")
        print(f"  Combined: {train_stats['loss']:.4f}")

        # Validation metrics with distribution
        print(f"\nValidation Metrics:")
        print(f"  Dice: {val_stats['dice']:.4f} (min={val_stats['dice_min']:.3f}, max={val_stats['dice_max']:.3f}, std={val_stats['dice_std']:.3f})")
        print(f"  IoU:  {val_stats['iou']:.4f} (min={val_stats['iou_min']:.3f}, max={val_stats['iou_max']:.3f}, std={val_stats['iou_std']:.3f})")

        # Precision/Recall/F1
        print(f"\nPrecision/Recall/F1:")
        print(f"  Precision: {val_stats['precision']:.4f} | Recall: {val_stats['recall']:.4f} | F1: {val_stats['f1']:.4f}")
        print(f"  TP: {val_stats['tp']:,} | FP: {val_stats['fp']:,} | TN: {val_stats['tn']:,} | FN: {val_stats['fn']:,}")

        # Prediction distribution
        dist = val_stats['pred_distribution']
        total_preds = max(dist['total'], 1)
        print(f"\nPrediction Distribution:")
        print(f"  Low [0-0.1]:       {dist['low']/total_preds*100:5.1f}%  ({dist['low']:,} pixels)")
        print(f"  Uncertain [0.4-0.6]: {dist['uncertain']/total_preds*100:5.1f}%  ({dist['uncertain']:,} pixels)")
        print(f"  High [0.9-1.0]:    {dist['high']/total_preds*100:5.1f}%  ({dist['high']:,} pixels)")

        # Threshold analysis
        current_thresh = val_stats.get('threshold_used', 0.5)
        print(f"\nThreshold Analysis (Dice):")
        for thresh, dice_val in sorted(val_stats['dice_at_thresholds'].items()):
            marker = " <-- current" if abs(thresh - current_thresh) < 0.01 else ""
            print(f"  @{thresh}: {dice_val:.4f}{marker}")

        # Class balance
        gt_pct = val_stats['gt_positive_pct']
        pred_pct = val_stats['pred_positive_pct']
        ratio = pred_pct / max(gt_pct, 0.01)
        print(f"\nClass Balance:")
        print(f"  GT positive:   {gt_pct:.2f}%")
        print(f"  Pred positive: {pred_pct:.2f}%")
        print(f"  Ratio (pred/gt): {ratio:.2f}x {'(under-predicting)' if ratio < 0.9 else '(over-predicting)' if ratio > 1.1 else '(balanced)'}")

        # Gradient stats
        print(f"\nGradient Stats:")
        print(f"  Mean: {train_stats['grad_mean']:.6f} | Max: {train_stats['grad_max']:.4f} | Min: {train_stats['grad_min']:.6f}")

        # Mask Size Analysis
        msa = val_stats.get('mask_size_analysis', {})
        print(f"\nMask Size vs Dice:")
        if msa.get('empty', {}).get('count', 0) > 0:
            print(f"  Empty (0%):     {msa['empty']['count']:3d} samples, avg Dice={msa['empty']['avg_dice']:.4f}")
        if msa.get('small', {}).get('count', 0) > 0:
            print(f"  Small (<0.3%):  {msa['small']['count']:3d} samples, avg Dice={msa['small']['avg_dice']:.4f}")
        if msa.get('medium', {}).get('count', 0) > 0:
            print(f"  Medium (0.3-1%): {msa['medium']['count']:3d} samples, avg Dice={msa['medium']['avg_dice']:.4f}")
        if msa.get('large', {}).get('count', 0) > 0:
            print(f"  Large (>1%):    {msa['large']['count']:3d} samples, avg Dice={msa['large']['avg_dice']:.4f}")

        # Worst samples
        worst = val_stats.get('worst_samples', [])
        if worst:
            print(f"\nWorst 5 Samples:")
            for i, s in enumerate(worst[:5]):
                print(f"  {i+1}. batch={s['batch_idx']}, sample={s['sample_idx']}, Dice={s['dice']:.4f}, mask={s['mask_size_pct']:.3f}%")

        # Best samples
        best = val_stats.get('best_samples', [])
        if best:
            print(f"\nBest 5 Samples:")
            for i, s in enumerate(best[:5]):
                print(f"  {i+1}. batch={s['batch_idx']}, sample={s['sample_idx']}, Dice={s['dice']:.4f}, mask={s['mask_size_pct']:.3f}%")

        print(f"{'='*60}\n")

        # Save worst sample visualizations every 10 epochs
        if (epoch + 1) % 10 == 0:
            sample_metrics = val_stats.get('sample_metrics', [])
            if sample_metrics:
                save_worst_samples(model, val_loader, device, epoch, sample_metrics, n=5)

        # Check for improvement
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0

            # Save best model
            CONFIG["model_save_dir"].mkdir(parents=True, exist_ok=True)
            save_path = CONFIG.get("model_save_path", CONFIG["model_save_dir"] / "enemy_segmentation.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice,
                "config": CONFIG,
            }, save_path)
            print(f"  ✓ New best! Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")

        # Early stopping
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Validation Dice: {best_dice:.4f}")
    
    if best_dice >= 0.8:
        print("✓ Target Dice score (0.8+) ACHIEVED!")
    else:
        print(f"✗ Target Dice score (0.8+) not reached. Consider:")
        print("  - Adding more training data")
        print("  - Adjusting augmentations")
        print("  - Fine-tuning hyperparameters")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
