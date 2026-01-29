"""
Preprocess Labeled Data into Tiles
===================================
Extracts 256x256 tiles from labeled images and their masks.
Filters to maintain 80/20 positive/negative tile ratio.

Usage:
    python src/preprocess_tiles.py

Output:
    data/tiles/
        tile_0000_image.png
        tile_0000_mask.png
        ...
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

LABELED_DIR = Path("data/labeled")
MASKS_DIR = Path("data/masks")
OUTPUT_DIR = Path("data/tiles")

TILE_SIZE = 256
STRIDE = 128  # 50% overlap

# Target ratio of positive tiles (tiles with mask pixels)
POSITIVE_RATIO = 0.8
NEGATIVE_RATIO = 0.2


# =============================================================================
# Mask Extraction
# =============================================================================

def extract_mask(masked: np.ndarray, unmasked: np.ndarray, threshold: int = 30) -> np.ndarray:
    """Extract binary mask by comparing masked vs unmasked images."""
    diff = np.abs(masked.astype(np.int16) - unmasked.astype(np.int16))
    diff_max = np.max(diff, axis=2)
    mask = (diff_max > threshold).astype(np.uint8)
    return mask


def get_or_compute_mask(unmasked_path: Path, masked_path: Path) -> np.ndarray:
    """Load cached mask or compute from images."""
    mask_name = unmasked_path.name.replace("_unmasked.png", "_mask.png")
    mask_path = MASKS_DIR / mask_name
    
    if mask_path.exists():
        mask = np.array(Image.open(mask_path).convert("L"))
        return (mask > 127).astype(np.uint8)
    
    # Compute mask
    unmasked = np.array(Image.open(unmasked_path).convert("RGB"))
    masked = np.array(Image.open(masked_path).convert("RGB"))
    mask = extract_mask(masked, unmasked)
    
    # Cache it
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask * 255).save(mask_path)
    
    return mask


# =============================================================================
# Tile Extraction
# =============================================================================

def extract_tiles_from_image(
    image: np.ndarray,
    mask: np.ndarray,
    tile_size: int = TILE_SIZE,
    stride: int = STRIDE,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[bool]]:
    """Extract all tiles from an image and its mask.
    
    Returns:
        tiles: List of image tiles
        masks: List of mask tiles
        is_positive: List of booleans indicating if tile contains mask pixels
    """
    h, w = image.shape[:2]
    tiles = []
    masks = []
    is_positive = []
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile_img = image[y:y+tile_size, x:x+tile_size]
            tile_mask = mask[y:y+tile_size, x:x+tile_size]
            
            tiles.append(tile_img)
            masks.append(tile_mask)
            is_positive.append(tile_mask.sum() > 0)
    
    return tiles, masks, is_positive


# =============================================================================
# Main Processing
# =============================================================================

def find_image_pairs(labeled_dir: Path) -> List[Tuple[Path, Path]]:
    """Find all unmasked/masked image pairs in labeled directory."""
    pairs = []
    for f in sorted(labeled_dir.glob("*_unmasked.png")):
        masked = f.parent / f.name.replace("_unmasked.png", "_masked.png")
        if masked.exists():
            pairs.append((f, masked))
    return pairs


def main():
    print("=" * 60)
    print("Tile Preprocessing")
    print("=" * 60)
    
    # Find all image pairs
    pairs = find_image_pairs(LABELED_DIR)
    print(f"Found {len(pairs)} image pairs in {LABELED_DIR}")
    
    if not pairs:
        print("No image pairs found!")
        return
    
    # Extract all tiles
    all_positive_tiles = []  # (image, mask)
    all_negative_tiles = []  # (image, mask)
    
    for unmasked_path, masked_path in tqdm(pairs, desc="Extracting tiles"):
        # Load image and mask
        image = np.array(Image.open(unmasked_path).convert("RGB"))
        mask = get_or_compute_mask(unmasked_path, masked_path)
        
        # Extract tiles
        tiles, masks_list, is_positive = extract_tiles_from_image(image, mask)
        
        for tile, tile_mask, positive in zip(tiles, masks_list, is_positive):
            if positive:
                all_positive_tiles.append((tile, tile_mask))
            else:
                all_negative_tiles.append((tile, tile_mask))
    
    print(f"\nExtracted tiles:")
    print(f"  Positive (has enemy): {len(all_positive_tiles)}")
    print(f"  Negative (no enemy):  {len(all_negative_tiles)}")
    
    # Calculate target counts for 80/20 split
    total_positive = len(all_positive_tiles)
    target_negative = int(total_positive * (NEGATIVE_RATIO / POSITIVE_RATIO))
    
    print(f"\nTarget split ({int(POSITIVE_RATIO*100)}/{int(NEGATIVE_RATIO*100)}):")
    print(f"  Positive: {total_positive}")
    print(f"  Negative: {target_negative}")
    
    # Sample negatives if we have too many
    if len(all_negative_tiles) > target_negative:
        print(f"  Sampling {target_negative} negatives from {len(all_negative_tiles)}")
        all_negative_tiles = random.sample(all_negative_tiles, target_negative)
    
    # Combine and shuffle
    all_tiles = all_positive_tiles + all_negative_tiles
    random.shuffle(all_tiles)
    
    print(f"\nTotal tiles to save: {len(all_tiles)}")
    
    # Save tiles
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear existing tiles
    for f in OUTPUT_DIR.glob("*.png"):
        f.unlink()
    
    for idx, (tile, tile_mask) in enumerate(tqdm(all_tiles, desc="Saving tiles")):
        tile_name = f"tile_{idx:06d}_image.png"
        mask_name = f"tile_{idx:06d}_mask.png"
        
        Image.fromarray(tile).save(OUTPUT_DIR / tile_name)
        Image.fromarray(tile_mask * 255).save(OUTPUT_DIR / mask_name)
    
    print(f"\nSaved {len(all_tiles)} tiles to {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
