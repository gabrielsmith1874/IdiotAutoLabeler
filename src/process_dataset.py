
import os
import shutil
import random
from pathlib import Path

def process_dataset(data_dir, target_ratio=0.2):
    """
    Process dataset to maintain a specific ratio of negative samples.
    target_ratio: Proportion of negatives in the total dataset (e.g., 0.2 for 20% negatives)
    """
    data_dir = Path(data_dir)
    extra_dir = Path("data/extra_negatives")
    extra_dir.mkdir(exist_ok=True)
    
    positives = []
    negatives = []
    
    unmasked_files = sorted(list(data_dir.glob("*_unmasked.png")))
    for file in unmasked_files:
        masked_name = file.name.replace("_unmasked.png", "_masked.png")
        masked_file = data_dir / masked_name
        
        if masked_file.exists():
            # Quick check: same file size usually means no change (negative sample)
            if os.path.getsize(file) == os.path.getsize(masked_file):
                negatives.append((file, masked_file))
            else:
                positives.append((file, masked_file))
    
    num_pos = len(positives)
    num_neg = len(negatives)
    
    print(f"Current State:")
    print(f"  Positives: {num_pos}")
    print(f"  Negatives: {num_neg}")
    print(f"  Total: {num_pos + num_neg}")
    print(f"  Negative %: {num_neg / (num_pos + num_neg) * 100:.2f}%")
    
    # Calculate target number of negatives
    # Ratio = Neg / (Pos + Neg) => Ratio * Pos + Ratio * Neg = Neg => Ratio * Pos = Neg * (1 - Ratio)
    # Neg = (Ratio * Pos) / (1 - Ratio)
    # For 80/20 (Ratio = 0.2): Neg = (0.2 * Pos) / 0.8 = 0.25 * Pos
    target_num_neg = int((target_ratio * num_pos) / (1 - target_ratio))
    
    print(f"\nTarget State (80/20 split):")
    print(f"  Desired Negatives: {target_num_neg}")
    
    if num_neg > target_num_neg:
        # Shuffle negatives and pick ones to move
        random.shuffle(negatives)
        to_keep = negatives[:target_num_neg]
        to_move = negatives[target_num_neg:]
        
        print(f"  Moving {len(to_move)} negative pairs to {extra_dir}...")
        
        for unmasked, masked in to_move:
            shutil.move(str(unmasked), str(extra_dir / unmasked.name))
            shutil.move(str(masked), str(extra_dir / masked.name))
        
        print(f"\nSuccess! Kept {len(to_keep)} negatives.")
    else:
        print(f"\nNo processing needed. Already at or below target negative count.")

if __name__ == "__main__":
    # Seed for reproducibility if needed, but random is fine here
    random.seed(42)
    process_dataset("data/labeled", target_ratio=0.2)
