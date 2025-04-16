from pathlib import Path
import random

def split_dataset(image_dir, val_ratio=0.2, seed=42):
    all_images = sorted(Path(image_dir).glob("*.png"))
    random.seed(seed)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * (1 - val_ratio))
    train_files = [p.name for p in all_images[:split_idx]]
    val_files = [p.name for p in all_images[split_idx:]]

    return train_files, val_files
