import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler


def compute_rarity_scores(mask_paths, imbalance_ratios):
    rarity_boost = {cls: np.log(r + 1) for cls, r in imbalance_ratios.items()}
    rarity_scores = {}

    for mask_path in tqdm(mask_paths, desc="Calculating rarity scores"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        unique_classes = np.unique(mask)
        score = sum(rarity_boost.get(cls, 0) for cls in unique_classes)
        rarity_scores[mask_path.name] = score

    return rarity_scores

def build_weighted_sampler(dataset, rarity_scores):
    weights = [rarity_scores.get(p.name, 1.0) for p in dataset.image_paths]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler
