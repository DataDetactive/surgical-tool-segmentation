from dataset import SurgicalSegmentationDataset
from sampler import compute_rarity_scores, build_weighted_sampler
from utilis import split_dataset
from loss import DiceCELoss, FocalCELoss
from metric import compute_dice_score, compute_iou_score
from pathlib import Path
import numpy as np 
from torch.utils.data import DataLoader
from model import get_unet_model
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import csv

# ----- Config -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = "processed_dataset/train/images"
mask_dir = "processed_dataset/train/masks"
batch_size = 4
num_epochs = 20
num_classes = 10
checkpoint_dir = "/home.local/stagiaire/Music/T/code/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)





image_dir ="/home.local/stagiaire/Music/T/processed_dataset/train/images"
mask_dir ="/home.local/stagiaire/Music/T/processed_dataset/train/masks"
mask_paths = sorted(Path(mask_dir).glob("*.png"))

# Split
train_files, val_files = split_dataset(image_dir, val_ratio=0.2)

# Initialize datasets
train_dataset = SurgicalSegmentationDataset(image_dir, mask_dir, train_files)
val_dataset   = SurgicalSegmentationDataset(image_dir, mask_dir, val_files)

# I alredy calculated this from my analysis
imbalance_ratios = {
    0: 1.0,          # Background
    1: 32.36,        # Tool Clasper
    2: 21.95,        # Tool Wrist
    3: 7.03,         # Tool Shaft	
    4: 153.33,       # Needle
    5: 69.51,        # Thread
    6: 187.09,       # Suction Tool
    7: 325.69,       # Needle Holder
    8: 417.37,       # Clamps
    9: 1021.68       # Catheter
}

# log-scale boost to avoid huge differences
rarity_boost = {k: np.log(v + 1) for k, v in imbalance_ratios.items()}
rarity_scores = compute_rarity_scores(mask_paths, rarity_boost)
train_sampler = build_weighted_sampler(train_dataset, rarity_scores)

train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)



# ----- Model + Loss + Optimizer -----
model = get_unet_model("resnet18", num_classes=num_classes).to(device)
weights = torch.tensor(list(imbalance_ratios.values()), dtype=torch.float)
weights = weights / weights.sum()
loss_fn = DiceCELoss(weight=weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=1e-4)





# ----- Training Loop -----
best_val_dice = 0.0
results_file = "training_metrics.csv"

with open(results_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Mean Dice", "Mean IoU"])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # ----- Validation -----
        model.eval()
        val_loss = 0
        all_dice_scores = []
        all_iou_scores = []

        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc="Validating"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item() * images.size(0)

                batch_dice = compute_dice_score(outputs, masks, num_classes)
                batch_iou = compute_iou_score(outputs, masks, num_classes)
                all_dice_scores.append(batch_dice)
                all_iou_scores.append(batch_iou)

        val_loss /= len(val_loader.dataset)
        mean_dice = np.nanmean(np.array(all_dice_scores), axis=0)
        mean_iou = np.nanmean(np.array(all_iou_scores), axis=0)
        avg_dice = np.nanmean(mean_dice)
        avg_iou = np.nanmean(mean_iou)

        print(f"\n Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Mean Dice = {avg_dice:.4f} | Mean IoU = {avg_iou:.4f}")

        writer.writerow([epoch+1, train_loss, val_loss, avg_dice, avg_iou])

        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            print("Best model updated.")
