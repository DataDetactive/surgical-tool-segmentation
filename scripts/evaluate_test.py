import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import get_unet_model
from src.dataset import SurgicalSegmentationDataset
from src.metric import compute_dice_score, compute_iou_score


# ----- Config -----
image_dir = "/home.local/stagiaire/Music/T/processed_dataset/test/images"
mask_dir = "/home.local/stagiaire/Music/T/processed_dataset/test/masks"
checkpoint_path = "/home.local/stagiaire/Music/T/code/checkpoints_focal/best_model.pth"
save_dir = "test_predictions_focal"
os.makedirs(save_dir, exist_ok=True)



batch_size = 1
num_classes = 10
input_size = (1024, 1024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load Test Data -----
test_dataset = SurgicalSegmentationDataset(image_dir, mask_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ----- Load Model -----
model = get_unet_model("resnet18", num_classes=num_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ----- Inference + Metrics -----
all_dice_scores = []
all_iou_scores = []

with torch.no_grad():
    for idx, (image, mask, name) in enumerate(tqdm(test_loader, desc="Evaluating Test Set")):
        image = image.to(device)
        mask = mask.to(device)

        output = model(image)
        pred = torch.argmax(output, dim=1)

        # Metrics
        dice = compute_dice_score(output, mask, num_classes)
        iou = compute_iou_score(output, mask, num_classes)
        all_dice_scores.append(dice)
        all_iou_scores.append(iou)

        # Save visualization
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        gt_np = mask[0].cpu().numpy()
        pred_np = pred[0].cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img_np)
        axs[0].set_title("Input Image")
        axs[1].imshow(gt_np, cmap="nipy_spectral", vmin=0, vmax=num_classes-1)
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred_np, cmap="nipy_spectral", vmin=0, vmax=num_classes-1)
        axs[2].set_title("Prediction")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name[0].replace('.png', '')}_compare.png"))
        plt.close()

# ----- Report Final Scores -----
mean_dice = np.nanmean(np.array(all_dice_scores), axis=0)
mean_iou = np.nanmean(np.array(all_iou_scores), axis=0)
overall_dice = np.nanmean(mean_dice)
overall_iou = np.nanmean(mean_iou)

print("\n Final Test Results")
print("----------------------")
print(f"Mean Dice: {overall_dice:.4f}")
print(f"Mean IoU : {overall_iou:.4f}")

# Optional: save per-class scores
np.savetxt("test_dice_per_class_focal.csv", mean_dice, delimiter=",", header="Per-class Dice", comments="")
np.savetxt("test_iou_per_class_focal.csv", mean_iou, delimiter=",", header="Per-class IoU", comments="")