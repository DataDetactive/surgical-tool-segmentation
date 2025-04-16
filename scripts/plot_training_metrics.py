import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
csv_path = "/home.local/stagiaire/Music/T/code/training_metrics_focal.csv"
df = pd.read_csv(csv_path)

# Plot Train/Val Loss
plt.figure(figsize=(10, 5))
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", marker='o')
plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss", marker='o')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# Plot Mean Dice and Mean IoU
plt.figure(figsize=(10, 5))
plt.plot(df["Epoch"], df["Mean Dice"], label="Mean Dice", marker='o')
plt.plot(df["Epoch"], df["Mean IoU"], label="Mean IoU", marker='o')
plt.title("Segmentation Metrics over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dice_iou_curve.png")
plt.show()
