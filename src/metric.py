import torch
import numpy as np

def compute_dice_score(preds, targets, num_classes):
    dice_scores = []
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        if union == 0:
            dice_scores.append(np.nan)
        else:
            dice_scores.append(2 * intersection / (union + 1e-8))
    return dice_scores

def compute_iou_score(preds, targets, num_classes):
    iou_scores = []
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        if union == 0:
            iou_scores.append(np.nan)
        else:
            iou_scores.append(intersection / (union + 1e-8))
    return iou_scores
