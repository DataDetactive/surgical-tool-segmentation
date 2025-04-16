import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Dice Loss --------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        

        if targets.ndim == 4:
            targets = targets.squeeze(1) 
        
        

        num_classes = inputs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        inputs_soft = F.softmax(inputs, dim=1)

        inputs_flat = inputs_soft.contiguous().view(inputs.size(0), num_classes, -1)
        targets_flat = targets_one_hot.contiguous().view(inputs.size(0), num_classes, -1)

        intersection = (inputs_flat * targets_flat).sum(2)
        union = inputs_flat.sum(2) + targets_flat.sum(2)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


# -------- Composite Loss --------
class DiceCELoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5, ce_weight=0.5):
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, inputs, targets):
        return self.dice_weight * self.dice(inputs, targets) + self.ce_weight * self.ce(inputs, targets)

# --------  Focal Loss --------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        logpt = -F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(logpt)
        focal = -((1 - pt) ** self.gamma) * logpt
        return self.alpha * focal.mean()

# -------- Composite Loss --------
class FocalCELoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, ce_weight=0.5, focal_weight=0.5):
        super(FocalCELoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, weight=weight)
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        loss_ce = self.ce(inputs, targets)
        loss_focal = self.focal(inputs, targets)
        return self.ce_weight * loss_ce + self.focal_weight * loss_focal