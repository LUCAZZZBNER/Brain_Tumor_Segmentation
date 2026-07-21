from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float, dice_weight: float, smooth: float = 1.0) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        total = bce_weight + dice_weight
        self.bce_weight = bce_weight / total
        self.dice_weight = dice_weight / total
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"Logit/target shape mismatch: {logits.shape} vs {targets.shape}")
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = soft_dice_loss(logits, targets, self.smooth)
        return self.bce_weight * bce + self.dice_weight * dice


def build_loss(config: dict[str, object]) -> nn.Module:
    name = str(config.get("name", "bce_dice")).lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name != "bce_dice":
        raise ValueError(f"Unsupported loss: {name}")
    return BCEDiceLoss(
        bce_weight=float(config["bce_weight"]),
        dice_weight=float(config["dice_weight"]),
        smooth=float(config.get("smooth", 1.0)),
    )
