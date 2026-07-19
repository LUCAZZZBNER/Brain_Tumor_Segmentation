from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinarySegmentationMeter:
    """Accumulate foreground-only IoU and Dice without averaging batches."""

    threshold: float = 0.5
    epsilon: float = 1e-7

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.intersection = 0.0
        self.union = 0.0
        self.predicted = 0.0
        self.target = 0.0
        self.macro_iou_sum = 0.0
        self.macro_dice_sum = 0.0
        self.num_images = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if logits.shape != targets.shape:
            raise ValueError(f"Logit/target shape mismatch: {logits.shape} vs {targets.shape}")
        predictions = torch.sigmoid(logits) >= self.threshold
        targets_bool = targets >= 0.5
        reduce_dims = tuple(range(1, predictions.ndim))
        intersection = (predictions & targets_bool).sum(dim=reduce_dims).double()
        union = (predictions | targets_bool).sum(dim=reduce_dims).double()
        predicted = predictions.sum(dim=reduce_dims).double()
        target = targets_bool.sum(dim=reduce_dims).double()

        per_image_iou = torch.where(
            union > 0, intersection / union.clamp_min(self.epsilon), torch.ones_like(union)
        )
        dice_denominator = predicted + target
        per_image_dice = torch.where(
            dice_denominator > 0,
            2.0 * intersection / dice_denominator.clamp_min(self.epsilon),
            torch.ones_like(dice_denominator),
        )
        self.intersection += intersection.sum().item()
        self.union += union.sum().item()
        self.predicted += predicted.sum().item()
        self.target += target.sum().item()
        self.macro_iou_sum += per_image_iou.sum().item()
        self.macro_dice_sum += per_image_dice.sum().item()
        self.num_images += int(logits.shape[0])

    def compute(self) -> dict[str, float]:
        if self.num_images == 0:
            raise RuntimeError("No samples were added to the metric meter")
        micro_iou = (
            self.intersection / self.union if self.union > 0 else 1.0
        )
        dice_denominator = self.predicted + self.target
        micro_dice = (
            2.0 * self.intersection / dice_denominator if dice_denominator > 0 else 1.0
        )
        return {
            "macro_iou": self.macro_iou_sum / self.num_images,
            "micro_iou": micro_iou,
            "macro_dice": self.macro_dice_sum / self.num_images,
            "micro_dice": micro_dice,
        }

