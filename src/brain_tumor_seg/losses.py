from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def binary_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Numerically stable focal BCE; alpha is the foreground class weight."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Focal alpha must be between 0 and 1")
    if gamma < 0:
        raise ValueError("Focal gamma must be non-negative")
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    probability_true = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_true = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_true * (1.0 - probability_true).pow(gamma) * cross_entropy).mean()


def soft_boundary_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    kernel_size: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Dice loss on differentiable morphological boundary maps."""
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Boundary kernel size must be an odd integer of at least 3")
    padding = kernel_size // 2

    def boundary(values: torch.Tensor) -> torch.Tensor:
        dilation = F.max_pool2d(values, kernel_size, stride=1, padding=padding)
        erosion = -F.max_pool2d(-values, kernel_size, stride=1, padding=padding)
        return dilation - erosion

    predicted_boundary = boundary(torch.sigmoid(logits)).flatten(start_dim=1)
    target_boundary = boundary(targets).flatten(start_dim=1)
    intersection = (predicted_boundary * target_boundary).sum(dim=1)
    denominator = predicted_boundary.sum(dim=1) + target_boundary.sum(dim=1)
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


class BCEPositiveDiceLoss(BCEDiceLoss):
    """BCE on every slice and Dice only on slices containing foreground."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"Logit/target shape mismatch: {logits.shape} vs {targets.shape}")
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        positive = targets.flatten(start_dim=1).sum(dim=1) > 0
        if positive.any():
            dice = soft_dice_loss(logits[positive], targets[positive], self.smooth)
        else:
            # Keep the result attached to the graph for empty-only batches.
            dice = logits.sum() * 0.0
        return self.bce_weight * bce + self.dice_weight * dice


class DiceFocalBoundaryLoss(nn.Module):
    """Optimize overlap, hard pixels, and boundary alignment in one objective."""

    def __init__(
        self,
        dice_weight: float = 0.55,
        focal_weight: float = 0.30,
        boundary_weight: float = 0.15,
        *,
        smooth: float = 1.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        boundary_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        weights = (dice_weight, focal_weight, boundary_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Loss weights must be non-negative and not all zero")
        total = sum(weights)
        self.dice_weight = dice_weight / total
        self.focal_weight = focal_weight / total
        self.boundary_weight = boundary_weight / total
        self.smooth = smooth
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.boundary_kernel_size = boundary_kernel_size

        # Validate values at construction time instead of failing after training starts.
        if not 0.0 <= focal_alpha <= 1.0:
            raise ValueError("Focal alpha must be between 0 and 1")
        if focal_gamma < 0:
            raise ValueError("Focal gamma must be non-negative")
        if boundary_kernel_size < 3 or boundary_kernel_size % 2 == 0:
            raise ValueError("Boundary kernel size must be an odd integer of at least 3")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(f"Logit/target shape mismatch: {logits.shape} vs {targets.shape}")
        dice = soft_dice_loss(logits, targets, self.smooth)
        focal = binary_focal_loss(
            logits,
            targets,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        boundary = soft_boundary_dice_loss(
            logits,
            targets,
            kernel_size=self.boundary_kernel_size,
            smooth=self.smooth,
        )
        return (
            self.dice_weight * dice
            + self.focal_weight * focal
            + self.boundary_weight * boundary
        )


def build_loss(config: dict[str, object]) -> nn.Module:
    name = str(config.get("name", "bce_dice")).lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name in {"bce_positive_dice", "bce_pos_dice"}:
        return BCEPositiveDiceLoss(
            bce_weight=float(config.get("bce_weight", 0.5)),
            dice_weight=float(config.get("dice_weight", 0.5)),
            smooth=float(config.get("smooth", 1.0)),
        )
    if name in {"dice_focal_boundary", "focal_dice_boundary"}:
        return DiceFocalBoundaryLoss(
            dice_weight=float(config.get("dice_weight", 0.55)),
            focal_weight=float(config.get("focal_weight", 0.30)),
            boundary_weight=float(config.get("boundary_weight", 0.15)),
            smooth=float(config.get("smooth", 1.0)),
            focal_alpha=float(config.get("focal_alpha", 0.75)),
            focal_gamma=float(config.get("focal_gamma", 2.0)),
            boundary_kernel_size=int(config.get("boundary_kernel_size", 3)),
        )
    if name != "bce_dice":
        raise ValueError(f"Unsupported loss: {name}")
    return BCEDiceLoss(
        bce_weight=float(config["bce_weight"]),
        dice_weight=float(config["dice_weight"]),
        smooth=float(config.get("smooth", 1.0)),
    )
