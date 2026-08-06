from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@torch.no_grad()
def binary_metrics_per_sample(
    logits: torch.Tensor, targets: torch.Tensor, *, threshold: float = 0.5
) -> dict[str, torch.Tensor]:
    """Return per-image binary segmentation counts and metrics."""
    if logits.shape != targets.shape:
        raise ValueError(f"Logit/target shape mismatch: {logits.shape} vs {targets.shape}")
    predictions = torch.sigmoid(logits) >= threshold
    targets_bool = targets >= 0.5
    reduce_dims = tuple(range(1, predictions.ndim))
    true_positive = (predictions & targets_bool).sum(dim=reduce_dims).double()
    false_positive = (predictions & ~targets_bool).sum(dim=reduce_dims).double()
    false_negative = (~predictions & targets_bool).sum(dim=reduce_dims).double()
    true_negative = (~predictions & ~targets_bool).sum(dim=reduce_dims).double()
    predicted = true_positive + false_positive
    target = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    total = true_positive + false_positive + false_negative + true_negative

    iou = torch.where(union > 0, true_positive / union, torch.ones_like(union))
    dice_denominator = predicted + target
    dice = torch.where(
        dice_denominator > 0,
        2.0 * true_positive / dice_denominator,
        torch.ones_like(dice_denominator),
    )
    precision = torch.where(
        predicted > 0,
        true_positive / predicted,
        (target == 0).double(),
    )
    recall = torch.where(
        target > 0,
        true_positive / target,
        (predicted == 0).double(),
    )
    negative = true_negative + false_positive
    specificity = torch.where(
        negative > 0,
        true_negative / negative,
        torch.ones_like(negative),
    )
    accuracy = (true_positive + true_negative) / total.clamp_min(1.0)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
    }


def select_best_threshold(
    candidates: Iterable[dict[str, Any]],
    metric: str,
    *,
    reference_threshold: float = 0.5,
) -> dict[str, Any]:
    """Select a validation threshold, resolving ties toward the configured reference."""
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("Threshold search requires at least one candidate")
    for candidate in candidate_list:
        if "threshold" not in candidate or metric not in candidate:
            raise ValueError(f"Threshold candidate must contain threshold and {metric}")
    return max(
        candidate_list,
        key=lambda candidate: (
            float(candidate[metric]),
            -abs(float(candidate["threshold"]) - reference_threshold),
            -float(candidate["threshold"]),
        ),
    )


@dataclass
class BinarySegmentationMeter:
    """Accumulate macro and micro binary segmentation metrics across batches."""

    threshold: float = 0.5
    epsilon: float = 1e-7

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.intersection = 0.0
        self.union = 0.0
        self.predicted = 0.0
        self.target = 0.0
        self.true_negative = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0
        self.total_pixels = 0.0
        self.macro_iou_sum = 0.0
        self.macro_dice_sum = 0.0
        self.macro_precision_sum = 0.0
        self.macro_recall_sum = 0.0
        self.macro_specificity_sum = 0.0
        self.macro_accuracy_sum = 0.0
        self.positive_iou_sum = 0.0
        self.positive_dice_sum = 0.0
        self.empty_false_positive_images = 0
        self.empty_predicted_fraction_sum = 0.0
        self.num_positive_images = 0
        self.num_empty_images = 0
        self.num_images = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        values = binary_metrics_per_sample(logits, targets, threshold=self.threshold)
        true_positive = values["true_positive"]
        false_positive = values["false_positive"]
        false_negative = values["false_negative"]
        true_negative = values["true_negative"]
        self.intersection += true_positive.sum().item()
        self.union += (true_positive + false_positive + false_negative).sum().item()
        self.predicted += (true_positive + false_positive).sum().item()
        self.target += (true_positive + false_negative).sum().item()
        self.true_negative += true_negative.sum().item()
        self.false_positive += false_positive.sum().item()
        self.false_negative += false_negative.sum().item()
        self.total_pixels += (
            true_positive + false_positive + false_negative + true_negative
        ).sum().item()
        self.macro_iou_sum += values["iou"].sum().item()
        self.macro_dice_sum += values["dice"].sum().item()
        self.macro_precision_sum += values["precision"].sum().item()
        self.macro_recall_sum += values["recall"].sum().item()
        self.macro_specificity_sum += values["specificity"].sum().item()
        self.macro_accuracy_sum += values["accuracy"].sum().item()
        target_pixels = true_positive + false_negative
        predicted_pixels = true_positive + false_positive
        positive = target_pixels > 0
        empty = ~positive
        self.positive_iou_sum += values["iou"][positive].sum().item()
        self.positive_dice_sum += values["dice"][positive].sum().item()
        self.num_positive_images += int(positive.sum().item())
        self.num_empty_images += int(empty.sum().item())
        self.empty_false_positive_images += int((predicted_pixels[empty] > 0).sum().item())
        pixels_per_image = float(targets[0].numel())
        self.empty_predicted_fraction_sum += (
            predicted_pixels[empty] / pixels_per_image
        ).sum().item()
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
        micro_precision = (
            self.intersection / self.predicted
            if self.predicted > 0
            else float(self.target == 0)
        )
        micro_recall = (
            self.intersection / self.target
            if self.target > 0
            else float(self.predicted == 0)
        )
        negative = self.true_negative + self.false_positive
        micro_specificity = self.true_negative / negative if negative > 0 else 1.0
        micro_accuracy = (
            (self.intersection + self.true_negative) / self.total_pixels
            if self.total_pixels > 0
            else 1.0
        )
        return {
            "macro_iou": self.macro_iou_sum / self.num_images,
            "micro_iou": micro_iou,
            "macro_dice": self.macro_dice_sum / self.num_images,
            "micro_dice": micro_dice,
            "macro_precision": self.macro_precision_sum / self.num_images,
            "micro_precision": micro_precision,
            "macro_recall": self.macro_recall_sum / self.num_images,
            "micro_recall": micro_recall,
            "macro_specificity": self.macro_specificity_sum / self.num_images,
            "micro_specificity": micro_specificity,
            "macro_accuracy": self.macro_accuracy_sum / self.num_images,
            "micro_accuracy": micro_accuracy,
            "positive_macro_iou": (
                self.positive_iou_sum / self.num_positive_images
                if self.num_positive_images > 0
                else 0.0
            ),
            "positive_macro_dice": (
                self.positive_dice_sum / self.num_positive_images
                if self.num_positive_images > 0
                else 0.0
            ),
            "empty_slice_false_positive_rate": (
                self.empty_false_positive_images / self.num_empty_images
                if self.num_empty_images > 0
                else 0.0
            ),
            "empty_slice_mean_predicted_fraction": (
                self.empty_predicted_fraction_sum / self.num_empty_images
                if self.num_empty_images > 0
                else 0.0
            ),
            "num_positive_images": float(self.num_positive_images),
            "num_empty_images": float(self.num_empty_images),
        }
