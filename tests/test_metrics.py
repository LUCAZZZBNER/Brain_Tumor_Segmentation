from __future__ import annotations

import pytest
import torch

from brain_tumor_seg.metrics import (
    BinarySegmentationMeter,
    binary_metrics_per_sample,
    select_best_threshold,
)


def test_binary_segmentation_metrics_are_computed_per_image_and_globally() -> None:
    predictions = torch.tensor(
        [
            [[[1, 0], [0, 0]]],
            [[[1, 1], [0, 0]]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [[[1, 0], [0, 0]]],
            [[[1, 0], [1, 0]]],
        ],
        dtype=torch.float32,
    )
    logits = torch.where(predictions > 0, torch.tensor(10.0), torch.tensor(-10.0))
    meter = BinarySegmentationMeter(threshold=0.5)
    meter.update(logits, targets)
    metrics = meter.compute()
    assert metrics["macro_iou"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert metrics["micro_iou"] == pytest.approx(0.5)
    assert metrics["macro_dice"] == pytest.approx(0.75)
    assert metrics["micro_dice"] == pytest.approx(2.0 / 3.0)
    assert metrics["macro_precision"] == pytest.approx(0.75)
    assert metrics["micro_precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["micro_recall"] == pytest.approx(2.0 / 3.0)
    assert metrics["macro_specificity"] == pytest.approx(0.75)
    assert metrics["micro_specificity"] == pytest.approx(0.8)
    assert metrics["macro_accuracy"] == pytest.approx(0.75)
    assert metrics["micro_accuracy"] == pytest.approx(0.75)

    per_sample = binary_metrics_per_sample(logits, targets)
    assert per_sample["iou"].tolist() == pytest.approx([1.0, 1.0 / 3.0])
    assert per_sample["precision"].tolist() == pytest.approx([1.0, 0.5])
    assert per_sample["recall"].tolist() == pytest.approx([1.0, 0.5])


def test_threshold_selection_uses_validation_metric_and_resolves_ties_toward_reference() -> None:
    candidates = [
        {"threshold": 0.35, "macro_iou": 0.72},
        {"threshold": 0.45, "macro_iou": 0.75},
        {"threshold": 0.55, "macro_iou": 0.75},
    ]
    selected = select_best_threshold(candidates, "macro_iou", reference_threshold=0.5)
    assert selected["threshold"] == 0.45


def test_positive_metrics_exclude_empty_targets_and_empty_false_positives_are_reported() -> None:
    predictions = torch.tensor(
        [
            [[[1, 0], [0, 0]]],
            [[[1, 0], [0, 0]]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [[[1, 0], [0, 0]]],
            [[[0, 0], [0, 0]]],
        ],
        dtype=torch.float32,
    )
    logits = torch.where(predictions > 0, torch.tensor(10.0), torch.tensor(-10.0))
    meter = BinarySegmentationMeter()
    meter.update(logits, targets)
    metrics = meter.compute()
    assert metrics["positive_macro_iou"] == pytest.approx(1.0)
    assert metrics["positive_macro_dice"] == pytest.approx(1.0)
    assert metrics["empty_slice_false_positive_rate"] == pytest.approx(1.0)
    assert metrics["empty_slice_mean_predicted_fraction"] == pytest.approx(0.25)
    assert metrics["num_positive_images"] == 1
    assert metrics["num_empty_images"] == 1
