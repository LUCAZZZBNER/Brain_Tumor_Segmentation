from __future__ import annotations

import pytest
import torch

from brain_tumor_seg.metrics import BinarySegmentationMeter


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

