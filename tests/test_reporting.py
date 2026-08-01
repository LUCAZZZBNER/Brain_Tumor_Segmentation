from __future__ import annotations

import csv

from brain_tumor_seg.reporting import flatten_epoch_record, update_training_artifacts
from brain_tumor_seg.utils import append_jsonl


def test_training_history_is_flattened_and_written(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    record = {
        "epoch": 1,
        "learning_rate": 0.001,
        "selection_threshold": 0.5,
        "best_threshold": 0.5,
        "best_val_metric": 0.75,
        "elapsed_seconds": 12.0,
        "train": {"loss": 0.8, "macro_iou": 0.6, "macro_dice": 0.7},
        "val": {"loss": 0.7, "macro_iou": 0.65, "macro_dice": 0.75},
    }
    row = flatten_epoch_record(record)
    assert row["train_loss"] == 0.8
    assert row["val_macro_iou"] == 0.65

    append_jsonl(metrics_path, record)
    update_training_artifacts(metrics_path, tmp_path / "history")
    with (tmp_path / "history" / "epochs.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["epoch"] == "1"
    assert float(rows[0]["val_macro_dice"]) == 0.75
    assert (tmp_path / "history" / "curves" / "loss.png").is_file()
    assert (tmp_path / "history" / "curves" / "iou.png").is_file()
