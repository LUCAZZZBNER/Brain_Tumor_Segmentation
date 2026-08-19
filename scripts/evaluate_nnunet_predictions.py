from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def case_identifier(sample_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    value = "k3m_" + "".join(character if character in allowed else "_" for character in sample_id)
    return value + "case" if value[-1] == "_" else value


def divide(numerator: float, denominator: float, empty_value: float = 1.0) -> float:
    return numerator / denominator if denominator > 0 else empty_value


def sample_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = float(np.logical_and(prediction, target).sum())
    fp = float(np.logical_and(prediction, np.logical_not(target)).sum())
    fn = float(np.logical_and(np.logical_not(prediction), target).sum())
    tn = float(np.logical_and(np.logical_not(prediction), np.logical_not(target)).sum())
    predicted = tp + fp
    target_count = tp + fn
    union = tp + fp + fn
    total = tp + fp + fn + tn
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "iou": divide(tp, union),
        "dice": divide(2.0 * tp, predicted + target_count),
        "precision": divide(tp, predicted, float(target_count == 0)),
        "recall": divide(tp, target_count, float(predicted == 0)),
        "specificity": divide(tn, tn + fp),
        "accuracy": divide(tp + tn, total),
        "target_pixels": target_count,
        "predicted_pixels": predicted,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No prediction rows were evaluated")
    metric_names = ("iou", "dice", "precision", "recall", "specificity", "accuracy")
    values: dict[str, float] = {
        f"macro_{name}": float(np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    }
    totals = {
        key: sum(float(row[key]) for row in rows)
        for key in ("true_positive", "false_positive", "false_negative", "true_negative")
    }
    tp, fp, fn, tn = (
        totals["true_positive"],
        totals["false_positive"],
        totals["false_negative"],
        totals["true_negative"],
    )
    values.update(
        {
            "micro_iou": divide(tp, tp + fp + fn),
            "micro_dice": divide(2.0 * tp, 2.0 * tp + fp + fn),
            "micro_precision": divide(tp, tp + fp, float(tp + fn == 0)),
            "micro_recall": divide(tp, tp + fn, float(tp + fp == 0)),
            "micro_specificity": divide(tn, tn + fp),
            "micro_accuracy": divide(tp + tn, tp + fp + fn + tn),
        }
    )
    positive = [row for row in rows if float(row["target_pixels"]) > 0]
    empty = [row for row in rows if float(row["target_pixels"]) == 0]
    values.update(
        {
            "positive_macro_iou": float(np.mean([row["iou"] for row in positive])),
            "positive_macro_dice": float(np.mean([row["dice"] for row in positive])),
            "empty_slice_false_positive_rate": float(
                np.mean([float(row["predicted_pixels"] > 0) for row in empty])
            ),
            "empty_slice_mean_predicted_fraction": float(
                np.mean([float(row["predicted_pixels"]) / (256.0 * 256.0) for row in empty])
            ),
            "num_samples": float(len(rows)),
            "num_positive_images": float(len(positive)),
            "num_empty_images": float(len(empty)),
        }
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate official nnU-Net predictions with the project metrics"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_config = config["data"]
    manifest_path = project_root / data_config["manifest"]
    metadata_path = project_root / data_config["split_metadata"]
    data_root = project_root / data_config["root"]
    predictions_dir = Path(args.predictions).resolve()
    with metadata_path.open("r", encoding="utf-8") as handle:
        split_metadata = json.load(handle)
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != split_metadata.get("manifest_sha256"):
        raise ValueError("Manifest hash differs from split metadata")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        test_samples = [row for row in csv.DictReader(handle) if row["split"] == "test"]

    rows: list[dict[str, Any]] = []
    for sample in test_samples:
        prediction_path = predictions_dir / f"{case_identifier(sample['sample_id'])}.png"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing nnU-Net prediction: {prediction_path}")
        with Image.open(prediction_path) as prediction_file:
            prediction = np.asarray(prediction_file.convert("L"), dtype=np.uint8) > 0
        with Image.open(data_root / sample["mask_path"]) as mask_file:
            target = np.asarray(mask_file.convert("L"), dtype=np.uint8) >= 128
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch for {sample['sample_id']}: "
                f"{prediction.shape} vs {target.shape}"
            )
        rows.append({**sample_metrics(prediction, target), **sample})

    metrics = aggregate(rows)
    patient_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if float(row["target_pixels"]) > 0:
            patient_values[str(row["group_id"])].append(float(row["iou"]))
    metrics["patient_positive_macro_iou"] = float(
        np.mean([np.mean(values) for values in patient_values.values()])
    )
    metrics["num_patients"] = float(len({row["group_id"] for row in rows}))

    output_dir = project_root / config["project"]["output_dir"]
    evaluation_dir = output_dir / "evaluation" / "test"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with (evaluation_dir / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "model": "official_nnunetv2_2d",
        "seed": int(config["project"]["seed"]),
        "trainer": config["nnunet"]["trainer"],
        "augmentation": "disabled_by_trainer",
        "manifest_sha256": manifest_hash,
        "split_level": split_metadata["split_level"],
        "split": "test",
        "metrics": metrics,
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
