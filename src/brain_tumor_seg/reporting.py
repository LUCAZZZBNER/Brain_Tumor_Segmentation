from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METRIC_NAMES = (
    "loss",
    "macro_iou",
    "micro_iou",
    "macro_dice",
    "micro_dice",
    "macro_precision",
    "micro_precision",
    "macro_recall",
    "micro_recall",
    "macro_specificity",
    "micro_specificity",
    "macro_accuracy",
    "micro_accuracy",
    "positive_macro_iou",
    "positive_macro_dice",
    "empty_slice_false_positive_rate",
    "empty_slice_mean_predicted_fraction",
    "num_positive_images",
    "num_empty_images",
)


def write_csv_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    rows = [dict(row) for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered_fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered_fields:
                    ordered_fields.append(key)
        fieldnames = ordered_fields
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on JSONL line {line_number}")
            records.append(value)
    return records


def flatten_epoch_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": record["epoch"],
        "learning_rate": record.get("learning_rate"),
        "selection_threshold": record.get("selection_threshold"),
        "selection_metric_value": record.get("selection_metric_value"),
        "best_threshold": record.get("best_threshold"),
        "best_val_metric": record.get("best_val_metric"),
        "elapsed_seconds": record.get("elapsed_seconds"),
    }
    for split in ("train", "val"):
        metrics = record.get(split, {})
        if not isinstance(metrics, Mapping):
            continue
        for name in METRIC_NAMES:
            value = metrics.get(name)
            if isinstance(value, (int, float)):
                row[f"{split}_{name}"] = value
    return row


def update_training_artifacts(metrics_path: str | Path, history_dir: str | Path) -> None:
    records = read_jsonl(metrics_path)
    rows = [flatten_epoch_record(record) for record in records]
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(history_dir / "epochs.csv", rows)
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    curve_specs = {
        "loss.png": (
            "Loss",
            (("train_loss", "Train loss"), ("val_loss", "Validation loss")),
        ),
        "iou.png": (
            "Intersection over Union",
            (
                ("train_macro_iou", "Train macro IoU"),
                ("val_macro_iou", "Validation macro IoU"),
                ("train_micro_iou", "Train micro IoU"),
                ("val_micro_iou", "Validation micro IoU"),
            ),
        ),
        "dice.png": (
            "Dice",
            (
                ("train_macro_dice", "Train macro Dice"),
                ("val_macro_dice", "Validation macro Dice"),
                ("train_micro_dice", "Train micro Dice"),
                ("val_micro_dice", "Validation micro Dice"),
            ),
        ),
        "precision_recall.png": (
            "Precision and recall",
            (
                ("train_macro_precision", "Train precision"),
                ("val_macro_precision", "Validation precision"),
                ("train_macro_recall", "Train recall"),
                ("val_macro_recall", "Validation recall"),
            ),
        ),
        "specificity_accuracy.png": (
            "Specificity and accuracy",
            (
                ("train_macro_specificity", "Train specificity"),
                ("val_macro_specificity", "Validation specificity"),
                ("train_macro_accuracy", "Train accuracy"),
                ("val_macro_accuracy", "Validation accuracy"),
            ),
        ),
        "learning_rate.png": (
            "Learning rate",
            (("learning_rate", "Learning rate"),),
        ),
        "model_selection.png": (
            "Validation model selection",
            (
                ("selection_metric_value", "Current selected-threshold metric"),
                ("best_val_metric", "Best validation metric"),
                ("selection_threshold", "Selected threshold"),
            ),
        ),
    }
    curves_dir = history_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    for filename, (title, series) in curve_specs.items():
        available = [(key, label) for key, label in series if any(key in row for row in rows)]
        if available:
            _save_line_plot(curves_dir / filename, epochs, rows, title, available)


def write_per_class_metrics(metrics: Mapping[str, Any], path: str | Path) -> None:
    rows = []
    for tumor_type, values in sorted(metrics.get("per_class", {}).items()):
        rows.append({"tumor_type": tumor_type, **values})
    write_csv_rows(path, rows)


def save_evaluation_plots(
    sample_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pyplot = _pyplot()

    figure, axes = pyplot.subplots(2, 2, figsize=(11, 8))
    for axis, name, title in zip(
        axes.flat,
        ("iou", "dice", "precision", "recall"),
        ("Per-sample IoU", "Per-sample Dice", "Per-sample precision", "Per-sample recall"),
        strict=True,
    ):
        values = [float(row[name]) for row in sample_rows if name in row]
        axis.hist(values, bins=20, range=(0.0, 1.0), color="#4472c4", alpha=0.85)
        axis.set_title(title)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel(name.capitalize())
        axis.set_ylabel("Samples")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    _atomic_save_figure(figure, output_dir / "sample_metric_distributions.png")
    pyplot.close(figure)

    per_class = metrics.get("per_class", {})
    if per_class:
        classes = sorted(per_class)
        positions = list(range(len(classes)))
        width = 0.35
        figure, axis = pyplot.subplots(figsize=(max(8, len(classes) * 2.2), 5))
        iou = [float(per_class[name]["macro_iou"]) for name in classes]
        dice = [float(per_class[name]["macro_dice"]) for name in classes]
        axis.bar([value - width / 2 for value in positions], iou, width, label="Macro IoU")
        axis.bar([value + width / 2 for value in positions], dice, width, label="Macro Dice")
        axis.set_xticks(positions, classes, rotation=15, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Score")
        axis.set_title("Metrics by tumor type")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        figure.tight_layout()
        _atomic_save_figure(figure, output_dir / "per_class_metrics.png")
        pyplot.close(figure)


def save_segmentation_comparison(
    image_path: str | Path,
    mask_path: str | Path,
    probability: Any,
    *,
    threshold: float,
    output_path: str | Path,
    channel_mode: str = "grayscale",
) -> None:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
        if channel_mode.lower() == "flair_green":
            image = image_file.convert("RGB").getchannel("G")
        elif channel_mode.lower() == "rgb_multimodal":
            image = image_file.convert("RGB")
        elif channel_mode.lower() == "grayscale":
            image = image_file.convert("L")
        else:
            raise ValueError(f"Unsupported image channel mode: {channel_mode}")
        truth = (np.asarray(mask_file.convert("L"), dtype=np.uint8) >= 128).astype(np.uint8)
    probability_image = Image.fromarray(np.asarray(probability, dtype=np.float32), mode="F")
    probability_image = probability_image.resize(image.size, resample=Image.Resampling.BILINEAR)
    probability_array = np.asarray(probability_image, dtype=np.float32)
    prediction = probability_array >= threshold
    image_array = np.asarray(image, dtype=np.uint8)

    overlay = (
        np.stack([image_array, image_array, image_array], axis=-1)
        if image_array.ndim == 2
        else image_array.copy()
    ).astype(np.float32)
    true_positive = prediction & (truth > 0)
    false_positive = prediction & (truth == 0)
    false_negative = ~prediction & (truth > 0)
    overlay[true_positive] = 0.45 * overlay[true_positive] + 0.55 * np.array([0, 255, 0])
    overlay[false_positive] = 0.45 * overlay[false_positive] + 0.55 * np.array([255, 0, 0])
    overlay[false_negative] = 0.45 * overlay[false_negative] + 0.55 * np.array([255, 215, 0])

    pyplot = _pyplot()
    figure, axes = pyplot.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(image_array, cmap="gray" if image_array.ndim == 2 else None)
    axes[0].set_title("Original MRI")
    axes[1].imshow(truth, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground truth")
    axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (t={threshold:.2f})")
    axes[3].imshow(overlay.astype(np.uint8))
    axes[3].set_title("Overlay: TP green / FP red / FN yellow")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    _atomic_save_figure(figure, Path(output_path))
    pyplot.close(figure)


def _save_line_plot(
    path: Path,
    epochs: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    title: str,
    series: Sequence[tuple[str, str]],
) -> None:
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(9, 5.5))
    for key, label in series:
        values = [
            float(row[key]) if row.get(key) not in (None, "") else float("nan")
            for row in rows
        ]
        axis.plot(epochs, values, marker="o", markersize=3, linewidth=1.6, label=label)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _atomic_save_figure(figure, path)
    pyplot.close(figure)


def _atomic_save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=160, bbox_inches="tight")
    temporary.replace(path)


@lru_cache(maxsize=1)
def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot
    except ImportError as error:
        raise RuntimeError(
            "Plot generation requires matplotlib. Install the project requirements first."
        ) from error
    return pyplot
