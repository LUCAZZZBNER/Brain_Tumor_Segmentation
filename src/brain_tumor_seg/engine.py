from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import BinarySegmentationMeter, binary_metrics_per_sample
from .reporting import save_segmentation_comparison


BATCH_LOG_FIELDS = (
    "batch",
    "batch_size",
    "processed_samples",
    "batch_loss",
    "running_loss",
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

SAMPLE_LOG_FIELDS = (
    "sample_id",
    "tumor_type",
    "image_path",
    "mask_path",
    "iou",
    "dice",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    threshold: float,
    amp: bool,
    gradient_clip_norm: float | None,
    epoch: int,
    batch_log_path: str | Path | None = None,
) -> dict[str, float]:
    model.train()
    meter = BinarySegmentationMeter(threshold=threshold)
    loss_sum = 0.0
    num_samples = 0
    batch_log_handle, batch_log_writer = _open_csv_log(batch_log_path, BATCH_LOG_FIELDS)
    try:
        progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
        for batch_index, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                logits = model(images)
                loss = criterion(logits, masks)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, batch={batch_index}. "
                    "Training was stopped before updating weights or saving a checkpoint."
                )
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None and gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.shape[0]
            batch_loss = loss.detach().item()
            loss_sum += batch_loss * batch_size
            num_samples += batch_size
            meter.update(logits.detach(), masks)
            running_loss = loss_sum / num_samples
            if batch_log_writer is not None:
                running_metrics = meter.compute()
                batch_log_writer.writerow(
                    {
                        "batch": batch_index,
                        "batch_size": batch_size,
                        "processed_samples": num_samples,
                        "batch_loss": batch_loss,
                        "running_loss": running_loss,
                        **running_metrics,
                    }
                )
                batch_log_handle.flush()
            progress.set_postfix(loss=f"{running_loss:.4f}")
    finally:
        if batch_log_handle is not None:
            batch_log_handle.close()
    metrics = meter.compute()
    metrics["loss"] = loss_sum / num_samples
    return metrics


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    threshold: float,
    amp: bool,
    description: str,
    predictions_dir: str | Path | None = None,
    max_saved_predictions: int | None = None,
    threshold_search: list[float] | None = None,
    batch_log_path: str | Path | None = None,
    sample_log_path: str | Path | None = None,
    data_root: str | Path | None = None,
    comparisons_dir: str | Path | None = None,
    save_probability_maps: bool = True,
    channel_mode: str = "grayscale",
) -> dict[str, Any]:
    model.eval()
    meter = BinarySegmentationMeter(threshold=threshold)
    threshold_meters = {
        candidate: BinarySegmentationMeter(threshold=candidate)
        for candidate in sorted(set(threshold_search or []))
    }
    class_meters: dict[str, BinarySegmentationMeter] = {}
    loss_sum = 0.0
    num_samples = 0
    num_saved = 0
    prediction_root = Path(predictions_dir) if predictions_dir is not None else None
    comparison_root = Path(comparisons_dir) if comparisons_dir is not None else None
    data_root_path = Path(data_root) if data_root is not None else None
    if comparison_root is not None and data_root_path is None:
        raise ValueError("data_root is required when saving comparison figures")

    batch_log_handle, batch_log_writer = _open_csv_log(batch_log_path, BATCH_LOG_FIELDS)
    sample_log_handle, sample_log_writer = _open_csv_log(sample_log_path, SAMPLE_LOG_FIELDS)

    try:
        progress = tqdm(loader, desc=description, leave=False)
        for batch_index, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with _autocast(device, amp):
                logits = model(images)
                loss = criterion(logits, masks)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite evaluation loss in {description}, batch={batch_index}. "
                    "Evaluation was stopped to avoid reporting invalid all-background metrics."
                )

            batch_size = images.shape[0]
            batch_loss = loss.item()
            loss_sum += batch_loss * batch_size
            num_samples += batch_size
            meter.update(logits, masks)
            for threshold_meter in threshold_meters.values():
                threshold_meter.update(logits, masks)

            tumor_types: list[str] = list(batch["tumor_type"])
            for index, tumor_type in enumerate(tumor_types):
                class_meter = class_meters.setdefault(
                    tumor_type, BinarySegmentationMeter(threshold=threshold)
                )
                class_meter.update(logits[index : index + 1], masks[index : index + 1])

            running_loss = loss_sum / num_samples
            if batch_log_writer is not None:
                batch_log_writer.writerow(
                    {
                        "batch": batch_index,
                        "batch_size": batch_size,
                        "processed_samples": num_samples,
                        "batch_loss": batch_loss,
                        "running_loss": running_loss,
                        **meter.compute(),
                    }
                )
                batch_log_handle.flush()

            per_sample = binary_metrics_per_sample(logits, masks, threshold=threshold)
            if sample_log_writer is not None:
                cpu_values = {name: values.detach().cpu() for name, values in per_sample.items()}
                for index in range(batch_size):
                    sample_log_writer.writerow(
                        {
                            "sample_id": batch["sample_id"][index],
                            "tumor_type": tumor_types[index],
                            "image_path": batch["image_path"][index],
                            "mask_path": batch["mask_path"][index],
                            **{
                                name: float(cpu_values[name][index].item())
                                for name in (
                                    "iou",
                                    "dice",
                                    "precision",
                                    "recall",
                                    "specificity",
                                    "accuracy",
                                )
                            },
                            **{
                                name: int(cpu_values[name][index].item())
                                for name in (
                                    "true_positive",
                                    "false_positive",
                                    "false_negative",
                                    "true_negative",
                                )
                            },
                        }
                    )
                sample_log_handle.flush()

            if prediction_root is not None or comparison_root is not None:
                probabilities = torch.sigmoid(logits).detach().cpu()
                for index in range(batch_size):
                    if max_saved_predictions is not None and num_saved >= max_saved_predictions:
                        break
                    probability = probabilities[index, 0].numpy()
                    prediction = (probability >= threshold).astype("uint8") * 255
                    # DataLoader collates (height, width) into two tensors.
                    original_height = int(batch["original_size"][0][index])
                    original_width = int(batch["original_size"][1][index])
                    safe_name = (
                        str(batch["sample_id"][index]).replace("/", "_").replace("\\", "_")
                    )
                    if prediction_root is not None:
                        category_dir = prediction_root / tumor_types[index]
                        category_dir.mkdir(parents=True, exist_ok=True)
                        output = Image.fromarray(prediction, mode="L").resize(
                            (original_width, original_height),
                            resample=Image.Resampling.NEAREST,
                        )
                        output.save(category_dir / f"{safe_name}_pred.png")
                        if save_probability_maps:
                            probability_output = Image.fromarray(
                                np.clip(probability * 255.0, 0, 255).astype("uint8"), mode="L"
                            ).resize(
                                (original_width, original_height),
                                resample=Image.Resampling.BILINEAR,
                            )
                            probability_output.save(category_dir / f"{safe_name}_prob.png")
                    if comparison_root is not None:
                        comparison_path = comparison_root / tumor_types[index] / f"{safe_name}.png"
                        save_segmentation_comparison(
                            data_root_path / batch["image_path"][index],
                            data_root_path / batch["mask_path"][index],
                            probability,
                            threshold=threshold,
                            output_path=comparison_path,
                            channel_mode=channel_mode,
                        )
                    num_saved += 1

            progress.set_postfix(loss=f"{running_loss:.4f}")
    finally:
        if batch_log_handle is not None:
            batch_log_handle.close()
        if sample_log_handle is not None:
            sample_log_handle.close()

    metrics: dict[str, Any] = meter.compute()
    metrics["loss"] = loss_sum / num_samples
    metrics["num_samples"] = num_samples
    metrics["per_class"] = {
        tumor_type: {**class_meter.compute(), "num_samples": class_meter.num_images}
        for tumor_type, class_meter in sorted(class_meters.items())
    }
    if threshold_meters:
        metrics["threshold_search"] = [
            {"threshold": candidate, **threshold_meter.compute()}
            for candidate, threshold_meter in threshold_meters.items()
        ]
    if prediction_root is not None or comparison_root is not None:
        metrics["num_saved_predictions"] = num_saved
    return metrics


def _open_csv_log(path: str | Path | None, fieldnames: tuple[str, ...]):
    if path is None:
        return None, None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer
