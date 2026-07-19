from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import BinarySegmentationMeter


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
) -> dict[str, float]:
    model.train()
    meter = BinarySegmentationMeter(threshold=threshold)
    loss_sum = 0.0
    num_samples = 0
    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        loss_sum += loss.detach().item() * batch_size
        num_samples += batch_size
        meter.update(logits.detach(), masks)
        progress.set_postfix(loss=f"{loss_sum / num_samples:.4f}")
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
) -> dict[str, Any]:
    model.eval()
    meter = BinarySegmentationMeter(threshold=threshold)
    class_meters: dict[str, BinarySegmentationMeter] = {}
    loss_sum = 0.0
    num_samples = 0
    num_saved = 0
    prediction_root = Path(predictions_dir) if predictions_dir is not None else None

    progress = tqdm(loader, desc=description, leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        num_samples += batch_size
        meter.update(logits, masks)

        tumor_types: list[str] = list(batch["tumor_type"])
        for index, tumor_type in enumerate(tumor_types):
            class_meter = class_meters.setdefault(
                tumor_type, BinarySegmentationMeter(threshold=threshold)
            )
            class_meter.update(logits[index : index + 1], masks[index : index + 1])

        if prediction_root is not None:
            probabilities = torch.sigmoid(logits).cpu()
            for index in range(batch_size):
                if max_saved_predictions is not None and num_saved >= max_saved_predictions:
                    break
                prediction = (probabilities[index, 0] >= threshold).numpy().astype("uint8") * 255
                # DataLoader collates (height, width) into two tensors.
                original_height = int(batch["original_size"][0][index])
                original_width = int(batch["original_size"][1][index])
                output = Image.fromarray(prediction, mode="L").resize(
                    (original_width, original_height), resample=Image.Resampling.NEAREST
                )
                category_dir = prediction_root / tumor_types[index]
                category_dir.mkdir(parents=True, exist_ok=True)
                safe_name = str(batch["sample_id"][index]).replace("/", "_").replace("\\", "_")
                output.save(category_dir / f"{safe_name}_pred.png")
                num_saved += 1

        progress.set_postfix(loss=f"{loss_sum / num_samples:.4f}")

    metrics: dict[str, Any] = meter.compute()
    metrics["loss"] = loss_sum / num_samples
    metrics["num_samples"] = num_samples
    metrics["per_class"] = {
        tumor_type: {**class_meter.compute(), "num_samples": class_meter.num_images}
        for tumor_type, class_meter in sorted(class_meters.items())
    }
    if prediction_root is not None:
        metrics["num_saved_predictions"] = num_saved
    return metrics

