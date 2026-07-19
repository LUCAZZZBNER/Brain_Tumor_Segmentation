from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import load_config, project_path
from .data import BrainTumorDataset, build_transform, seed_worker
from .engine import evaluate_one_epoch, train_one_epoch
from .losses import build_loss
from .model import build_model
from .splits import read_manifest, sha256_file, verify_manifest_files
from .utils import (
    append_jsonl,
    atomic_torch_save,
    environment_info,
    select_device,
    set_reproducibility,
    strip_internal_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the U-Net baseline using train/val only")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--resume", default=None, help="Checkpoint path; overrides training.resume")
    return parser.parse_args()


def _load_and_verify_split(config: dict[str, Any]):
    data_config = config["data"]
    manifest_path = project_path(config, data_config["manifest"])
    metadata_path = project_path(config, data_config["split_metadata"])
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "The fixed split does not exist. Run: python -m brain_tumor_seg.make_splits "
            f"--config {config['_config_path']}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != metadata.get("manifest_sha256"):
        raise ValueError("Manifest hash differs from split metadata; the fixed split was modified")
    samples = read_manifest(manifest_path)
    data_root = project_path(config, data_config["root"])
    # Verify only train/val content here. Test image bytes are first read by evaluate.py.
    development_samples = [sample for sample in samples if sample.split in {"train", "val"}]
    verify_manifest_files(
        development_samples,
        data_root,
        verify_hashes=bool(data_config.get("verify_file_hashes", False)),
    )
    return samples, data_root, actual_manifest_hash, metadata


def _build_loader(
    dataset: BrainTumorDataset,
    config: dict[str, Any],
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    data_config = config["data"]
    num_workers = int(data_config["num_workers"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=bool(data_config.get("persistent_workers", True)) and num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    deterministic = bool(config["training"].get("deterministic", True))
    set_reproducibility(seed, deterministic)
    device = select_device(args.device)
    samples, data_root, manifest_hash, split_metadata = _load_and_verify_split(config)

    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    # The test subset is deliberately not instantiated or evaluated in this program.
    train_dataset = BrainTumorDataset(
        data_root, train_samples, build_transform(config["data"], train=True)
    )
    val_dataset = BrainTumorDataset(
        data_root, val_samples, build_transform(config["data"], train=False)
    )
    train_loader = _build_loader(train_dataset, config, shuffle=True, seed=seed)
    val_loader = _build_loader(val_dataset, config, shuffle=False, seed=seed + 1)

    model = build_model(config["model"]).to(device)
    criterion = build_loss(config["loss"])
    optimizer_config = config["optimizer"]
    if str(optimizer_config["name"]).lower() != "adamw":
        raise ValueError("The baseline currently supports optimizer.name=adamw")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler_config = config["scheduler"]
    if str(scheduler_config["name"]).lower() != "reduce_on_plateau":
        raise ValueError("The baseline currently supports scheduler.name=reduce_on_plateau")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(scheduler_config["min_learning_rate"]),
    )
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    output_dir = project_path(config, config["project"]["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    resume_value = args.resume or config["training"].get("resume")
    if metrics_path.exists() and not resume_value:
        raise FileExistsError(
            f"Experiment output already exists: {metrics_path}. Change project.output_dir or resume "
            "from a checkpoint; existing results will not be silently overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", strip_internal_config(config))
    write_json(output_dir / "environment.json", environment_info())
    write_json(output_dir / "split_metadata_snapshot.json", split_metadata)

    primary_metric = str(config["metrics"].get("primary", "macro_iou"))
    threshold = float(config["metrics"]["threshold"])
    start_epoch = 1
    best_metric = float("-inf")
    bad_epochs = 0
    if resume_value:
        resume_path = project_path(config, resume_value)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise ValueError("Resume checkpoint was trained with a different data split")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        bad_epochs = int(checkpoint.get("bad_epochs", 0))

    print(
        f"device={device}; train={len(train_dataset)}; val={len(val_dataset)}; "
        f"split_level={split_metadata['split_level']}"
    )
    epochs = int(config["training"]["epochs"])
    early_stopping_patience = int(config["training"]["early_stopping_patience"])
    gradient_clip = config["training"].get("gradient_clip_norm")
    started = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            threshold=threshold,
            amp=amp_enabled,
            gradient_clip_norm=float(gradient_clip) if gradient_clip is not None else None,
            epoch=epoch,
        )
        val_metrics = evaluate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold,
            amp=amp_enabled,
            description=f"val   {epoch:03d}",
        )
        current_metric = float(val_metrics[primary_metric])
        scheduler.step(current_metric)
        improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
            bad_epochs = 0
        else:
            bad_epochs += 1

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "primary_metric": primary_metric,
            "best_val_metric": best_metric,
            "elapsed_seconds": time.time() - started,
        }
        append_jsonl(metrics_path, record)
        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
            "primary_metric": primary_metric,
            "manifest_sha256": manifest_hash,
            "config": strip_internal_config(config),
        }
        atomic_torch_save(checkpoint_payload, checkpoints_dir / "last.pt")
        if improved:
            atomic_torch_save(checkpoint_payload, checkpoints_dir / "best.pt")
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_{primary_metric}={current_metric:.4f} "
            f"best={best_metric:.4f}"
        )
        if bad_epochs >= early_stopping_patience:
            print(f"Early stopping after {bad_epochs} epochs without validation improvement")
            break

    write_json(
        output_dir / "training_summary.json",
        {
            "last_epoch": last_epoch,
            "best_validation_metric": best_metric,
            "primary_metric": primary_metric,
            "manifest_sha256": manifest_hash,
            "test_evaluated": False,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"Training complete. Best checkpoint: {checkpoints_dir / 'best.pt'}")
    print("The test set has not been touched. Run the separate evaluate command exactly once.")


if __name__ == "__main__":
    main()
