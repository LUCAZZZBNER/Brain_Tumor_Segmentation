from __future__ import annotations

import argparse
import csv
import json
import shutil

import torch
from torch.utils.data import DataLoader

from .config import load_config, project_path
from .data import BrainTumorDataset, build_transform, seed_worker
from .engine import evaluate_one_epoch
from .losses import build_loss
from .model import build_model
from .reporting import save_evaluation_plots, write_csv_rows, write_per_class_metrics
from .splits import read_manifest, sha256_file, verify_manifest_files
from .utils import select_device, set_reproducibility, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen checkpoint on val or test")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--checkpoint", default=None, help="Defaults to output_dir/checkpoints/best.pt"
    )
    parser.add_argument("--split", choices=("val", "test"), default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Explicitly allow replacing an existing split metrics file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    set_reproducibility(seed, bool(config["training"].get("deterministic", True)))
    device = select_device(args.device)
    data_config = config["data"]
    manifest_path = project_path(config, data_config["manifest"])
    metadata_path = project_path(config, data_config["split_metadata"])
    with metadata_path.open("r", encoding="utf-8") as handle:
        split_metadata = json.load(handle)
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != split_metadata.get("manifest_sha256"):
        raise ValueError("Manifest hash differs from split metadata")
    all_samples = read_manifest(manifest_path)
    data_root = project_path(config, data_config["root"])
    split = args.split or str(config["evaluation"].get("split", "test"))
    samples = [sample for sample in all_samples if sample.split == split]
    verify_manifest_files(
        samples,
        data_root,
        verify_hashes=bool(data_config.get("verify_file_hashes", False)),
    )
    dataset = BrainTumorDataset(
        data_root,
        samples,
        build_transform(data_config, train=False),
        channel_mode=str(data_config.get("channel_mode", "grayscale")),
    )
    num_workers = int(data_config["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(data_config.get("pin_memory", True)),
        persistent_workers=bool(data_config.get("persistent_workers", True)) and num_workers > 0,
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    output_dir = project_path(config, config["project"]["output_dir"])
    result_path = output_dir / f"{split}_metrics.json"
    evaluation_dir = output_dir / "evaluation" / split
    comparisons_dir = output_dir / "comparisons" / split
    predictions_artifact_dir = output_dir / "predictions" / split
    existing_artifacts = [
        path
        for path in (result_path, evaluation_dir, comparisons_dir, predictions_artifact_dir)
        if path.exists()
    ]
    if existing_artifacts and not args.overwrite_results:
        raise FileExistsError(
            f"Evaluation artifacts already exist: {existing_artifacts}. Refusing to overwrite. "
            "Pass --overwrite-results only when intentionally reproducing the same evaluation."
        )
    checkpoint_path = (
        project_path(config, args.checkpoint)
        if args.checkpoint
        else output_dir / "checkpoints" / "best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise ValueError("Checkpoint and current manifest use different train/val/test partitions")
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    criterion = build_loss(config["loss"])
    threshold = float(checkpoint.get("threshold", config["metrics"]["threshold"]))
    threshold_search = [
        float(value) for value in config["metrics"].get("threshold_search", [])
    ]

    save_predictions = bool(config["evaluation"].get("save_predictions", True))
    save_predictions = save_predictions and not args.no_save_predictions
    predictions_dir = predictions_artifact_dir if save_predictions else None
    save_comparisons = bool(config["evaluation"].get("save_comparison_figures", True))
    save_comparisons = save_comparisons and not args.no_save_predictions
    maximum_value = config["evaluation"].get("max_saved_predictions", 100)
    max_saved_predictions = None if maximum_value is None else int(maximum_value)
    if args.overwrite_results:
        if result_path.exists():
            result_path.unlink()
        for artifact_dir in (evaluation_dir, comparisons_dir, predictions_artifact_dir):
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
    metrics = evaluate_one_epoch(
        model,
        loader,
        criterion,
        device,
        threshold=threshold,
        amp=bool(config["training"].get("amp", True)),
        description=split,
        threshold_search=threshold_search,
        predictions_dir=predictions_dir,
        max_saved_predictions=(
            max_saved_predictions if (save_predictions or save_comparisons) else 0
        ),
        batch_log_path=evaluation_dir / "batches.csv",
        sample_log_path=evaluation_dir / "samples.csv",
        data_root=data_root,
        comparisons_dir=comparisons_dir if save_comparisons else None,
        save_probability_maps=bool(config["evaluation"].get("save_probability_maps", True)),
        channel_mode=str(data_config.get("channel_mode", "grayscale")),
    )
    write_per_class_metrics(metrics, evaluation_dir / "per_class.csv")
    with (evaluation_dir / "samples.csv").open("r", encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle))
    save_evaluation_plots(sample_rows, metrics, evaluation_dir / "plots")
    write_csv_rows(
        evaluation_dir / "summary.csv",
        [
            {
                "split": split,
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "threshold": threshold,
                **{
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
            }
        ],
    )
    result = {
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_metric": checkpoint.get("primary_metric"),
        "best_validation_metric": float(checkpoint["best_metric"]),
        "manifest_sha256": manifest_hash,
        "split_level": split_metadata["split_level"],
        "threshold": threshold,
        "metrics": metrics,
    }
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main()
