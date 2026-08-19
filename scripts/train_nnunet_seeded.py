from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an official nnU-Net v2 trainer with an explicit random seed"
    )
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--trainer", default="nnUNetTrainerNoDA")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Runtime batch-size override (resource-only; planned default may be too large)",
    )
    parser.add_argument(
        "--num-data-workers",
        type=int,
        default=0,
        help="nnU-Net data-loader workers; use 0 on Windows to avoid shared-memory failures",
    )
    parser.add_argument(
        "--num-export-workers",
        type=int,
        default=1,
        help="Default nnU-Net preprocessing/export process count",
    )
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Instantiate the configured trainer and exit without training",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one real forward/backward batch in an isolated temporary results folder",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_data_workers < 0:
        parser.error("--num-data-workers cannot be negative")
    if args.num_export_workers < 1:
        parser.error("--num-export-workers must be at least 1")
    if args.validate_only and args.smoke_test:
        parser.error("--validate-only and --smoke-test are mutually exclusive")

    # These variables must be set before importing nnunetv2 because its
    # configuration module reads them at import time. A zero DA worker count
    # selects SingleThreadedAugmenter and avoids Windows shared-file mappings.
    os.environ["nnUNet_n_proc_DA"] = str(args.num_data_workers)
    os.environ["nnUNet_def_n_proc"] = str(args.num_export_workers)
    temporary_results: tempfile.TemporaryDirectory[str] | None = None
    if args.smoke_test:
        temporary_results = tempfile.TemporaryDirectory(prefix="nnunet_smoke_results_")
        os.environ["nnUNet_results"] = temporary_results.name

    try:
        import nnunetv2
        from batchgenerators.utilities.file_and_folder_operations import join, load_json
        from nnunetv2.paths import nnUNet_preprocessed
        from nnunetv2.utilities.dataset_name_id_conversion import (
            maybe_convert_to_dataset_name,
        )
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    except ImportError as error:
        raise ImportError(
            "Official nnU-Net v2 is unavailable. Install requirements-nnunet.txt "
            "inside the active project environment."
        ) from error

    deterministic = not args.non_deterministic
    set_seed(args.seed, deterministic)
    dataset_name = maybe_convert_to_dataset_name(args.dataset_id)
    package_root = Path(nnunetv2.__path__[0])
    trainer_class = recursive_find_python_class(
        str(package_root / "training" / "nnUNetTrainer"),
        args.trainer,
        "nnunetv2.training.nnUNetTrainer",
    )
    if trainer_class is None:
        raise RuntimeError(
            f"nnU-Net trainer {args.trainer!r} was not found. "
            "Use an nnunetv2 release that includes nnUNetTrainerNoDA."
        )
    preprocessed_dataset = join(nnUNet_preprocessed, dataset_name)
    plans = load_json(join(preprocessed_dataset, f"{args.plans}.json"))
    # Current nnU-Net v2 releases pass this runtime flag through the plans
    # mapping and consume it in nnUNetTrainer.__init__. It is intentionally not
    # persisted back to nnUNetPlans.json.
    plans["continue_training"] = bool(args.continue_training)
    try:
        planned_batch_size = int(plans["configurations"][args.configuration]["batch_size"])
        plans["configurations"][args.configuration]["batch_size"] = args.batch_size
    except KeyError as error:
        raise KeyError(
            f"Configuration {args.configuration!r} or its batch_size is missing from "
            f"{args.plans}.json"
        ) from error
    dataset_json = load_json(join(preprocessed_dataset, "dataset.json"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for nnU-Net but is unavailable")
    trainer_arguments = {
        "plans": plans,
        "configuration": args.configuration,
        "fold": args.fold,
        "dataset_json": dataset_json,
        "device": device,
    }
    trainer_parameters = inspect.signature(trainer_class.__init__).parameters
    # nnU-Net v2 releases before the dataset-unpacking refactor accepted this
    # flag. Newer releases unpack through the training lifecycle and removed it.
    if "unpack_dataset" in trainer_parameters:
        trainer_arguments["unpack_dataset"] = True
    trainer = trainer_class(**trainer_arguments)
    if args.validate_only:
        print(
            f"Trainer initialization validated: {trainer_class.__name__}; "
            f"dataset={dataset_name}; configuration={args.configuration}; fold={args.fold}; "
            f"batch_size={args.batch_size} (planned={planned_batch_size}); "
            f"data_workers={args.num_data_workers}; export_workers={args.num_export_workers}"
        )
        if temporary_results is not None:
            temporary_results.cleanup()
        return
    # The trainer constructor may initialize internal random state. Reapply the
    # experiment seed immediately before initialization/training.
    set_seed(args.seed, deterministic)
    if args.smoke_test:
        try:
            trainer.on_train_start()
            trainer.on_train_epoch_start()
            smoke_result = trainer.train_step(next(trainer.dataloader_train))
            print(
                "One-batch nnU-Net smoke test passed: "
                f"batch_size={args.batch_size}; data_workers={args.num_data_workers}; "
                f"export_workers={args.num_export_workers}; result={smoke_result}"
            )
        finally:
            # Do not call on_train_end: it writes checkpoint_final.pth. The
            # isolated temporary results tree is removed instead, so a smoke
            # test can never make the real run look complete.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if temporary_results is not None:
                temporary_results.cleanup()
        return
    if args.continue_training:
        latest_checkpoint = Path(trainer.output_folder) / "checkpoint_latest.pth"
        if not latest_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot continue; checkpoint not found: {latest_checkpoint}")
        trainer.load_checkpoint(str(latest_checkpoint))
        set_seed(args.seed, deterministic)
    trainer.run_training()

    output_folder = Path(trainer.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with (output_folder / "explicit_seed.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": args.seed,
                "deterministic": deterministic,
                "dataset": dataset_name,
                "configuration": args.configuration,
                "fold": args.fold,
                "trainer": args.trainer,
                "plans": args.plans,
                "batch_size": args.batch_size,
                "planned_batch_size": planned_batch_size,
                "num_data_workers": args.num_data_workers,
                "num_export_workers": args.num_export_workers,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
