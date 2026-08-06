from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from summarize_kaggle3m_clean_m4_p_final_component_ablation import (
    EXPERIMENTS,
    ROOT,
    run_dir,
)

CURVES = (
    ("train", "positive_macro_iou", "Train Positive Macro IoU", "Positive Macro IoU"),
    ("val", "positive_macro_iou", "Validation Positive Macro IoU", "Positive Macro IoU"),
    ("train", "loss", "Train Loss", "Loss"),
    ("val", "loss", "Validation Loss", "Loss"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2x2 three-seed training curves for the M4-P ablation"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument(
        "--output",
        default=(
            "reports/figures/"
            "kaggle_3m_multimodal_only_m4_p_training_curves_seeds42_123_2026.png"
        ),
    )
    parser.add_argument(
        "--pdf-output",
        default=(
            "reports/figures/"
            "kaggle_3m_multimodal_only_m4_p_training_curves_seeds42_123_2026.pdf"
        ),
    )
    return parser.parse_args()


def read_history(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training history not found: {path}")
    history: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            history[int(row["epoch"])] = row
    if not history:
        raise ValueError(f"Training history is empty: {path}")
    return history


def resolve_output(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for mean and SD curves")

    histories: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    for experiment in EXPERIMENTS:
        code = experiment["code"]
        histories[code] = {
            seed: read_history(run_dir(experiment, seed) / "metrics.jsonl")
            for seed in seeds
        }

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(16, 10.5), constrained_layout=False)
    colors = plt.get_cmap("tab10").colors

    for axis, (split, metric, title, ylabel) in zip(axes.flat, CURVES, strict=True):
        for index, experiment in enumerate(EXPERIMENTS):
            code = experiment["code"]
            common_epochs = sorted(
                set.intersection(
                    *(set(histories[code][seed]) for seed in seeds)
                )
            )
            if not common_epochs:
                raise ValueError(f"No common epochs across seeds for {code}")
            values = np.asarray(
                [
                    [
                        float(histories[code][seed][epoch][split][metric])
                        for epoch in common_epochs
                    ]
                    for seed in seeds
                ],
                dtype=np.float64,
            )
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            epochs = np.asarray(common_epochs)
            color = colors[index % len(colors)]
            axis.plot(
                epochs,
                mean,
                color=color,
                linewidth=1.8,
                label=code,
            )
            lower = mean - std
            upper = mean + std
            if metric == "positive_macro_iou":
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)
            else:
                lower = np.clip(lower, 0.0, None)
            axis.fill_between(epochs, lower, upper, color=color, alpha=0.10)

        axis.set_title(title, fontsize=13)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        if metric == "positive_macro_iou":
            axis.set_ylim(0.0, 1.0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.suptitle(
        "M4-P Final-Component Ablation: Training Curves (Mean ± SD, 3 Seeds)",
        fontsize=16,
        y=0.985,
    )
    figure.text(
        0.5,
        0.055,
        "Each model is shown through its last epoch shared by all three seeds; "
        "shading denotes sample SD.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.15, hspace=0.30)

    output_path = resolve_output(args.output)
    pdf_path = resolve_output(args.pdf_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved PNG to {output_path}")
    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    main()
