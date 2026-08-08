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
    (
        "train",
        "positive_macro_iou",
        "Train Positive Macro IoU",
        "Positive Macro IoU",
        "train_iou",
    ),
    (
        "val",
        "positive_macro_iou",
        "Validation Positive Macro IoU",
        "Positive Macro IoU",
        "validation_iou",
    ),
    ("train", "loss", "Train Loss", "Loss", "train_loss"),
    ("val", "loss", "Validation Loss", "Loss", "validation_loss"),
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
    parser.add_argument(
        "--separate-output-dir",
        default="reports/figures",
        help="Directory for the four separate PNG and PDF figures",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Model codes to plot; default: every model in the report",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=8.0)
    parser.add_argument("--font-size", type=float, default=16.0)
    parser.add_argument("--title-size", type=float, default=22.0)
    parser.add_argument("--legend-size", type=float, default=14.0)
    parser.add_argument("--line-width", type=float, default=2.4)
    parser.add_argument(
        "--sd-alpha",
        type=float,
        default=0.12,
        help="Standard-deviation shading opacity; use 0 to hide shading",
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


def select_experiments(requested: list[str] | None) -> tuple[dict[str, str], ...]:
    if not requested:
        return EXPERIMENTS
    aliases = {"M4-P-A": "M4-P−A", "M4-P-B": "M4-P−B"}
    requested_codes = [aliases.get(code, code) for code in requested]
    by_code = {experiment["code"]: experiment for experiment in EXPERIMENTS}
    unknown = [code for code in requested_codes if code not in by_code]
    if unknown:
        raise ValueError(f"Unknown model codes: {', '.join(unknown)}")
    return tuple(by_code[code] for code in requested_codes)


def curve_statistics(
    histories: dict[str, dict[int, dict[int, dict[str, Any]]]],
    code: str,
    seeds: list[int],
    split: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_epochs = sorted(
        set.intersection(*(set(histories[code][seed]) for seed in seeds))
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
    return (
        np.asarray(common_epochs),
        values.mean(axis=0),
        values.std(axis=0, ddof=1),
    )


def draw_curve(
    axis: Any,
    experiments: tuple[dict[str, str], ...],
    histories: dict[str, dict[int, dict[int, dict[str, Any]]]],
    seeds: list[int],
    split: str,
    metric: str,
    title: str,
    ylabel: str,
    args: argparse.Namespace,
) -> None:
    colors = plt.get_cmap("tab10").colors
    for index, experiment in enumerate(experiments):
        code = experiment["code"]
        epochs, mean, std = curve_statistics(
            histories, code, seeds, split, metric
        )
        color = colors[index % len(colors)]
        axis.plot(
            epochs,
            mean,
            color=color,
            linewidth=args.line_width,
            label=code,
        )
        if args.sd_alpha > 0.0:
            lower = mean - std
            upper = mean + std
            if metric == "positive_macro_iou":
                lower = np.clip(lower, 0.0, 1.0)
                upper = np.clip(upper, 0.0, 1.0)
            else:
                lower = np.clip(lower, 0.0, None)
            axis.fill_between(
                epochs,
                lower,
                upper,
                color=color,
                alpha=args.sd_alpha,
            )

    axis.set_title(title, fontsize=args.title_size)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="both", labelsize=args.font_size)
    axis.grid(True, color="#d8d8d8", linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    if metric == "positive_macro_iou":
        axis.set_ylim(0.0, 1.0)


def save_separate_figures(
    experiments: tuple[dict[str, str], ...],
    histories: dict[str, dict[int, dict[int, dict[str, Any]]]],
    seeds: list[int],
    args: argparse.Namespace,
) -> None:
    output_dir = resolve_output(args.separate_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_label = "_".join(str(seed) for seed in seeds)
    for split, metric, title, ylabel, slug in CURVES:
        figure, axis = plt.subplots(figsize=(args.width, args.height))
        draw_curve(
            axis,
            experiments,
            histories,
            seeds,
            split,
            metric,
            title,
            ylabel,
            args,
        )
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=min(4, len(experiments)),
            frameon=False,
            fontsize=args.legend_size,
        )
        figure.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.25)
        stem = (
            "kaggle_3m_multimodal_only_m4_p_training_curve_"
            f"{slug}_seeds{seed_label}"
        )
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved separate PNG to {png_path}")
        print(f"Saved separate PDF to {pdf_path}")


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for mean and SD curves")
    if args.dpi <= 0 or args.width <= 0.0 or args.height <= 0.0:
        raise ValueError("DPI, width, and height must be positive")
    if args.font_size <= 0.0 or args.title_size <= 0.0 or args.legend_size <= 0.0:
        raise ValueError("Every font size must be positive")
    if args.line_width <= 0.0:
        raise ValueError("Line width must be positive")
    if not 0.0 <= args.sd_alpha <= 1.0:
        raise ValueError("SD alpha must be between 0 and 1")

    experiments = select_experiments(args.models)

    histories: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    for experiment in experiments:
        code = experiment["code"]
        histories[code] = {
            seed: read_history(run_dir(experiment, seed) / "metrics.jsonl")
            for seed in seeds
        }

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": args.font_size,
            "axes.labelsize": args.font_size,
            "xtick.labelsize": args.font_size,
            "ytick.labelsize": args.font_size,
        }
    )
    save_separate_figures(experiments, histories, seeds, args)

    figure, axes = plt.subplots(2, 2, figsize=(16, 10.5), constrained_layout=False)
    for axis, curve in zip(axes.flat, CURVES, strict=True):
        split, metric, title, ylabel, _slug = curve
        combined_args = argparse.Namespace(**vars(args))
        combined_args.title_size = 15.0
        combined_args.font_size = 11.0
        combined_args.line_width = min(args.line_width, 2.0)
        draw_curve(
            axis,
            experiments,
            histories,
            seeds,
            split,
            metric,
            title,
            ylabel,
            combined_args,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(experiments)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.suptitle(
        "M4-P Final-Component Ablation: Training Curves (Mean ± SD, 3 Seeds)",
        fontsize=17,
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
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved PNG to {output_path}")
    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    main()
