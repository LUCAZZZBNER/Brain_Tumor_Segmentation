from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    {
        "code": "E0",
        "label": "单通道 FLAIR，无增强，普通 U-Net",
        "run_stem": "kaggle_3m_multimodal_only_e0_flair_unet",
    },
    {
        "code": "E1-A",
        "label": "三通道多模态，无增强，普通 U-Net",
        "run_stem": "kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation",
    },
    {
        "code": "E2-B",
        "label": "单通道 FLAIR，轻量增强，普通 U-Net",
        "run_stem": "kaggle_3m_multimodal_only_e2_flair_unet_augmentation",
    },
    {
        "code": "M0-AB",
        "label": "三通道多模态，轻量增强，普通 U-Net",
        "run_stem": "kaggle_3m_multimodal_only_m0_rgb_unet",
    },
    {
        "code": "M4-NP",
        "label": "三通道多模态，轻量增强，ResNet34，无预训练",
        "run_stem": (
            "kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet"
        ),
    },
    {
        "code": "M4-P",
        "label": "三通道多模态，轻量增强，ResNet34，ImageNet 预训练",
        "run_stem": "kaggle_3m_multimodal_only_m4_rgb_resnet34_unet",
    },
)

METRICS = (
    ("positive_macro_iou", "Positive IoU"),
    ("positive_macro_dice", "Positive Dice"),
    ("micro_iou", "Micro IoU"),
    ("micro_precision", "Precision"),
    ("micro_recall", "Recall"),
    ("empty_slice_false_positive_rate", "空切片误报率"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize six-model multi-seed ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "kaggle_3m_multimodal_only_complete_ablation_seeds42_123_2026.md"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def summary_cell(values: list[float], *, percentage: bool = False) -> str:
    mean, std = mean_std(values)
    scale = 100.0 if percentage else 1.0
    suffix = "%" if percentage else ""
    return f"{mean * scale:.4f} ± {std * scale:.4f}{suffix}"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def run_dir(experiment: dict[str, str], seed: int) -> Path:
    return ROOT / "runs" / f"{experiment['run_stem']}_seed{seed}"


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for a multi-seed report")

    metadata = read_json(ROOT / "splits" / "kaggle_3m_multimodal_only_seed42.meta.json")
    results: dict[str, dict[int, dict[str, Any]]] = {}
    for experiment in EXPERIMENTS:
        per_seed: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            directory = run_dir(experiment, seed)
            summary = read_json(directory / "training_summary.json")
            test_result = read_json(directory / "test_metrics.json")
            per_seed[seed] = {
                "summary": summary,
                "metrics": test_result["metrics"],
            }
        results[experiment["code"]] = per_seed

    def metric_values(code: str, key: str) -> list[float]:
        return [float(results[code][seed]["metrics"][key]) for seed in seeds]

    def test_iou(code: str, seed: int) -> float:
        return float(results[code][seed]["metrics"]["positive_macro_iou"])

    effect_definitions = (
        (
            "A：E1-A − E0",
            lambda seed: test_iou("E1-A", seed) - test_iou("E0", seed),
        ),
        (
            "B：E2-B − E0",
            lambda seed: test_iou("E2-B", seed) - test_iou("E0", seed),
        ),
        (
            "A+B：M0-AB − E0",
            lambda seed: test_iou("M0-AB", seed) - test_iou("E0", seed),
        ),
        (
            "A×B 交互",
            lambda seed: (
                test_iou("M0-AB", seed)
                - test_iou("E1-A", seed)
                - test_iou("E2-B", seed)
                + test_iou("E0", seed)
            ),
        ),
        (
            "结构：M4-NP − M0-AB",
            lambda seed: test_iou("M4-NP", seed) - test_iou("M0-AB", seed),
        ),
        (
            "预训练：M4-P − M4-NP",
            lambda seed: test_iou("M4-P", seed) - test_iou("M4-NP", seed),
        ),
        (
            "总变化：M4-P − E0",
            lambda seed: test_iou("M4-P", seed) - test_iou("E0", seed),
        ),
    )

    lines: list[str] = [
        "# Kaggle 3M 纯多通道患者队列六模型多随机种子消融报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
        "",
        "## 1. 实验范围",
        "",
        f"本报告汇总训练随机种子 {', '.join(str(seed) for seed in seeds)}。"
        "所有运行使用同一个 clean patient-level manifest，只改变训练随机种子和输出目录。",
        "",
        f"数据包含 {metadata['num_patients']} 名患者、{metadata['num_samples']} 张切片；"
        f"测试集包含 {metadata['per_split']['test']['patients']} 名患者、"
        f"{metadata['per_split']['test']['samples']} 张切片。",
        "",
        "## 2. 实验矩阵",
        "",
        "| 模型 | 定义 |",
        "|---|---|",
    ]
    for experiment in EXPERIMENTS:
        lines.append(f"| {experiment['code']} | {experiment['label']} |")

    lines.extend(
        [
            "",
            "## 3. 各随机种子 Test Positive Macro IoU",
            "",
            "| 模型 | " + " | ".join(f"Seed {seed}" for seed in seeds) + " | Mean ± SD |",
            "|---|" + "---:|" * (len(seeds) + 1),
        ]
    )
    for experiment in EXPERIMENTS:
        values = metric_values(experiment["code"], "positive_macro_iou")
        lines.append(
            f"| {experiment['code']} | "
            + " | ".join(f"{value:.4f}" for value in values)
            + f" | {summary_cell(values)} |"
        )

    lines.extend(
        [
            "",
            "## 4. 测试指标 Mean ± SD",
            "",
            "| 模型 | Positive IoU | Positive Dice | Micro IoU | Precision | Recall | "
            "空切片误报率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for experiment in EXPERIMENTS:
        code = experiment["code"]
        cells = []
        for key, _label in METRICS:
            cells.append(
                summary_cell(
                    metric_values(code, key),
                    percentage=key == "empty_slice_false_positive_rate",
                )
            )
        lines.append(f"| {code} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 5. 训练与验证稳定性",
            "",
            "| 模型 | Best Epoch | Val Positive IoU | Train–Val Gap | 训练时间 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for experiment in EXPERIMENTS:
        code = experiment["code"]
        epochs = [float(results[code][seed]["summary"]["best_epoch"]) for seed in seeds]
        val_values = [
            float(results[code][seed]["summary"]["validation_metric_at_best_epoch"])
            for seed in seeds
        ]
        gaps = [
            float(results[code][seed]["summary"]["training_metric_at_best_epoch"])
            - float(results[code][seed]["summary"]["validation_metric_at_best_epoch"])
            for seed in seeds
        ]
        minutes = [
            float(results[code][seed]["summary"]["elapsed_seconds"]) / 60.0
            for seed in seeds
        ]
        lines.append(
            f"| {code} | {summary_cell(epochs)} | {summary_cell(val_values)} | "
            f"{summary_cell(gaps)} | {summary_cell(minutes)} min |"
        )

    lines.extend(
        [
            "",
            "## 6. 逐随机种子配对消融效应",
            "",
            "| 对比 | " + " | ".join(f"Seed {seed}" for seed in seeds) + " | Mean ± SD |",
            "|---|" + "---:|" * (len(seeds) + 1),
        ]
    )
    for label, function in effect_definitions:
        values = [function(seed) for seed in seeds]
        lines.append(
            f"| {label} | "
            + " | ".join(signed(value) for value in values)
            + f" | {summary_cell(values)} |"
        )

    mean_ranking = sorted(
        (
            (
                experiment["code"],
                mean_std(metric_values(experiment["code"], "positive_macro_iou"))[0],
            )
            for experiment in EXPERIMENTS
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    lines.extend(["", "## 7. 综合排名与结论", ""])
    for index, (code, mean_value) in enumerate(mean_ranking, start=1):
        lines.append(f"{index}. {code}：平均 Test Positive Macro IoU {mean_value:.4f}")
    lines.extend(
        [
            "",
            "只有当逐 seed 配对效应方向一致、均值大于其随机波动时，才应将对应因素写成"
            "稳定结论。不得从三个 seed 中挑选最高一次作为最终结果。",
            "",
            "多随机种子只评估固定患者划分上的训练随机性；测试集仍只有少量患者，"
            "最终模型还需要 patient-level 交叉验证或外部验证。",
            "",
            "## 8. 原始结果",
            "",
        ]
    )
    for experiment in EXPERIMENTS:
        lines.append(f"### {experiment['code']}")
        lines.append("")
        for seed in seeds:
            relative_run = f"../runs/{experiment['run_stem']}_seed{seed}"
            lines.append(
                f"- Seed {seed}：[训练总结]({relative_run}/training_summary.json)；"
                f"[测试指标]({relative_run}/test_metrics.json)"
            )
        lines.append("")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
