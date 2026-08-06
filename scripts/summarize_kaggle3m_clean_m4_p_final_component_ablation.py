from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    {
        "code": "E0",
        "label": "单通道 FLAIR，无增强，普通 U-Net",
        "config": "configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml",
        "run_stem": "kaggle_3m_multimodal_only_e0_flair_unet",
    },
    {
        "code": "E1-A",
        "label": "三通道，无增强，普通 U-Net",
        "config": (
            "configs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation.yaml"
        ),
        "run_stem": "kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation",
    },
    {
        "code": "E2-B",
        "label": "单通道 FLAIR，轻量增强，普通 U-Net",
        "config": (
            "configs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation.yaml"
        ),
        "run_stem": "kaggle_3m_multimodal_only_e2_flair_unet_augmentation",
    },
    {
        "code": "M0-AB",
        "label": "三通道，轻量增强，普通 U-Net",
        "config": "configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml",
        "run_stem": "kaggle_3m_multimodal_only_m0_rgb_unet",
    },
    {
        "code": "M4-NP",
        "label": "三通道，轻量增强，ResNet34，无预训练",
        "config": (
            "configs/"
            "kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet.yaml"
        ),
        "run_stem": (
            "kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet"
        ),
    },
    {
        "code": "M4-P−A",
        "label": "单通道 FLAIR，轻量增强，ResNet34，ImageNet 预训练",
        "config": (
            "configs/"
            "kaggle_3m_multimodal_only_m4_p_minus_a_flair_resnet34_unet.yaml"
        ),
        "run_stem": (
            "kaggle_3m_multimodal_only_m4_p_minus_a_flair_resnet34_unet"
        ),
    },
    {
        "code": "M4-P−B",
        "label": "三通道，无增强，ResNet34，ImageNet 预训练",
        "config": (
            "configs/"
            "kaggle_3m_multimodal_only_m4_p_minus_b_rgb_resnet34_unet_"
            "no_augmentation.yaml"
        ),
        "run_stem": (
            "kaggle_3m_multimodal_only_m4_p_minus_b_rgb_resnet34_unet_"
            "no_augmentation"
        ),
    },
    {
        "code": "M4-P",
        "label": "三通道，轻量增强，ResNet34，ImageNet 预训练",
        "config": "configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml",
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

EFFECTS = (
    ("完整方案", "M4-P", "E0", "M4-P − E0"),
    ("最终模型中的三通道输入 A", "M4-P", "M4-P−A", "M4-P − M4-P−A"),
    ("最终模型中的轻量增强 B", "M4-P", "M4-P−B", "M4-P − M4-P−B"),
    ("ResNet34 结构 C", "M4-NP", "M0-AB", "M4-NP − M0-AB"),
    ("ImageNet 预训练 D", "M4-P", "M4-NP", "M4-P − M4-NP"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize clean M4-P final-component ablation"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument(
        "--output",
        default=(
            "reports/"
            "kaggle_3m_multimodal_only_m4_p_final_component_ablation_"
            "seeds42_123_2026.md"
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


def patient_positive_iou(directory: Path) -> dict[str, float]:
    sample_path = directory / "evaluation" / "test" / "samples.csv"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Required patient-level result not found: {sample_path}")
    values: dict[str, list[float]] = {}
    with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            true_pixels = int(row["true_positive"]) + int(row["false_negative"])
            if true_pixels <= 0:
                continue
            patient = row["sample_id"].split("__slice_", maxsplit=1)[0]
            values.setdefault(patient, []).append(float(row["iou"]))
    return {
        patient: statistics.mean(patient_values)
        for patient, patient_values in values.items()
    }


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def bootstrap_interval(
    differences: list[float], *, samples: int = 10_000
) -> tuple[float, float]:
    generator = random.Random(42)
    sample_size = len(differences)
    estimates = sorted(
        statistics.mean(generator.choices(differences, k=sample_size))
        for _ in range(samples)
    )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for a multi-seed report")

    metadata = read_json(
        ROOT / "splits" / "kaggle_3m_multimodal_only_seed42.meta.json"
    )
    results: dict[str, dict[int, dict[str, Any]]] = {}
    for experiment in EXPERIMENTS:
        per_seed: dict[int, dict[str, Any]] = {}
        for seed in seeds:
            directory = run_dir(experiment, seed)
            test_result = read_json(directory / "test_metrics.json")
            per_seed[seed] = {
                "summary": read_json(directory / "training_summary.json"),
                "metrics": test_result["metrics"],
                "patients": patient_positive_iou(directory),
            }
        results[experiment["code"]] = per_seed

    def metric_values(code: str, key: str) -> list[float]:
        return [float(results[code][seed]["metrics"][key]) for seed in seeds]

    def test_iou(code: str, seed: int) -> float:
        return float(results[code][seed]["metrics"]["positive_macro_iou"])

    def effect_values(left: str, right: str) -> list[float]:
        return [test_iou(left, seed) - test_iou(right, seed) for seed in seeds]

    def patient_differences(left: str, right: str) -> list[float]:
        patient_sets = []
        for code in (left, right):
            for seed in seeds:
                patient_sets.append(set(results[code][seed]["patients"]))
        patients = sorted(set.intersection(*patient_sets))
        differences = []
        for patient in patients:
            left_mean = statistics.mean(
                float(results[left][seed]["patients"][patient]) for seed in seeds
            )
            right_mean = statistics.mean(
                float(results[right][seed]["patients"][patient]) for seed in seeds
            )
            differences.append(left_mean - right_mean)
        return differences

    lines: list[str] = [
        "# Kaggle 3M M4-P 最终组件三随机种子消融报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
        "",
        "## 1. 实验目的",
        "",
        "以 E0 为正式 baseline，以 M4-P 为最终模型，通过严格单变量比较验证三通道"
        "输入（A）、轻量增强（B）、ResNet34 结构（C）和 ImageNet 预训练（D）。",
        "",
        f"训练随机种子：{', '.join(str(seed) for seed in seeds)}。所有实验使用同一个 "
        "clean patient-level manifest，六名灰度等价患者均已排除。",
        "",
        f"数据包含 {metadata['num_patients']} 名患者、{metadata['num_samples']} 张切片；"
        f"测试集包含 {metadata['per_split']['test']['patients']} 名患者、"
        f"{metadata['per_split']['test']['samples']} 张切片。",
        "",
        "## 2. 实验矩阵",
        "",
        "| 模型 | 定义 | 配置 |",
        "|---|---|---|",
    ]
    for experiment in EXPERIMENTS:
        lines.append(
            f"| {experiment['code']} | {experiment['label']} | "
            f"[{experiment['config']}](../{experiment['config']}) |"
        )

    lines.extend(
        [
            "",
            "## 3. 各随机种子 Test Positive Macro IoU",
            "",
            "| 模型 | "
            + " | ".join(f"Seed {seed}" for seed in seeds)
            + " | Mean ± SD |",
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
        cells = [
            summary_cell(
                metric_values(code, key),
                percentage=key == "empty_slice_false_positive_rate",
            )
            for key, _label in METRICS
        ]
        lines.append(f"| {code} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 5. 逐随机种子严格配对消融效应",
            "",
            "正值表示加入对应组件后 Test Positive Macro IoU 提高。",
            "",
            "| 组件 | 严格比较 | "
            + " | ".join(f"Seed {seed}" for seed in seeds)
            + " | Mean ± SD | 正向种子 |",
            "|---|---|" + "---:|" * (len(seeds) + 1) + "---:|",
        ]
    )
    for label, left, right, comparison in EFFECTS:
        values = effect_values(left, right)
        positive = sum(value > 0.0 for value in values)
        lines.append(
            f"| {label} | {comparison} | "
            + " | ".join(signed(value) for value in values)
            + f" | {summary_cell(values)} | {positive}/{len(seeds)} |"
        )

    lines.extend(
        [
            "",
            "## 6. 患者级配对分析",
            "",
            "先对每名患者的正切片 IoU 求均值，再对训练种子求均值；95% CI 使用患者"
            "作为重采样单位进行 10,000 次配对 bootstrap。测试患者较少，区间仅用于"
            "描述不确定性。",
            "",
            "| 组件 | 患者级平均差值 | 改善患者 | 配对 bootstrap 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    patient_effects: dict[str, tuple[float, int, int, float, float]] = {}
    for label, left, right, _comparison in EFFECTS:
        differences = patient_differences(left, right)
        lower, upper = bootstrap_interval(differences)
        mean = statistics.mean(differences)
        positive = sum(value > 0.0 for value in differences)
        patient_effects[label] = (mean, positive, len(differences), lower, upper)
        lines.append(
            f"| {label} | {signed(mean)} | {positive}/{len(differences)} | "
            f"[{lower:+.4f}, {upper:+.4f}] |"
        )

    lines.extend(["", "## 7. 训练与验证稳定性", ""])
    lines.extend(
        [
            "| 模型 | Best Epoch | Val Positive IoU | Train–Val Gap | 训练时间 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for experiment in EXPERIMENTS:
        code = experiment["code"]
        summaries = [results[code][seed]["summary"] for seed in seeds]
        epochs = [float(summary["best_epoch"]) for summary in summaries]
        val_values = [
            float(summary["validation_metric_at_best_epoch"])
            for summary in summaries
        ]
        gaps = [
            float(summary["training_metric_at_best_epoch"])
            - float(summary["validation_metric_at_best_epoch"])
            for summary in summaries
        ]
        minutes = [float(summary["elapsed_seconds"]) / 60.0 for summary in summaries]
        lines.append(
            f"| {code} | {summary_cell(epochs)} | {summary_cell(val_values)} | "
            f"{summary_cell(gaps)} | {summary_cell(minutes)} min |"
        )

    seed_label = "_".join(str(seed) for seed in seeds)
    curve_figure = (
        "figures/"
        "kaggle_3m_multimodal_only_m4_p_training_curves_"
        f"seeds{seed_label}.png"
    )
    lines.extend(
        [
            "",
            "## 8. 三随机种子训练曲线",
            "",
            "曲线为三个训练种子在共同 epoch 上的均值，阴影为样本标准差。每个模型"
            "只显示到三个种子均具有记录的最后一个 epoch。",
            "",
            f"![八模型 Train/Validation IoU 与 Loss 曲线]({curve_figure})",
            "",
            "## 9. 自动结论",
            "",
        ]
    )
    for label, left, right, comparison in EFFECTS:
        values = effect_values(left, right)
        mean, std = mean_std(values)
        positive = sum(value > 0.0 for value in values)
        if positive == len(values):
            interpretation = "所有训练种子方向一致，支持该组件带来正向贡献"
        elif mean > 0.0:
            interpretation = "平均效应为正，但训练种子方向不完全一致"
        elif mean < 0.0:
            interpretation = "平均效应为负，现有结果不支持该组件有效"
        else:
            interpretation = "平均效应为零，现有结果不支持该组件有效"
        patient_mean, patient_positive, patient_count, lower, upper = patient_effects[
            label
        ]
        lines.append(
            f"- {label}（{comparison}）：seed 配对效应 {mean:+.4f} ± {std:.4f}，"
            f"{positive}/{len(values)} 个种子为正；{interpretation}。患者级平均差值"
            f" {patient_mean:+.4f}，{patient_positive}/{patient_count} 名患者改善，"
            f"95% CI [{lower:+.4f}, {upper:+.4f}]。"
        )

    lines.extend(
        [
            "",
            "只有当组件的逐 seed 配对效应、患者级方向和不确定性共同支持时，才应在"
            "论文中写成稳定贡献。固定测试集只有少量患者，不能用训练种子替代患者级"
            "交叉验证或外部验证。",
            "",
            "## 10. 原始结果",
            "",
        ]
    )
    for experiment in EXPERIMENTS:
        lines.extend([f"### {experiment['code']}", ""])
        for seed in seeds:
            relative_run = f"../runs/{experiment['run_stem']}_seed{seed}"
            lines.append(
                f"- Seed {seed}：[训练总结]({relative_run}/training_summary.json)；"
                f"[测试指标]({relative_run}/test_metrics.json)；"
                f"[逐切片结果]({relative_run}/evaluation/test/samples.csv)"
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
