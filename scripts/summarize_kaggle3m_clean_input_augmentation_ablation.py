from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    {
        "code": "E0",
        "label": "单通道 FLAIR，无增强",
        "channels": "单通道",
        "augmentation": "无",
        "config": "configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml",
        "run": "runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42",
    },
    {
        "code": "E1-A",
        "label": "三通道多模态，无增强",
        "channels": "三通道",
        "augmentation": "无",
        "config": "configs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation.yaml",
        "run": "runs/kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation_seed42",
    },
    {
        "code": "E2-B",
        "label": "单通道 FLAIR，轻量增强",
        "channels": "单通道",
        "augmentation": "有",
        "config": "configs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation.yaml",
        "run": "runs/kaggle_3m_multimodal_only_e2_flair_unet_augmentation_seed42",
    },
    {
        "code": "M0-AB",
        "label": "三通道多模态，轻量增强",
        "channels": "三通道",
        "augmentation": "有",
        "config": "configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml",
        "run": "runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42",
    },
)

SIZE_BINS = (
    ("tiny", "微小 1–255 px", 1, 255),
    ("small", "小型 256–1023 px", 256, 1023),
    ("medium", "中型 1024–4095 px", 1024, 4095),
    ("large", "大型 ≥4096 px", 4096, None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize clean input/augmentation ablation")
    parser.add_argument(
        "--output",
        default="reports/kaggle_3m_multimodal_only_input_augmentation_ablation_seed42.md",
    )
    return parser.parse_args()


def read_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required result not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def number(value: Any) -> str:
    return f"{float(value):.4f}"


def percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def size_metrics(run: str) -> dict[str, tuple[int, float | None]]:
    sample_path = ROOT / run / "evaluation" / "test" / "samples.csv"
    values: dict[str, list[float]] = {key: [] for key, *_ in SIZE_BINS}
    with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            true_pixels = int(row["true_positive"]) + int(row["false_negative"])
            for key, _label, lower, upper in SIZE_BINS:
                if true_pixels >= lower and (upper is None or true_pixels <= upper):
                    values[key].append(float(row["iou"]))
                    break
    return {
        key: (len(items), sum(items) / len(items) if items else None)
        for key, items in values.items()
    }


def effect_text(name: str, value: float) -> str:
    direction = "提升" if value > 0 else "下降" if value < 0 else "无变化"
    return f"{name}使 Test Positive Macro IoU {direction} {abs(value):.4f}。"


def main() -> None:
    args = parse_args()
    metadata = read_json("splits/kaggle_3m_multimodal_only_seed42.meta.json")
    results: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        summary = read_json(f"{experiment['run']}/training_summary.json")
        test_result = read_json(f"{experiment['run']}/test_metrics.json")
        results.append(
            {
                **experiment,
                "summary": summary,
                "metrics": test_result["metrics"],
                "sizes": size_metrics(experiment["run"]),
            }
        )

    by_code = {result["code"]: result for result in results}
    positive_iou = {
        code: float(result["metrics"]["positive_macro_iou"])
        for code, result in by_code.items()
    }
    effect_a = positive_iou["E1-A"] - positive_iou["E0"]
    effect_b = positive_iou["E2-B"] - positive_iou["E0"]
    effect_ab = positive_iou["M0-AB"] - positive_iou["E0"]
    interaction = (
        positive_iou["M0-AB"]
        - positive_iou["E1-A"]
        - positive_iou["E2-B"]
        + positive_iou["E0"]
    )

    lines: list[str] = [
        "# Kaggle 3M 纯多通道患者队列输入与增强 2×2 消融报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
        "",
        "## 1. 实验目的",
        "",
        "以 E0 单通道 FLAIR、无增强 U-Net 为 baseline，在相同 clean patient-level split "
        "上分别验证三通道多模态输入（A）、轻量数据增强（B）及二者交互作用。",
        "",
        "## 2. 数据与实验定义",
        "",
        f"使用 {metadata['num_patients']} 名患者、{metadata['num_samples']} 张切片；"
        f"测试集包含 {metadata['per_split']['test']['patients']} 名患者、"
        f"{metadata['per_split']['test']['samples']} 张切片。六名单通道患者已按患者级排除。",
        "",
        "| 模型 | 输入 | 增强 | 配置 |",
        "|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['code']} | {result['channels']} | {result['augmentation']} | "
            f"[{result['config']}](../{result['config']}) |"
        )

    lines.extend(
        [
            "",
            "## 3. 训练与验证结果",
            "",
            "| 模型 | 最佳 Epoch | Train Positive IoU | Val Positive IoU | "
            "Train–Val Gap | 时间 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        summary = result["summary"]
        train_iou = float(summary["training_metric_at_best_epoch"])
        val_iou = float(summary["validation_metric_at_best_epoch"])
        lines.append(
            f"| {result['code']} | {summary['best_epoch']} | {number(train_iou)} | "
            f"{number(val_iou)} | {number(train_iou - val_iou)} | "
            f"{float(summary['elapsed_seconds']) / 60.0:.1f} min |"
        )

    lines.extend(
        [
            "",
            "## 4. 测试集结果",
            "",
            "| 模型 | Positive IoU | Positive Dice | Micro IoU | Precision | Recall | "
            "空切片误报率 | Loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['code']} | {number(metrics['positive_macro_iou'])} | "
            f"{number(metrics['positive_macro_dice'])} | {number(metrics['micro_iou'])} | "
            f"{number(metrics['micro_precision'])} | {number(metrics['micro_recall'])} | "
            f"{percent(metrics['empty_slice_false_positive_rate'])} | "
            f"{number(metrics['loss'])} |"
        )

    lines.extend(
        [
            "",
            "## 5. 独立效应与交互作用",
            "",
            "| 对比 | 含义 | Positive IoU 变化 |",
            "|---|---|---:|",
            f"| E1-A − E0 | 三通道输入独立效应 | {signed(effect_a)} |",
            f"| E2-B − E0 | 数据增强独立效应 | {signed(effect_b)} |",
            f"| M0-AB − E0 | 三通道与增强总变化 | {signed(effect_ab)} |",
            f"| M0-AB − E1-A − E2-B + E0 | A×B 交互作用 | {signed(interaction)} |",
            "",
            "## 6. 不同病灶面积的 Positive IoU",
            "",
            "| 模型 | 微小 1–255 px | 小型 256–1023 px | "
            "中型 1024–4095 px | 大型 ≥4096 px |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        values = []
        for key, _label, _lower, _upper in SIZE_BINS:
            count, average = result["sizes"][key]
            values.append("—" if average is None else f"{number(average)} (n={count})")
        lines.append(f"| {result['code']} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 7. 自动结论",
            "",
            effect_text("三通道输入单独使用", effect_a),
            effect_text("轻量数据增强单独使用", effect_b),
            effect_text("三通道与增强组合", effect_ab),
            effect_text("A×B 交互作用", interaction),
            "",
            "本报告来自 seed 42 单次实验，正式结论需要在相同 clean split 上追加 seed 123 "
            "和 2026，并报告 mean ± std。",
            "",
            "## 8. 原始结果",
            "",
        ]
    )
    for result in results:
        lines.append(
            f"- {result['code']}：[训练总结](../{result['run']}/training_summary.json)；"
            f"[测试指标](../{result['run']}/test_metrics.json)"
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

