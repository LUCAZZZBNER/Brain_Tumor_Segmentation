from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    {
        "code": "M0",
        "label": "普通 U-Net",
        "encoder": "DoubleConv",
        "pretrained": "否",
        "config": "configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml",
        "run": "runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42",
    },
    {
        "code": "M4-NP",
        "label": "ResNet34 U-Net（随机初始化）",
        "encoder": "ResNet34",
        "pretrained": "否",
        "config": (
            "configs/"
            "kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet.yaml"
        ),
        "run": (
            "runs/"
            "kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet_seed42"
        ),
    },
    {
        "code": "M4-P",
        "label": "ResNet34 U-Net（ImageNet 预训练）",
        "encoder": "ResNet34",
        "pretrained": "是",
        "config": "configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml",
        "run": "runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize M4 pretraining ablation")
    parser.add_argument(
        "--output",
        default="reports/kaggle_3m_m4_pretraining_ablation.md",
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


def patient_positive_iou(run: str) -> dict[str, float]:
    sample_path = ROOT / run / "evaluation" / "test" / "samples.csv"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Required patient-level results not found: {sample_path}")
    values: dict[str, list[float]] = defaultdict(list)
    with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            true_pixels = int(row["true_positive"]) + int(row["false_negative"])
            if true_pixels <= 0:
                continue
            patient = row["sample_id"].split("__slice_", maxsplit=1)[0]
            values[patient].append(float(row["iou"]))
    return {
        patient: sum(patient_values) / len(patient_values)
        for patient, patient_values in values.items()
    }


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
                "patients": patient_positive_iou(experiment["run"]),
            }
        )

    by_code = {result["code"]: result for result in results}
    m0_iou = float(by_code["M0"]["metrics"]["positive_macro_iou"])
    no_pretrain_iou = float(by_code["M4-NP"]["metrics"]["positive_macro_iou"])
    pretrained_iou = float(by_code["M4-P"]["metrics"]["positive_macro_iou"])
    architecture_effect = no_pretrain_iou - m0_iou
    pretraining_effect = pretrained_iou - no_pretrain_iou
    total_effect = pretrained_iou - m0_iou

    patients = sorted(
        set.intersection(*(set(result["patients"]) for result in results))
    )
    lines: list[str] = [
        "# Kaggle 3M M4 ResNet34 预训练单变量消融报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
        "",
        "## 1. 实验目的",
        "",
        "本实验在纯多通道患者队列上分离 ResNet34 encoder 结构与 ImageNet 预训练的贡献。"
        "M4-NP 与 M4-P 的唯一差异是 pretrained=false/true；其余数据、增强、损失函数、"
        "优化器、随机种子和训练策略完全一致。",
        "",
        "## 2. 数据与实验定义",
        "",
        f"共使用 {metadata['num_patients']} 名患者、{metadata['num_samples']} 张切片；"
        f"测试集包含 {metadata['per_split']['test']['patients']} 名患者、"
        f"{metadata['per_split']['test']['samples']} 张切片。",
        "",
        "| 模型 | Encoder | ImageNet 预训练 | 配置 |",
        "|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['code']} | {result['encoder']} | {result['pretrained']} | "
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
            "## 5. 单变量贡献",
            "",
            "| 对比 | 含义 | Positive IoU 变化 |",
            "|---|---|---:|",
            f"| M4-NP − M0 | ResNet34 结构贡献 | {signed(architecture_effect)} |",
            f"| M4-P − M4-NP | ImageNet 预训练贡献 | {signed(pretraining_effect)} |",
            f"| M4-P − M0 | 结构与预训练的总变化 | {signed(total_effect)} |",
            "",
            "这些差值是模型级消融结果，不应解释为严格可加的因果效应。",
            "",
            "## 6. 患者级 Positive IoU",
            "",
            "| 患者 | M0 | M4-NP | M4-P | M4-NP − M0 | M4-P − M4-NP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for patient in patients:
        m0 = by_code["M0"]["patients"][patient]
        no_pretrain = by_code["M4-NP"]["patients"][patient]
        pretrained = by_code["M4-P"]["patients"][patient]
        lines.append(
            f"| {patient} | {number(m0)} | {number(no_pretrain)} | "
            f"{number(pretrained)} | {signed(no_pretrain - m0)} | "
            f"{signed(pretrained - no_pretrain)} |"
        )

    if pretraining_effect > 0 and architecture_effect > 0:
        conclusion = "ResNet34 结构与 ImageNet 预训练均带来正向提升。"
    elif pretraining_effect > 0:
        conclusion = "提升主要来自 ImageNet 预训练，随机初始化 ResNet34 未超过 M0。"
    elif architecture_effect > 0:
        conclusion = "提升主要来自 ResNet34 结构，ImageNet 预训练没有进一步提升。"
    else:
        conclusion = "当前单次实验未证明 ResNet34 结构或 ImageNet 预训练的正向贡献。"
    lines.extend(
        [
            "",
            "## 7. 自动结论",
            "",
            conclusion,
            "",
            "本实验仍为 seed 42 单次运行。确定最终论文结论前，应在固定 clean split 上追加"
            "多个训练随机种子，并报告 mean ± std。",
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

