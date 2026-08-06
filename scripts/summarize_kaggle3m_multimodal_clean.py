from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    {
        "code": "E0",
        "label": "单通道 FLAIR U-Net（无增强）",
        "config": "configs/kaggle_3m_multimodal_only_e0_flair_unet.yaml",
        "run": "runs/kaggle_3m_multimodal_only_e0_flair_unet_seed42",
        "original_run": "runs/kaggle_3m_e0_baseline_flair_no_augmentation_seed42",
    },
    {
        "code": "M0",
        "label": "三通道多模态 U-Net（轻量增强）",
        "config": "configs/kaggle_3m_multimodal_only_m0_rgb_unet.yaml",
        "run": "runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42",
        "original_run": "runs/kaggle_3m_e1_a_rgb_seed42",
    },
    {
        "code": "M4",
        "label": "三通道多模态 ResNet34 U-Net（轻量增强）",
        "config": "configs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet.yaml",
        "run": "runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42",
        "original_run": "runs/kaggle_3m_m4_rgb_resnet34_unet_seed42",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize clean Kaggle 3M E0/M0/M4 runs")
    parser.add_argument(
        "--output",
        default="reports/kaggle_3m_multimodal_only_e0_m0_m4.md",
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


def number(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def result_link(run: str, filename: str) -> str:
    return f"../{run}/{filename}"


def main() -> None:
    args = parse_args()
    metadata = read_json("splits/kaggle_3m_multimodal_only_seed42.meta.json")
    results: list[dict[str, Any]] = []

    for experiment in EXPERIMENTS:
        summary = read_json(f"{experiment['run']}/training_summary.json")
        test_result = read_json(f"{experiment['run']}/test_metrics.json")
        metrics = test_result["metrics"]
        original_path = ROOT / experiment["original_run"] / "test_metrics.json"
        original_metrics = None
        if original_path.is_file():
            with original_path.open("r", encoding="utf-8-sig") as handle:
                original_metrics = json.load(handle)["metrics"]
        results.append(
            {
                **experiment,
                "summary": summary,
                "test": test_result,
                "metrics": metrics,
                "original_metrics": original_metrics,
            }
        )

    ranked = sorted(
        results,
        key=lambda item: float(item["metrics"]["positive_macro_iou"]),
        reverse=True,
    )
    best = ranked[0]
    lines: list[str] = []
    lines.extend(
        [
            "# Kaggle 3M 纯多通道队列 E0、M0、M4 对比报告",
            "",
            f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。",
            "",
            "## 1. 实验目的",
            "",
            "本实验从 Kaggle 3M 队列中按患者级排除包含灰度等价切片（R=G=B）的患者，"
            "保留其余患者原有的 train/val/test 归属，然后从头训练并测试 E0、M0 和 M4。"
            "除数据清理、输出目录和实验名称外，各模型设置与原实验保持一致。",
            "",
            "## 2. 数据清理结果",
            "",
            f"- 保留患者：{metadata['num_patients']} 名",
            f"- 保留切片：{metadata['num_samples']} 张",
            f"- 排除患者：{metadata['num_excluded_patients']} 名",
            f"- 排除切片：{metadata['num_excluded_samples']} 张",
            "- 划分策略：过滤原 manifest，未重新随机分配患者",
            "",
            "| Split | 患者 | 切片 | 正切片 | 空切片 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "val", "test"):
        row = metadata["per_split"][split]
        lines.append(
            f"| {split} | {row['patients']} | {row['samples']} | "
            f"{row['positive_masks']} | {row['empty_masks']} |"
        )

    lines.extend(
        [
            "",
            "排除患者明细：",
            "",
            "| 患者 | 原 Split | 切片 | 正切片 | 空切片 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for patient in metadata["excluded_patients"]:
        lines.append(
            f"| {patient['patient_id']} | {patient['split']} | {patient['samples']} | "
            f"{patient['positive_masks']} | {patient['empty_masks']} |"
        )

    lines.extend(
        [
            "",
            "## 3. 实验定义",
            "",
            "| 编号 | 模型 | 配置 |",
            "|---|---|---|",
        ]
    )
    for item in results:
        lines.append(f"| {item['code']} | {item['label']} | [{item['config']}]"
                     f"(../{item['config']}) |")

    lines.extend(
        [
            "",
            "三组实验均使用随机种子 42、patient-level 固定划分、256×256 输入、"
            "batch size 4、AdamW、BCE + Positive Dice loss，并以 Validation Positive "
            "Macro IoU 选择最佳 checkpoint。M0 与 M4 使用相同的水平翻转和 ±5° 旋转。",
            "",
            "## 4. 训练与验证结果",
            "",
            "| 模型 | 最佳 Epoch | Train Positive IoU | Val Positive IoU | Train–Val Gap | 训练时间 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        summary = item["summary"]
        train_iou = float(summary["training_metric_at_best_epoch"])
        val_iou = float(summary["validation_metric_at_best_epoch"])
        lines.append(
            f"| {item['code']} | {summary['best_epoch']} | {number(train_iou)} | "
            f"{number(val_iou)} | {number(train_iou - val_iou)} | "
            f"{float(summary['elapsed_seconds']) / 60.0:.1f} min |"
        )

    lines.extend(
        [
            "",
            "## 5. 测试集结果",
            "",
            "| 模型 | Positive Macro IoU | Positive Macro Dice | Micro IoU | "
            "Precision | Recall | 空切片误报率 | Test Loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results:
        metrics = item["metrics"]
        lines.append(
            f"| {item['code']} | {number(metrics['positive_macro_iou'])} | "
            f"{number(metrics['positive_macro_dice'])} | {number(metrics['micro_iou'])} | "
            f"{number(metrics['micro_precision'])} | {number(metrics['micro_recall'])} | "
            f"{percent(metrics['empty_slice_false_positive_rate'])} | "
            f"{number(metrics['loss'])} |"
        )

    lines.extend(
        [
            "",
            "按 Test Positive Macro IoU 排名：",
            "",
        ]
    )
    for index, item in enumerate(ranked, start=1):
        lines.append(
            f"{index}. {item['code']}：{number(item['metrics']['positive_macro_iou'])}"
        )

    if all(item["original_metrics"] is not None for item in results):
        lines.extend(
            [
                "",
                "## 6. 相对原混合队列的变化",
                "",
                "下表用于观察清除单通道患者后重新训练的变化。由于测试集由 11 名患者变为 "
                "10 名患者，同时模型也从头训练，因此该变化不能只解释为删除某一患者的直接贡献。",
                "",
                "| 模型 | 原 Positive IoU | 清理后 Positive IoU | 变化 | 原空切片误报率 | 清理后误报率 | 变化 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in results:
            old = item["original_metrics"]
            new = item["metrics"]
            iou_delta = float(new["positive_macro_iou"]) - float(old["positive_macro_iou"])
            fp_delta = float(new["empty_slice_false_positive_rate"]) - float(
                old["empty_slice_false_positive_rate"]
            )
            lines.append(
                f"| {item['code']} | {number(old['positive_macro_iou'])} | "
                f"{number(new['positive_macro_iou'])} | {signed(iou_delta)} | "
                f"{percent(old['empty_slice_false_positive_rate'])} | "
                f"{percent(new['empty_slice_false_positive_rate'])} | "
                f"{fp_delta * 100.0:+.2f} pp |"
            )

    lines.extend(
        [
            "",
            "## 7. 自动结论",
            "",
            f"在纯多通道患者队列上，{best['code']} 获得最高 Test Positive Macro IoU："
            f"{number(best['metrics']['positive_macro_iou'])}。",
            "",
            "本报告的模型选择以 Positive Macro IoU 为主，同时应结合 Recall、Precision 和"
            "空切片误报率判断临床取舍。所有结果仍来自单次 seed 42 实验，后续显著性结论需要"
            "额外随机种子或 patient-level 交叉验证。",
            "",
            "## 8. 原始结果",
            "",
        ]
    )
    for item in results:
        lines.append(
            f"- {item['code']}：[训练总结]({result_link(item['run'], 'training_summary.json')})；"
            f"[测试指标]({result_link(item['run'], 'test_metrics.json')})"
        )
    lines.extend(
        [
            "- [纯多通道 split 元数据](../splits/kaggle_3m_multimodal_only_seed42.meta.json)",
            "",
        ]
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
