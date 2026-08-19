from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_COLUMNS = (
    ("positive_macro_iou", "Positive Macro IoU"),
    ("positive_macro_dice", "Positive Macro Dice"),
    ("micro_iou", "Micro IoU"),
    ("micro_precision", "Micro Precision"),
    ("micro_recall", "Micro Recall"),
    ("empty_slice_false_positive_rate", "Empty-slice FPR"),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_metric(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the fixed-split seed-42 modern-baseline Markdown report"
    )
    parser.add_argument(
        "--output", default="reports/kaggle_3m_modern_baselines_seed42.md"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    metadata_path = project_root / "splits/kaggle_3m_multimodal_only_seed42.meta.json"
    metadata = load_json(metadata_path)
    expected_hash = str(metadata["manifest_sha256"])
    expected_samples = int(metadata["counts"]["test"])

    # Clean M0 is the comparison anchor because it uses this exact cleaned
    # manifest. The older E0 result used the pre-cleaning cohort and must not be
    # used to compute deltas for these modern baselines.
    specifications = (
        (
            "Clean M0 U-Net (reference)",
            "runs/kaggle_3m_multimodal_only_m0_rgb_unet_seed42/test_metrics.json",
        ),
        (
            "M4-P / clean ResNet34 U-Net",
            "runs/kaggle_3m_multimodal_only_m4_rgb_resnet34_unet_seed42/test_metrics.json",
        ),
        (
            "Official nnU-Net v2 2D (NoDA)",
            "runs/nnunetv2_2d_kaggle_3m_clean_no_augmentation_seed42/test_metrics.json",
        ),
        (
            "Basic TransUNet 2D",
            "runs/kaggle_3m_multimodal_only_transunet_2d_basic_no_augmentation_seed42/test_metrics.json",
        ),
    )

    results: list[dict[str, Any]] = []
    for model_name, relative_path in specifications:
        result_path = project_root / relative_path
        entry: dict[str, Any] = {
            "name": model_name,
            "path": relative_path,
            "status": "待运行",
            "metrics": {},
        }
        if result_path.is_file():
            payload = load_json(result_path)
            manifest_hash = str(payload.get("manifest_sha256", ""))
            metrics = payload.get("metrics", {})
            sample_count = int(float(metrics.get("num_samples", -1)))
            if manifest_hash != expected_hash:
                entry["status"] = "不可比较：manifest hash 不同"
            elif sample_count != expected_samples:
                entry["status"] = f"不可比较：测试切片数为 {sample_count}"
            else:
                entry["status"] = "完成"
                entry["metrics"] = metrics
        results.append(entry)

    reference_iou = results[0]["metrics"].get("positive_macro_iou")
    table_header = ["模型", "状态", *(label for _, label in METRIC_COLUMNS), "ΔIoU vs M0"]
    lines = [
        "# Kaggle 3M 清洗患者集：Seed-42 现代基线报告",
        "",
        f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## 实验一致性",
        "",
        f"- 患者级固定划分：训练 {metadata['patient_counts']['train']}、验证 "
        f"{metadata['patient_counts']['val']}、测试 {metadata['patient_counts']['test']}，"
        "同一患者不会跨集合。",
        f"- 切片数：训练 {metadata['counts']['train']}、验证 {metadata['counts']['val']}、"
        f"测试 {metadata['counts']['test']}。",
        f"- 随机种子：{metadata['seed']}；清洗后 manifest SHA-256：`{expected_hash}`。",
        "- nnU-Net 和 TransUNet 均禁用训练时数据增强；测试集仅用于最终冻结评估。",
        "",
        "## 测试集结果",
        "",
        "| " + " | ".join(table_header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(table_header) - 1)) + " |",
    ]
    for entry in results:
        metrics = entry["metrics"]
        iou = metrics.get("positive_macro_iou")
        delta = (
            f"{float(iou) - float(reference_iou):+.4f}"
            if isinstance(iou, (int, float)) and isinstance(reference_iou, (int, float))
            else "—"
        )
        row = [
            entry["name"],
            entry["status"],
            *(format_metric(metrics.get(key)) for key, _ in METRIC_COLUMNS),
            delta,
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## 现代基线运行配置",
            "",
            "| 项目 | nnU-Net v2 2D | Basic TransUNet 2D |",
            "| --- | --- | --- |",
            "| 输入 | 3 通道，256×256 | 3 通道，256×256 |",
            "| 初始化 | nnU-Net 官方随机初始化 | 随机初始化 |",
            "| 数据增强 | `nnUNetTrainerNoDA`，禁用 | 禁用 |",
            "| Batch size | 4（仅资源覆盖；官方计划值保留） | 2 |",
            "| 数据进程 | 0（官方单线程 augmenter） | 0（Windows 安全设置） |",
            "| 推理预处理/导出进程 | 1 / 1 | 不适用 |",
            "| 测试增强 | 禁用 TTA | 禁用 |",
            "",
            "## 可重复性与解释说明",
            "",
            "- 表中只有 manifest hash 与 525 张固定测试切片同时匹配的结果才会显示指标并参与 ΔIoU 计算。",
            "- `Clean M0 U-Net` 是当前清洗患者集上的同清单参考。旧 E0 使用清洗前 manifest，不能与这里的现代基线直接计算差值。",
            "- nnU-Net 的 batch size 4、数据进程 0 和导出进程 1 是 Windows 资源安全设置；网络、损失、样本、固定划分、种子及 NoDA 策略不变。",
            "- `Basic TransUNet 2D` 是从零训练的基础对照，并非论文原始的 R50-ViT-B/16 预训练版本；投稿表格中应明确这一点。",
            "",
            "## 结果文件",
            "",
        ]
    )
    for entry in results:
        lines.append(f"- {entry['name']}：`{entry['path']}`（{entry['status']}）")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Modern-baseline report written to {output_path}")


if __name__ == "__main__":
    main()
