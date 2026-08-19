from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
EXPERIMENTS = (
    ("E0", "kaggle_3m_multimodal_only_e0_flair_unet"),
    ("E1-A", "kaggle_3m_multimodal_only_e1_rgb_unet_no_augmentation"),
    ("E2-B", "kaggle_3m_multimodal_only_e2_flair_unet_augmentation"),
    ("M0-AB", "kaggle_3m_multimodal_only_m0_rgb_unet"),
    ("M4-NP", "kaggle_3m_multimodal_only_m4_no_pretrain_rgb_resnet34_unet"),
    ("M4-P−A", "kaggle_3m_multimodal_only_m4_p_minus_a_flair_resnet34_unet"),
    ("M4-P−B", "kaggle_3m_multimodal_only_m4_p_minus_b_rgb_resnet34_unet_no_augmentation"),
    ("M4-P", "kaggle_3m_multimodal_only_m4_rgb_resnet34_unet"),
)
METRICS = (
    "positive_macro_iou",
    "positive_macro_dice",
    "micro_iou",
    "micro_precision",
    "micro_recall",
    "empty_slice_false_positive_rate",
)
EFFECTS = (
    ("完整方案", "M4-P", "E0"),
    ("三通道输入 A", "M4-P", "M4-P−A"),
    ("轻量增强 B", "M4-P", "M4-P−B"),
    ("ResNet34 结构 C", "M4-NP", "M0-AB"),
    ("ImageNet 预训练 D", "M4-P", "M4-NP"),
)


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def patient_values(path: Path):
    grouped: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["true_positive"]) + int(row["false_negative"]) <= 0:
                continue
            patient = row["sample_id"].split("__slice_", 1)[0]
            grouped.setdefault(patient, []).append(float(row["iou"]))
    return {key: statistics.mean(values) for key, values in grouped.items()}


def percentile(values, probability):
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def bootstrap(differences):
    rng = random.Random(42)
    estimates = sorted(
        statistics.mean(rng.choices(differences, k=len(differences)))
        for _ in range(10_000)
    )
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def normalized_config(config):
    value = json.loads(json.dumps(config))
    value["project"].pop("seed", None)
    value["project"].pop("output_dir", None)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(output: Path):
    metadata = read_json(ROOT / "splits/kaggle_3m_multimodal_only_seed42.meta.json")
    results = {}
    problems = []
    manifest_hashes = Counter()
    environments = Counter()
    run_rows = []
    for code, stem in EXPERIMENTS:
        results[code] = {}
        base_config = None
        for seed in SEEDS:
            run = ROOT / "runs" / f"{stem}_seed{seed}"
            required = (
                "resolved_config.json", "training_summary.json", "test_metrics.json",
                "evaluation/test/samples.csv", "environment.json"
            )
            missing = [name for name in required if not (run / name).is_file()]
            if missing:
                problems.append({"severity": "严重", "item": str(run), "problem": f"缺少 {missing}"})
                continue
            config = read_json(run / "resolved_config.json")
            summary = read_json(run / "training_summary.json")
            test = read_json(run / "test_metrics.json")
            env = read_json(run / "environment.json")
            metrics = test["metrics"]
            patients = patient_values(run / "evaluation/test/samples.csv")
            results[code][seed] = {"config": config, "summary": summary, "test": test, "metrics": metrics, "patients": patients}
            manifest_hashes[test["manifest_sha256"]] += 1
            environments[(tuple(env.get("gpu_names", [])), env.get("cuda_version"), env.get("torch"))] += 1
            if base_config is None:
                base_config = normalized_config(config)
            elif normalized_config(config) != base_config:
                problems.append({"severity": "严重", "item": f"{code}/seed{seed}", "problem": "同模型不同种子除种子和输出目录外配置不一致"})
            if int(metrics["num_samples"]) != 525 or int(metrics["num_positive_images"]) != 173 or int(metrics["num_empty_images"]) != 352:
                problems.append({"severity": "严重", "item": f"{code}/seed{seed}", "problem": "测试样本统计不一致"})
            if test.get("threshold") != 0.5 or test.get("selection_metric") != "positive_macro_iou":
                problems.append({"severity": "严重", "item": f"{code}/seed{seed}", "problem": "阈值或模型选择指标不一致"})
            if test.get("checkpoint_epoch") != summary.get("best_epoch"):
                problems.append({"severity": "严重", "item": f"{code}/seed{seed}", "problem": "测试检查点不是训练总结中的最佳 epoch"})
            run_rows.append({
                "model": code, "seed": seed, "best_epoch": summary["best_epoch"],
                "test_positive_iou": metrics["positive_macro_iou"],
                "manifest_sha256": test["manifest_sha256"], "samples": metrics["num_samples"],
                "positive": metrics["num_positive_images"], "empty": metrics["num_empty_images"]
            })

    summaries = {}
    for code, _stem in EXPERIMENTS:
        summaries[code] = {}
        for metric in METRICS:
            values = [float(results[code][seed]["metrics"][metric]) for seed in SEEDS]
            mean, sd = mean_sd(values)
            summaries[code][metric] = {"values": values, "mean": mean, "sd": sd}

    effects = {}
    for label, left, right in EFFECTS:
        seed_diffs = [
            float(results[left][seed]["metrics"]["positive_macro_iou"])
            - float(results[right][seed]["metrics"]["positive_macro_iou"])
            for seed in SEEDS
        ]
        patients = sorted(set.intersection(*[
            set(results[code][seed]["patients"])
            for code in (left, right) for seed in SEEDS
        ]))
        patient_diffs = []
        for patient in patients:
            lval = statistics.mean(results[left][seed]["patients"][patient] for seed in SEEDS)
            rval = statistics.mean(results[right][seed]["patients"][patient] for seed in SEEDS)
            patient_diffs.append(lval - rval)
        low, high = bootstrap(patient_diffs)
        effects[label] = {
            "comparison": f"{left} − {right}", "seed_values": seed_diffs,
            "seed_mean": statistics.mean(seed_diffs), "seed_sd": statistics.stdev(seed_diffs),
            "positive_seeds": sum(value > 0 for value in seed_diffs),
            "patient_mean": statistics.mean(patient_diffs),
            "positive_patients": sum(value > 0 for value in patient_diffs),
            "patient_count": len(patient_diffs), "ci95": [low, high]
        }

    expected_model_fields = {
        "E0": ("flair_green", "unet", False, False),
        "E1-A": ("rgb_multimodal", "unet", False, False),
        "E2-B": ("flair_green", "unet", True, False),
        "M0-AB": ("rgb_multimodal", "unet", True, False),
        "M4-NP": ("rgb_multimodal", "resnet34_unet", True, False),
        "M4-P−A": ("flair_green", "resnet34_unet", True, True),
        "M4-P−B": ("rgb_multimodal", "resnet34_unet", False, True),
        "M4-P": ("rgb_multimodal", "resnet34_unet", True, True),
    }
    config_matrix = []
    for code, _stem in EXPERIMENTS:
        cfg = results[code][42]["config"]
        actual = (
            cfg["data"]["channel_mode"], cfg["model"]["name"],
            bool(cfg["data"].get("augmentation")), bool(cfg["model"].get("pretrained", False))
        )
        config_matrix.append({"model": code, "actual": actual, "expected": expected_model_fields[code], "matches": actual == expected_model_fields[code]})
        if actual != expected_model_fields[code]:
            problems.append({"severity": "严重", "item": code, "problem": f"配置矩阵不符：{actual}"})

    sample_checks = {}
    for model, stem in (("M4-P", EXPERIMENTS[-1][1]), ("E0", EXPERIMENTS[0][1])):
        path = ROOT / "runs" / f"{stem}_seed42" / "evaluation/test/samples.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = {row["sample_id"]: row for row in csv.DictReader(handle)}
        for sample in (
            "TCGA_DU_5872_19950223__slice_45", "TCGA_DU_6408_19860521__slice_22",
            "TCGA_HT_7881_19981015__slice_23", "TCGA_FG_8189_20030516__slice_51"
        ):
            if sample in rows:
                sample_checks[f"{model}:{sample}"] = rows[sample]

    source_text = (ROOT / "src/brain_tumor_seg/data.py").read_text(encoding="utf-8")
    missing_fill_implemented = "missing" in source_text.lower() and "fill" in source_text.lower()
    report = {
        "metadata": metadata, "run_rows": run_rows, "manifest_hash_counts": dict(manifest_hashes),
        "environments": [{"gpu": list(key[0]), "cuda": key[1], "torch": key[2], "count": value} for key, value in environments.items()],
        "summaries": summaries, "effects": effects, "config_matrix": config_matrix,
        "problems": problems, "sample_checks": sample_checks,
        "data_loader_missing_modality_fill_implemented": missing_fill_implemented,
        "data_loader_sha256": sha256(ROOT / "src/brain_tumor_seg/data.py"),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
