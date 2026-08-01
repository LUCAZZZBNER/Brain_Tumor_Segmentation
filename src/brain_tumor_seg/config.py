from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and validate the fields used by the baseline."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {
        "project",
        "data",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "metrics",
        "evaluation",
    }
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    ratios = config["data"].get("split_ratios", {})
    if set(ratios) != {"train", "val", "test"}:
        raise ValueError("data.split_ratios must contain exactly train, val, and test")
    if any(float(value) <= 0 for value in ratios.values()):
        raise ValueError("All split ratios must be positive")
    if abs(sum(float(value) for value in ratios.values()) - 1.0) > 1e-8:
        raise ValueError("data.split_ratios must sum to 1.0")

    image_size = config["data"].get("image_size")
    if not (
        isinstance(image_size, list)
        and len(image_size) == 2
        and all(isinstance(value, int) and value > 0 for value in image_size)
    ):
        raise ValueError("data.image_size must be a list of two positive integers")
    if any(value % 16 != 0 for value in image_size):
        raise ValueError("Each image dimension must be divisible by 16 for this U-Net")
    if (
        str(config["model"].get("encoder", "double_conv")).lower() == "resnet34"
        and any(value % 32 != 0 for value in image_size)
    ):
        raise ValueError("ResNet-34 U-Net requires each image dimension to be divisible by 32")

    if config["model"].get("in_channels") != 1:
        raise ValueError("This dataset is grayscale; model.in_channels must be 1")
    if config["model"].get("out_channels") != 1:
        raise ValueError("This baseline performs binary segmentation; out_channels must be 1")
    threshold = float(config["metrics"].get("threshold", 0.5))
    if not 0.0 < threshold < 1.0:
        raise ValueError("metrics.threshold must be between 0 and 1")
    threshold_search = config["metrics"].get("threshold_search", [])
    if not isinstance(threshold_search, list):
        raise ValueError("metrics.threshold_search must be a list")
    if any(not 0.0 < float(value) < 1.0 for value in threshold_search):
        raise ValueError("Every metrics.threshold_search value must be between 0 and 1")

    maximum_predictions = config["evaluation"].get("max_saved_predictions", 100)
    if maximum_predictions is not None and int(maximum_predictions) <= 0:
        raise ValueError("evaluation.max_saved_predictions must be positive or null")
    for key in ("save_predictions", "save_probability_maps", "save_comparison_figures"):
        value = config["evaluation"].get(key, True)
        if not isinstance(value, bool):
            raise ValueError(f"evaluation.{key} must be a boolean")

    augmentation = config["data"].get("augmentation", {})
    for key in (
        "horizontal_flip_probability",
        "gaussian_blur_probability",
        "gaussian_noise_probability",
    ):
        probability = float(augmentation.get(key, 0.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"data.augmentation.{key} must be between 0 and 1")


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path
