from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "group_id", "split", "image_path", "mask_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}: {path}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {row['sample_id']}: {row['split']}")
        group_splits[row["group_id"]].add(row["split"])
    leaking = {group: values for group, values in group_splits.items() if len(values) > 1}
    if leaking:
        raise ValueError(f"Patient leakage in source manifest: {leaking}")
    return rows


def case_identifier(sample_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    value = "k3m_" + "".join(character if character in allowed else "_" for character in sample_id)
    if not value or value[-1] == "_":
        value += "case"
    return value


def save_case(
    row: dict[str, str],
    *,
    data_root: Path,
    images_dir: Path,
    labels_dir: Path,
) -> str:
    case_id = case_identifier(row["sample_id"])
    image_path = data_root / row["image_path"]
    mask_path = data_root / row["mask_path"]
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
        if image.size != (256, 256):
            raise ValueError(f"Expected 256x256 image, got {image.size}: {image_path}")
        channels = image.split()
        for index, channel in enumerate(channels):
            channel.save(images_dir / f"{case_id}_{index:04d}.png")
    with Image.open(mask_path) as mask_file:
        mask_array = np.asarray(mask_file.convert("L"), dtype=np.uint8)
    if mask_array.shape != (256, 256):
        raise ValueError(f"Expected 256x256 mask, got {mask_array.shape}: {mask_path}")
    binary_mask = (mask_array >= 128).astype(np.uint8)
    Image.fromarray(binary_mask, mode="L").save(labels_dir / f"{case_id}.png")
    return case_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the fixed clean kaggle_3m manifest to nnU-Net v2 2D format"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = read_yaml(config_path)
    seed = int(config["project"]["seed"])
    if seed != 42:
        raise ValueError(f"This baseline is frozen to seed 42, got {seed}")
    data_config = config["data"]
    nnunet_config = config["nnunet"]
    manifest_path = resolve(project_root, str(data_config["manifest"]))
    metadata_path = resolve(project_root, str(data_config["split_metadata"]))
    data_root = resolve(project_root, str(data_config["root"]))
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != metadata.get("manifest_sha256"):
        raise ValueError("Manifest hash differs from split metadata")
    if metadata.get("split_level") != "patient":
        raise ValueError("The source manifest is not declared patient-level")
    rows = load_manifest(manifest_path)

    output_dir = resolve(project_root, str(config["project"]["output_dir"]))
    workspace_root = output_dir / "workspace"
    dataset_id = int(nnunet_config["dataset_id"])
    dataset_name = str(nnunet_config["dataset_name"])
    dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
    raw_root = workspace_root / "nnUNet_raw"
    dataset_root = raw_root / dataset_folder
    marker_path = dataset_root / "conversion_metadata.json"
    if dataset_root.exists():
        if marker_path.is_file() and not args.overwrite:
            with marker_path.open("r", encoding="utf-8") as handle:
                marker = json.load(handle)
            if marker.get("manifest_sha256") == manifest_hash:
                print(f"Dataset conversion already matches the fixed manifest: {dataset_root}")
                return
        if not args.overwrite:
            raise FileExistsError(
                f"nnU-Net raw dataset already exists: {dataset_root}. "
                "Pass --overwrite only to rebuild this exact derived dataset."
            )
        shutil.rmtree(dataset_root)

    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_ts = dataset_root / "imagesTs"
    labels_ts = dataset_root / "labelsTs"
    for directory in (images_tr, labels_tr, images_ts, labels_ts):
        directory.mkdir(parents=True, exist_ok=True)

    split_cases = {"train": [], "val": [], "test": []}
    case_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        is_test = row["split"] == "test"
        case_id = save_case(
            row,
            data_root=data_root,
            images_dir=images_ts if is_test else images_tr,
            labels_dir=labels_ts if is_test else labels_tr,
        )
        split_cases[row["split"]].append(case_id)
        case_rows.append(
            {
                "case_id": case_id,
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "split": row["split"],
            }
        )
        if index % 250 == 0 or index == len(rows):
            print(f"Converted {index}/{len(rows)} slices")

    channel_names = list(data_config.get("channel_names", []))
    if len(channel_names) != 3:
        raise ValueError("data.channel_names must contain the three MRI channel names")
    dataset_json = {
        "channel_names": {str(index): name for index, name in enumerate(channel_names)},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(split_cases["train"]) + len(split_cases["val"]),
        "file_ending": str(data_config.get("file_ending", ".png")),
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    write_json(dataset_root / "dataset.json", dataset_json)
    write_json(
        dataset_root / "splits_final.json",
        [{"train": sorted(split_cases["train"]), "val": sorted(split_cases["val"])}],
    )
    with (dataset_root / "case_map.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("case_id", "sample_id", "group_id", "split")
        )
        writer.writeheader()
        writer.writerows(case_rows)
    write_json(
        marker_path,
        {
            "manifest_sha256": manifest_hash,
            "split_level": metadata["split_level"],
            "seed": seed,
            "dataset_folder": dataset_folder,
            "sample_counts": dict(Counter(row["split"] for row in rows)),
            "patient_counts": {
                split: len({row["group_id"] for row in rows if row["split"] == split})
                for split in ("train", "val", "test")
            },
        },
    )
    print(f"Prepared official nnU-Net v2 dataset: {dataset_root}")
    print(f"Patient-disjoint split file: {dataset_root / 'splits_final.json'}")


if __name__ == "__main__":
    main()
