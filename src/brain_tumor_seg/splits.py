from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image


FILE_PATTERN = re.compile(r"enh_(\d+)(_mask)?\.png$")
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = (
    "sample_id",
    "source_id",
    "tumor_type",
    "group_id",
    "split",
    "image_path",
    "mask_path",
    "image_sha256",
    "mask_sha256",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_id: int
    tumor_type: str
    group_id: str
    split: str
    image_path: str
    mask_path: str
    image_sha256: str = ""
    mask_sha256: str = ""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pair(image_path: Path, mask_path: Path) -> None:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.format != "PNG" or mask.format != "PNG":
            raise ValueError(f"Only PNG files are supported: {image_path}, {mask_path}")
        if image.mode != "L" or mask.mode != "L":
            raise ValueError(f"Expected grayscale L images: {image_path}, {mask_path}")
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {image_path}, {mask_path}")
        values = set(int(value) for value in np.unique(np.asarray(mask)))
        if not values.issubset({0, 3, 255}):
            raise ValueError(f"Unexpected mask values {sorted(values)} in {mask_path}")
        binary_values = set(int(value) for value in np.unique(np.asarray(mask) >= 128))
        if binary_values != {0, 1}:
            raise ValueError(f"Mask must contain both foreground and background: {mask_path}")


def discover_samples(
    data_root: str | Path,
    *,
    validate_images: bool = True,
    compute_hashes: bool = True,
) -> list[Sample]:
    """Discover image/mask pairs and fail fast on malformed or incomplete data."""
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    samples: list[Sample] = []
    seen_ids: set[str] = set()
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No tumor-type directories found under {root}")

    for class_dir in class_dirs:
        images: dict[int, Path] = {}
        masks: dict[int, Path] = {}
        unexpected: list[str] = []
        for path in sorted(class_dir.glob("*.png")):
            match = FILE_PATTERN.fullmatch(path.name)
            if match is None:
                unexpected.append(path.name)
                continue
            source_id = int(match.group(1))
            destination = masks if match.group(2) else images
            if source_id in destination:
                raise ValueError(f"Duplicate source ID {source_id} in {class_dir}")
            destination[source_id] = path
        if unexpected:
            raise ValueError(f"Unexpected PNG names in {class_dir}: {unexpected[:10]}")
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        if missing_masks or missing_images:
            raise ValueError(
                f"Unpaired files in {class_dir}: missing masks={missing_masks[:10]}, "
                f"missing images={missing_images[:10]}"
            )

        for source_id in sorted(images):
            image_path, mask_path = images[source_id], masks[source_id]
            sample_id = f"{class_dir.name}__enh_{source_id}"
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            seen_ids.add(sample_id)
            if validate_images:
                _validate_pair(image_path, mask_path)
            samples.append(
                Sample(
                    sample_id=sample_id,
                    source_id=source_id,
                    tumor_type=class_dir.name,
                    group_id=sample_id,
                    split="",
                    image_path=image_path.relative_to(root).as_posix(),
                    mask_path=mask_path.relative_to(root).as_posix(),
                    image_sha256=sha256_file(image_path) if compute_hashes else "",
                    mask_sha256=sha256_file(mask_path) if compute_hashes else "",
                )
            )
    if not samples:
        raise ValueError(f"No image/mask pairs found under {root}")
    return samples


def load_group_mapping(path: str | Path) -> dict[str, str]:
    """Load an explicit sample_id -> patient/group mapping."""
    mapping: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"sample_id", "group_id"}.issubset(reader.fieldnames):
            raise ValueError("Group CSV must contain sample_id and group_id columns")
        for row in reader:
            sample_id, group_id = row["sample_id"].strip(), row["group_id"].strip()
            if not sample_id or not group_id:
                raise ValueError("sample_id and group_id must be non-empty")
            if sample_id in mapping:
                raise ValueError(f"Duplicate sample_id in group CSV: {sample_id}")
            mapping[sample_id] = group_id
    return mapping


def apply_group_mapping(samples: Iterable[Sample], mapping: Mapping[str, str]) -> list[Sample]:
    samples = list(samples)
    sample_ids = {sample.sample_id for sample in samples}
    missing = sorted(sample_ids - set(mapping))
    extra = sorted(set(mapping) - sample_ids)
    if missing or extra:
        raise ValueError(
            f"Group mapping must match the dataset exactly; missing={missing[:10]}, extra={extra[:10]}"
        )
    return [replace(sample, group_id=mapping[sample.sample_id]) for sample in samples]


def _stable_class_seed(seed: int, tumor_type: str) -> int:
    suffix = int(hashlib.sha256(tumor_type.encode("utf-8")).hexdigest()[:8], 16)
    return seed + suffix


def stratified_group_split(
    samples: Iterable[Sample], ratios: Mapping[str, float], seed: int
) -> list[Sample]:
    """Assign whole groups to splits while approximately preserving each class ratio."""
    samples = list(samples)
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("ratios must contain train/val/test and sum to 1")

    group_classes: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        group_classes[sample.group_id].add(sample.tumor_type)
    mixed = {group: classes for group, classes in group_classes.items() if len(classes) != 1}
    if mixed:
        raise ValueError(f"Each group must belong to one tumor type; mixed groups: {mixed}")

    assignments: dict[str, str] = {}
    by_class: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        by_class[sample.tumor_type][sample.group_id].append(sample)

    for tumor_type, groups_by_id in sorted(by_class.items()):
        if len(groups_by_id) < len(SPLITS):
            raise ValueError(
                f"Tumor type {tumor_type!r} has only {len(groups_by_id)} groups; at least 3 are "
                "required for train/val/test"
            )
        rng = random.Random(_stable_class_seed(seed, tumor_type))
        groups = list(groups_by_id.items())
        rng.shuffle(groups)
        groups.sort(key=lambda item: len(item[1]), reverse=True)
        target = {split: ratios[split] * sum(len(value) for value in groups_by_id.values()) for split in SPLITS}
        counts = {split: 0 for split in SPLITS}

        # Greedy largest-deficit allocation is deterministic after the seeded tie shuffle.
        for group_id, members in groups:
            split_order = list(SPLITS)
            rng.shuffle(split_order)
            chosen = max(
                split_order,
                key=lambda split: (target[split] - counts[split]) / max(target[split], 1.0),
            )
            assignments[group_id] = chosen
            counts[chosen] += len(members)

        if any(counts[split] == 0 for split in SPLITS):
            raise RuntimeError(f"Could not create non-empty splits for class {tumor_type}: {counts}")

    result = [replace(sample, split=assignments[sample.group_id]) for sample in samples]
    validate_assignments(result)
    return sorted(result, key=lambda sample: (SPLITS.index(sample.split), sample.tumor_type, sample.source_id))


def validate_assignments(samples: Iterable[Sample]) -> None:
    samples = list(samples)
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in split assignment")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.split not in SPLITS:
            raise ValueError(f"Invalid split for {sample.sample_id}: {sample.split}")
        group_splits[sample.group_id].add(sample.split)
    leaking = {group: values for group, values in group_splits.items() if len(values) > 1}
    if leaking:
        raise ValueError(f"Group leakage across splits: {leaking}")
    split_counts = Counter(sample.split for sample in samples)
    if set(split_counts) != set(SPLITS):
        raise ValueError(f"Every split must be non-empty: {dict(split_counts)}")


def write_manifest(samples: Iterable[Sample], path: str | Path) -> str:
    samples = list(samples)
    validate_assignments(samples)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))
    return sha256_file(manifest_path)


def read_manifest(path: str | Path) -> list[Sample]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(MANIFEST_FIELDS):
            raise ValueError(f"Manifest columns must be exactly {list(MANIFEST_FIELDS)}")
        samples = [
            Sample(
                sample_id=row["sample_id"],
                source_id=int(row["source_id"]),
                tumor_type=row["tumor_type"],
                group_id=row["group_id"],
                split=row["split"],
                image_path=row["image_path"],
                mask_path=row["mask_path"],
                image_sha256=row["image_sha256"],
                mask_sha256=row["mask_sha256"],
            )
            for row in reader
        ]
    validate_assignments(samples)
    return samples


def verify_manifest_files(
    samples: Iterable[Sample], data_root: str | Path, *, verify_hashes: bool = False
) -> None:
    root = Path(data_root)
    for sample in samples:
        image_path, mask_path = root / sample.image_path, root / sample.mask_path
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing files for {sample.sample_id}: {image_path}, {mask_path}")
        if verify_hashes:
            if not sample.image_sha256 or not sample.mask_sha256:
                raise ValueError("Manifest has no hashes but hash verification was requested")
            if sha256_file(image_path) != sample.image_sha256:
                raise ValueError(f"Image hash mismatch: {image_path}")
            if sha256_file(mask_path) != sample.mask_sha256:
                raise ValueError(f"Mask hash mismatch: {mask_path}")


def dataset_fingerprint(samples: Iterable[Sample]) -> str:
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda item: item.sample_id):
        payload = (
            f"{sample.sample_id}\0{sample.image_path}\0{sample.mask_path}\0"
            f"{sample.image_sha256}\0{sample.mask_sha256}\n"
        )
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def build_metadata(
    samples: Iterable[Sample],
    *,
    seed: int,
    ratios: Mapping[str, float],
    split_level: str,
    manifest_sha256: str,
    group_csv: str | None,
) -> dict[str, object]:
    samples = list(samples)
    counts = Counter(sample.split for sample in samples)
    per_class: dict[str, dict[str, int]] = {}
    for tumor_type in sorted({sample.tumor_type for sample in samples}):
        per_class[tumor_type] = dict(
            Counter(sample.split for sample in samples if sample.tumor_type == tumor_type)
        )
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "ratios": dict(ratios),
        "split_level": split_level,
        "group_csv": group_csv,
        "num_samples": len(samples),
        "counts": dict(counts),
        "per_class_counts": per_class,
        "dataset_fingerprint": dataset_fingerprint(samples),
        "manifest_sha256": manifest_sha256,
        "mask_binarization": "mask >= 128",
    }


def write_metadata(metadata: Mapping[str, object], path: str | Path) -> None:
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
