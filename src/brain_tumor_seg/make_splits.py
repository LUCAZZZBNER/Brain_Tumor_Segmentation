from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, project_path
from .splits import (
    apply_group_mapping,
    build_metadata,
    discover_samples,
    load_group_mapping,
    stratified_group_split,
    write_manifest,
    write_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an immutable train/val/test manifest")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to YAML config")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing manifest and metadata file"
    )
    parser.add_argument(
        "--skip-content-hashes",
        action="store_true",
        help="Create a faster manifest without per-file SHA-256 hashes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    data_root = project_path(config, data_config["root"])
    manifest_path = project_path(config, data_config["manifest"])
    metadata_path = project_path(config, data_config["split_metadata"])
    if not args.overwrite and (manifest_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Split files already exist. Refusing to change the experiment partition: "
            f"{manifest_path}, {metadata_path}. Pass --overwrite only intentionally."
        )

    samples = discover_samples(
        data_root, validate_images=True, compute_hashes=not args.skip_content_hashes
    )
    group_csv_value = data_config.get("group_csv")
    if group_csv_value:
        group_csv = project_path(config, group_csv_value)
        samples = apply_group_mapping(samples, load_group_mapping(group_csv))
        split_level = "group"
        group_csv_text: str | None = str(Path(group_csv_value).as_posix())
    else:
        split_level = "sample"
        group_csv_text = None

    ratios = {key: float(value) for key, value in data_config["split_ratios"].items()}
    assigned = stratified_group_split(samples, ratios, int(config["project"]["seed"]))
    manifest_hash = write_manifest(assigned, manifest_path)
    metadata = build_metadata(
        assigned,
        seed=int(config["project"]["seed"]),
        ratios=ratios,
        split_level=split_level,
        manifest_sha256=manifest_hash,
        group_csv=group_csv_text,
    )
    write_metadata(metadata, metadata_path)
    print(f"Wrote {len(assigned)} samples to {manifest_path}")
    print(f"Split level: {split_level}; counts: {metadata['counts']}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
