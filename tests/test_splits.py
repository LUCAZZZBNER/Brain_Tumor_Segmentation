from __future__ import annotations

from collections import defaultdict

from brain_tumor_seg.splits import Sample, read_manifest, stratified_group_split, write_manifest


def make_samples() -> list[Sample]:
    samples = []
    for tumor_type in ("Glioma", "Meningioma", "Pituitary tumor"):
        for group_index in range(6):
            group_id = f"{tumor_type}-patient-{group_index}"
            for slice_index in range(2):
                source_id = group_index * 10 + slice_index
                sample_id = f"{tumor_type}-{source_id}"
                samples.append(
                    Sample(
                        sample_id=sample_id,
                        source_id=source_id,
                        tumor_type=tumor_type,
                        group_id=group_id,
                        split="",
                        image_path=f"{sample_id}.png",
                        mask_path=f"{sample_id}_mask.png",
                    )
                )
    return samples


def test_stratified_group_split_is_reproducible_and_has_no_group_leakage() -> None:
    ratios = {"train": 0.5, "val": 0.25, "test": 0.25}
    first = stratified_group_split(make_samples(), ratios, seed=42)
    second = stratified_group_split(make_samples(), ratios, seed=42)
    assert [(sample.sample_id, sample.split) for sample in first] == [
        (sample.sample_id, sample.split) for sample in second
    ]

    group_splits: dict[str, set[str]] = defaultdict(set)
    class_splits: dict[str, set[str]] = defaultdict(set)
    for sample in first:
        group_splits[sample.group_id].add(sample.split)
        class_splits[sample.tumor_type].add(sample.split)
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert all(splits == {"train", "val", "test"} for splits in class_splits.values())


def test_manifest_round_trip(tmp_path) -> None:
    assigned = stratified_group_split(
        make_samples(), {"train": 0.5, "val": 0.25, "test": 0.25}, seed=7
    )
    path = tmp_path / "manifest.csv"
    digest = write_manifest(assigned, path)
    restored = read_manifest(path)
    assert len(digest) == 64
    assert restored == assigned

