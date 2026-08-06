from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from brain_tumor_seg.data import BrainTumorDataset, SegmentationTransform
from brain_tumor_seg.splits import Sample
from brain_tumor_seg.train import build_balanced_mask_weights


def test_mask_background_value_three_is_binarized_as_background(tmp_path) -> None:
    class_dir = tmp_path / "Glioma"
    class_dir.mkdir()
    image = np.full((8, 8), 127, dtype=np.uint8)
    mask = np.full((8, 8), 3, dtype=np.uint8)
    mask[2:5, 3:7] = 255
    Image.fromarray(image).save(class_dir / "enh_1.png")
    Image.fromarray(mask).save(class_dir / "enh_1_mask.png")
    sample = Sample(
        sample_id="Glioma__enh_1",
        source_id=1,
        tumor_type="Glioma",
        group_id="Glioma__enh_1",
        split="train",
        image_path="Glioma/enh_1.png",
        mask_path="Glioma/enh_1_mask.png",
    )
    transform = SegmentationTransform((8, 8), train=False, mean=0.5, std=0.5)
    item = BrainTumorDataset(tmp_path, [sample], transform)[0]
    assert set(item["mask"].unique().tolist()) == {0.0, 1.0}
    assert item["mask"].sum().item() == 12
    assert item["image"].shape == (1, 8, 8)
    assert item["image_path"] == "Glioma/enh_1.png"
    assert item["mask_path"] == "Glioma/enh_1_mask.png"


def test_mri_noise_augmentation_changes_only_the_image() -> None:
    random.seed(7)
    np.random.seed(7)
    image = Image.fromarray(np.full((16, 16), 127, dtype=np.uint8))
    mask_array = np.full((16, 16), 3, dtype=np.uint8)
    mask_array[4:12, 4:12] = 255
    mask = Image.fromarray(mask_array)
    plain = SegmentationTransform((16, 16), train=False, mean=0.5, std=0.5)
    augmented = SegmentationTransform(
        (16, 16),
        train=True,
        mean=0.5,
        std=0.5,
        augmentation={
            "gaussian_noise_probability": 1.0,
            "gaussian_noise_std": 0.05,
        },
    )
    plain_image, plain_mask = plain(image, mask)
    augmented_image, augmented_mask = augmented(image, mask)
    assert not np.array_equal(plain_image.numpy(), augmented_image.numpy())
    assert np.array_equal(plain_mask.numpy(), augmented_mask.numpy())


def test_flair_green_channel_mode_uses_only_green_rgb_channel(tmp_path) -> None:
    patient_dir = tmp_path / "patient"
    patient_dir.mkdir()
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 10
    rgb[:, :, 1] = 128
    rgb[:, :, 2] = 240
    Image.fromarray(rgb, mode="RGB").save(patient_dir / "slice.tif")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(patient_dir / "slice_mask.tif")
    sample = Sample(
        sample_id="patient__slice_1",
        source_id=1,
        tumor_type="LGG",
        group_id="patient",
        split="train",
        image_path="patient/slice.tif",
        mask_path="patient/slice_mask.tif",
    )
    transform = SegmentationTransform((8, 8), train=False, mean=0.0, std=1.0)
    item = BrainTumorDataset(
        tmp_path, [sample], transform, channel_mode="flair_green"
    )[0]
    assert item["image"].mean().item() == pytest.approx(128.0 / 255.0)


def test_rgb_multimodal_channel_mode_preserves_all_three_channels(tmp_path) -> None:
    patient_dir = tmp_path / "patient"
    patient_dir.mkdir()
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 10
    rgb[:, :, 1] = 128
    rgb[:, :, 2] = 240
    Image.fromarray(rgb, mode="RGB").save(patient_dir / "slice.tif")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(patient_dir / "slice_mask.tif")
    sample = Sample(
        sample_id="patient__slice_1",
        source_id=1,
        tumor_type="LGG",
        group_id="patient",
        split="train",
        image_path="patient/slice.tif",
        mask_path="patient/slice_mask.tif",
    )
    transform = SegmentationTransform((8, 8), train=False, mean=0.0, std=1.0)
    item = BrainTumorDataset(
        tmp_path, [sample], transform, channel_mode="rgb_multimodal"
    )[0]
    assert item["image"].shape == (3, 8, 8)
    assert item["image"][:, 0, 0].tolist() == pytest.approx(
        [10.0 / 255.0, 128.0 / 255.0, 240.0 / 255.0]
    )


def test_balanced_mask_weights_assign_requested_total_probability(tmp_path) -> None:
    samples = []
    for index, positive in enumerate((True, False, False), start=1):
        mask = np.zeros((8, 8), dtype=np.uint8)
        if positive:
            mask[2:4, 3:5] = 255
        path = tmp_path / f"mask_{index}.tif"
        Image.fromarray(mask).save(path)
        samples.append(
            Sample(
                sample_id=f"sample_{index}",
                source_id=index,
                tumor_type="LGG",
                group_id=f"group_{index}",
                split="train",
                image_path=f"image_{index}.tif",
                mask_path=path.name,
            )
        )
    weights = build_balanced_mask_weights(samples, tmp_path, positive_fraction=0.5)
    assert weights[0].item() == pytest.approx(0.5)
    assert weights[1:].sum().item() == pytest.approx(0.5)
