from __future__ import annotations

import random

import numpy as np
from PIL import Image

from brain_tumor_seg.data import BrainTumorDataset, SegmentationTransform
from brain_tumor_seg.splits import Sample


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
