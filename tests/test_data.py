from __future__ import annotations

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

