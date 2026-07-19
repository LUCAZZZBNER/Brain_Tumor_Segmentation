from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .splits import Sample


class SegmentationTransform:
    """Apply spatially aligned image/mask transforms and convert them to tensors."""

    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        train: bool,
        mean: float,
        std: float,
        augmentation: dict[str, float] | None = None,
    ) -> None:
        if std <= 0:
            raise ValueError("Normalization std must be positive")
        self.height, self.width = image_size
        self.train = train
        self.mean = mean
        self.std = std
        self.augmentation = augmentation or {}

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.resize((self.width, self.height), resample=Image.Resampling.BILINEAR)
        mask = mask.resize((self.width, self.height), resample=Image.Resampling.NEAREST)

        if self.train:
            if random.random() < float(
                self.augmentation.get("horizontal_flip_probability", 0.0)
            ):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            max_rotation = float(self.augmentation.get("rotation_degrees", 0.0))
            if max_rotation > 0:
                angle = random.uniform(-max_rotation, max_rotation)
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
                # The stored mask background is 3; it remains background after thresholding.
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=3)

            brightness = float(self.augmentation.get("brightness", 0.0))
            if brightness > 0:
                image = ImageEnhance.Brightness(image).enhance(
                    random.uniform(1.0 - brightness, 1.0 + brightness)
                )
            contrast = float(self.augmentation.get("contrast", 0.0))
            if contrast > 0:
                image = ImageEnhance.Contrast(image).enhance(
                    random.uniform(1.0 - contrast, 1.0 + contrast)
                )

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        # Do not use mask > 0: this dataset stores background as value 3.
        mask_array = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.copy()).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array.copy()).unsqueeze(0)
        image_tensor = (image_tensor - self.mean) / self.std
        return image_tensor, mask_tensor


class BrainTumorDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        data_root: str | Path,
        samples: Iterable[Sample],
        transform: SegmentationTransform,
    ) -> None:
        self.data_root = Path(data_root)
        self.samples = list(samples)
        self.transform = transform
        if not self.samples:
            raise ValueError("Dataset split is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = self.data_root / sample.image_path
        mask_path = self.data_root / sample.mask_path
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            image = image_file.convert("L")
            mask = mask_file.convert("L")
            image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
            "source_id": sample.source_id,
            "tumor_type": sample.tumor_type,
            "image_path": sample.image_path,
            "original_size": (image.height, image.width),
        }


def seed_worker(worker_id: int) -> None:
    """Make Python/NumPy augmentation RNGs follow PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transform(data_config: dict[str, Any], *, train: bool) -> SegmentationTransform:
    normalization = data_config["normalization"]
    return SegmentationTransform(
        tuple(int(value) for value in data_config["image_size"]),
        train=train,
        mean=float(normalization["mean"]),
        std=float(normalization["std"]),
        augmentation=data_config.get("augmentation") if train else None,
    )

