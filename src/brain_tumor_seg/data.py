from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
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

            gamma = float(self.augmentation.get("gamma", 0.0))
            if gamma > 0:
                exponent = random.uniform(max(1.0 - gamma, 0.1), 1.0 + gamma)
                lookup = [round(255.0 * ((value / 255.0) ** exponent)) for value in range(256)]
                image = image.point(lookup)

            blur_probability = float(
                self.augmentation.get("gaussian_blur_probability", 0.0)
            )
            if blur_probability > 0 and random.random() < blur_probability:
                blur_radius = float(self.augmentation.get("gaussian_blur_radius", 0.75))
                image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if self.train:
            noise_probability = float(
                self.augmentation.get("gaussian_noise_probability", 0.0)
            )
            if noise_probability > 0 and random.random() < noise_probability:
                noise_std = float(self.augmentation.get("gaussian_noise_std", 0.02))
                noise = np.random.normal(0.0, noise_std, size=image_array.shape).astype(np.float32)
                image_array = np.clip(image_array + noise, 0.0, 1.0)
        # Do not use mask > 0: this dataset stores background as value 3.
        mask_array = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.float32)
        if image_array.ndim == 2:
            image_tensor = torch.from_numpy(image_array.copy()).unsqueeze(0)
        elif image_array.ndim == 3 and image_array.shape[2] == 3:
            image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1)
        else:
            raise ValueError(f"Expected a grayscale or RGB image, got shape {image_array.shape}")
        mask_tensor = torch.from_numpy(mask_array.copy()).unsqueeze(0)
        image_tensor = (image_tensor - self.mean) / self.std
        return image_tensor, mask_tensor


class BrainTumorDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        data_root: str | Path,
        samples: Iterable[Sample],
        transform: SegmentationTransform,
        *,
        channel_mode: str = "grayscale",
    ) -> None:
        self.data_root = Path(data_root)
        self.samples = list(samples)
        self.transform = transform
        self.channel_mode = channel_mode.lower()
        if self.channel_mode not in {"grayscale", "flair_green", "rgb_multimodal"}:
            raise ValueError(f"Unsupported image channel mode: {channel_mode}")
        if not self.samples:
            raise ValueError("Dataset split is empty")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = self.data_root / sample.image_path
        mask_path = self.data_root / sample.mask_path
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            if self.channel_mode == "flair_green":
                # kaggle_3m stores pre-, FLAIR-, and post-contrast MRI in RGB.
                # The green channel is the FLAIR sequence used by the original dataset.
                image = image_file.convert("RGB").getchannel("G")
            elif self.channel_mode == "rgb_multimodal":
                image = image_file.convert("RGB")
            else:
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
            "mask_path": sample.mask_path,
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
