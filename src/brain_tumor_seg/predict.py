from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import load_config, project_path
from .model import build_model
from .utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict binary masks for one or more MRI PNGs")
    parser.add_argument("images", nargs="+", help="Input image path(s)")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = select_device(args.device)
    experiment_dir = project_path(config, config["project"]["output_dir"])
    checkpoint_path = (
        project_path(config, args.checkpoint)
        if args.checkpoint
        else experiment_dir / "checkpoints" / "best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    height, width = (int(value) for value in config["data"]["image_size"])
    mean = float(config["data"]["normalization"]["mean"])
    std = float(config["data"]["normalization"]["std"])
    threshold = float(config["metrics"]["threshold"])
    output_dir = project_path(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_value in args.images:
        image_path = Path(image_value)
        with Image.open(image_path) as file:
            image = file.convert("L")
            original_size = image.size
            resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).unsqueeze(0).unsqueeze(0)
        tensor = ((tensor - mean) / std).to(device)
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        probability_image = Image.fromarray(np.round(probability * 255).astype(np.uint8), mode="L")
        probability_image = probability_image.resize(original_size, Image.Resampling.BILINEAR)
        mask = np.asarray(probability_image, dtype=np.uint8) >= round(threshold * 255)
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        probability_path = output_dir / f"{image_path.stem}_probability.png"
        mask_image.save(mask_path)
        probability_image.save(probability_path)
        print(f"{image_path} -> {mask_path}")


if __name__ == "__main__":
    main()

