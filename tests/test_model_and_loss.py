from __future__ import annotations

import torch

from brain_tumor_seg.losses import BCEDiceLoss
from brain_tumor_seg.model import UNet


def test_unet_preserves_spatial_shape() -> None:
    model = UNet(base_channels=4, dropout=0.0)
    output = model(torch.randn(2, 1, 64, 80))
    assert output.shape == (2, 1, 64, 80)


def test_bce_dice_loss_rewards_correct_logits() -> None:
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    good_logits = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    bad_logits = -good_logits
    criterion = BCEDiceLoss(0.5, 0.5)
    assert criterion(good_logits, target) < criterion(bad_logits, target)

