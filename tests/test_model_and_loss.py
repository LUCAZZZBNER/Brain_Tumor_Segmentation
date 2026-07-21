from __future__ import annotations

import torch

from brain_tumor_seg.losses import BCEDiceLoss, build_loss
from brain_tumor_seg.model import UNet


def test_unet_preserves_spatial_shape() -> None:
    model = UNet(base_channels=4, dropout=0.0)
    output = model(torch.randn(2, 1, 64, 80))
    assert output.shape == (2, 1, 64, 80)


def test_vanilla_unet_can_omit_batch_norm_and_dropout() -> None:
    model = UNet(base_channels=4, dropout=0.0, batch_norm=False)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert not any(isinstance(module, torch.nn.Dropout2d) for module in model.modules())


def test_plain_bce_loss_can_be_built() -> None:
    assert isinstance(build_loss({"name": "bce"}), torch.nn.BCEWithLogitsLoss)


def test_bce_dice_loss_rewards_correct_logits() -> None:
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    good_logits = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    bad_logits = -good_logits
    criterion = BCEDiceLoss(0.5, 0.5)
    assert criterion(good_logits, target) < criterion(bad_logits, target)
