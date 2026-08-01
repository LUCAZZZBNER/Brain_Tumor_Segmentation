from __future__ import annotations

import torch

from brain_tumor_seg.losses import BCEDiceLoss, DiceFocalBoundaryLoss, build_loss
from brain_tumor_seg.model import (
    ASPP,
    ASPPUNet,
    AttentionGate,
    AttentionUNet,
    ResidualAttentionASPPUNet,
    ResidualConvBlock,
    UNet,
    UNetPlusPlus,
    build_model,
)


def test_unet_preserves_spatial_shape() -> None:
    model = UNet(base_channels=4, dropout=0.0)
    output = model(torch.randn(2, 1, 64, 80))
    assert output.shape == (2, 1, 64, 80)


def test_vanilla_unet_can_omit_batch_norm_and_dropout() -> None:
    model = UNet(base_channels=4, dropout=0.0, batch_norm=False)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert not any(isinstance(module, torch.nn.Dropout2d) for module in model.modules())


def test_aspp_unet_preserves_shape_for_a_single_training_sample() -> None:
    model = ASPPUNet(
        base_channels=4,
        dropout=0.0,
        aspp_branch_channels=4,
        aspp_dilation_rates=(1, 2, 4),
    )
    model.train()
    output = model(torch.randn(1, 1, 64, 80))
    assert output.shape == (1, 1, 64, 80)
    assert isinstance(model.aspp, ASPP)
    assert model.aspp.dilation_rates == (1, 2, 4)
    assert len(model.aspp.branches) == 4


def test_aspp_unet_can_be_built_from_config() -> None:
    model = build_model(
        {
            "name": "aspp_unet",
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 4,
            "batch_norm": False,
            "dropout": 0.0,
            "aspp_branch_channels": 4,
            "aspp_dilation_rates": [1, 2, 4, 8],
            "aspp_dropout": 0.0,
        }
    )
    assert isinstance(model, ASPPUNet)
    assert model.aspp.dilation_rates == (1, 2, 4, 8)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())


def test_attention_unet_preserves_spatial_shape_and_has_four_gates() -> None:
    model = AttentionUNet(base_channels=4, dropout=0.0)
    output = model(torch.randn(2, 1, 64, 80))
    assert output.shape == (2, 1, 64, 80)
    assert sum(isinstance(module, AttentionGate) for module in model.modules()) == 4


def test_unet_plus_plus_preserves_spatial_shape_and_has_ten_nested_nodes() -> None:
    model = UNetPlusPlus(base_channels=4, dropout=0.0)
    output = model(torch.randn(2, 1, 64, 80))
    assert output.shape == (2, 1, 64, 80)
    assert len(model.nested_nodes) == 10


def test_residual_attention_aspp_unet_combines_all_three_modules() -> None:
    model = ResidualAttentionASPPUNet(
        base_channels=4,
        dropout=0.0,
        aspp_branch_channels=4,
        aspp_dilation_rates=(1, 2, 4),
    )
    output = model(torch.randn(2, 1, 64, 80))
    assert output.shape == (2, 1, 64, 80)
    assert isinstance(model.aspp, ASPP)
    assert sum(isinstance(module, AttentionGate) for module in model.modules()) == 4
    assert sum(isinstance(module, ResidualConvBlock) for module in model.modules()) == 9


def test_residual_attention_aspp_unet_can_be_built_from_config() -> None:
    model = build_model(
        {
            "name": "res_attention_aspp_unet",
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 4,
            "batch_norm": False,
            "dropout": 0.0,
            "aspp_branch_channels": 4,
            "aspp_dilation_rates": [1, 2, 4],
            "aspp_dropout": 0.0,
        }
    )
    assert isinstance(model, ResidualAttentionASPPUNet)
    assert not any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())


def test_plain_bce_loss_can_be_built() -> None:
    assert isinstance(build_loss({"name": "bce"}), torch.nn.BCEWithLogitsLoss)


def test_bce_dice_loss_rewards_correct_logits() -> None:
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    good_logits = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    bad_logits = -good_logits
    criterion = BCEDiceLoss(0.5, 0.5)
    assert criterion(good_logits, target) < criterion(bad_logits, target)


def test_dice_focal_boundary_loss_rewards_overlap_and_aligned_edges() -> None:
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 5:11] = 1.0
    good_logits = torch.where(target > 0, torch.tensor(8.0), torch.tensor(-8.0))
    bad_logits = -good_logits
    criterion = DiceFocalBoundaryLoss(
        dice_weight=0.55,
        focal_weight=0.30,
        boundary_weight=0.15,
    )
    assert criterion(good_logits, target) < criterion(bad_logits, target)


def test_dice_focal_boundary_loss_can_be_built_from_config() -> None:
    criterion = build_loss(
        {
            "name": "dice_focal_boundary",
            "dice_weight": 0.5,
            "focal_weight": 0.3,
            "boundary_weight": 0.2,
            "focal_alpha": 0.75,
            "focal_gamma": 2.0,
            "boundary_kernel_size": 3,
        }
    )
    assert isinstance(criterion, DiceFocalBoundaryLoss)
